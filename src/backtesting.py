"""
Backtesting Module
=====================================
Walk-forward (rolling-origin) backtesting for the ARIMA and VAR forecasters,
benchmarked against a naive random-walk baseline.

A single train/test split only ever scores a model against one market
regime and is prone to lucky (or unlucky) splits. Walk-forward validation
instead re-fits each model at a sequence of rolling origins across the
backtest window and scores only its true out-of-sample forecast at each
origin — this is the standard now used to report forecast accuracy for
both classical econometric models and time-series foundation models.

Metrics reported per origin, then averaged across origins:
  - MAE / RMSE / MAPE  — standard scale-dependent / percentage errors
  - MASE               — Mean Absolute Scaled Error. Scales MAE by the
                          in-sample naive (last-value) error, giving a
                          scale-free score where MASE < 1 means the model
                          beats a naive random walk and MASE > 1 means it
                          doesn't — the recommended metric for comparing
                          models across series with different volatility.
  - Directional Accuracy — % of origins where the forecast correctly
                          predicted the sign of the price move.
"""

import numpy as np
import pandas as pd
import warnings
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR

from preprocessing import log_returns

warnings.filterwarnings("ignore")


def naive_forecast(history: pd.Series, steps: int) -> np.ndarray:
    """Random-walk baseline: every future step held flat at the last observed price."""
    return np.full(steps, history.iloc[-1])


def naive_mae_in_sample(train_prices: pd.Series) -> float:
    """Mean absolute one-step change over the training window — the MASE scaling factor."""
    return train_prices.diff().abs().mean()


def forecast_errors(
    last_known: float,
    actual: np.ndarray,
    predicted: np.ndarray,
    naive_mae: float,
) -> dict:
    """
    Compute MAE / RMSE / MAPE / MASE / directional accuracy for one origin's
    h-step-ahead forecast.

    Parameters
    ----------
    last_known : float   last observed price before the forecast window
    actual     : ndarray  realized prices over the forecast horizon
    predicted  : ndarray  forecast prices over the same horizon
    naive_mae  : float    in-sample naive MAE used to scale MASE
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    errors = actual - predicted

    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))
    mape = np.mean(np.abs(errors / actual)) * 100
    mase = mae / naive_mae if naive_mae > 0 else np.nan

    actual_change = np.diff(np.concatenate(([last_known], actual)))
    predicted_change = np.diff(np.concatenate(([last_known], predicted)))
    directional_accuracy = np.mean(np.sign(actual_change) == np.sign(predicted_change)) * 100

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "MASE": mase,
        "Directional_Accuracy": directional_accuracy,
    }


def rolling_origins(n_obs: int, min_train: int, horizon: int, step: int):
    """
    Yield expanding-window origin indices for walk-forward validation.
    Each origin trains on rows [0, origin) and forecasts [origin, origin + horizon).
    """
    origin = min_train
    while origin + horizon <= n_obs:
        yield origin
        origin += step


def backtest_naive(
    prices: pd.Series,
    horizon: int = 7,
    min_train_frac: float = 0.8,
    step: int = 14,
) -> pd.DataFrame:
    """Walk-forward evaluation of the naive random-walk baseline."""
    n = len(prices)
    min_train = int(n * min_train_frac)
    rows = []
    for origin in rolling_origins(n, min_train, horizon, step):
        train = prices.iloc[:origin]
        test = prices.iloc[origin: origin + horizon]

        predicted = naive_forecast(train, horizon)
        naive_mae = naive_mae_in_sample(train)
        metrics = forecast_errors(train.iloc[-1], test.values, predicted, naive_mae)
        metrics["Origin_Date"] = train.index[-1]
        rows.append(metrics)

    return pd.DataFrame(rows)


def backtest_arima(
    prices: pd.Series,
    horizon: int = 7,
    min_train_frac: float = 0.8,
    step: int = 14,
    order: tuple = None,
) -> pd.DataFrame:
    """
    Walk-forward evaluation of ARIMA.

    The (p, d, q) order is selected once via auto_arima on the initial
    training window and then held fixed while the model is re-fit at every
    rolling origin — standard practice to keep walk-forward backtests
    computationally tractable without giving later origins an unfair
    "peek" at future data through repeated order search.
    """
    n = len(prices)
    min_train = int(n * min_train_frac)

    if order is None:
        import pmdarima as pm

        auto = pm.auto_arima(
            np.log(prices.iloc[:min_train]),
            start_p=0, start_q=0,
            max_p=5, max_q=5,
            d=None,
            seasonal=False,
            information_criterion="aic",
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            trace=False,
        )
        order = auto.order
        print(f"  ARIMA order fixed for backtest at {order} (selected on initial training window)")

    rows = []
    for origin in rolling_origins(n, min_train, horizon, step):
        train = prices.iloc[:origin]
        test = prices.iloc[origin: origin + horizon]

        fitted = ARIMA(np.log(train), order=order).fit()
        predicted = np.exp(fitted.forecast(steps=horizon).values)

        naive_mae = naive_mae_in_sample(train)
        metrics = forecast_errors(train.iloc[-1], test.values, predicted, naive_mae)
        metrics["Origin_Date"] = train.index[-1]
        rows.append(metrics)

    return pd.DataFrame(rows)


def backtest_var(
    prices: pd.DataFrame,
    horizon: int = 7,
    min_train_frac: float = 0.8,
    step: int = 14,
    maxlags: int = 15,
) -> pd.DataFrame:
    """
    Walk-forward evaluation of VAR.

    Log-returns, lag-order selection (AIC) and the forecast are all
    recomputed from scratch at every rolling origin using only data up to
    that origin, so no future information leaks into earlier forecasts.
    """
    n = len(prices)
    min_train = int(n * min_train_frac)

    rows = []
    for origin in rolling_origins(n, min_train, horizon, step):
        train_prices = prices.iloc[:origin]
        test_prices = prices["Bitcoin"].iloc[origin: origin + horizon]
        train_returns = log_returns(train_prices)

        model = VAR(train_returns)
        try:
            lag = max(model.select_order(maxlags=maxlags).selected_orders["aic"], 1)
        except Exception:
            lag = 1
        fitted = model.fit(lag)

        y_past = train_returns.values[-lag:]
        fc_returns = fitted.forecast(y=y_past, steps=horizon)
        btc_idx = train_returns.columns.tolist().index("Bitcoin")
        cum_returns = np.cumsum(fc_returns[:, btc_idx])

        last_log_price = np.log(train_prices["Bitcoin"].iloc[-1])
        predicted = np.exp(last_log_price + cum_returns)

        naive_mae = naive_mae_in_sample(train_prices["Bitcoin"])
        metrics = forecast_errors(train_prices["Bitcoin"].iloc[-1], test_prices.values, predicted, naive_mae)
        metrics["Origin_Date"] = train_prices.index[-1]
        rows.append(metrics)

    return pd.DataFrame(rows)


def summarize_backtest(results: dict) -> pd.DataFrame:
    """Average each model's per-origin metrics into a single comparison table."""
    rows = []
    for name, df in results.items():
        rows.append(
            {
                "Model": name,
                "N_Origins": len(df),
                "MAE": df["MAE"].mean(),
                "RMSE": df["RMSE"].mean(),
                "MAPE": df["MAPE"].mean(),
                "MASE": df["MASE"].mean(),
                "Directional_Accuracy": df["Directional_Accuracy"].mean(),
            }
        )
    return pd.DataFrame(rows).set_index("Model").sort_values("MASE")


def run_backtest(
    prices: pd.DataFrame,
    horizon: int = 7,
    min_train_frac: float = 0.8,
    step: int = 14,
) -> tuple:
    """
    Run walk-forward backtests for Naive, ARIMA and VAR over the same
    rolling origins and return (per-model detail frames, summary table).
    """
    print(f"  Walk-forward backtest: horizon={horizon}d, step={step}d, min_train_frac={min_train_frac}")

    naive_df = backtest_naive(prices["Bitcoin"], horizon, min_train_frac, step)
    print(f"  Naive baseline : {len(naive_df)} rolling origins evaluated")

    arima_df = backtest_arima(prices["Bitcoin"], horizon, min_train_frac, step)
    print(f"  ARIMA          : {len(arima_df)} rolling origins evaluated")

    var_df = backtest_var(prices, horizon, min_train_frac, step)
    print(f"  VAR            : {len(var_df)} rolling origins evaluated")

    results = {"Naive": naive_df, "ARIMA": arima_df, "VAR": var_df}
    summary = summarize_backtest(results)
    return results, summary


if __name__ == "__main__":
    from data_collection import download_data

    prices = download_data()
    _, summary = run_backtest(prices)
    print("\n  ─── Walk-Forward Backtest Summary (mean across rolling origins) ───")
    print(summary.round(4).to_string())
