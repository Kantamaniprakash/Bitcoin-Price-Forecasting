"""
VAR Forecasting Module
=======================
Fits a Vector Autoregression (VAR) model to the multivariate log-return
system [Bitcoin, Euro, Gold, S&P500, CrudeOil], selects the optimal lag
order via AIC, then forecasts the next 30 days of Bitcoin log-returns and
converts them back to price levels with a 95 % confidence interval.
"""

import numpy as np
import pandas as pd
import warnings
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")


def fit_var(
    returns: pd.DataFrame,
    maxlags: int = 15,
    ic: str = "aic",
) -> tuple:
    """
    Fit a VAR(p) model to the system of log-returns.

    Parameters
    ----------
    returns : pd.DataFrame  stationary log-return series
    maxlags : int           maximum lag to consider
    ic      : str           information criterion ('aic', 'bic', 'hqic')

    Returns
    -------
    (VARResultsWrapper, int)  fitted model, selected lag order
    """
    model = VAR(returns)
    lag_results = model.select_order(maxlags=maxlags)
    selected_lag = lag_results.selected_orders[ic]
    print(f"  VAR lag order selected by {ic.upper()}: {selected_lag}")

    fitted = model.fit(selected_lag)
    print(fitted.summary())
    return fitted, selected_lag


def forecast_var(
    fitted_model,
    returns: pd.DataFrame,
    prices: pd.DataFrame,
    steps: int = 30,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Forecast the next `steps` days using the fitted VAR model,
    then reconstruct Bitcoin price levels from log-returns.

    The confidence interval is built by propagating the VAR forecast
    covariance matrix through the cumulative-sum back-transformation.

    Parameters
    ----------
    fitted_model : VARResultsWrapper
    returns      : pd.DataFrame  full log-return history (for the y_past window)
    prices       : pd.DataFrame  full price history (for the last BTC price)
    steps        : int
    alpha        : float

    Returns
    -------
    pd.DataFrame  [Forecast, Lower_CI, Upper_CI]  in USD price levels
    """
    lag = fitted_model.k_ar
    y_past = returns.values[-lag:]            # shape (lag, k)

    # Point forecast (log-return space)
    fc_array = fitted_model.forecast(y=y_past, steps=steps)  # (steps, k)
    fc_df = pd.DataFrame(fc_array, columns=returns.columns)

    # Forecast covariance for confidence intervals
    fc_cov = fitted_model.forecast_cov(steps=steps)          # (steps, k, k)

    btc_idx = returns.columns.tolist().index("Bitcoin")
    z = 1.959964                                              # 97.5th percentile → 95% CI

    # Cumulative sum of log-returns → log-price forecast
    last_log_price = np.log(prices["Bitcoin"].iloc[-1])
    cum_returns    = fc_df["Bitcoin"].cumsum().values

    # Standard error grows with cumulative sum (assumes independence across steps)
    # Σ of cumulative return variance = Σ_{t=1}^{h} Var[r_t]
    step_var   = np.array([fc_cov[t][btc_idx, btc_idx] for t in range(steps)])
    cum_var    = np.cumsum(step_var)
    cum_se     = np.sqrt(cum_var)

    log_fc     = last_log_price + cum_returns
    log_lower  = log_fc - z * cum_se
    log_upper  = log_fc + z * cum_se

    # Back-transform
    last_date    = returns.index[-1]
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1), periods=steps, freq="D"
    )

    result = pd.DataFrame(
        {
            "Forecast": np.exp(log_fc),
            "Lower_CI": np.exp(log_lower),
            "Upper_CI": np.exp(log_upper),
        },
        index=future_dates,
    )
    result.index.name = "Date"
    return result


if __name__ == "__main__":
    from data_collection import download_data
    from preprocessing import prepare_var_data

    prices  = download_data()
    returns = prepare_var_data(prices)
    fitted, lag = fit_var(returns)
    fc = forecast_var(fitted, returns, prices)
    print(fc)
