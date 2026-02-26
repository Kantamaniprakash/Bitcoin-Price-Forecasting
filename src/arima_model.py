"""
ARIMA Forecasting Module
=========================
Fits an ARIMA(p,d,q) model to Bitcoin log-prices (levels) using
pmdarima's auto_arima for automatic order selection, then forecasts
the next 30 days with a 95 % confidence interval.

The forecast is converted back to price levels for interpretability.
"""

import numpy as np
import pandas as pd
import warnings
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")


def fit_arima(
    bitcoin_prices: pd.Series,
    seasonal: bool = False,
) -> tuple:
    """
    Fit ARIMA to log(Bitcoin price) using auto_arima order selection.

    Parameters
    ----------
    bitcoin_prices : pd.Series  raw BTC-USD closing prices
    seasonal       : bool       whether to include seasonal component

    Returns
    -------
    (fitted_model, (p, d, q))
    """
    log_prices = np.log(bitcoin_prices)

    print("  Running auto_arima order selection ...")
    auto = pm.auto_arima(
        log_prices,
        start_p=0, start_q=0,
        max_p=5,   max_q=5,
        d=None,
        seasonal=seasonal,
        information_criterion="aic",
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        trace=False,
    )
    order = auto.order
    print(f"  Best ARIMA order selected: {order}")

    # Refit with statsmodels for richer in-sample diagnostics / CI
    model = ARIMA(log_prices, order=order)
    fitted = model.fit()
    print(fitted.summary())
    return fitted, order


def forecast_arima(
    fitted_model,
    last_price: float,
    steps: int = 30,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Produce an h-step-ahead forecast with (1-alpha) confidence interval,
    then back-transform from log-space to USD price levels.

    Parameters
    ----------
    fitted_model : statsmodels ARIMA result
    last_price   : float  last observed BTC price (for display only)
    steps        : int    forecast horizon (default 30 days)
    alpha        : float  significance level (default 0.05 → 95% CI)

    Returns
    -------
    pd.DataFrame  columns: [Forecast, Lower_CI, Upper_CI]
    """
    forecast_obj = fitted_model.get_forecast(steps=steps)
    mean_log = forecast_obj.predicted_mean
    ci_log   = forecast_obj.conf_int(alpha=alpha)

    # Back-transform
    forecast_price = np.exp(mean_log)
    lower_price    = np.exp(ci_log.iloc[:, 0])
    upper_price    = np.exp(ci_log.iloc[:, 1])

    # Build a date index starting the day after the last observed date
    last_date = fitted_model.model.data.dates[-1]
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1), periods=steps, freq="D"
    )

    result = pd.DataFrame(
        {
            "Forecast":  forecast_price.values,
            "Lower_CI":  lower_price.values,
            "Upper_CI":  upper_price.values,
        },
        index=future_dates,
    )
    result.index.name = "Date"
    return result


if __name__ == "__main__":
    from data_collection import download_data

    prices = download_data()
    btc = prices["Bitcoin"]
    fitted, order = fit_arima(btc)
    fc = forecast_arima(fitted, last_price=btc.iloc[-1])
    print(fc)
