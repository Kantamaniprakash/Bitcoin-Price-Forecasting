"""
Preprocessing & Stationarity Module
=====================================
- Computes log returns
- Runs Augmented Dickey-Fuller (ADF) tests
- Differencing utilities
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller


def log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert price levels to log returns: r_t = ln(P_t / P_{t-1}).
    First row will be NaN and is dropped.
    """
    return np.log(df / df.shift(1)).dropna()


def adf_test(series: pd.Series, label: str = "") -> dict:
    """
    Augmented Dickey-Fuller test for stationarity.

    Returns a dict with test stat, p-value, critical values, and verdict.
    """
    result = adfuller(series.dropna(), autolag="AIC")
    verdict = "Stationary" if result[1] < 0.05 else "Non-Stationary"
    out = {
        "Series":       label or series.name,
        "ADF Statistic": round(result[0], 4),
        "p-value":       round(result[1], 4),
        "1%":            round(result[4]["1%"], 4),
        "5%":            round(result[4]["5%"], 4),
        "10%":           round(result[4]["10%"], 4),
        "Result":        verdict,
    }
    return out


def run_adf_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Run ADF on every column and return a summary DataFrame."""
    rows = [adf_test(df[col], col) for col in df.columns]
    return pd.DataFrame(rows).set_index("Series")


def prepare_var_data(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Return stationary log-return series ready for VAR estimation.
    Drops the first NaN row introduced by differencing.
    """
    returns = log_returns(prices)
    adf_summary = run_adf_tests(returns)
    non_stat = adf_summary[adf_summary["Result"] == "Non-Stationary"]
    if not non_stat.empty:
        print("WARNING: the following log-return series are still non-stationary:")
        print(non_stat)
    return returns


if __name__ == "__main__":
    from data_collection import download_data
    prices = download_data()
    returns = prepare_var_data(prices)
    print(run_adf_tests(returns))
