"""
Granger Causality Module
=========================
Tests whether lagged values of each macroeconomic variable (Euro, Gold,
S&P 500, Crude Oil) Granger-cause Bitcoin log-returns, and vice-versa.

A variable X Granger-causes Y if including X's past values significantly
improves the prediction of Y over a model that uses only Y's own past.
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
import warnings

warnings.filterwarnings("ignore")


def granger_matrix(
    returns: pd.DataFrame,
    max_lag: int = 5,
    significance: float = 0.05,
) -> pd.DataFrame:
    """
    Build a pairwise Granger causality matrix.

    Cell [i, j] is True if column i Granger-causes column j
    (i.e., past values of i improve the forecast of j).

    Parameters
    ----------
    returns     : pd.DataFrame   log-return series (stationary)
    max_lag     : int            maximum number of lags to test
    significance: float          α level (default 0.05)

    Returns
    -------
    pd.DataFrame  boolean matrix (True = significant causality)
    """
    cols = returns.columns.tolist()
    n = len(cols)
    matrix = pd.DataFrame(False, index=cols, columns=cols)
    p_matrix = pd.DataFrame(np.nan, index=cols, columns=cols)

    for cause in cols:
        for effect in cols:
            if cause == effect:
                continue
            data = returns[[effect, cause]].dropna()
            try:
                result = grangercausalitytests(data, maxlag=max_lag, verbose=False)
                # Use the minimum p-value across all tested lags (F-test)
                min_p = min(
                    result[lag][0]["ssr_ftest"][1] for lag in range(1, max_lag + 1)
                )
                p_matrix.loc[cause, effect] = round(min_p, 4)
                if min_p < significance:
                    matrix.loc[cause, effect] = True
            except Exception as e:
                print(f"  Granger test failed for {cause} → {effect}: {e}")

    return matrix, p_matrix


def granger_summary(
    returns: pd.DataFrame,
    max_lag: int = 5,
    significance: float = 0.05,
) -> pd.DataFrame:
    """
    Return a tidy DataFrame listing all significant Granger relationships.

    Columns: Cause | Effect | Min_p_value | Significant
    """
    _, p_matrix = granger_matrix(returns, max_lag, significance)
    cols = returns.columns.tolist()
    rows = []
    for cause in cols:
        for effect in cols:
            if cause == effect:
                continue
            p = p_matrix.loc[cause, effect]
            rows.append(
                {
                    "Cause":       cause,
                    "Effect":      effect,
                    "Min_p_value": p,
                    "Significant": "Yes" if p < significance else "No",
                }
            )
    df = pd.DataFrame(rows)
    df = df.sort_values(["Significant", "Min_p_value"], ascending=[False, True])
    return df.reset_index(drop=True)


if __name__ == "__main__":
    from data_collection import download_data
    from preprocessing import prepare_var_data

    prices = download_data()
    returns = prepare_var_data(prices)
    summary = granger_summary(returns)
    print(summary)
