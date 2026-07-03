"""
Bitcoin Price Forecasting – Walk-Forward Backtest
====================================================
Evaluates ARIMA and VAR out-of-sample forecast accuracy using rolling-origin
(walk-forward) validation instead of a single train/test split, benchmarked
against a naive random-walk baseline. Reports RMSE, MAE, MAPE, MASE and
directional accuracy averaged across all rolling origins.

A single train/test split only ever scores a model against one market
regime; walk-forward validation re-fits and re-scores the models across many
origins spanning the backtest window, which is how forecast accuracy is now
reported for both classical econometric models and time-series foundation
models.

Usage
-----
  python backtest.py

Dependencies
------------
  pip install -r requirements.txt
"""

import os
import sys
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

# ── allow imports from src/ ────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_collection import download_data, load_data
from backtesting      import run_backtest
from visualization     import plot_backtest_comparison

DATA_PATH = os.path.join("data", "raw_prices.csv")


def main():
    print("=" * 65)
    print("  Bitcoin Price Forecasting – Walk-Forward Backtest")
    print("=" * 65)

    print("\n[1/3] Loading data ...")
    if os.path.exists(DATA_PATH):
        print(f"  Loading cached data from {DATA_PATH}")
        prices = load_data(DATA_PATH)
    else:
        prices = download_data(save_path=DATA_PATH)

    print("\n[2/3] Running walk-forward backtest (Naive vs ARIMA vs VAR) ...")
    results, summary = run_backtest(prices, horizon=7, min_train_frac=0.8, step=14)

    print("\n  ─── Walk-Forward Backtest Summary (mean across rolling origins) ───")
    print(summary.round(4).to_string())

    print("\n[3/3] Saving results ...")
    plot_backtest_comparison(summary)

    summary_path = os.path.join("results", "backtest_summary.csv")
    summary.round(4).to_csv(summary_path)
    print(f"  Saved → {summary_path}")

    detail = pd.concat(
        {name: df.set_index("Origin_Date") for name, df in results.items()},
        names=["Model", "Origin_Date"],
    )
    detail_path = os.path.join("results", "backtest_detail.csv")
    detail.round(4).to_csv(detail_path)
    print(f"  Saved → {detail_path}")

    print("\n" + "=" * 65)
    print("  Backtest complete. Results saved in the results/ directory.")
    print("=" * 65)

    return results, summary


if __name__ == "__main__":
    main()
