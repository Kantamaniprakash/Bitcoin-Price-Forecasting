"""
Bitcoin Price Forecasting – Main Pipeline
==========================================
Runs the complete analysis end-to-end:

  1. Download historical data (BTC, EUR/USD, Gold, S&P 500, Crude Oil)
  2. Exploratory Data Analysis & correlation analysis
  3. Stationarity testing (ADF)
  4. Granger causality tests
  5. ARIMA model → 30-day forecast + 95% CI
  6. VAR model   → 30-day forecast + 95% CI
  7. Save all plots to results/

Usage
-----
  python main.py

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

from data_collection  import download_data, load_data
from preprocessing    import prepare_var_data, run_adf_tests
from granger_causality import granger_matrix, granger_summary
from arima_model      import fit_arima, forecast_arima
from var_model        import fit_var, forecast_var
from visualization    import (
    plot_prices,
    plot_correlation,
    plot_granger,
    plot_arima_forecast,
    plot_var_forecast,
    plot_combined_forecast,
    plot_return_distribution,
)

DATA_PATH = os.path.join("data", "raw_prices.csv")


def main():
    print("=" * 65)
    print("  Bitcoin Price Forecasting – ARIMA & VAR with Granger Causality")
    print("=" * 65)

    # ── 1. Data ──────────────────────────────────────────────────────────
    print("\n[1/7] Collecting data ...")
    if os.path.exists(DATA_PATH):
        print(f"  Loading cached data from {DATA_PATH}")
        prices = load_data(DATA_PATH)
    else:
        prices = download_data(save_path=DATA_PATH)

    print(f"\n  Date range : {prices.index[0].date()} → {prices.index[-1].date()}")
    print(f"  Shape      : {prices.shape}")
    print(prices.tail(3).to_string())

    # ── 2. EDA ───────────────────────────────────────────────────────────
    print("\n[2/7] Exploratory Data Analysis ...")
    returns = prepare_var_data(prices)

    print("\n  ─── ADF Stationarity Tests on Log-Returns ───")
    adf_summary = run_adf_tests(returns)
    print(adf_summary.to_string())

    plot_prices(prices)
    plot_return_distribution(returns)
    plot_correlation(returns)

    # ── 3. Granger Causality ─────────────────────────────────────────────
    print("\n[3/7] Granger Causality Analysis ...")
    bool_matrix, p_matrix = granger_matrix(returns, max_lag=5)
    gc_summary = granger_summary(returns, max_lag=5)

    print("\n  ─── Granger Causality p-value Matrix (row → column) ───")
    print(p_matrix.to_string())
    print("\n  ─── Significant Relationships (α = 0.05) ───")
    sig = gc_summary[gc_summary["Significant"] == "Yes"]
    print(sig.to_string(index=False) if not sig.empty else "  None found at α=0.05")

    plot_granger(p_matrix)

    # ── 4. ARIMA ─────────────────────────────────────────────────────────
    print("\n[4/7] ARIMA Model ...")
    arima_fitted, arima_order = fit_arima(prices["Bitcoin"])
    arima_fc = forecast_arima(arima_fitted, last_price=prices["Bitcoin"].iloc[-1])

    print("\n  ─── ARIMA 30-Day Forecast (USD) ───")
    print(arima_fc.round(2).to_string())

    plot_arima_forecast(prices, arima_fc)

    # ── 5. VAR ───────────────────────────────────────────────────────────
    print("\n[5/7] VAR Model ...")
    var_fitted, var_lag = fit_var(returns)
    var_fc = forecast_var(var_fitted, returns, prices)

    print("\n  ─── VAR 30-Day Forecast (USD) ───")
    print(var_fc.round(2).to_string())

    plot_var_forecast(prices, var_fc)

    # ── 6. Combined Plot ─────────────────────────────────────────────────
    print("\n[6/7] Generating combined forecast plot ...")
    plot_combined_forecast(prices, arima_fc, var_fc)

    # ── 7. Save Summary CSV ──────────────────────────────────────────────
    print("\n[7/7] Saving forecast summary ...")
    arima_fc_labeled         = arima_fc.copy()
    arima_fc_labeled.columns = [f"ARIMA_{c}" for c in arima_fc.columns]
    var_fc_labeled           = var_fc.copy()
    var_fc_labeled.columns   = [f"VAR_{c}"   for c in var_fc.columns]
    combined = pd.concat([arima_fc_labeled, var_fc_labeled], axis=1)
    out_path = os.path.join("results", "forecast_summary.csv")
    combined.round(2).to_csv(out_path)
    print(f"  Saved → {out_path}")

    print("\n" + "=" * 65)
    print("  All done!  Results saved in the results/ directory.")
    print("=" * 65)

    return arima_fc, var_fc, gc_summary


if __name__ == "__main__":
    main()
