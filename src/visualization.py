"""
Visualization Module
=====================
All plotting functions for the Bitcoin Price Forecasting project.
Uses matplotlib / seaborn for static publication-quality figures.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

sns.set_theme(style="darkgrid", palette="muted")
SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(SAVE_DIR, exist_ok=True)


def _save(fig, name: str):
    path = os.path.join(SAVE_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close(fig)


# ── 1. Historical Prices ─────────────────────────────────────────────────────

def plot_prices(prices: pd.DataFrame, save: bool = True):
    """Plot normalized price series (rebased to 100) on a single axis."""
    rebased = prices / prices.iloc[0] * 100
    fig, ax = plt.subplots(figsize=(14, 5))
    for col in rebased.columns:
        ax.plot(rebased.index, rebased[col], label=col, linewidth=1.3)
    ax.set_title("Normalized Asset Prices (Base = 100, Jan 2018)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Rebased Price (USD)")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    fig.tight_layout()
    if save:
        _save(fig, "01_normalized_prices.png")
    return fig


# ── 2. Correlation Heat-map ───────────────────────────────────────────────────

def plot_correlation(returns: pd.DataFrame, save: bool = True):
    """Correlation heat-map of log-return series."""
    corr = returns.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
        linewidths=0.5, ax=ax, vmin=-1, vmax=1,
        annot_kws={"size": 11},
    )
    ax.set_title("Pearson Correlation – Log Returns", fontsize=13, fontweight="bold")
    fig.tight_layout()
    if save:
        _save(fig, "02_correlation_heatmap.png")
    return fig


# ── 3. Granger Causality Heat-map ────────────────────────────────────────────

def plot_granger(p_matrix: pd.DataFrame, alpha: float = 0.05, save: bool = True):
    """
    Heat-map of Granger p-values.
    Cells below alpha are highlighted (significant causality).
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    annot = p_matrix.applymap(lambda x: f"{x:.3f}" if pd.notna(x) else "–")
    cmap  = sns.diverging_palette(10, 130, as_cmap=True)
    sns.heatmap(
        p_matrix.astype(float), annot=annot, fmt="", cmap=cmap,
        linewidths=0.5, ax=ax, vmin=0, vmax=0.2,
        annot_kws={"size": 9},
    )
    ax.set_title(
        f"Granger Causality p-values  (α = {alpha})  –  Row Granger-causes Column",
        fontsize=11, fontweight="bold",
    )
    ax.set_xlabel("Effect")
    ax.set_ylabel("Cause")
    fig.tight_layout()
    if save:
        _save(fig, "03_granger_heatmap.png")
    return fig


# ── 4. ARIMA Forecast ────────────────────────────────────────────────────────

def plot_arima_forecast(
    prices: pd.DataFrame,
    arima_fc: pd.DataFrame,
    history_days: int = 90,
    save: bool = True,
):
    """Plot ARIMA 30-day forecast with 95 % confidence band."""
    btc = prices["Bitcoin"].iloc[-history_days:]
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(btc.index, btc, color="#1f77b4", linewidth=1.5, label="Historical BTC Price")
    ax.plot(arima_fc.index, arima_fc["Forecast"], color="darkorange",
            linewidth=2, label="ARIMA Forecast (30-day)")
    ax.fill_between(
        arima_fc.index,
        arima_fc["Lower_CI"],
        arima_fc["Upper_CI"],
        color="darkorange", alpha=0.25, label="95% Confidence Interval",
    )
    ax.axvline(btc.index[-1], color="grey", linestyle="--", linewidth=1, label="Forecast Start")
    ax.set_title("Bitcoin Price – ARIMA 30-Day Forecast", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    fig.tight_layout()
    if save:
        _save(fig, "04_arima_forecast.png")
    return fig


# ── 5. VAR Forecast ──────────────────────────────────────────────────────────

def plot_var_forecast(
    prices: pd.DataFrame,
    var_fc: pd.DataFrame,
    history_days: int = 90,
    save: bool = True,
):
    """Plot VAR 30-day forecast with 95 % confidence band."""
    btc = prices["Bitcoin"].iloc[-history_days:]
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(btc.index, btc, color="#1f77b4", linewidth=1.5, label="Historical BTC Price")
    ax.plot(var_fc.index, var_fc["Forecast"], color="green",
            linewidth=2, label="VAR Forecast (30-day)")
    ax.fill_between(
        var_fc.index,
        var_fc["Lower_CI"],
        var_fc["Upper_CI"],
        color="green", alpha=0.20, label="95% Confidence Interval",
    )
    ax.axvline(btc.index[-1], color="grey", linestyle="--", linewidth=1, label="Forecast Start")
    ax.set_title("Bitcoin Price – VAR 30-Day Forecast", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    fig.tight_layout()
    if save:
        _save(fig, "05_var_forecast.png")
    return fig


# ── 6. Combined Forecast Comparison ──────────────────────────────────────────

def plot_combined_forecast(
    prices: pd.DataFrame,
    arima_fc: pd.DataFrame,
    var_fc: pd.DataFrame,
    history_days: int = 90,
    save: bool = True,
):
    """Overlay ARIMA and VAR forecasts for easy comparison."""
    btc = prices["Bitcoin"].iloc[-history_days:]
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(btc.index, btc, color="#1f77b4", linewidth=1.5, label="Historical BTC Price")

    ax.plot(arima_fc.index, arima_fc["Forecast"], color="darkorange",
            linewidth=2, linestyle="-", label="ARIMA Forecast")
    ax.fill_between(arima_fc.index, arima_fc["Lower_CI"], arima_fc["Upper_CI"],
                    color="darkorange", alpha=0.20, label="ARIMA 95% CI")

    ax.plot(var_fc.index, var_fc["Forecast"], color="green",
            linewidth=2, linestyle="--", label="VAR Forecast")
    ax.fill_between(var_fc.index, var_fc["Lower_CI"], var_fc["Upper_CI"],
                    color="green", alpha=0.15, label="VAR 95% CI")

    ax.axvline(btc.index[-1], color="grey", linestyle=":", linewidth=1, label="Forecast Start")
    ax.set_title("Bitcoin Price – ARIMA vs VAR 30-Day Forecast Comparison",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    fig.tight_layout()
    if save:
        _save(fig, "06_combined_forecast.png")
    return fig


# ── 7. Log-Return Distribution ────────────────────────────────────────────────

def plot_return_distribution(returns: pd.DataFrame, save: bool = True):
    """KDE + histogram of Bitcoin daily log-returns."""
    fig, ax = plt.subplots(figsize=(10, 4))
    btc_ret = returns["Bitcoin"]
    sns.histplot(btc_ret, kde=True, bins=80, color="#1f77b4", ax=ax, stat="density")
    ax.axvline(btc_ret.mean(), color="red",    linestyle="--", linewidth=1.5, label=f"Mean: {btc_ret.mean():.4f}")
    ax.axvline(btc_ret.std(),  color="orange", linestyle="--", linewidth=1.5, label=f"Std:  {btc_ret.std():.4f}")
    ax.set_title("Bitcoin Daily Log-Return Distribution", fontsize=13, fontweight="bold")
    ax.set_xlabel("Log Return")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    if save:
        _save(fig, "07_return_distribution.png")
    return fig
