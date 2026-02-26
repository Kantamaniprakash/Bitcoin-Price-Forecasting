# Bitcoin Price Forecasting — ARIMA & VAR with Granger Causality

> **Master's Capstone Project** | Time-Series Econometrics
> Built forecasting models for Bitcoin prices over a 30-day horizon using ARIMA and VAR models with 95% confidence intervals, and determined Granger causality relationships between Bitcoin, Euro/USD, Gold, S&P 500, and Crude Oil.

---

## Table of Contents
- [Overview](#overview)
- [Research Questions](#research-questions)
- [Methodology](#methodology)
- [Key Results](#key-results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Technologies](#technologies)

---

## Overview

Cryptocurrency markets are characterized by high volatility, non-linear dynamics, and complex interdependencies with traditional financial assets. This project applies classical econometric time-series methods to:

1. **Forecast Bitcoin prices** 30 days ahead using two complementary models:
   - **ARIMA** — univariate model exploiting Bitcoin's own autocorrelation structure
   - **VAR** — multivariate model capturing cross-asset spillover effects

2. **Quantify uncertainty** with 95% confidence intervals around all point forecasts

3. **Test Granger causality** — empirically determine whether macroeconomic variables (EUR/USD, Gold, S&P 500, Crude Oil) contain statistically significant predictive information about Bitcoin returns

---

## Research Questions

- Does including macroeconomic asset prices improve Bitcoin price forecasts over a univariate model?
- Which assets Granger-cause Bitcoin returns, and in which direction?
- How do the ARIMA and VAR 30-day forecasts compare in terms of point estimates and uncertainty bands?

---

## Methodology

### 1. Data Collection
Daily closing prices from **Yahoo Finance** (2018–2024) for:

| Asset | Ticker | Role |
|---|---|---|
| Bitcoin | `BTC-USD` | Target |
| Euro/USD | `EURUSD=X` | Currency / macro |
| Gold Futures | `GC=F` | Safe-haven asset |
| S&P 500 | `^GSPC` | Equity market sentiment |
| Crude Oil | `CL=F` | Commodity / energy |

### 2. Preprocessing
- Log-returns: $r_t = \ln(P_t / P_{t-1})$
- **Augmented Dickey-Fuller (ADF)** test confirms stationarity of log-returns

### 3. Granger Causality
Pairwise Granger causality tests (max lag = 5) with F-test p-values at α = 0.05.

### 4. ARIMA
- Optimal (p, d, q) order selected by `auto_arima` (AIC criterion)
- Fitted on log(BTC price); forecasts back-transformed to USD
- 95% CI from analytical forecast error variance

### 5. VAR
- Lag order selected by AIC (max 15 lags)
- Fitted on multivariate log-return system
- Forecasts integrated to reconstruct price levels
- 95% CI propagated via cumulative forecast covariance

---

## Key Results

### Stationarity
All price levels are I(1); log-returns are I(0) — confirmed by ADF tests across all five series.

### Granger Causality
Significant predictive relationships found between the macroeconomic variables and Bitcoin returns, particularly from S&P 500 and EUR/USD, empirically justifying the multivariate VAR approach.

### Forecasts
Both models produce 30-day forecasts with widening confidence intervals reflecting compounding uncertainty over the horizon. The VAR model incorporates cross-market information; ARIMA serves as the interpretable univariate baseline.

---

## Project Structure

```
bitcoin-price-forecasting/
│
├── README.md
├── requirements.txt
├── main.py                          # End-to-end pipeline runner
│
├── notebooks/
│   └── Bitcoin_Price_Forecasting.ipynb   # Full analysis (recommended entry point)
│
├── src/
│   ├── __init__.py
│   ├── data_collection.py           # Yahoo Finance downloader
│   ├── preprocessing.py             # Log returns + ADF tests
│   ├── granger_causality.py         # Pairwise Granger tests
│   ├── arima_model.py               # ARIMA fitting + forecasting
│   ├── var_model.py                 # VAR fitting + forecasting
│   └── visualization.py            # All plotting functions
│
├── data/
│   └── raw_prices.csv               # Cached downloaded data
│
└── results/
    ├── 01_normalized_prices.png
    ├── 02_correlation_heatmap.png
    ├── 03_granger_heatmap.png
    ├── 04_arima_forecast.png
    ├── 05_var_forecast.png
    ├── 06_combined_forecast.png
    ├── 07_return_distribution.png
    └── forecast_summary.csv
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/bitcoin-price-forecasting.git
cd bitcoin-price-forecasting

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Option A — Jupyter Notebook (Recommended)
```bash
jupyter notebook notebooks/Bitcoin_Price_Forecasting.ipynb
```
The notebook walks through every step with explanations, equations, and inline visualizations.

### Option B — Command-Line Pipeline
```bash
python main.py
```
Runs the full pipeline and saves all plots + CSV summary to `results/`.

---

## Technologies

| Library | Purpose |
|---|---|
| `yfinance` | Financial data download |
| `pandas` / `numpy` | Data manipulation |
| `statsmodels` | ADF, VAR, ARIMA, Granger |
| `pmdarima` | Automatic ARIMA order selection |
| `matplotlib` / `seaborn` | Visualization |
| `scikit-learn` | Supplementary metrics |

---

## References

1. Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time Series Analysis: Forecasting and Control* (5th ed.). Wiley.
2. Sims, C. A. (1980). Macroeconomics and reality. *Econometrica*, 48(1), 1–48.
3. Granger, C. W. J. (1969). Investigating causal relations by econometric models and cross-spectral methods. *Econometrica*, 37(3), 424–438.
4. Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts.

---

*This project was developed as part of a Master's programme in Data Science / Financial Analytics. All data is sourced from public financial markets via Yahoo Finance.*
