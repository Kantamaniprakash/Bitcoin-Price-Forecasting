# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-07-05

### Added
- ARIMA and VAR time-series models that forecast Bitcoin prices 30 days ahead with 95% confidence intervals, using data downloaded from Yahoo Finance (BTC-USD, EUR/USD, Gold, S&P 500, Crude Oil).
- Pairwise Granger causality testing to determine which macroeconomic assets carry statistically significant predictive information about Bitcoin returns.
- Preprocessing pipeline with log-returns transformation and Augmented Dickey-Fuller (ADF) stationarity tests.
- Walk-forward (rolling-origin) backtesting via `backtest.py`, scoring each model against a naive random-walk baseline on MAE, RMSE, MAPE, MASE, and directional accuracy.
- End-to-end command-line pipeline (`main.py`) and a companion Jupyter notebook that produce all forecast plots and CSV summaries under `results/`.
- Modular `src/` package separating data collection, preprocessing, Granger causality, ARIMA, VAR, backtesting, and visualization.
- CI workflow testing across Python 3.10, 3.11, and 3.12, plus MIT license, Dependabot configuration, and smoke tests.

[0.1.0]: https://github.com/Kantamaniprakash/Bitcoin-Price-Forecasting/releases/tag/v0.1.0
