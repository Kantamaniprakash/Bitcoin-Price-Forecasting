"""
Data Collection Module
======================
Downloads historical price data for Bitcoin and macroeconomic variables:
  - Bitcoin (BTC-USD)
  - Euro/USD exchange rate (EURUSD=X)
  - Gold Futures (GC=F)
  - S&P 500 Index (^GSPC)
  - Crude Oil Futures (CL=F)
"""

import yfinance as yf
import pandas as pd
import os


TICKERS = {
    "Bitcoin":   "BTC-USD",
    "Euro":      "EURUSD=X",
    "Gold":      "GC=F",
    "SP500":     "^GSPC",
    "CrudeOil":  "CL=F",
}

START_DATE = "2018-01-01"
END_DATE   = "2024-12-31"


def download_data(
    tickers: dict = TICKERS,
    start: str = START_DATE,
    end: str = END_DATE,
    save_path: str = None,
) -> pd.DataFrame:
    """
    Download adjusted closing prices for all tickers and merge into one DataFrame.

    Parameters
    ----------
    tickers : dict  {label: yfinance_symbol}
    start   : str   start date  (YYYY-MM-DD)
    end     : str   end date    (YYYY-MM-DD)
    save_path : str optional CSV output path

    Returns
    -------
    pd.DataFrame  daily closing prices, forward-filled, dropna
    """
    frames = {}
    for label, symbol in tickers.items():
        print(f"  Downloading {label} ({symbol}) ...")
        raw = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
        if raw.empty:
            print(f"    WARNING: no data returned for {symbol}")
            continue
        # yfinance may return MultiIndex columns – flatten
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        frames[label] = raw["Close"].rename(label)

    df = pd.concat(frames.values(), axis=1)
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"

    # Forward-fill weekends / holidays, then drop any remaining NaN rows
    df = df.ffill().dropna()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path)
        print(f"  Data saved to {save_path}")

    print(f"  Dataset shape: {df.shape}  |  {df.index[0].date()} → {df.index[-1].date()}")
    return df


def load_data(csv_path: str) -> pd.DataFrame:
    """Load a previously saved CSV."""
    df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
    return df


if __name__ == "__main__":
    df = download_data(save_path="../data/raw_prices.csv")
    print(df.tail())
