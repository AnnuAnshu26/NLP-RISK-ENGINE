"""Price loader: yfinance (live) with an offline synthetic fallback in /data."""
import logging
from typing import List

import pandas as pd

from src.config import PRICES_FILE, TICKERS

log = logging.getLogger(__name__)


def load_offline_prices() -> pd.DataFrame:
    df = pd.read_csv(PRICES_FILE, index_col="date", parse_dates=True)
    return df


def load_prices(tickers: List[str] | None = None, start: str | None = None, end: str | None = None,
                live: bool = False) -> pd.DataFrame:
    """Daily close prices (rows = dates, cols = tickers).

    live=True pulls from Yahoo Finance via yfinance; on any failure (offline, rate-limit,
    empty frame) we log a warning and fall back to the bundled synthetic prices so the
    demo never breaks.
    """
    tickers = tickers or TICKERS
    if live:
        try:
            import yfinance as yf
            df = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)["Close"]
            df = df.dropna(how="all")
            if not df.empty and set(tickers) <= set(df.columns):
                df.index = pd.to_datetime(df.index).tz_localize(None)
                return df[tickers].ffill()
            log.warning("yfinance returned incomplete data; using offline prices")
        except Exception as e:
            log.warning("yfinance failed (%s); using offline prices", e)
    return load_offline_prices()[tickers]
