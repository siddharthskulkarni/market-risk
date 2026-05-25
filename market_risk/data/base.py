"""DataSource ABC and CSV helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd


class DataSource(ABC):
    """Pluggable market data fetcher returning a domain object or DataFrame."""

    @abstractmethod
    def fetch(self, start: str, end: str) -> Any:
        """Fetch data for [start, end] inclusive (YYYY-MM-DD strings)."""


def write_ohlcv_csv(df: pd.DataFrame, path: str | Path) -> Path:
    """Write standard stock schema: date,ticker,open,high,low,close,volume."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def write_yields_csv(df: pd.DataFrame, path: str | Path) -> Path:
    """Write standard bond schema: date,ticker,rate."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def parse_dates(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    out = df.copy()
    out[col] = pd.to_datetime(out[col])
    return out


def default_sp500_tickers(n_equity: int = 15) -> list[str]:
    """First n symbols from Wikipedia S&P 500 table + GLD, SLV."""
    from io import StringIO

    import requests

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; market-risk/0.2)",
        "X-Requested-With": "XMLHttpRequest",
    }
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    table = pd.read_html(StringIO(r.text))
    tickers = list(table[0]["Symbol"])
    return tickers[:n_equity] + ["GLD", "SLV"]
