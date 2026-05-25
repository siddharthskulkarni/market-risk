"""TradingView CSV normalization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from market_risk.data.base import DataSource


def load_tradingview_csv(path: str | Path, ticker: str) -> pd.DataFrame:
    """Normalize TradingView export to long OHLCV schema."""
    df = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date")
    close_col = "Close/Last" if "Close/Last" in df.columns else "Close"
    return pd.DataFrame(
        {
            "date": df["Date"].dt.strftime("%Y-%m-%d"),
            "ticker": ticker,
            "open": df.get("Open", df[close_col]),
            "high": df.get("High", df[close_col]),
            "low": df.get("Low", df[close_col]),
            "close": df[close_col],
            "volume": df.get("Volume", np.nan),
        }
    )


class TradingViewCsvSource(DataSource):
    """Load a single-symbol TradingView daily export."""

    def __init__(self, path: str | Path, ticker: str):
        self.path = Path(path)
        self.ticker = ticker

    def fetch(self, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        df = load_tradingview_csv(self.path, self.ticker)
        if start:
            df = df[df["date"] >= start]
        if end:
            df = df[df["date"] <= end]
        return df
