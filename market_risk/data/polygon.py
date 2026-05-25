"""Polygon.io / Massive API data sources."""

from __future__ import annotations

import os
import time
from datetime import datetime

import pandas as pd
from massive import RESTClient

from market_risk.data._env import load_project_dotenv
from market_risk.data.base import DataSource, write_ohlcv_csv, write_yields_csv

BOND_YIELD_ATTRS = {
    "1YR": "yield_1_year",
    "5YR": "yield_5_year",
    "10YR": "yield_10_year",
}


class PolygonEquitySource(DataSource):
    """Daily adjusted OHLCV for a list of tickers."""

    def __init__(
        self,
        tickers: list[str],
        api_key: str | None = None,
        sleep_every_n: int = 4,
        sleep_seconds: int = 60,
    ):
        load_project_dotenv()
        self.tickers = tickers
        self.api_key = api_key or os.getenv("MASSIVE_API_KEY")
        self.sleep_every_n = sleep_every_n
        self.sleep_seconds = sleep_seconds

    def fetch(self, start: str, end: str) -> pd.DataFrame:
        if not self.api_key:
            raise ValueError("MASSIVE_API_KEY is required for PolygonEquitySource")
        client = RESTClient(api_key=self.api_key)
        rows = []
        for i, ticker in enumerate(self.tickers):
            for a in client.list_aggs(
                ticker, 1, "day", start, end, adjusted="true", sort="asc"
            ):
                date = datetime.fromtimestamp(int(a.timestamp / 1000)).strftime("%Y-%m-%d")
                rows.append([date, ticker, a.open, a.high, a.low, a.close, a.volume])
            if i % self.sleep_every_n == 0 and i != 0:
                time.sleep(self.sleep_seconds)
        df = pd.DataFrame(
            rows, columns=["date", "ticker", "open", "high", "low", "close", "volume"]
        )
        return df.drop_duplicates(subset=["date", "ticker"])

    def fetch_to_csv(self, start: str, end: str, path: str) -> pd.DataFrame:
        df = self.fetch(start, end)
        write_ohlcv_csv(df, path)
        return df


class PolygonTreasurySource(DataSource):
    """Constant-maturity Treasury yields from Polygon."""

    def __init__(self, api_key: str | None = None, bond_map: dict[str, str] | None = None):
        load_project_dotenv()
        self.api_key = api_key or os.getenv("MASSIVE_API_KEY")
        self.bond_map = bond_map or BOND_YIELD_ATTRS

    def fetch(self, start: str, end: str) -> pd.DataFrame:
        if not self.api_key:
            raise ValueError("MASSIVE_API_KEY is required for PolygonTreasurySource")
        client = RESTClient(api_key=self.api_key)
        rows = []
        seen_dates: set[str] = set()
        batch = list(
            client.list_treasury_yields(
                date_gte=start, date_lte=end, limit=50000, sort="date.asc"
            )
        )
        for date_row in batch:
            d = date_row.date
            if d in seen_dates:
                continue
            seen_dates.add(d)
            for label, attr in self.bond_map.items():
                rows.append([d, label, getattr(date_row, attr)])
        return pd.DataFrame(rows, columns=["date", "ticker", "rate"])

    def fetch_to_csv(self, start: str, end: str, path: str) -> pd.DataFrame:
        df = self.fetch(start, end)
        write_yields_csv(df, path)
        return df
