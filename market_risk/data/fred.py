"""FRED Treasury yield backup."""

from __future__ import annotations

import os

import pandas as pd

from market_risk.data._env import load_project_dotenv
from market_risk.data.base import DataSource, write_yields_csv

FRED_SERIES = {"1YR": "DGS1", "5YR": "DGS5", "10YR": "DGS10"}


class FredTreasurySource(DataSource):
    """DGS1, DGS5, DGS10 from FRED."""

    def __init__(self, series_map: dict[str, str] | None = None, api_key: str | None = None):
        load_project_dotenv()
        self.series_map = series_map or FRED_SERIES
        self.api_key = api_key or os.getenv("FRED_API_KEY")

    def fetch(self, start: str, end: str) -> pd.DataFrame:
        import pandas_datareader as pdr

        rows = []
        kwargs = {"api_key": self.api_key} if self.api_key else {}
        for label, series in self.series_map.items():
            s = pdr.get_data_fred(series, start=start, end=end, **kwargs)
            s = s.reset_index()
            date_col = "DATE" if "DATE" in s.columns else s.columns[0]
            s = s.rename(columns={date_col: "date", series: "rate"})
            if not pd.api.types.is_string_dtype(s["date"]):
                s["date"] = pd.to_datetime(s["date"]).dt.strftime("%Y-%m-%d")
            s["ticker"] = label
            rows.append(s[["date", "ticker", "rate"]])
        out = pd.concat(rows, ignore_index=True)
        return out.dropna(subset=["rate"])


def merge_fred_into_polygon(
    bonds_polygon: pd.DataFrame, bonds_fred: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    """Prefer Polygon; fill rows only in FRED. Returns merged df and stats."""
    stats: dict = {}
    merged = bonds_polygon.merge(
        bonds_fred,
        on=["date", "ticker"],
        how="outer",
        suffixes=("_polygon", "_fred"),
    )
    both = merged.dropna(subset=["rate_polygon", "rate_fred"])
    if len(both):
        diff = (both["rate_polygon"] - both["rate_fred"]).abs()
        stats["overlap"] = len(both)
        stats["max_abs_diff"] = float(diff.max())
    missing_poly = bonds_fred[
        ~bonds_fred.set_index(["date", "ticker"]).index.isin(
            bonds_polygon.set_index(["date", "ticker"]).index
        )
    ]
    if len(missing_poly):
        stats["fred_only_rows"] = len(missing_poly)
        return pd.concat([bonds_polygon, missing_poly], ignore_index=True), stats
    return bonds_polygon, stats
