"""Optional disk cache wrapper (placeholder for future use)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from market_risk.data.base import DataSource


class CachedDataSource(DataSource):
    """Return cached CSV if present and fresh; otherwise delegate to inner source."""

    def __init__(self, inner: DataSource, cache_path: Path, force_refresh: bool = False):
        self.inner = inner
        self.cache_path = Path(cache_path)
        self.force_refresh = force_refresh

    def fetch(self, start: str, end: str) -> Any:
        if not self.force_refresh and self.cache_path.is_file():
            import pandas as pd

            return pd.read_csv(self.cache_path)
        result = self.inner.fetch(start, end)
        if hasattr(result, "to_csv"):
            result.to_csv(self.cache_path, index=False)
        return result
