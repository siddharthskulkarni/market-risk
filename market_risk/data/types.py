"""Immutable domain types for market data and return panels."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ReturnPanel:
    """Aligned multi-asset log-return panel."""

    dates: pd.DatetimeIndex
    asset_names: tuple[str, ...]
    log_returns: np.ndarray  # shape (T, N)

    @property
    def n_days(self) -> int:
        return self.log_returns.shape[0]

    @property
    def n_assets(self) -> int:
        return self.log_returns.shape[1]

    def equal_weight_portfolio(self) -> np.ndarray:
        w = np.ones(self.n_assets) / self.n_assets
        return (self.log_returns @ w.reshape(-1, 1)).flatten()

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(self.log_returns, columns=list(self.asset_names))
        df.insert(0, "date", self.dates.strftime("%Y-%m-%d"))
        return df


@dataclass(frozen=True)
class PortfolioPaths:
    """Output paths for ingest pipeline."""

    stocks_csv: str = "data/portfolio_stocks_2yrs_hist_dret.csv"
    bonds_csv: str = "data/portfolio_bonds_2yrs_hist_dret.csv"
    log_returns_csv: str = "data/portfolio_log_returns_2y.csv"
    pf_log_returns_csv: str = "data/portfolio_pf_log_returns_2y.csv"
    manifest_json: str = "data/data_manifest.json"
    backtest_summary_csv: str = "data/portfolio_backtest_summary.csv"


@dataclass(frozen=True)
class IngestManifest:
    generated_at: str
    start_date: str
    end_date: str
    n_return_days: int
    n_assets: int
    assets: list[str]
    sources: dict[str, str]
    files: dict[str, str]

    def to_dict(self) -> dict:
        return {
            "generated_at": self.generated_at,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "n_return_days": self.n_return_days,
            "n_assets": self.n_assets,
            "assets": self.assets,
            "sources": self.sources,
            "files": self.files,
        }

    @classmethod
    def now(
        cls,
        panel: ReturnPanel,
        paths: PortfolioPaths,
        sources: dict[str, str] | None = None,
    ) -> IngestManifest:
        return cls(
            generated_at=datetime.now().isoformat(),
            start_date=str(panel.dates[0].date()),
            end_date=str(panel.dates[-1].date()),
            n_return_days=panel.n_days,
            n_assets=panel.n_assets,
            assets=list(panel.asset_names),
            sources=sources or {},
            files={
                "stocks": paths.stocks_csv,
                "bonds": paths.bonds_csv,
                "log_returns": paths.log_returns_csv,
                "portfolio_log_returns": paths.pf_log_returns_csv,
            },
        )
