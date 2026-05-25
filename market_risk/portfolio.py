"""Portfolio return panel construction and persistence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from market_risk.data.base import parse_dates
from market_risk.data.types import IngestManifest, PortfolioPaths, ReturnPanel
from market_risk.returns import bond_gross_returns, gross_to_log_returns, stock_gross_returns

BOND_TICKERS = ("1YR", "5YR", "10YR")


def build_return_panel(
    stocks_df: pd.DataFrame,
    bonds_df: pd.DataFrame,
    stock_tickers: list[str],
    bond_tickers: tuple[str, ...] = BOND_TICKERS,
) -> ReturnPanel:
    """Align calendars and build (T-1) x N log-return matrix."""
    stocks_df = parse_dates(stocks_df)
    bonds_df = parse_dates(bonds_df)
    dates = sorted(set(stocks_df["date"].unique()) & set(bonds_df["date"].unique()))
    asset_names = tuple(list(stock_tickers) + list(bond_tickers))

    stocks_df = stocks_df[stocks_df["date"].isin(dates)]
    cols = []
    for ticker in stock_tickers:
        col = stocks_df[stocks_df["ticker"] == ticker].sort_values("date")["close"].to_numpy()
        cols.append(col)
    stock_prices = np.stack(cols, axis=1)
    stock_rets = stock_gross_returns(stock_prices)

    bonds_df = bonds_df[bonds_df["date"].isin(dates)]
    cols = []
    for ticker in bond_tickers:
        col = bonds_df[bonds_df["ticker"] == ticker].sort_values("date")["rate"].to_numpy()
        cols.append(col)
    bond_rates = np.stack(cols, axis=1)
    bond_rets = bond_gross_returns(bond_rates)

    rets = np.hstack([stock_rets, bond_rets])
    logrets = gross_to_log_returns(rets)
    return_dates = pd.DatetimeIndex(dates)[1:]

    if np.isnan(logrets).any():
        nan_pct = np.isnan(logrets).mean(axis=0)
        for name, pct in zip(asset_names, nan_pct):
            if pct > 0.05:
                raise ValueError(f"{name} missing {pct*100:.1f}% of return days")

    return ReturnPanel(
        dates=return_dates,
        asset_names=asset_names,
        log_returns=logrets,
    )


def build_portfolio_panel(
    stocks_csv: str | Path,
    bonds_csv: str | Path,
    stock_tickers: list[str] | None = None,
    bond_tickers: tuple[str, ...] = BOND_TICKERS,
) -> ReturnPanel:
    """Load raw CSVs and build aligned return panel."""
    stocks_df = pd.read_csv(stocks_csv)
    bonds_df = pd.read_csv(bonds_csv)
    if stock_tickers is None:
        stock_tickers = sorted(
            t for t in stocks_df["ticker"].unique() if t not in bond_tickers and t != "ticker"
        )
    return build_return_panel(stocks_df, bonds_df, stock_tickers, bond_tickers)


def save_panel_outputs(
    panel: ReturnPanel,
    paths: PortfolioPaths | None = None,
    sources: dict[str, str] | None = None,
) -> IngestManifest:
    """Write log-return CSVs, portfolio series, and manifest."""
    paths = paths or PortfolioPaths()
    panel.to_dataframe().to_csv(paths.log_returns_csv, index=False)
    pf = panel.equal_weight_portfolio()
    pd.DataFrame(
        {"date": panel.dates.strftime("%Y-%m-%d"), "log_return": pf}
    ).to_csv(paths.pf_log_returns_csv, index=False)

    manifest = IngestManifest.now(panel, paths, sources=sources)
    Path(paths.manifest_json).parent.mkdir(parents=True, exist_ok=True)
    with open(paths.manifest_json, "w") as f:
        json.dump(manifest.to_dict(), f, indent=2)
    return manifest


def load_portfolio_log_returns(path: str | Path) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Load equal-weight portfolio log returns from CSV."""
    df = pd.read_csv(path, parse_dates=["date"])
    return df["log_return"].to_numpy(), pd.DatetimeIndex(df["date"])
