"""CLI entry point for market data ingest."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta

from market_risk.data._env import load_project_dotenv
from market_risk.data.base import default_sp500_tickers
from market_risk.data.fred import FredTreasurySource, merge_fred_into_polygon
from market_risk.data.polygon import PolygonEquitySource, PolygonTreasurySource
from market_risk.data.types import PortfolioPaths
from market_risk.portfolio import build_return_panel, save_panel_outputs


def main() -> None:
    load_project_dotenv()
    parser = argparse.ArgumentParser(description="Fetch market data and build portfolio panel")
    parser.add_argument("--lookback", type=int, default=720, help="Calendar days lookback")
    parser.add_argument("--skip-fetch", action="store_true", help="Only rebuild panel from cached CSVs")
    parser.add_argument("--fred-backup", action="store_true", help="Merge FRED yields into bond file")
    args = parser.parse_args()

    paths = PortfolioPaths()
    today = datetime.today()
    start = (today - timedelta(days=args.lookback)).strftime("%Y-%m-%d")
    end = today.strftime("%Y-%m-%d")

    import pandas as pd

    if not args.skip_fetch:
        stock_tickers = default_sp500_tickers(15)
        print(f"Fetching equities {start} -> {end}")
        PolygonEquitySource(stock_tickers).fetch_to_csv(start, end, paths.stocks_csv)
        print("Fetching Treasury yields")
        bonds = PolygonTreasurySource().fetch(start, end)
        if args.fred_backup:
            try:
                fred = FredTreasurySource().fetch(start, end)
                bonds, stats = merge_fred_into_polygon(bonds, fred)
                print(f"FRED merge stats: {stats}")
            except Exception as e:
                print(f"FRED backup skipped: {e}")
        bonds.to_csv(paths.bonds_csv, index=False)
    else:
        stocks_df = pd.read_csv(paths.stocks_csv)
        stock_tickers = sorted(
            t for t in stocks_df["ticker"].unique() if t not in ("1YR", "5YR", "10YR", "ticker")
        )

    panel = build_return_panel(
        pd.read_csv(paths.stocks_csv),
        pd.read_csv(paths.bonds_csv),
        stock_tickers,
    )
    manifest = save_panel_outputs(
        panel,
        paths,
        sources={"equities": "polygon_massive_adjusted", "bonds": "polygon_treasury_fred_backup"},
    )
    print(f"Panel: {panel.n_days} days x {panel.n_assets} assets")
    print(f"Manifest: {paths.manifest_json} ({manifest.start_date} .. {manifest.end_date})")


if __name__ == "__main__":
    main()
