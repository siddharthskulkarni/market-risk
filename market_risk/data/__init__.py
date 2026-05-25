"""Market data ingestion."""

from market_risk.data.base import DataSource, default_sp500_tickers
from market_risk.data.fred import FredTreasurySource, merge_fred_into_polygon
from market_risk.data.polygon import PolygonEquitySource, PolygonTreasurySource
from market_risk.data.tradingview import TradingViewCsvSource, load_tradingview_csv
from market_risk.data.types import IngestManifest, PortfolioPaths, ReturnPanel

__all__ = [
    "DataSource",
    "default_sp500_tickers",
    "FredTreasurySource",
    "merge_fred_into_polygon",
    "PolygonEquitySource",
    "PolygonTreasurySource",
    "TradingViewCsvSource",
    "load_tradingview_csv",
    "IngestManifest",
    "PortfolioPaths",
    "ReturnPanel",
]
