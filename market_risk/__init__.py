"""
Market risk library: VaR, ES, GARCH, and regulatory backtests.

Install: pip install -e ".[data,models,viz,dev]"
"""

from market_risk.backtest import (
    RollingBacktestResult,
    basel_traffic_light,
    christoffersen_conditional_coverage,
    christoffersen_independence,
    kupiec_test,
    mcneil_frey_es_test,
    rolling_var_backtest,
    rolling_var_es_paths,
    summarize_backtest,
)
from market_risk.data import (
    DataSource,
    FredTreasurySource,
    IngestManifest,
    PolygonEquitySource,
    PolygonTreasurySource,
    PortfolioPaths,
    ReturnPanel,
    default_sp500_tickers,
    load_tradingview_csv,
)
from market_risk.portfolio import (
    build_portfolio_panel,
    build_return_panel,
    load_portfolio_log_returns,
    save_panel_outputs,
)
from market_risk.returns import ALPHA, CONFIDENCE, log_var_to_loss, loss_to_log_var
from market_risk.var import historical_var_es, parametric_var_es_normal, parametric_var_es_t

__all__ = [
    "ALPHA",
    "CONFIDENCE",
    "DataSource",
    "FredTreasurySource",
    "IngestManifest",
    "PolygonEquitySource",
    "PolygonTreasurySource",
    "PortfolioPaths",
    "ReturnPanel",
    "RollingBacktestResult",
    "basel_traffic_light",
    "build_portfolio_panel",
    "build_return_panel",
    "christoffersen_conditional_coverage",
    "christoffersen_independence",
    "default_sp500_tickers",
    "historical_var_es",
    "kupiec_test",
    "load_portfolio_log_returns",
    "load_tradingview_csv",
    "log_var_to_loss",
    "loss_to_log_var",
    "mcneil_frey_es_test",
    "parametric_var_es_normal",
    "parametric_var_es_t",
    "rolling_var_backtest",
    "rolling_var_es_paths",
    "save_panel_outputs",
    "summarize_backtest",
]
