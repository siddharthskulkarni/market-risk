"""Deprecated: use `market_risk` package. This shim will be removed in a future release."""

import warnings

warnings.warn(
    "risk_utils is deprecated; import from market_risk instead.",
    DeprecationWarning,
    stacklevel=2,
)

from market_risk.backtest import (  # noqa: F401
    basel_traffic_light,
    christoffersen_conditional_coverage,
    christoffersen_independence,
    kupiec_test,
    mcneil_frey_es_test,
)
from market_risk.data.tradingview import load_tradingview_csv  # noqa: F401
from market_risk.returns import ALPHA, CONFIDENCE, log_var_to_loss, loss_to_log_var  # noqa: F401
from market_risk.var import (  # noqa: F401
    historical_var_es,
    parametric_var_es_normal,
    parametric_var_es_t,
)
