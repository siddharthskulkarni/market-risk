"""Value-at-Risk and Expected Shortfall estimators."""

from __future__ import annotations

import numpy as np
import scipy as sp

from market_risk.returns import ALPHA


def historical_var_es(log_returns: np.ndarray, alpha: float = ALPHA) -> tuple[float, float]:
    """Historical simulation VaR/ES on log returns (loss convention)."""
    r = np.asarray(log_returns).flatten()
    q = np.quantile(r, alpha)
    tail = r[r <= q]
    es_log = float(tail.mean()) if len(tail) else q
    return float(q), es_log


def parametric_var_es_normal(log_returns: np.ndarray, alpha: float = ALPHA) -> tuple[float, float]:
    mu, sigma = sp.stats.norm.fit(log_returns)
    var_log = sp.stats.norm.ppf(alpha, loc=mu, scale=sigma)
    es_log = mu - sigma * sp.stats.norm.pdf(sp.stats.norm.ppf(alpha)) / alpha
    return float(var_log), float(es_log)


def parametric_var_es_t(log_returns: np.ndarray, alpha: float = ALPHA) -> tuple[float, float]:
    df, loc, scale = sp.stats.t.fit(log_returns)
    var_log = sp.stats.t.ppf(alpha, df=df, loc=loc, scale=scale)
    q = sp.stats.t.ppf(alpha, df=df)
    es_log = loc - scale * sp.stats.t.pdf(q, df=df) / alpha * (df + q**2) / (df - 1)
    return float(var_log), float(es_log)
