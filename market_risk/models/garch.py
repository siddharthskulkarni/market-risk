"""DCC-GARCH (reduced universe) and GARCH-t portfolio VaR/ES."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import scipy as sp

from market_risk.returns import ALPHA


def dcc_reduced_var_es(
    log_df: pd.DataFrame,
    asset_names: list[str] | None = None,
    alpha: float = ALPHA,
    dcc_assets: list[str] | None = None,
) -> dict[str, Any]:
    """
    Fit univariate GARCH-t per asset in reduced universe; DCC-style correlation.
    Returns var_log, es_log, and assets used.
    """
    if dcc_assets is None:
        candidates = ["GLD", "SLV", "1YR", "5YR", "10YR", "A", "MSFT"]
        if asset_names and "MSFT" not in asset_names:
            candidates = ["GLD", "SLV", "1YR", "5YR", "10YR", "A", "ADBE"]
        dcc_assets = [a for a in candidates if a in log_df.columns]
    sub = log_df[dcc_assets].to_numpy()
    from arch import arch_model

    resid = np.zeros_like(sub)
    vols = []
    for j in range(sub.shape[1]):
        am = arch_model(sub[:, j] * 100, mean="Constant", vol="GARCH", p=1, q=1, dist="t")
        res = am.fit(disp="off")
        resid[:, j] = res.resid / res.conditional_volatility
        vols.append(res.conditional_volatility[-1] / 100)
    corr = np.corrcoef(resid.T)
    w = np.ones(len(dcc_assets)) / len(dcc_assets)
    port_vol = float(np.sqrt(w @ (np.diag(vols) @ corr @ np.diag(vols)) @ w))
    port_mean = float(sub.mean(axis=0) @ w)
    var_log = port_mean + sp.stats.t.ppf(alpha, df=8) * port_vol
    tail = sp.stats.t.ppf(alpha, df=8)
    es_factor = -sp.stats.t.pdf(tail, df=8) / alpha * (8 + tail**2) / 7
    es_log = port_mean + es_factor * port_vol
    return {"var_log": float(var_log), "es_log": float(es_log), "assets": dcc_assets}


def garch_t_portfolio_var_es(
    pf_logrets: np.ndarray,
    alpha: float = ALPHA,
) -> dict[str, float]:
    """Univariate GARCH(1,1)-t on equal-weight portfolio log returns."""
    from arch import arch_model

    am = arch_model(np.asarray(pf_logrets).flatten() * 100, mean="Constant", vol="GARCH", p=1, q=1, dist="t")
    garch_res = am.fit(disp="off")
    fcast = garch_res.forecast(horizon=1, reindex=False)
    mean_h = float(garch_res.params.get("mu", garch_res.params.get("Const", 0))) / 100
    vol_h = float(np.sqrt(fcast.variance.values[-1, 0])) / 100
    nu = float(garch_res.params.get("nu", 8))
    var_log = mean_h + sp.stats.t.ppf(alpha, df=nu) * vol_h
    tail = sp.stats.t.ppf(alpha, df=nu)
    es_log = mean_h + (-sp.stats.t.pdf(tail, df=nu) / alpha * (nu + tail**2) / (nu - 1)) * vol_h
    return {"var_log": float(var_log), "es_log": float(es_log)}


def fit_dcc_with_fallback(
    log_df: pd.DataFrame,
    pf_logrets: np.ndarray,
    alpha: float = ALPHA,
) -> dict[str, dict[str, Any]]:
    """Try reduced DCC; always include GARCH-t portfolio fallback."""
    out: dict[str, dict[str, Any]] = {}
    try:
        out["dcc_garch"] = dcc_reduced_var_es(log_df, alpha=alpha)
    except Exception as exc:
        out["dcc_garch_error"] = {"error": str(exc)}
    out["garch_t_portfolio"] = garch_t_portfolio_var_es(pf_logrets, alpha=alpha)
    return out
