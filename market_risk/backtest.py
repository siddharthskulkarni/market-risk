"""VaR backtests: Kupiec, Christoffersen, Basel, rolling OOS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import scipy as sp

from market_risk.returns import ALPHA
from market_risk.var import historical_var_es, parametric_var_es_normal, parametric_var_es_t

RollingMethod = Literal["historical", "t", "garch", "normal"]


def kupiec_test(breaches: int, n: int, p0: float = ALPHA) -> dict:
    """Kupiec (1995) unconditional coverage LR test."""
    if n <= 0:
        return {"lr": np.nan, "p_value": np.nan, "reject_5pct": False, "breaches": breaches, "n": n}
    y = int(breaches)
    p_hat = y / n
    if y == 0:
        lr = -2 * n * np.log(1 - p0)
    elif y == n:
        lr = -2 * n * np.log(p0)
    else:
        num = (p0**y) * ((1 - p0) ** (n - y))
        den = (p_hat**y) * ((1 - p_hat) ** (n - y))
        lr = -2 * np.log(num / den)
    p_value = 1 - sp.stats.chi2.cdf(lr, df=1)
    return {
        "lr": float(lr),
        "p_value": float(p_value),
        "reject_5pct": bool(p_value < 0.05),
        "breaches": y,
        "n": n,
        "rate": float(p_hat),
    }


def christoffersen_independence(breach_ind: np.ndarray) -> dict:
    """Christoffersen (1998) independence test on breach indicator series."""
    ind = np.asarray(breach_ind, dtype=int).flatten()
    n = len(ind)
    if n < 2:
        return {"lr": np.nan, "p_value": np.nan, "reject_5pct": False}

    n00 = n01 = n10 = n11 = 0
    for t in range(1, n):
        i0, i1 = ind[t - 1], ind[t]
        if i0 == 0 and i1 == 0:
            n00 += 1
        elif i0 == 0 and i1 == 1:
            n01 += 1
        elif i0 == 1 and i1 == 0:
            n10 += 1
        else:
            n11 += 1

    pi0 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi1 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    def _ll(n0, n1, p):
        if n0 + n1 == 0:
            return 0.0
        if p <= 0:
            return n0 * np.log(1 - 1e-16)
        if p >= 1:
            return n1 * np.log(1e-16)
        return n0 * np.log(1 - p) + n1 * np.log(p)

    lr_ind = -2 * (
        _ll(n00, n01, pi0) + _ll(n10, n11, pi1) - _ll(n00 + n10, n01 + n11, pi)
    )
    p_value = 1 - sp.stats.chi2.cdf(lr_ind, df=1)
    return {"lr": float(lr_ind), "p_value": float(p_value), "reject_5pct": bool(p_value < 0.05)}


def christoffersen_conditional_coverage(breach_ind: np.ndarray, p0: float = ALPHA) -> dict:
    """Joint unconditional + independence test."""
    ind = np.asarray(breach_ind, dtype=int).flatten()
    kup = kupiec_test(int(ind.sum()), len(ind), p0=p0)
    indep = christoffersen_independence(ind)
    if np.isnan(kup["lr"]) or np.isnan(indep["lr"]):
        return {
            "lr": np.nan,
            "p_value": np.nan,
            "reject_5pct": False,
            "kupiec": kup,
            "independence": indep,
        }
    lr_cc = kup["lr"] + indep["lr"]
    p_value = 1 - sp.stats.chi2.cdf(lr_cc, df=2)
    return {
        "lr": float(lr_cc),
        "p_value": float(p_value),
        "reject_5pct": bool(p_value < 0.05),
        "kupiec": kup,
        "independence": indep,
    }


def basel_traffic_light(breach_ind: np.ndarray, window: int = 250) -> dict:
    """Basel traffic-light zones on rolling windows of VaR exceptions (99%)."""
    ind = np.asarray(breach_ind, dtype=int).flatten()
    if len(ind) < window:
        return {"green_pct": np.nan, "yellow_pct": np.nan, "red_pct": np.nan, "n_windows": 0}

    zones = []
    for start in range(0, len(ind) - window + 1):
        exc = int(ind[start : start + window].sum())
        if exc < 5:
            zones.append("green")
        elif exc <= 9:
            zones.append("yellow")
        else:
            zones.append("red")
    n_w = len(zones)
    return {
        "green_pct": zones.count("green") / n_w,
        "yellow_pct": zones.count("yellow") / n_w,
        "red_pct": zones.count("red") / n_w,
        "n_windows": n_w,
    }


def mcneil_frey_es_test(
    log_returns: np.ndarray, var_log: float, es_log: float, alpha: float = ALPHA
) -> dict:
    """McNeil-Frey style: average exceedance vs ES on VaR breach days."""
    r = np.asarray(log_returns).flatten()
    breaches = r <= var_log
    n_b = int(breaches.sum())
    if n_b == 0:
        return {"n_breaches": 0, "avg_exceedance": np.nan, "es_violation_rate": np.nan}
    avg_exc = float(r[breaches].mean())
    return {
        "n_breaches": n_b,
        "avg_exceedance": avg_exc,
        "es_violation_rate": float(avg_exc < es_log),
    }


def _rolling_var_es_one(
    sample: np.ndarray,
    method: RollingMethod,
    alpha: float,
) -> tuple[float, float]:
    if method == "historical":
        return historical_var_es(sample, alpha)
    if method == "t":
        return parametric_var_es_t(sample, alpha)
    if method == "garch":
        try:
            from arch import arch_model

            am = arch_model(sample * 100, mean="Constant", vol="GARCH", p=1, q=1, dist="t")
            res = am.fit(disp="off")
            fc = res.forecast(horizon=1, reindex=False)
            mean_h = float(res.params.get("mu", res.params.get("Const", 0))) / 100
            vol_h = float(np.sqrt(fc.variance.values[-1, 0])) / 100
            nu = float(res.params.get("nu", 8))
            v = mean_h + sp.stats.t.ppf(alpha, df=nu) * vol_h
            tail = sp.stats.t.ppf(alpha, df=nu)
            e = mean_h + (-sp.stats.t.pdf(tail, df=nu) / alpha * (nu + tail**2) / (nu - 1)) * vol_h
            return float(v), float(e)
        except Exception:
            return parametric_var_es_t(sample, alpha)
    return parametric_var_es_normal(sample, alpha)


def rolling_var_es_paths(
    log_returns: np.ndarray,
    method: RollingMethod,
    est_window: int = 252,
    alpha: float = ALPHA,
) -> tuple[np.ndarray, np.ndarray]:
    """Rolling 1-step-ahead VaR/ES paths (NaN before est_window)."""
    r = np.asarray(log_returns).flatten()
    n = len(r)
    var_path = np.full(n, np.nan)
    es_path = np.full(n, np.nan)
    for t in range(est_window, n):
        v, e = _rolling_var_es_one(r[t - est_window : t], method, alpha)
        var_path[t] = v
        es_path[t] = e
    return var_path, es_path


@dataclass(frozen=True)
class RollingBacktestResult:
    method: str
    var_log: np.ndarray
    es_log: np.ndarray
    breach: np.ndarray
    dates: np.ndarray | None


def summarize_backtest(
    log_returns: np.ndarray,
    var_log: np.ndarray,
    es_log: np.ndarray,
    method: str,
) -> dict:
    """Kupiec, Christoffersen, Basel, McNeil-Frey for one OOS series."""
    r = np.asarray(log_returns).flatten()
    v = np.asarray(var_log).flatten()
    valid = ~np.isnan(v)
    realized = r[valid]
    vv = v[valid]
    ee = np.asarray(es_log).flatten()[valid]
    breach = realized <= vv
    kup = kupiec_test(int(breach.sum()), len(breach))
    cc = christoffersen_conditional_coverage(breach)
    bl = basel_traffic_light(breach)
    mf = mcneil_frey_es_test(realized, float(np.median(vv)), float(np.median(ee)))
    return {
        "method": method,
        "kupiec_lr": kup["lr"],
        "kupiec_p": kup["p_value"],
        "kupiec_reject": kup["reject_5pct"],
        "cc_lr": cc["lr"],
        "cc_p": cc["p_value"],
        "cc_reject": cc["reject_5pct"],
        "basel_green_pct": bl["green_pct"],
        "basel_yellow_pct": bl["yellow_pct"],
        "basel_red_pct": bl["red_pct"],
        "es_violation_rate": mf["es_violation_rate"],
        "n_oos": len(breach),
        "breach_rate": float(breach.mean()),
    }


def rolling_var_backtest(
    log_returns: np.ndarray,
    methods: list[RollingMethod] | None = None,
    est_window: int = 252,
    dates: np.ndarray | None = None,
    alpha: float = ALPHA,
) -> tuple[dict[str, RollingBacktestResult], pd.DataFrame]:
    """Run rolling OOS backtests for multiple VaR methods."""
    if methods is None:
        methods = ["historical", "t", "garch"]
    r = np.asarray(log_returns).flatten()
    results: dict[str, RollingBacktestResult] = {}
    rows = []
    for method in methods:
        var_path, es_path = rolling_var_es_paths(r, method, est_window, alpha)
        valid = ~np.isnan(var_path)
        breach = r[valid] <= var_path[valid]
        d = dates[valid] if dates is not None else None
        results[method] = RollingBacktestResult(
            method=method,
            var_log=var_path[valid],
            es_log=es_path[valid],
            breach=breach,
            dates=d,
        )
        rows.append(summarize_backtest(r[valid], var_path[valid], es_path[valid], method))
    return results, pd.DataFrame(rows)
