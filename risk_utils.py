"""Shared risk metrics and backtests for baseline.ipynb and workflow.ipynb."""

from __future__ import annotations

import numpy as np
import scipy as sp


ALPHA = 0.01
CONFIDENCE = 1 - ALPHA


def log_var_to_loss(var_log: float) -> float:
    """Map log-return VaR threshold to positive loss fraction."""
    return float(-(np.exp(var_log) - 1))


def loss_to_log_var(loss: float) -> float:
    """Map positive loss fraction to log-return VaR threshold."""
    return float(np.log(1 - loss))


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


def load_tradingview_csv(path: str, ticker: str) -> "pd.DataFrame":
    """Normalize TradingView export to long OHLCV schema."""
    import pandas as pd

    df = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date")
    close_col = "Close/Last" if "Close/Last" in df.columns else "Close"
    out = pd.DataFrame(
        {
            "date": df["Date"].dt.strftime("%Y-%m-%d"),
            "ticker": ticker,
            "open": df.get("Open", df[close_col]),
            "high": df.get("High", df[close_col]),
            "low": df.get("Low", df[close_col]),
            "close": df[close_col],
            "volume": df.get("Volume", np.nan),
        }
    )
    return out
