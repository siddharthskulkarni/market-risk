"""Return conventions and VaR loss mapping."""

from __future__ import annotations

import numpy as np

ALPHA = 0.01
CONFIDENCE = 1 - ALPHA


def log_var_to_loss(var_log: float) -> float:
    """Map log-return VaR threshold to positive loss fraction."""
    return float(-(np.exp(var_log) - 1))


def loss_to_log_var(loss: float) -> float:
    """Map positive loss fraction to log-return VaR threshold."""
    return float(np.log(1 - loss))


def stock_gross_returns(prices: np.ndarray) -> np.ndarray:
    """close[t] / close[t-1] along axis 0."""
    return prices[1:, :] / prices[:-1, :]


def bond_gross_returns(rates: np.ndarray) -> np.ndarray:
    """Yield change factor: ((Δy)/100) + 1."""
    return ((rates[1:, :] - rates[:-1, :]) / 100) + 1


def gross_to_log_returns(rets: np.ndarray) -> np.ndarray:
    return np.log(rets)
