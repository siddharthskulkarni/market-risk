"""Matplotlib figures for notebooks (optional [viz] extra)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def portfolio_eda_figure(pf_logrets: np.ndarray, dates, figsize=(12, 4)):
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    ax1.hist(pf_logrets, bins=50)
    ax1.set_title("Equal-weight portfolio log returns")
    ax2.plot(dates, pf_logrets)
    fig.autofmt_xdate()
    plt.tight_layout()
    return fig


def portfolio_distribution_figure(pf_logrets: np.ndarray, save_path: str | Path | None = None):
    import matplotlib.pyplot as plt
    import scipy as sp

    mu, std = sp.stats.norm.fit(pf_logrets)
    df_t, loc, scale = sp.stats.t.fit(pf_logrets)
    kde = sp.stats.gaussian_kde(np.asarray(pf_logrets).flatten())
    x = np.linspace(pf_logrets.min(), pf_logrets.max(), 200)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    for ax, pdf, title in [
        (ax1, sp.stats.norm.pdf(x, mu, std), "Normal"),
        (ax2, sp.stats.t.pdf(x, df_t, loc, scale), "Student-t"),
        (ax3, kde(x), "KDE"),
    ]:
        ax.hist(pf_logrets, bins=50, density=True, alpha=0.6)
        ax.plot(x, pdf, "k")
        ax.set_title(title)
    ax4.set_visible(False)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def rolling_var_figure(dates, realized_loss_pct, var_loss_pct, title: str = "Rolling OOS VaR", save_path=None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dates, realized_loss_pct, alpha=0.5, label="Realized loss %")
    ax.plot(dates, var_loss_pct, color="r", label="Rolling 99% VaR %")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def gld_returns_figure(dates, prices, rets, logrets, figsize=(20, 10)):
    import matplotlib.pyplot as plt

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    ax1.plot(dates, prices)
    ax1.set_title("GLD Closing Prices")
    ax2.plot(dates[1:], rets)
    ax2.set_title("GLD Daily Returns")
    ax3.plot(dates[1:], logrets)
    ax3.set_title("GLD Daily Log Returns")
    ax4.hist(logrets, bins=100)
    ax4.set_title("GLD Log Returns Histogram")
    plt.tight_layout()
    return fig
