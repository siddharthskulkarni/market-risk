import numpy as np
import pandas as pd

from market_risk.portfolio import build_return_panel


def _mini_stocks():
    dates = pd.date_range("2024-01-02", periods=5, freq="B")
    rows = []
    for t in ["AAA", "BBB"]:
        for i, d in enumerate(dates):
            rows.append([d.strftime("%Y-%m-%d"), t, 100 + i, 101, 99, 100 + i, 1e6])
    return pd.DataFrame(rows, columns=["date", "ticker", "open", "high", "low", "close", "volume"])


def _mini_bonds():
    dates = pd.date_range("2024-01-02", periods=5, freq="B")
    rows = []
    for t in ["1YR", "5YR", "10YR"]:
        for i, d in enumerate(dates):
            rows.append([d.strftime("%Y-%m-%d"), t, 4.0 + 0.01 * i])
    return pd.DataFrame(rows, columns=["date", "ticker", "rate"])


def test_build_return_panel_shape():
    panel = build_return_panel(_mini_stocks(), _mini_bonds(), ["AAA", "BBB"])
    assert panel.n_assets == 5
    assert panel.n_days == 4
    assert panel.log_returns.shape == (4, 5)
    pf = panel.equal_weight_portfolio()
    assert len(pf) == 4
    assert not np.isnan(pf).any()
