import numpy as np

from market_risk.backtest import kupiec_test, christoffersen_conditional_coverage, basel_traffic_light


def test_kupiec_zero_breaches():
    out = kupiec_test(0, 250, p0=0.01)
    assert out["breaches"] == 0
    assert out["p_value"] > 0


def test_kupiec_expected_breaches():
    out = kupiec_test(3, 300, p0=0.01)
    assert "lr" in out
    assert 0 <= out["p_value"] <= 1


def test_christoffersen_runs():
    ind = np.array([0, 0, 1, 0, 0, 0, 1, 0] * 30)
    cc = christoffersen_conditional_coverage(ind)
    assert "p_value" in cc


def test_basel_traffic_light():
    ind = np.zeros(300, dtype=int)
    ind[:10] = 1
    bl = basel_traffic_light(ind, window=250)
    assert bl["n_windows"] > 0
