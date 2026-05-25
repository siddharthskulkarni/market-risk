import numpy as np

from market_risk.returns import ALPHA, log_var_to_loss
from market_risk.var import historical_var_es, parametric_var_es_t


def test_log_var_to_loss_roundtrip():
    v = -0.02
    loss = log_var_to_loss(v)
    assert loss > 0


def test_historical_var_es_tail():
    rng = np.random.default_rng(42)
    r = rng.normal(-0.0005, 0.01, 500)
    var_log, es_log = historical_var_es(r, ALPHA)
    assert es_log <= var_log or np.isclose(var_log, es_log)


def test_parametric_t_var_finite():
    rng = np.random.default_rng(0)
    r = rng.standard_t(5, 300) * 0.01
    var_log, es_log = parametric_var_es_t(r, ALPHA)
    assert np.isfinite(var_log) and np.isfinite(es_log)
    assert es_log <= var_log
