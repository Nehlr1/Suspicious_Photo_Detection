import numpy as np

from outlet_audit.calibrate import fit_lr


def _model(prior=0.05, seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.normal(0.7, 0.05, 2000)
    neg = rng.normal(0.2, 0.08, 2000)
    return fit_lr(pos, neg, prior=prior)


def test_posterior_bounds_and_separation():
    m = _model()
    x = np.linspace(0.0, 1.0, 101)
    p = m.posterior(x)
    assert ((p >= 0) & (p <= 1)).all()
    assert m.posterior(0.2) > 0.9 and m.posterior(0.7) < 0.1


def test_log_lr_monotone_between_modes():
    m = _model()
    x = np.linspace(0.2, 0.7, 51)
    d = np.diff(m.log_lr(x))
    assert (d <= 1e-9).all()


def test_floor_between_modes_and_prior_effect():
    m = _model()
    floor = m.solve_floor(0.5)
    assert 0.2 < floor < 0.7
    m2 = _model(prior=0.5)
    assert (m2.posterior(np.array([0.3, 0.45, 0.6])) > m.posterior(np.array([0.3, 0.45, 0.6]))).all()


def test_extra_log_lr_shifts_posterior():
    m = _model()
    assert m.posterior(0.45, extra_log_lr=3.0) > m.posterior(0.45) > m.posterior(0.45, extra_log_lr=-3.0)
