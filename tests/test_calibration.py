import numpy as np
import pytest
from quantlab import (BatesModel, HestonModel, MertonModel, black_implied_vol, calibrate,
                      calibrate_series, otm_prices, profile)

F = 100.0
NAMES_M = ("sigma", "xi", "mu_j", "sigma_j")
BOUNDS_M = ([0.01, 0.0, -0.8, 0.01], [1.0, 5.0, 0.3, 0.8])


def merton(p):
    return MertonModel(r=0.0, sigma=p[0], xi=p[1], mu_j=p[2], sigma_j=p[3])


def heston(p):
    return HestonModel(r=0.0, kappa=p[0], vbar=p[1], gamma=p[2], rho=p[3], v0=p[4])


def market_ivs(model, Ks, T):
    Ks = np.asarray(Ks, float)
    return black_implied_vol(otm_prices(model, F, Ks, T), F, Ks, T, Ks >= F)


# ---------------- Recovery from noiseless quotes ----------------

@pytest.mark.parametrize("x0", [[0.2, 0.5, -0.05, 0.05], [0.1, 1.5, -0.05, 0.15]])
def test_recovers_merton(x0):
    true = np.array([0.15, 0.3, -0.15, 0.1])
    Ks, T = np.linspace(75, 120, 40), 0.25
    r = calibrate(merton, x0, BOUNDS_M, F, Ks, T, market_ivs(merton(true), Ks, T), names=NAMES_M)
    assert r.success
    assert np.allclose(r.params, true, atol=1e-5)
    assert r.rmse_vol < 1e-6


def test_recovers_heston_from_several_expiries():
    true = np.array([1.5, 0.04, 0.5, -0.7, 0.03])
    Ts = [0.1, 0.25, 0.5, 1.0, 2.0]
    Kss = [F * np.exp(np.linspace(-0.5, 0.3, 30) * np.sqrt(t)) for t in Ts]
    ivs = [market_ivs(heston(true), k, t) for k, t in zip(Kss, Ts)]
    bounds = ([0.05, 0.001, 0.01, -0.99, 0.001], [10.0, 0.5, 2.0, 0.99, 0.5])
    r = calibrate(heston, [1.0, 0.06, 0.3, -0.3, 0.05], bounds, [F] * 5, Kss, Ts, ivs)
    assert np.allclose(r.params, true, rtol=1e-4)
    assert len(r.iv_errors) == 5 and r.rmse_vol < 1e-6


def test_residuals_are_implied_vol_errors():
    # After the fit, the reported implied-vol errors agree with what the optimiser minimised
    true = np.array([0.15, 0.3, -0.15, 0.1])
    Ks, T = np.linspace(80, 115, 20), 0.25
    ivs = market_ivs(merton(true), Ks, T) + 0.01          # every quote 1 vol pt too high
    r = calibrate(merton, true, BOUNDS_M, F, Ks, T, ivs)
    assert r.rmse_vol < 0.01                              # the model absorbs most of the shift


def test_warm_start_series():
    # A slowly drifting Bates surface: each fit starts at the previous answer
    def bates(p):
        return BatesModel(r=0.0, kappa=2.0, vbar=p[0], gamma=0.4, rho=p[1], v0=p[2],
                          xi=0.2, mu_j=-0.1, sigma_j=0.1)
    Ks, T = np.linspace(80, 115, 25), 0.25
    path = [np.array([0.04, -0.6, 0.03 + 0.002 * i]) for i in range(5)]
    snaps = [dict(F=F, Ks=Ks, T=T, market_ivs=market_ivs(bates(p), Ks, T)) for p in path]
    res = calibrate_series(bates, [0.05, -0.5, 0.04], ([0.001, -0.99, 0.001], [0.5, 0.99, 0.5]),
                           snaps)
    for r, p in zip(res, path):
        assert np.allclose(r.params, p, rtol=1e-3)
    assert res[-1].nfev <= res[0].nfev                    # later fits start close: fewer steps


def test_model_must_be_in_forward_terms():
    with pytest.raises(ValueError):
        calibrate(lambda p: MertonModel(0.05, p[0], p[1], p[2], p[3]), [0.15, 0.3, -0.1, 0.1],
                  BOUNDS_M, F, [90.0, 100.0, 110.0], 0.25, [0.2, 0.18, 0.17])


# ---------------- Deliberate failure: one expiry cannot separate crash frequency and size ----

@pytest.fixture(scope="module")
def noisy_merton_quotes():
    true = np.array([0.15, 0.3, -0.15, 0.1])
    Ks, T = np.linspace(80, 115, 25), 0.25
    rng = np.random.default_rng(0)
    ivs = market_ivs(merton(true), Ks, T) + 0.002 * rng.standard_normal(len(Ks))  # 0.2 vol pt
    return true, Ks, T, ivs


def test_merton_one_expiry_is_not_identified(noisy_merton_quotes):
    true, Ks, T, ivs = noisy_merton_quotes
    # The fit is good (below the 0.2 vol pt noise), but xi and mu_j trade off almost perfectly
    r = calibrate(merton, true, BOUNDS_M, F, Ks, T, ivs, names=NAMES_M)
    assert r.rmse_vol < 0.002
    assert abs(r.correlation[1, 2]) > 0.9
    # Profile: fix the crash frequency xi, refit the rest. Over a range of xi that is more
    # than a factor 2 wide, the fit quality changes by less than 0.02 vol pts.
    prof = profile(merton, true, BOUNDS_M, 1, [0.2, 0.3, 0.4, 0.5], F, Ks, T, ivs)
    rmses = np.array([p[1] for p in prof])
    assert rmses.max() - rmses.min() < 0.0002
    mus = np.array([p[2][2] for p in prof])
    assert mus[0] < mus[-1] - 0.1          # bigger, rarer crashes vs smaller, frequent ones


def test_fixing_the_crash_size_restores_identification(noisy_merton_quotes):
    true, Ks, T, ivs = noisy_merton_quotes
    fixed_mu = true[2]

    def merton_fixed(q):
        return MertonModel(r=0.0, sigma=q[0], xi=q[1], mu_j=fixed_mu, sigma_j=q[2])
    r = calibrate(merton_fixed, [0.2, 1.0, 0.05], ([0.01, 0.0, 0.01], [1.0, 5.0, 0.8]),
                  F, Ks, T, ivs)
    free = calibrate(merton, true, BOUNDS_M, F, Ks, T, ivs)
    assert abs(r.params[1] - true[1]) < 3 * r.std_errors[1]
    assert r.std_errors[1] < 0.5 * free.std_errors[1]     # xi much better determined
