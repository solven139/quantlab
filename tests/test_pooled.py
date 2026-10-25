"""Pooled calibration: recover known shared and per-snapshot parameters from synthetic smiles."""
import numpy as np
import pytest

from quantlab import (BatesModel, MertonModel, black_implied_vol, calibrate, calibrate_pooled,
                      otm_prices)

F = 100.0
KS = np.linspace(75.0, 120.0, 19)


def synthetic_snapshot(model, T, **extra):
    """A snapshot whose market vols are exactly the model's: perfect data, known answer."""
    ivs = black_implied_vol(otm_prices(model, F, KS, T), F, KS, T, KS >= F)
    return dict(F=F, Ks=KS, T=T, market_ivs=ivs, **extra)


def merton(shared, per, snap):
    mu_j, sigma_j = shared
    sigma, lam = per
    return MertonModel(0.0, sigma, lam, mu_j, sigma_j)


MERTON_SHARED = np.array([-0.15, 0.10])
MERTON_PER = np.array([[0.15, 0.5], [0.20, 1.0], [0.30, 2.0], [0.25, 3.0]])
MERTON_T = [0.05, 0.05, 0.10, 0.10]


@pytest.fixture(scope="module")
def merton_fit():
    snaps = [synthetic_snapshot(merton(MERTON_SHARED, per, None), T)
             for per, T in zip(MERTON_PER, MERTON_T)]
    return calibrate_pooled(merton, [-0.25, 0.20], [0.20, 1.0],
                            ([-0.6, 0.01], [0.0, 0.5]), ([0.01, 0.0], [1.0, 20.0]), snaps)


def test_merton_recovers_shared_jump_size(merton_fit):
    assert merton_fit.shared == pytest.approx(MERTON_SHARED, abs=1e-4)


def test_merton_recovers_per_snapshot_parameters(merton_fit):
    assert merton_fit.per == pytest.approx(MERTON_PER, rel=1e-3)
    assert merton_fit.rmse_vol_all < 1e-6


def bates(shared, per, snap):
    kappa, mu_j, sigma_j = shared
    rho, gamma, lam = per
    # v0 is not fitted: it comes from the snapshot (as when it is anchored to realized variance)
    return BatesModel(0.0, kappa, 0.04, gamma, rho, snap["v0"], lam, mu_j, sigma_j)


BATES_SHARED = np.array([3.0, -0.25, 0.15])
BATES_PER = np.array([[-0.7, 1.0, 0.2], [-0.6, 1.5, 0.5], [-0.8, 2.0, 1.0],
                      [-0.7, 2.5, 0.8], [-0.75, 1.2, 0.3]])
BATES_V0 = [0.03, 0.05, 0.12, 0.20, 0.06]
BATES_T = [0.04, 0.04, 0.10, 0.10, 0.10]


def test_bates_with_fixed_v0_recovers_crash_intensity():
    snaps = [synthetic_snapshot(bates(BATES_SHARED, per, dict(v0=v0)), T, v0=v0)
             for per, v0, T in zip(BATES_PER, BATES_V0, BATES_T)]
    fit = calibrate_pooled(bates, [2.0, -0.15, 0.10], [-0.5, 1.0, 0.5],
                           ([0.1, -0.6, 0.01], [20.0, 0.0, 0.5]),
                           ([-0.999, 0.05, 0.0], [0.0, 10.0, 20.0]), snaps)
    assert fit.shared == pytest.approx(BATES_SHARED, rel=1e-2)
    assert fit.per[:, 2] == pytest.approx(BATES_PER[:, 2], rel=1e-2)    # lambda, per snapshot
    assert fit.rmse_vol_all < 1e-5


def test_no_shared_parameters_matches_calibrate():
    """With nothing shared, pooling one snapshot is ordinary calibration."""
    snap = synthetic_snapshot(MertonModel(0.0, 0.2, 1.0, -0.15, 0.1), 0.08)

    def factory(shared, per, s):
        return MertonModel(0.0, *per)

    bounds = ([0.01, 0.0, -0.6, 0.01], [1.0, 20.0, 0.0, 0.5])
    start = [0.15, 2.0, -0.1, 0.15]
    pooled = calibrate_pooled(factory, [], start, ([], []), bounds, [snap])
    single = calibrate(lambda p: MertonModel(0.0, *p), start, bounds, snap["F"], snap["Ks"],
                       snap["T"], snap["market_ivs"])
    assert pooled.rmse_vol[0] == pytest.approx(single.rmse_vol, abs=1e-6)


def test_bad_per0_shape_is_rejected():
    snap = synthetic_snapshot(MertonModel(0.0, 0.2, 1.0, -0.15, 0.1), 0.08)
    with pytest.raises(ValueError):
        calibrate_pooled(merton, [-0.2, 0.1], np.zeros((3, 2)),
                         ([-0.6, 0.01], [0.0, 0.5]), ([0.01, 0.0], [1.0, 20.0]), [snap, snap])