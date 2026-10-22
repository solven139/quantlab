import numpy as np
import pytest
from quantlab import (HestonModel, black_implied_vol, black_price, cos_density, implied_density,
                      otm_ivs_from_quotes, otm_prices)

F, T = 100.0, 0.5
H = HestonModel(r=0.0, kappa=1.5, vbar=0.04, gamma=0.5, rho=-0.7, v0=0.03)
GRID = np.linspace(30.0, 200.0, 3401)


def heston_ivs(Ks):
    Ks = np.asarray(Ks, float)
    return black_implied_vol(otm_prices(H, F, Ks, T), F, Ks, T, Ks >= F)


def heston_density(K):
    # cos_density is the density of X = log(S(T)/S0); change variables to S(T) = F e^X
    return np.array(cos_density(H, T, list(np.log(K / F)), 256, 12.0)) / K


@pytest.mark.parametrize("Ks", [np.linspace(55, 150, 40), np.linspace(75, 125, 25)])
def test_recovers_heston_density(Ks):
    d = implied_density(F, Ks, heston_ivs(Ks), T, K_grid=GRID)
    ref = heston_density(d.K)
    inside = (d.K >= Ks[0]) & (d.K <= Ks[-1])
    assert np.max(np.abs(d.density - ref)[inside]) < 0.005 * ref.max()
    assert np.trapezoid(np.abs(d.density - ref), d.K) < 0.005


def test_density_integrates_to_one_and_has_mean_F():
    Ks = np.linspace(55, 150, 40)
    d = implied_density(F, Ks, heston_ivs(Ks), T, K_grid=GRID)
    assert d.mass == pytest.approx(1.0, abs=1e-3)
    assert d.mean == pytest.approx(F, rel=1e-3)           # E[S(T)] = forward
    assert d.density.min() >= -1e-6
    assert d.cdf()[-1] == pytest.approx(d.mass)


def test_quotes_to_ivs_with_discounting():
    Ks = np.linspace(70, 130, 13)
    ivs, DF = heston_ivs(Ks), 0.97
    calls = DF * black_price(F, Ks, T, ivs, True)
    puts = DF * black_price(F, Ks, T, ivs, False)
    assert np.allclose(otm_ivs_from_quotes(F, DF, Ks, calls, puts, T), ivs, atol=1e-9)


def test_noise_needs_smoothing():
    # 0.1 vol pt of noise on the quotes: an interpolating spline turns it into wild,
    # negative "probabilities"; a smoothing spline does not.
    Ks = np.linspace(60, 145, 35)
    noisy = heston_ivs(Ks) + 0.001 * np.random.default_rng(3).standard_normal(len(Ks))
    raw = implied_density(F, Ks, noisy, T, K_grid=GRID, smoothing=0.0)
    smooth = implied_density(F, Ks, noisy, T, K_grid=GRID, smoothing=len(Ks) * 0.001**2)
    ref = heston_density(GRID)
    err_raw = np.trapezoid(np.abs(raw.density - ref), GRID)
    err_smooth = np.trapezoid(np.abs(smooth.density - ref), GRID)
    assert raw.density.min() < 0.0
    assert err_smooth < 0.5 * err_raw


def test_sabr_smoothing_runs():
    Ks = np.linspace(70, 130, 25)
    d = implied_density(F, Ks, heston_ivs(Ks), T, method="sabr", K_grid=GRID)
    assert d.mass == pytest.approx(1.0, abs=5e-3) and d.mean == pytest.approx(F, rel=5e-3)
