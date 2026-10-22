import numpy as np
import pytest
from quantlab import black_implied_vol, fit_sabr, sabr_implied_vol, sabr_implied_vols


def atm_formula(F, T, alpha, beta, rho, nu):
    # book Ch 4.3.3, the K = F limit of eq. 4.61
    om = 1.0 - beta
    return alpha / F**om * (1 + (om**2 / 24 * alpha**2 / F**(2 * om)
                                 + 0.25 * rho * beta * alpha * nu / F**om
                                 + (2 - 3 * rho**2) / 24 * nu**2) * T)


@pytest.mark.parametrize("beta, alpha", [(1.0, 0.2), (0.5, 2.0), (0.0, 20.0)])
@pytest.mark.parametrize("rho", [-0.7, 0.0, 0.4])
def test_atm_limit(beta, alpha, rho):
    assert sabr_implied_vol(100.0, 100.0, 1.5, alpha, beta, rho, 0.5) == pytest.approx(
        atm_formula(100.0, 1.5, alpha, beta, rho, 0.5), rel=1e-14)


@pytest.mark.parametrize("beta, alpha", [(1.0, 0.2), (0.5, 2.0)])
def test_continuous_through_the_money(beta, alpha):
    atm = sabr_implied_vol(100.0, 100.0, 1.0, alpha, beta, -0.4, 0.6)
    for eps in [1e-10, 1e-8, 1e-6, 1e-4]:
        for K in [100.0 * (1 - eps), 100.0 * (1 + eps)]:
            assert abs(sabr_implied_vol(100.0, K, 1.0, alpha, beta, -0.4, 0.6) - atm) < 5 * eps
    # No jump where the formula switches to its series form near z = 0: on a fine grid
    # through K = F the steps in vol are all equal (the smile's slope), with no outlier.
    Ks = 100.0 * (1 + np.linspace(-1e-5, 1e-5, 201))
    steps = np.diff(sabr_implied_vols(100.0, list(Ks), 1.0, alpha, beta, -0.4, 0.6))
    assert np.max(np.abs(steps - np.median(steps))) < 1e-11


def test_lognormal_limit():
    # beta = 1 and no vol of vol: Black's model with sigma = alpha at every strike
    v = sabr_implied_vols(100.0, [50.0, 80.0, 100.0, 130.0, 200.0], 2.0, 0.25, 1.0, 0.3, 1e-12)
    assert np.allclose(v, 0.25, atol=1e-12)


def test_parameter_effects():
    Ks = [80.0, 100.0, 120.0]
    base = sabr_implied_vols(100.0, Ks, 1.0, 0.2, 1.0, 0.0, 0.5)
    neg = sabr_implied_vols(100.0, Ks, 1.0, 0.2, 1.0, -0.6, 0.5)
    assert base[0] > base[1] < base[2]           # rho = 0: a symmetric-looking smile
    assert neg[0] - neg[2] > base[0] - base[2]   # rho < 0: downward skew


def test_book_example_4_3_3():
    # Book Table 4.1: USD/JPY, T = 1, forward 131.5. Hagan's parametrisation fits the 5 quotes.
    F, Ks = 131.5, np.array([110.0, 121.3, 131.5, 140.9, 151.4])
    ivs = np.array([14.2, 11.8, 10.3, 10.0, 10.7]) / 100
    fit = fit_sabr(F, Ks, 1.0, ivs, beta=1.0)
    assert fit.success and fit.rmse_vol < 0.0002          # < 0.02 vol pts
    assert np.max(np.abs(fit.vols(Ks) - ivs)) < 0.0003


def test_fit_recovers_parameters():
    Ks = np.linspace(70, 140, 25)
    ivs = sabr_implied_vols(100.0, list(Ks), 0.5, 0.22, 1.0, -0.55, 0.9)
    fit = fit_sabr(100.0, Ks, 0.5, ivs, beta=1.0)
    assert (fit.alpha, fit.rho, fit.nu) == pytest.approx((0.22, -0.55, 0.9), rel=1e-6)


@pytest.mark.parametrize("beta, alpha", [(1.0, 0.2), (0.5, 2.0)])
def test_hagan_matches_monte_carlo(beta, alpha):
    # Independent check: simulate the SABR SDE and invert the prices. Hagan's formula is an
    # asymptotic expansion, so agreement is approximate (it degrades as nu^2 T grows).
    F0, T, rho, nu, n, steps = 100.0, 1.0, -0.4, 0.4, 100_000, 200
    rng = np.random.default_rng(0)
    dt = T / steps
    Fp, s = np.full(n, F0), np.full(n, alpha)
    for _ in range(steps):
        z1 = rng.standard_normal(n)
        z2 = rho * z1 + np.sqrt(1 - rho**2) * rng.standard_normal(n)
        if beta == 1.0:
            Fp = Fp * np.exp(s * np.sqrt(dt) * z1 - 0.5 * s * s * dt)
        else:
            Fp = np.maximum(Fp + s * Fp**beta * np.sqrt(dt) * z1, 0.0)
        s = s * np.exp(nu * np.sqrt(dt) * z2 - 0.5 * nu * nu * dt)
    Ks = np.array([85.0, 95.0, 100.0, 105.0, 115.0])
    prices = np.array([np.maximum(K - Fp, 0).mean() if K < F0 else np.maximum(Fp - K, 0).mean()
                       for K in Ks])
    mc = black_implied_vol(prices, F0, Ks, T, Ks >= F0)
    hagan = sabr_implied_vols(F0, list(Ks), T, alpha, beta, rho, nu)
    assert np.max(np.abs(mc - hagan)) < 0.003             # within 0.3 vol pts


def test_invalid_inputs():
    with pytest.raises(ValueError):
        sabr_implied_vol(100.0, 100.0, 1.0, 0.2, 1.0, 1.0, 0.5)     # rho = 1
    with pytest.raises(ValueError):
        sabr_implied_vol(100.0, -5.0, 1.0, 0.2, 1.0, 0.0, 0.5)
