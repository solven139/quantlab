import numpy as np
import pytest
from quantlab import (BatesModel, CirScheme, HestonModel, MertonModel, OptionType, cos_price,
                      cos_prices, heston_terminal)

HP = dict(r=0.02, kappa=1.5768, vbar=0.0398, gamma=0.5751, rho=-0.5711, v0=0.0175)
JP = dict(xi=0.5, mu_j=-0.1, sigma_j=0.15)


@pytest.mark.parametrize("u", [0.0, 0.3, 1.0, 7.5, 40.0])
def test_no_jumps_is_heston_exactly(u):
    h = HestonModel(**HP)
    b = BatesModel(**HP, xi=0.0, mu_j=-0.1, sigma_j=0.15)
    assert b.char_fn(u, 1.0) == h.char_fn(u, 1.0)
    assert b.cumulant1(1.0) == h.cumulant1(1.0) and b.cumulant2(1.0) == h.cumulant2(1.0)


def test_no_jumps_prices_equal_heston():
    h = HestonModel(**HP)
    b = BatesModel(**HP, xi=0.0, mu_j=-0.1, sigma_j=0.15)
    Ks = list(np.linspace(70, 130, 13))
    assert np.array_equal(cos_prices(b, OptionType.Put, 100.0, Ks, 1.0, 256, 12.0),
                          cos_prices(h, OptionType.Put, 100.0, Ks, 1.0, 256, 12.0))


def test_characteristic_function_is_a_product():
    h = HestonModel(**HP)
    b = BatesModel(**HP, **JP)
    m = MertonModel(r=0.0, sigma=1e-12, **JP)   # (almost) pure jump part, r = 0
    for u in [0.5, 2.0, 10.0]:
        assert b.char_fn(u, 1.0) == pytest.approx(h.char_fn(u, 1.0) * m.char_fn(u, 1.0), rel=1e-10)


def test_martingale_and_parity():
    b = BatesModel(**HP, **JP)
    # Put-call parity holds only if the drift correction makes S(T) e^{-rT} a martingale
    for K in [80.0, 100.0, 120.0]:
        c = cos_price(b, OptionType.Call, 100.0, K, 1.0, 256, 12.0)
        p = cos_price(b, OptionType.Put, 100.0, K, 1.0, 256, 12.0)
        assert c - p == pytest.approx(100.0 - K * np.exp(-0.02), abs=1e-6)


def test_tiny_vol_of_vol_is_merton():
    # gamma -> 0 with v0 = vbar = sigma^2: Heston becomes Black-Scholes, so Bates becomes Merton
    b = BatesModel(r=0.02, kappa=1.5, vbar=0.04, gamma=1e-4, rho=0.0, v0=0.04, **JP)
    m = MertonModel(r=0.02, sigma=0.2, **JP)
    for K in [70.0, 100.0, 130.0]:
        assert cos_price(b, OptionType.Put, 100.0, K, 1.0, 256, 12.0) == pytest.approx(
            cos_price(m, OptionType.Put, 100.0, K, 1.0, 256, 12.0), abs=1e-6)


def test_cumulants_add():
    b = BatesModel(**HP, **JP)
    h = HestonModel(**HP)
    xi, mu, s = JP["xi"], JP["mu_j"], JP["sigma_j"]
    assert b.cumulant2(2.0) == pytest.approx(h.cumulant2(2.0) + xi * 2.0 * (mu**2 + s**2))
    assert b.cumulant4(2.0) == pytest.approx(2.0 * xi * (mu**4 + 6 * mu**2 * s**2 + 3 * s**4))


def test_matches_monte_carlo():
    # Bates = Heston paths (QE scheme, Step 6) x an independent compound Poisson jump factor
    b = BatesModel(**HP, **JP)
    T, n = 1.0, 400_000
    ST = heston_terminal(CirScheme.QE, 100.0, HP["r"], HP["kappa"], HP["vbar"], HP["gamma"],
                         HP["rho"], HP["v0"], T, 100, n, 5)
    rng = np.random.default_rng(1)
    nj = rng.poisson(JP["xi"] * T, n)
    jumps = nj * JP["mu_j"] + np.sqrt(nj) * JP["sigma_j"] * rng.standard_normal(n)
    ST = ST * np.exp(jumps - b.drift_correction() * T)
    for K in [80.0, 100.0, 120.0]:
        pay = np.exp(-HP["r"] * T) * np.maximum(K - ST, 0.0)
        se = pay.std() / np.sqrt(n)
        assert abs(pay.mean() - cos_price(b, OptionType.Put, 100.0, K, T, 256, 12.0)) < 4 * se


def test_invalid():
    with pytest.raises(ValueError):
        BatesModel(**HP, xi=-1.0, mu_j=0.0, sigma_j=0.1)
