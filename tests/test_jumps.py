import math

import numpy as np
import pytest
from quantlab import (BlackScholesModel, KouModel, MertonModel, OptionType,
                      VarianceGammaModel, bs_price, cos_price, simulate_merton)

S0, T, r = 100.0, 1.0, 0.05
MODELS = {
    "merton": MertonModel(r=r, sigma=0.15, xi=0.5, mu_j=-0.1, sigma_j=0.15),
    "kou": KouModel(r=r, sigma=0.15, xi=0.8, p=0.35, eta1=12.0, eta2=8.0),
    "vg": VarianceGammaModel(r=r, sigma=0.2, theta=-0.15, beta=0.2),
}


def merton_series(cp, K, sigma, xi, mu_j, sigma_j, n_terms=60):
    # Merton (1976) closed form: Poisson-weighted sum of Black-Scholes prices (book Ch 5.2.1)
    k = math.exp(mu_j + 0.5 * sigma_j**2) - 1.0
    lam = xi * (1.0 + k)
    total = 0.0
    for n in range(n_terms):
        sig_n = math.sqrt(sigma**2 + n * sigma_j**2 / T)
        r_n = r - xi * k + n * math.log(1.0 + k) / T
        w = math.exp(-lam * T) * (lam * T) ** n / math.factorial(n)
        total += w * bs_price(cp, S0, K, T, r_n, sig_n)
    return total


@pytest.mark.parametrize("name", list(MODELS))
def test_char_fn_is_one_at_zero(name):
    assert MODELS[name].char_fn(0.0, T) == pytest.approx(1.0, abs=1e-14)


@pytest.mark.parametrize("name", list(MODELS))
@pytest.mark.parametrize("K", [80.0, 100.0, 120.0])
def test_put_call_parity_needs_the_drift_correction(name, K):
    # C - P = S0 - K e^{-rT} only holds if the discounted stock is a martingale,
    # which is exactly what the drift correction omega guarantees.
    # L = 12: Kou's exponential tails are fatter than normal ones, so the default
    # L = 8 range misses ~4e-8 of value. Fat tails need a wider integration range.
    m = MODELS[name]
    c = cos_price(m, OptionType.Call, S0, K, T, N=512, L=12.0)
    p = cos_price(m, OptionType.Put, S0, K, T, N=512, L=12.0)
    assert c - p == pytest.approx(S0 - K * math.exp(-r * T), abs=1e-8)


@pytest.mark.parametrize("cp", [OptionType.Call, OptionType.Put])
@pytest.mark.parametrize("K", [70.0, 90.0, 100.0, 110.0, 140.0])
def test_merton_cos_matches_closed_form_series(cp, K):
    exact = merton_series(cp, K, 0.15, 0.5, -0.1, 0.15)
    assert cos_price(MODELS["merton"], cp, S0, K, T, N=256) == pytest.approx(exact, abs=1e-9)


def test_merton_without_jumps_is_black_scholes():
    m = MertonModel(r=r, sigma=0.2, xi=0.0, mu_j=-0.1, sigma_j=0.15)
    assert cos_price(m, OptionType.Call, S0, 105.0, T) == pytest.approx(
        bs_price(OptionType.Call, S0, 105.0, T, r, 0.2), abs=1e-10)


def test_vg_tends_to_black_scholes_for_small_beta():
    # beta -> 0: the Gamma clock becomes deterministic, VG becomes Brownian motion
    m = VarianceGammaModel(r=r, sigma=0.2, theta=0.0, beta=1e-6)
    assert cos_price(m, OptionType.Call, S0, 100.0, T) == pytest.approx(
        bs_price(OptionType.Call, S0, 100.0, T, r, 0.2), abs=1e-3)


def test_kou_cos_matches_monte_carlo():
    rng = np.random.default_rng(0)
    n, sigma, xi, p, e1, e2 = 400_000, 0.15, 0.8, 0.35, 12.0, 8.0
    omega = MODELS["kou"].drift_correction()
    X = (r - omega - 0.5 * sigma**2) * T + sigma * math.sqrt(T) * rng.standard_normal(n)
    nj = rng.poisson(xi * T, n)
    n_up = rng.binomial(nj, p)
    n_dn = nj - n_up
    # a sum of m Exponential(eta) variables is Gamma(m, 1/eta)
    X += np.where(n_up > 0, rng.gamma(np.maximum(n_up, 1), 1.0 / e1), 0.0)
    X -= np.where(n_dn > 0, rng.gamma(np.maximum(n_dn, 1), 1.0 / e2), 0.0)
    pay = math.exp(-r * T) * np.maximum(S0 * np.exp(X) - 100.0, 0.0)
    se = pay.std() / math.sqrt(n)
    assert abs(pay.mean() - cos_price(MODELS["kou"], OptionType.Call, S0, 100.0, T)) < 4 * se


def test_vg_cos_matches_monte_carlo():
    rng = np.random.default_rng(1)
    n, sigma, theta, beta = 400_000, 0.2, -0.15, 0.2
    omega = MODELS["vg"].drift_correction()
    G = rng.gamma(T / beta, beta, n)                  # the random business clock
    X = (r + omega) * T + theta * G + sigma * np.sqrt(G) * rng.standard_normal(n)
    pay = math.exp(-r * T) * np.maximum(S0 * np.exp(X) - 100.0, 0.0)
    se = pay.std() / math.sqrt(n)
    assert abs(pay.mean() - cos_price(MODELS["vg"], OptionType.Call, S0, 100.0, T)) < 4 * se


def test_merton_paths_are_a_martingale():
    ST = simulate_merton(S0, r, 0.15, 0.5, -0.1, 0.15, T, 50, 100_000, seed=3)[:, -1]
    se = ST.std() / math.sqrt(len(ST))
    assert abs(ST.mean() - S0 * math.exp(r * T)) < 4 * se


def test_invalid_parameters_are_rejected():
    with pytest.raises(ValueError):
        KouModel(r=r, sigma=0.2, xi=1.0, p=0.5, eta1=0.9, eta2=5.0)   # E[e^J] infinite
    with pytest.raises(ValueError):
        VarianceGammaModel(r=r, sigma=0.2, theta=3.0, beta=1.0)      # omega undefined
