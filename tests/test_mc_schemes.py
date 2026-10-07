import math

import numpy as np
import pytest
from quantlab import (CirScheme, HestonModel, OptionType, cir_step_samples, cos_price,
                      gbm_schemes, heston_terminal)


def fitted_order(dts, errors):
    # slope of log(error) against log(dt): error ~ C dt^order
    return np.polyfit(np.log(dts), np.log(errors), 1)[0]


# ---------------- GBM: strong and weak convergence (book Ch 9.2) ----------------
S0, r, sigma, T = 100.0, 0.06, 0.3, 1.0
STEPS = [8, 16, 32, 64, 128]


def strong_errors():
    eu, mi = [], []
    for n in STEPS:
        ex, e, m = gbm_schemes(S0, r, sigma, T, n, 20_000, seed=1)
        eu.append(np.mean(np.abs(e - ex)))
        mi.append(np.mean(np.abs(m - ex)))
    return np.array(eu), np.array(mi)


def test_same_brownian_paths_for_all_schemes():
    ex, eu, mi = gbm_schemes(S0, r, sigma, T, 2000, 500, seed=4)
    # with tiny steps all three schemes end in nearly the same place, path by path
    assert np.max(np.abs(mi - ex)) < 0.05
    assert np.max(np.abs(eu - ex)) < 2.0


def test_euler_strong_order_is_one_half():
    eu, _ = strong_errors()
    assert fitted_order(T / np.array(STEPS), eu) == pytest.approx(0.5, abs=0.1)


def test_milstein_strong_order_is_one():
    _, mi = strong_errors()
    assert fitted_order(T / np.array(STEPS), mi) == pytest.approx(1.0, abs=0.1)


def test_weak_error_is_small_for_both():
    ex, eu, mi = gbm_schemes(S0, r, sigma, T, 64, 200_000, seed=2)
    assert abs(eu.mean() - S0 * math.exp(r * T)) < 0.05
    assert abs(mi.mean() - S0 * math.exp(r * T)) < 0.05


# ---------------- CIR variance: exact, QE, Euler (book Ch 9.3) ----------------
def cir_mean_var(v0, kappa, vbar, gamma, dt):
    ek = math.exp(-kappa * dt)
    m = vbar + (v0 - vbar) * ek
    s2 = v0 * gamma**2 * ek * (1 - ek) / kappa + vbar * gamma**2 * (1 - ek) ** 2 / (2 * kappa)
    return m, s2


# Book Example 9.3.2: gamma = 0.2 uses the quadratic branch, gamma = 0.6 the exponential one
@pytest.mark.parametrize("gamma", [0.2, 0.6])
@pytest.mark.parametrize("scheme", [CirScheme.Exact, CirScheme.QE])
def test_cir_samples_match_exact_mean_and_variance(scheme, gamma):
    v0, kappa, vbar, dt = 0.3, 0.5, 0.05, 5.0
    x = cir_step_samples(scheme, v0, kappa, vbar, gamma, dt, 400_000, seed=2)
    m, s2 = cir_mean_var(v0, kappa, vbar, gamma, dt)
    assert np.all(x >= 0.0)
    assert x.mean() == pytest.approx(m, rel=0.01)
    assert x.var() == pytest.approx(s2, rel=0.03)


def test_euler_one_big_step_is_badly_wrong():
    # the reason better schemes exist: Euler's mean after one 5-year step is far off
    v0, kappa, vbar, gamma, dt = 0.3, 0.5, 0.05, 0.2, 5.0
    x = cir_step_samples(CirScheme.Euler, v0, kappa, vbar, gamma, dt, 100_000, seed=3)
    m, _ = cir_mean_var(v0, kappa, vbar, gamma, dt)
    assert abs(x.mean() - m) > 0.03


def test_qe_exponential_branch_puts_mass_at_zero():
    x = cir_step_samples(CirScheme.QE, 0.3, 0.5, 0.05, 0.6, 5.0, 100_000, seed=5)
    assert 0.5 < (x == 0.0).mean() < 0.9


# ---------------- Heston: AES with exact / QE variance vs Euler (book Ch 9.4) ----------------
FO = dict(kappa=1.5768, vbar=0.0398, gamma=0.5751, rho=-0.5711, v0=0.0175)  # Feller violated
REF = cos_price(HestonModel(r=0.0, **FO), OptionType.Call, 100.0, 100.0, 1.0, N=256, L=12.0)


def heston_mc(scheme, n_steps, n_paths=200_000, seed=3):
    ST = heston_terminal(scheme, 100.0, 0.0, FO["kappa"], FO["vbar"], FO["gamma"], FO["rho"],
                         FO["v0"], 1.0, n_steps, n_paths, seed)
    pay = np.maximum(ST - 100.0, 0.0)
    return pay.mean(), pay.std() / math.sqrt(len(pay))


def test_heston_qe_martingale_error_shrinks_with_smaller_steps():
    # AES/QE is not exactly a martingale for big steps (Andersen 2008 adds a "martingale
    # correction" for that). The error in E[S(T)] must shrink as the steps get smaller.
    def drift_error(n_steps):
        ST = heston_terminal(CirScheme.QE, 100.0, 0.03, FO["kappa"], FO["vbar"], FO["gamma"],
                             FO["rho"], FO["v0"], 1.0, n_steps, 400_000, seed=6)
        return abs(ST.mean() - 100.0 * math.exp(0.03)), ST.std() / math.sqrt(len(ST))

    big, _ = drift_error(4)
    small, se = drift_error(256)
    assert big > 0.2
    assert small < 4 * se


@pytest.mark.parametrize("scheme", [CirScheme.Exact, CirScheme.QE])
def test_heston_good_schemes_hit_the_cos_price(scheme):
    price, se = heston_mc(scheme, 16)
    assert abs(price - REF) < 4 * se


def test_heston_euler_is_biased_even_with_many_steps():
    price, se = heston_mc(CirScheme.Euler, 64)
    assert abs(price - REF) > 6 * se
