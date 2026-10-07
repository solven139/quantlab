import math

import numpy as np
import pytest
from quantlab import BlackScholesModel, OptionType, bs_price, cos_density, cos_price

S0, T, r, sigma = 100.0, 1.0, 0.05, 0.25
model = BlackScholesModel(r=r, sigma=sigma)


def test_char_fn_basic_properties():
    assert model.char_fn(0.0, T) == pytest.approx(1.0)               # phi(0) = E[1] = 1
    for u in [0.5, 2.0, 10.0]:
        assert abs(model.char_fn(u, T)) <= 1.0 + 1e-15               # |phi(u)| <= 1
        # phi(-u) is the complex conjugate of phi(u) for a real random variable
        assert model.char_fn(-u, T) == pytest.approx(model.char_fn(u, T).conjugate())


def test_char_fn_matches_normal_distribution():
    # X ~ N(mu, s^2) has phi(u) = exp(i u mu - s^2 u^2 / 2)
    mu, s2, u = (r - 0.5 * sigma**2) * T, sigma**2 * T, 1.7
    expected = complex(math.cos(u * mu), math.sin(u * mu)) * math.exp(-0.5 * s2 * u * u)
    assert model.char_fn(u, T) == pytest.approx(expected, abs=1e-14)


@pytest.mark.parametrize("cp", [OptionType.Call, OptionType.Put])
@pytest.mark.parametrize("K", [60.0, 90.0, 100.0, 110.0, 150.0])
def test_cos_matches_black_scholes(cp, K):
    exact = bs_price(cp, S0, K, T, r, sigma)
    assert cos_price(model, cp, S0, K, T, N=256) == pytest.approx(exact, abs=1e-10)


def test_error_falls_fast_with_N():
    # Book Ch 6.2.3: exponential convergence for smooth densities
    exact = bs_price(OptionType.Call, S0, 100.0, T, r, sigma)
    errs = [abs(cos_price(model, OptionType.Call, S0, 100.0, T, N=n) - exact)
            for n in (8, 16, 32, 64)]
    assert errs[0] > errs[1] > errs[2]
    assert errs[2] < 1e-5 and errs[3] < 1e-10


def test_put_call_parity_holds_for_cos():
    K = 105.0
    c = cos_price(model, OptionType.Call, S0, K, T)
    p = cos_price(model, OptionType.Put, S0, K, T)
    assert c - p == pytest.approx(S0 - K * math.exp(-r * T), abs=1e-10)


def test_density_recovery_matches_normal_pdf():
    xs = np.linspace(-1.0, 1.0, 41)
    f = np.array(cos_density(model, T, list(xs), N=128))
    mu, s = (r - 0.5 * sigma**2) * T, sigma * math.sqrt(T)
    exact = np.exp(-0.5 * ((xs - mu) / s) ** 2) / (s * math.sqrt(2 * math.pi))
    assert np.max(np.abs(f - exact)) < 1e-8
