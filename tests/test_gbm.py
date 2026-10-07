import math
import numpy as np
import pytest
from quantlab import OptionType, bs_price, simulate_gbm

S0, r, sigma, T = 100.0, 0.05, 0.2, 1.0


def test_shape_and_start():
    paths = simulate_gbm(S0, r, sigma, T, n_steps=50, n_paths=10)
    assert paths.shape == (10, 51)          # one extra column for t = 0
    assert np.all(paths[:, 0] == S0)        # every path starts at S0
    assert np.all(paths > 0)                # GBM never goes negative


def test_same_seed_same_paths():
    a = simulate_gbm(S0, r, sigma, T, 20, 5, seed=7)
    b = simulate_gbm(S0, r, sigma, T, 20, 5, seed=7)
    c = simulate_gbm(S0, r, sigma, T, 20, 5, seed=8)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_martingale_mean():
    # Under Q, E[S(T)] = S0 * exp(rT): the discounted price is a martingale (book Ch 2.3)
    ST = simulate_gbm(S0, r, sigma, T, 1, 200_000)[:, -1]
    stderr = ST.std() / math.sqrt(len(ST))
    assert abs(ST.mean() - S0 * math.exp(r * T)) < 4 * stderr


def test_log_return_variance():
    # log(S(T)/S0) ~ Normal((r - sigma^2/2) T, sigma^2 T)
    ST = simulate_gbm(S0, r, sigma, T, 1, 200_000)[:, -1]
    assert np.var(np.log(ST / S0)) == pytest.approx(sigma**2 * T, rel=0.02)


def test_monte_carlo_call_matches_black_scholes():
    K = 105.0
    ST = simulate_gbm(S0, r, sigma, T, 1, 200_000)[:, -1]
    payoff = np.exp(-r * T) * np.maximum(ST - K, 0.0)
    stderr = payoff.std() / math.sqrt(len(payoff))
    assert abs(payoff.mean() - bs_price(OptionType.Call, S0, K, T, r, sigma)) < 4 * stderr