import math
import numpy as np
import pytest
from quantlab import delta_hedge_pnl

# Book Example 3.3.2 parameters
S0, K, T, r, sigma = 1.0, 0.95, 1.0, 0.1, 0.2


def test_one_value_per_path_and_reproducible():
    a = delta_hedge_pnl(S0, K, T, r, sigma, sigma, 50, 100, seed=1)
    b = delta_hedge_pnl(S0, K, T, r, sigma, sigma, 50, 100, seed=1)
    assert a.shape == (100,)
    assert np.array_equal(a, b)


def test_hedge_is_fair_on_average():
    # Book Example 3.3.1: with the right volatility, E[P&L(T)] = 0
    pnl = delta_hedge_pnl(S0, K, T, r, sigma, sigma, 250, 20_000)
    stderr = pnl.std() / math.sqrt(len(pnl))
    assert abs(pnl.mean()) < 4 * stderr


def test_hedge_error_shrinks_like_one_over_sqrt_m():
    # 4x more rebalancing -> about half the spread of P&L
    s100 = delta_hedge_pnl(S0, K, T, r, sigma, sigma, 100, 20_000).std()
    s400 = delta_hedge_pnl(S0, K, T, r, sigma, sigma, 400, 20_000).std()
    assert s400 / s100 == pytest.approx(0.5, abs=0.07)


def test_hedging_hedge_far_better_than_no_hedge():
    # With one single rebalance date (m=1) we hold the initial delta all the way
    s1 = delta_hedge_pnl(S0, K, T, r, sigma, sigma, 1, 20_000).std()
    s250 = delta_hedge_pnl(S0, K, T, r, sigma, sigma, 250, 20_000).std()
    assert s250 < 0.2 * s1


def test_wrong_volatility_gives_systematic_pnl():
    # Sold at a vol ABOVE what the stock really does -> we make money on average
    rich = delta_hedge_pnl(S0, K, T, r, sigma_true=0.2, sigma_hedge=0.3,
                           n_rebalance=250, n_paths=20_000)
    # Sold at a vol BELOW what the stock really does -> we lose on average
    cheap = delta_hedge_pnl(S0, K, T, r, sigma_true=0.3, sigma_hedge=0.2,
                            n_rebalance=250, n_paths=20_000)
    assert rich.mean() > 0
    assert cheap.mean() < 0