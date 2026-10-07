import math

import pytest
from quantlab import OptionType, bs_price, implied_vol, implied_vol_iterates


def test_book_example_4_1_2():
    # S0=100, K=120, T=1, r=5%, call trades at 2  ->  sigma_imp = 0.161482728841394
    iv = implied_vol(OptionType.Call, 2.0, 100.0, 120.0, 1.0, 0.05)
    assert iv == pytest.approx(0.161482728841394, abs=1e-10)


@pytest.mark.parametrize("cp", [OptionType.Call, OptionType.Put])
@pytest.mark.parametrize("K", [60.0, 80.0, 100.0, 125.0, 160.0])
@pytest.mark.parametrize("sigma", [0.08, 0.2, 0.45, 1.2])
def test_round_trip(cp, K, sigma):
    # price with a known sigma, then recover it from the price
    S0, T, r = 100.0, 1.0, 0.03
    price = bs_price(cp, S0, K, T, r, sigma)
    # Time value = price above the no-arbitrage lower bound. Only that part depends on sigma.
    df = math.exp(-r * T)
    intrinsic = max(S0 - K * df, 0.0) if cp == OptionType.Call else max(K * df - S0, 0.0)
    if price - intrinsic < 1e-6:
        pytest.skip("almost no time value: the price carries no information about sigma")
    assert implied_vol(cp, price, S0, K, T, r) == pytest.approx(sigma, abs=1e-8)


def test_newton_converges_fast_at_the_money():
    price = bs_price(OptionType.Call, 100.0, 100.0, 1.0, 0.05, 0.3)
    steps = implied_vol_iterates(OptionType.Call, price, 100.0, 100.0, 1.0, 0.05)
    assert len(steps) <= 6          # quadratic convergence: a handful of steps
    assert steps[0] == 0.2          # starts from the initial guess


def test_deep_out_of_the_money_still_converges():
    # Tiny vega here: plain Newton would jump far away, bisection keeps it safe
    S0, K, T, r, sigma = 100.0, 250.0, 0.5, 0.0, 0.6
    price = bs_price(OptionType.Call, S0, K, T, r, sigma)
    assert implied_vol(OptionType.Call, price, S0, K, T, r) == pytest.approx(sigma, abs=1e-7)


def test_arbitrage_price_is_rejected():
    with pytest.raises(ValueError):
        implied_vol(OptionType.Call, 150.0, 100.0, 100.0, 1.0, 0.05)  # call worth more than the stock
    with pytest.raises(ValueError):
        implied_vol(OptionType.Call, 1.0, 100.0, 50.0, 1.0, 0.05)     # below intrinsic value