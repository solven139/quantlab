import math
import pytest
from scipy.stats import norm
from quantlab import OptionType, bs_price


def bs_reference(cp, S0, K, T, r, sigma):
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if cp == "C":
        return S0 * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)


@pytest.mark.parametrize("K", [60, 100, 140])
def test_matches_scipy(K):
    args = (100.0, K, 1.0, 0.05, 0.2)
    assert bs_price(OptionType.Call, *args) == pytest.approx(bs_reference("C", *args), abs=1e-10)
    assert bs_price(OptionType.Put, *args) == pytest.approx(bs_reference("P", *args), abs=1e-10)


def test_put_call_parity():
    S0, K, T, r, s = 100.0, 95.0, 2.0, 0.03, 0.25
    c = bs_price(OptionType.Call, S0, K, T, r, s)
    p = bs_price(OptionType.Put, S0, K, T, r, s)
    assert c - p == pytest.approx(S0 - K * math.exp(-r * T), abs=1e-10)


def test_bad_input_raises():
    with pytest.raises(ValueError):
        bs_price(OptionType.Call, 100.0, 100.0, 1.0, 0.05, -0.2)
