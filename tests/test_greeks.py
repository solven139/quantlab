import math
import pytest
from quantlab import OptionType, bs_delta, bs_gamma, bs_price, bs_vega, norm_pdf

# Market inputs to test at: in-, at- and out-of-the-money
CASES = [
    (100.0, 80.0, 1.0, 0.05, 0.20),
    (100.0, 100.0, 1.0, 0.05, 0.20),
    (100.0, 120.0, 0.5, 0.02, 0.35),
]
h = 1e-4  # bump size for finite differences


def test_norm_pdf():
    assert norm_pdf(0.0) == pytest.approx(1 / math.sqrt(2 * math.pi), abs=1e-12)
    assert norm_pdf(1.3) == pytest.approx(norm_pdf(-1.3), abs=1e-15)  # symmetric


@pytest.mark.parametrize("S0,K,T,r,s", CASES)
@pytest.mark.parametrize("cp", [OptionType.Call, OptionType.Put])
def test_delta_matches_bump(cp, S0, K, T, r, s):
    # central difference: [V(S+h) - V(S-h)] / 2h
    fd = (bs_price(cp, S0 + h, K, T, r, s) - bs_price(cp, S0 - h, K, T, r, s)) / (2 * h)
    assert bs_delta(cp, S0, K, T, r, s) == pytest.approx(fd, abs=1e-6)


@pytest.mark.parametrize("S0,K,T,r,s", CASES)
def test_gamma_matches_bump(S0, K, T, r, s):
    hh = 1e-2
    up = bs_price(OptionType.Call, S0 + hh, K, T, r, s)
    mid = bs_price(OptionType.Call, S0, K, T, r, s)
    dn = bs_price(OptionType.Call, S0 - hh, K, T, r, s)
    fd = (up - 2 * mid + dn) / hh**2
    assert bs_gamma(S0, K, T, r, s) == pytest.approx(fd, abs=1e-5)


@pytest.mark.parametrize("S0,K,T,r,s", CASES)
def test_vega_matches_bump(S0, K, T, r, s):
    fd = (bs_price(OptionType.Call, S0, K, T, r, s + h)
          - bs_price(OptionType.Call, S0, K, T, r, s - h)) / (2 * h)
    assert bs_vega(S0, K, T, r, s) == pytest.approx(fd, abs=1e-5)


def test_call_put_delta_relation():
    # From put-call parity: delta_call - delta_put = 1
    args = (100.0, 105.0, 1.0, 0.03, 0.25)
    assert bs_delta(OptionType.Call, *args) - bs_delta(OptionType.Put, *args) == pytest.approx(1.0, abs=1e-12)