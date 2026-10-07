import math

import numpy as np
import pytest
from quantlab import (HestonModel, OptionType, bs_price, cos_price, implied_vol,
                      simulate_heston)

# Fang & Oosterlee (2008) test case, used throughout the book's Heston chapters
FO = dict(r=0.0, kappa=1.5768, vbar=0.0398, gamma=0.5751, rho=-0.5711, v0=0.0175)


def test_char_fn_is_one_at_zero():
    assert HestonModel(**FO).char_fn(0.0, 1.0) == pytest.approx(1.0, abs=1e-14)


def test_fang_oosterlee_reference_price():
    # S0 = K = 100, T = 1: reference call value 5.785155450
    m = HestonModel(**FO)
    assert cos_price(m, OptionType.Call, 100.0, 100.0, 1.0, N=256, L=12.0) == pytest.approx(
        5.785155450, abs=1e-7)


@pytest.mark.parametrize("K", [80.0, 100.0, 120.0])
def test_put_call_parity(K):
    m = HestonModel(r=0.03, kappa=2.0, vbar=0.04, gamma=0.6, rho=-0.7, v0=0.05)
    # Heston's tails are fatter than normal: for 1e-8 accuracy the COS range must be wider
    c = cos_price(m, OptionType.Call, 100.0, K, 1.0, N=1024, L=20.0)
    p = cos_price(m, OptionType.Put, 100.0, K, 1.0, N=1024, L=20.0)
    assert c - p == pytest.approx(100.0 - K * math.exp(-0.03), abs=1e-8)


def test_small_vol_of_vol_gives_black_scholes():
    # gamma -> 0 with v0 = vbar: variance stays at vbar, so Heston = Black-Scholes(sqrt(vbar)).
    # rho = 0 removes the first-order skew effect (with rho != 0 the gap grows linearly in gamma).
    m = HestonModel(r=0.02, kappa=1.0, vbar=0.04, gamma=1e-2, rho=0.0, v0=0.04)
    assert cos_price(m, OptionType.Call, 100.0, 110.0, 1.0, N=256, L=12.0) == pytest.approx(
        bs_price(OptionType.Call, 100.0, 110.0, 1.0, 0.02, 0.2), abs=1e-3)


def test_cos_matches_monte_carlo():
    p = dict(r=0.03, kappa=2.0, vbar=0.04, gamma=0.4, rho=-0.7, v0=0.05)
    S, _, _ = simulate_heston(100.0, p["r"], p["kappa"], p["vbar"], p["gamma"], p["rho"],
                              p["v0"], 1.0, 250, 100_000, seed=5)
    pay = math.exp(-p["r"]) * np.maximum(S[:, -1] - 100.0, 0.0)
    se = pay.std() / math.sqrt(len(pay))
    v_cos = cos_price(HestonModel(**p), OptionType.Call, 100.0, 100.0, 1.0, N=256, L=12.0)
    assert abs(pay.mean() - v_cos) < 4 * se + 0.02   # + small Euler discretisation bias


@pytest.mark.parametrize("rho, sign", [(-0.8, +1), (0.6, -1)])
def test_correlation_sets_the_direction_of_the_skew(rho, sign):
    # rho < 0: falling stock -> rising vol -> expensive low strikes (negative skew)
    m = HestonModel(r=0.0, kappa=1.5, vbar=0.04, gamma=0.6, rho=rho, v0=0.04)

    def iv(K, typ):
        return implied_vol(typ, cos_price(m, typ, 100.0, K, 1.0, N=256, L=12.0), 100.0, K, 1.0, 0.0)

    low, high = iv(80.0, OptionType.Put), iv(120.0, OptionType.Call)
    assert sign * (low - high) > 0.01


def test_feller_condition():
    assert HestonModel(r=0.0, kappa=2.0, vbar=0.04, gamma=0.3, rho=0.0, v0=0.04).feller_satisfied()
    assert not HestonModel(**FO).feller_satisfied()   # 2*1.5768*0.0398 = 0.126 < 0.5751^2 = 0.331
