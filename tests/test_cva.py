import numpy as np
import pytest
from quantlab import (HullWhiteModel, NelsonSiegelCurve, Swap, cva, par_swap_rate,
                      simulate_exposure, swap_annuity, swap_value_today)

CURVE = NelsonSiegelCurve(b0=0.04, b1=-0.02, b2=0.01, tau=2.0)
HW = HullWhiteModel(lambda_=0.5, eta=0.01, curve=CURVE)


# ---------------- Today's swap value (TODO 1) ----------------

def test_swap_value_today_formula():
    # book eq. 12.12, evaluated independently in Python
    s = Swap(notional=100.0, K=0.03, start=1.0, end=6.0, tau=0.5)
    dates = np.arange(1.5, 6.0 + 1e-9, 0.5)
    annuity = sum(0.5 * CURVE.discount(T) for T in dates)
    expected = 100.0 * (CURVE.discount(1.0) - CURVE.discount(6.0) - 0.03 * annuity)
    assert swap_value_today(CURVE, s) == pytest.approx(expected, rel=1e-12)
    assert swap_annuity(CURVE, s) == pytest.approx(annuity, rel=1e-12)


def test_receiver_is_minus_payer():
    pay = Swap(notional=100.0, K=0.02, end=5.0, payer=True)
    rec = Swap(notional=100.0, K=0.02, end=5.0, payer=False)
    assert swap_value_today(CURVE, rec) == pytest.approx(-swap_value_today(CURVE, pay), rel=1e-12)


def test_par_rate_gives_zero_value():
    s = Swap(notional=1e4, K=0.0, end=10.0, tau=1.0)
    s.K = par_swap_rate(CURVE, s)
    assert abs(swap_value_today(CURVE, s)) < 1e-9
    # payer gains when the fixed rate is below par
    s.K -= 0.01
    assert swap_value_today(CURVE, s) > 0.0


def test_invalid_schedule():
    with pytest.raises(ValueError):
        Swap(start=0.0, end=10.0, tau=3.0)    # 10 is not a whole number of 3-year periods


# ---------------- Exposure simulation ----------------

@pytest.fixture(scope="module")
def sim():
    swaps = [Swap(notional=1e4, K=0.03, start=1.0, end=10.0, tau=0.5),
             Swap(notional=1e4, K=0.03, start=1.0, end=10.0, tau=0.5, payer=False)]
    return swaps, simulate_exposure(HW, swaps, 10.0, 24, 8000, 3)


def test_shapes_and_boundary_values(sim):
    swaps, (T, V, D) = sim
    assert V.shape == (2, 8000, 241) and D.shape == (8000, 241) and len(T) == 241
    assert np.allclose(V[0, :, 0], swap_value_today(CURVE, swaps[0]))   # t = 0: today's value
    assert np.allclose(V[:, :, -1], 0.0)                               # after the last payment
    assert np.allclose(V[1], -V[0])                                    # receiver = -payer, pathwise


def test_discounted_value_is_a_martingale(sim):
    # E[D(t) V(t)] = today's value of the payments still left after t
    swaps, (T, V, D) = sim
    s = swaps[0]
    X = D * V[0]
    for t in [0.5, 1.0, 2.25, 5.0, 8.75]:
        i = int(round(t * 24))
        start = s.start if t < s.start else s.start + np.floor((t - s.start) / s.tau + 1e-9) * s.tau
        rest = Swap(notional=s.notional, K=s.K, start=start, end=s.end, tau=s.tau)
        se = X[:, i].std() / np.sqrt(X.shape[0])
        assert abs(X[:, i].mean() - swap_value_today(CURVE, rest)) < 4 * se + 2.0


def test_netting_never_increases_exposure(sim):
    _, (T, V, D) = sim
    separate = (D * (np.maximum(V[0], 0) + np.maximum(V[1], 0))).mean(0)
    netted = (D * np.maximum(V[0] + V[1], 0)).mean(0)
    assert np.all(netted <= separate + 1e-12)
    assert np.allclose(netted, 0.0)   # payer + receiver on the same terms cancel exactly


# ---------------- CVA (TODO 2) ----------------

def test_cva_flat_exposure():
    # Constant EE: CVA = LGD * EE * P(default before T) = LGD * EE * (1 - e^{-hT})
    times = list(np.linspace(0.0, 5.0, 61))
    ee = [100.0] * len(times)
    assert cva(times, ee, 0.6, 0.02) == pytest.approx(0.6 * 100 * (1 - np.exp(-0.1)), rel=1e-12)


def test_cva_matches_sum():
    times = np.linspace(0.0, 10.0, 41)
    ee = 50.0 * np.sin(np.pi * times / 10.0)
    h, lgd = 0.03, 0.6
    pd = np.exp(-h * times[:-1]) - np.exp(-h * times[1:])
    assert cva(list(times), list(ee), lgd, h) == pytest.approx(lgd * np.sum(ee[1:] * pd), rel=1e-12)


def test_cva_zero_without_default_risk_and_grows_with_hazard():
    times = list(np.linspace(0.0, 10.0, 41))
    ee = [10.0 + t for t in times]
    assert cva(times, ee, 0.6, 0.0) == 0.0
    assert cva(times, ee, 0.6, 0.01) < cva(times, ee, 0.6, 0.05)
