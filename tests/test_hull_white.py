import numpy as np
import pytest
from quantlab import HullWhiteModel, NelsonSiegelCurve, simulate_hull_white

# An upward-sloping curve: short rate 2%, long rate 4%, with a small hump
CURVE = NelsonSiegelCurve(b0=0.04, b1=-0.02, b2=0.01, tau=2.0)
HW = HullWhiteModel(lambda_=0.5, eta=0.01, curve=CURVE)


@pytest.mark.parametrize("T", [0.5, 2.0, 7.0])
def test_forward_is_minus_log_discount_slope(T):
    h = 1e-5
    fd = -(np.log(CURVE.discount(T + h)) - np.log(CURVE.discount(T - h))) / (2 * h)
    assert CURVE.inst_forward(T) == pytest.approx(fd, abs=1e-8)


@pytest.mark.parametrize("T", [0.5, 3.0])
def test_forward_slope_is_derivative_of_forward(T):
    h = 1e-5
    fd = (CURVE.inst_forward(T + h) - CURVE.inst_forward(T - h)) / (2 * h)
    assert CURVE.forward_slope(T) == pytest.approx(fd, abs=1e-8)


def test_theta_formula():
    # book eq. 11.37, evaluated independently in Python
    l, eta = 0.5, 0.01
    for t in [0.0, 1.0, 5.0]:
        expected = (CURVE.inst_forward(t) + CURVE.forward_slope(t) / l
                    + eta**2 / (2 * l**2) * (1 - np.exp(-2 * l * t)))
        assert HW.theta(t) == pytest.approx(expected, abs=1e-14)


def test_r0_and_bond_at_time_zero():
    assert HW.r0() == pytest.approx(0.02, abs=1e-14)       # f(0,0) = b0 + b1
    for T in [1.0, 5.0, 10.0]:
        assert HW.zcb(0.0, T, HW.r0()) == pytest.approx(CURVE.discount(T), rel=1e-12)


@pytest.fixture(scope="module")
def paths():
    # 10 years, 500 steps, 20k paths
    return simulate_hull_white(HW, 10.0, 500, 20_000, 7)


@pytest.mark.parametrize("T", [1.0, 5.0, 10.0])
def test_model_reproduces_todays_curve(paths, T):
    # The point of theta(t): E[exp(-int_0^T r)] = P(0,T) for every maturity
    _, I = paths
    j = int(round(T / 10.0 * 500))
    D = np.exp(-I[:, j])
    se = D.std(ddof=1) / np.sqrt(len(D))
    assert abs(D.mean() - CURVE.discount(T)) < 4 * se + 2e-4   # MC error + Euler bias


@pytest.mark.parametrize("t", [1.0, 5.0])
def test_short_rate_mean_and_variance(paths, t):
    r, _ = paths
    j = int(round(t / 10.0 * 500))
    assert r[:, j].mean() == pytest.approx(HW.mean_r(t), abs=4e-4)
    assert r[:, j].var() == pytest.approx(HW.var_r(t), rel=0.05)


def test_bond_formula_matches_monte_carlo(paths):
    # Tower property: E[D(t) P(t,T)] = P(0,T) with P(t,T) from the closed form
    r, I = paths
    t, T = 3.0, 8.0
    j = int(round(t / 10.0 * 500))
    est = np.mean(np.exp(-I[:, j]) * np.array([HW.zcb(t, T, x) for x in r[:, j]]))
    assert est == pytest.approx(CURVE.discount(T), abs=1e-3)


def test_faster_mean_reversion_reduces_variance():
    slow = HullWhiteModel(lambda_=0.1, eta=0.01, curve=CURVE)
    fast = HullWhiteModel(lambda_=2.0, eta=0.01, curve=CURVE)
    assert fast.var_r(5.0) < slow.var_r(5.0)
    # long-run variance limit eta^2 / (2 lambda)
    assert fast.var_r(100.0) == pytest.approx(0.01**2 / 4.0, rel=1e-10)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        NelsonSiegelCurve(0.04, -0.02, 0.01, 0.0)
    with pytest.raises(ValueError):
        HullWhiteModel(lambda_=0.0, eta=0.01, curve=CURVE)
