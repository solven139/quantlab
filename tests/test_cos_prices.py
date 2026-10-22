import time

import numpy as np
import pytest
from quantlab import (BlackScholesModel, HestonModel, KouModel, MertonModel, OptionType,
                      VarianceGammaModel, bs_price, cos_price, cos_prices)

KS = list(np.linspace(60.0, 140.0, 60))
HESTON = HestonModel(r=0.02, kappa=1.5768, vbar=0.0398, gamma=0.5751, rho=-0.5711, v0=0.0175)
MODELS = {
    "bs": BlackScholesModel(r=0.03, sigma=0.25),
    "merton": MertonModel(r=0.02, sigma=0.15, xi=0.5, mu_j=-0.1, sigma_j=0.1),
    "kou": KouModel(r=0.02, sigma=0.15, xi=0.8, p=0.3, eta1=10.0, eta2=5.0),
    "vg": VarianceGammaModel(r=0.02, sigma=0.2, theta=-0.1, beta=0.2),
    "heston": HESTON,
}


@pytest.mark.parametrize("name", list(MODELS))
@pytest.mark.parametrize("typ", [OptionType.Call, OptionType.Put])
def test_matches_scalar_pricer(name, typ):
    m = MODELS[name]
    vec = cos_prices(m, typ, 100.0, KS, 1.0, 256, 12.0)
    ref = np.array([cos_price(m, typ, 100.0, K, 1.0, 256, 12.0) for K in KS])
    assert np.max(np.abs(vec - ref)) < 1e-12 * max(1.0, ref.max())


def test_black_scholes_exact():
    m = MODELS["bs"]
    vec = cos_prices(m, OptionType.Call, 100.0, KS, 0.5, 128, 10.0)
    exact = [bs_price(OptionType.Call, 100.0, K, 0.5, 0.03, 0.25) for K in KS]
    assert np.allclose(vec, exact, atol=1e-10)


def test_heston_reference_value():
    # Fang & Oosterlee reference: Heston call, S0 = K = 100, T = 1, r = 0 -> 5.785155450
    h = HestonModel(r=0.0, kappa=1.5768, vbar=0.0398, gamma=0.5751, rho=-0.5711, v0=0.0175)
    assert cos_prices(h, OptionType.Call, 100.0, [100.0], 1.0, 256, 12.0)[0] == pytest.approx(
        5.785155450, abs=1e-7)


def test_edge_cases():
    assert len(cos_prices(HESTON, OptionType.Put, 100.0, [], 1.0)) == 0
    with pytest.raises(ValueError):
        cos_prices(HESTON, OptionType.Put, 100.0, [100.0, -1.0], 1.0)


def test_faster_than_one_call_per_strike():
    # Benchmark: 60 strikes of Heston. phi is evaluated N times instead of 60 N times.
    def best_of(f, reps):
        best = np.inf
        for _ in range(reps):
            t = time.perf_counter(); f(); best = min(best, time.perf_counter() - t)
        return best
    t_scalar = best_of(lambda: [cos_price(HESTON, OptionType.Put, 100.0, K, 1.0, 256, 12.0)
                                for K in KS], 5)
    t_vector = best_of(lambda: cos_prices(HESTON, OptionType.Put, 100.0, KS, 1.0, 256, 12.0), 20)
    print(f"\n60 Heston strikes: scalar {t_scalar * 1e3:.2f} ms, vector {t_vector * 1e3:.3f} ms, "
          f"speed-up {t_scalar / t_vector:.0f}x")
    assert t_scalar / t_vector > 5.0
