"""Speed of cos_prices (characteristic function once) vs cos_price per strike.

Run:  python benchmarks/bench_cos_prices.py
"""
import time

import numpy as np
from quantlab import BatesModel, HestonModel, MertonModel, OptionType, cos_price, cos_prices

KS = list(np.linspace(60.0, 140.0, 60))
MODELS = {
    "Merton": MertonModel(r=0.0, sigma=0.15, xi=0.5, mu_j=-0.1, sigma_j=0.1),
    "Heston": HestonModel(r=0.0, kappa=1.5768, vbar=0.0398, gamma=0.5751, rho=-0.5711, v0=0.0175),
    "Bates": BatesModel(r=0.0, kappa=1.5768, vbar=0.0398, gamma=0.5751, rho=-0.5711, v0=0.0175,
                        xi=0.5, mu_j=-0.1, sigma_j=0.1),
}


def best_of(f, reps):
    best = np.inf
    for _ in range(reps):
        t = time.perf_counter()
        f()
        best = min(best, time.perf_counter() - t)
    return best


print(f"{'model':8s} {'N':>5s} {'per strike [ms]':>16s} {'all at once [ms]':>17s} {'speed-up':>9s}")
for name, m in MODELS.items():
    for N in (128, 256, 512):
        ts = best_of(lambda: [cos_price(m, OptionType.Put, 100.0, K, 1.0, N, 12.0) for K in KS], 5)
        tv = best_of(lambda: cos_prices(m, OptionType.Put, 100.0, KS, 1.0, N, 12.0), 50)
        print(f"{name:8s} {N:5d} {ts * 1e3:16.3f} {tv * 1e3:17.3f} {ts / tv:8.0f}x")
