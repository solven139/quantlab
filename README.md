# QuantLab

An interactive simulator for the option-pricing models in *Mathematical Modeling and Computation in Finance* (Oosterlee & Grzelak). It has a **C++20 pricing engine**, a thin **pybind11** binding to Python and a **Streamlit** front end.

Each page starts from one model and states the assumptions it makes. You can then relax an assumption (constant volatility, no jumps, continuous hedging, exact simulation) and see what happens to prices, P&L and implied volatility.

## What's inside

| Page | What it shows | Engine pieces |
|---|---|---|
| **Black-Scholes** | Prices, Greeks and simulated GBM paths, with the model's assumptions listed | `bs_price`, `bs_delta`, `bs_gamma`, `bs_vega`, `simulate_gbm` |
| **Delta Hedging** | You sell a call and hedge it on a discrete grid. P&L spread as rebalancing frequency and hedge-vol misspecification change | `delta_hedge_pnl` |
| **Implied Volatility** | Inverting Black-Scholes numerically: Newton-Raphson with a bisection fallback, with each iterate shown | `implied_vol`, `implied_vol_iterates` |
| **The COS Method** | Fourier-cosine pricing from the characteristic function: density recovery, convergence in N, and speed against closed form | `cos_price`, `cos_density` |
| **Jump Models** | Merton, Kou and Variance Gamma: fat tails, skew and the implied-vol smile that jumps produce | `MertonModel`, `KouModel`, `VarianceGammaModel`, `simulate_merton` |
| **Heston** | Stochastic variance correlated with the spot: paths, prices and the implied-vol surface | `HestonModel`, `simulate_heston` |
| **Monte Carlo Lab** | How wrong each time-stepping scheme is: Euler vs Milstein vs exact for GBM, and Euler vs QE vs exact for the CIR/Heston variance | `gbm_schemes`, `cir_step_samples`, `heston_terminal` |

## Architecture

```
cpp/include, cpp/src   →  quantlab_core   (static C++20 library, no Python dependency)
cpp/bindings           →  _core           (pybind11 module, thin wrapper only)
python/quantlab        →  import quantlab (public Python API)
app/                   →  Streamlit pages (UI only; every number comes from the C++ engine)
tests/                 →  pytest suite
```

All the maths lives in plain C++ that knows nothing about Python. The binding layer only converts types, for example copying path arrays into NumPy arrays. The models share one `Model` interface (characteristic function and parameters), so the COS pricer works on any of them without changes.

## Getting started

Requirements: Python ≥ 3.10, CMake ≥ 3.18 and a C++20 compiler (GCC 10+, Clang 12+ or MSVC 2019+).

```bash
git clone https://github.com/solven139/quantlab.git
cd quantlab
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[test]"        # builds the C++ extension via scikit-build-core
streamlit run app/Home.py
```

### Use the engine directly

```python
from quantlab import OptionType, bs_price, implied_vol, cos_price, BlackScholesModel, HestonModel

p = bs_price(OptionType.Call, S0=100, K=100, T=1.0, r=0.05, sigma=0.2)   # 10.4506
implied_vol(OptionType.Call, p, 100, 100, 1.0, 0.05)                    # 0.2

# The COS method recovers the closed-form price...
cos_price(BlackScholesModel(r=0.05, sigma=0.2), OptionType.Call, 100, 100, 1.0)  # 10.4506
# ...and prices models that have no simple closed form
heston = HestonModel(r=0.05, kappa=1.5, vbar=0.04, gamma=0.5, rho=-0.7, v0=0.04)
cos_price(heston, OptionType.Call, 100, 100, 1.0)                        # 10.0555
```

## Testing

```bash
pytest
```

The suite currently has 136 tests in 9 files. They check the engine against independent references:
- Greeks against finite differences
- Monte Carlo prices against Black-Scholes
- the martingale property of simulated paths
- COS prices against closed form
- implied-vol round trips
- convergence of the simulation schemes

## Reference

C. W. Oosterlee & L. A. Grzelak, *Mathematical Modeling and Computation in Finance: With Exercises and Python and MATLAB Computer Codes*, World Scientific, 2019. Chapter and equation references appear on each page of the app.
