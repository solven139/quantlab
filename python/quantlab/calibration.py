"""Calibration: from market implied vols back to model parameters.

Everything else in quantlab goes parameters -> prices. This module goes the other way.

Conventions
-----------
* Work in forward terms. The model must be built with r = 0 and is priced with S0 = F, so
  cos_prices returns undiscounted prices on the forward (the discount factor cancels out).
* Use out-of-the-money options only: puts for K < F, calls for K >= F. They are the liquid
  ones, and they have the time value that carries the information about the smile.
* Residuals are price errors divided by Black vega, which is a first-order approximation
  of the implied-vol error:  (V_model - V_market) / vega  ~  sigma_model - sigma_market.
  This avoids inverting the model price to an implied vol inside the optimiser.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from ._core import OptionType, cos_prices
from .black import black_implied_vol, black_price, black_vega


def otm_prices(model, F, Ks, T, N=256, L=12.0):
    """Undiscounted out-of-the-money prices: puts below the forward, calls at and above it."""
    Ks = np.asarray(Ks, float)
    put = Ks < F
    out = np.empty_like(Ks)
    if put.any():
        out[put] = cos_prices(model, OptionType.Put, F, list(Ks[put]), T, N, L)
    if (~put).any():
        out[~put] = cos_prices(model, OptionType.Call, F, list(Ks[~put]), T, N, L)
    return out


@dataclass
class CalibrationResult:
    params: np.ndarray            # fitted parameters
    model: object                 # model_factory(params)
    model_ivs: list               # model implied vols, one array per expiry slice
    iv_errors: list               # model - market implied vol, one array per slice
    rmse_vol: float               # root-mean-square implied-vol error (in vol units, 0.01 = 1 pt)
    std_errors: np.ndarray        # approximate standard errors of the parameters
    correlation: np.ndarray       # approximate correlation matrix of the parameter estimates
    condition_number: float       # of J^T J: large means some parameter combination is not identified
    success: bool
    message: str
    nfev: int
    names: tuple = field(default=())

    def summary(self):
        names = self.names or tuple(f"p{i}" for i in range(len(self.params)))
        rows = [f"{n:>10s} = {v: .6f}  (± {s:.2g})" for n, v, s in
                zip(names, self.params, self.std_errors)]
        return "\n".join(rows + [f"   RMSE    = {self.rmse_vol * 100:.4f} vol pts",
                                 f"   cond    = {self.condition_number:.3g}"])


def _as_slices(F, Ks, T, market_ivs):
    """Accept one expiry (scalars + arrays) or several (sequences of the same length)."""
    if np.ndim(T) == 0:
        return [(float(F), np.asarray(Ks, float), float(T), np.asarray(market_ivs, float))]
    return [(float(f), np.asarray(k, float), float(t), np.asarray(v, float))
            for f, k, t, v in zip(F, Ks, T, market_ivs)]


def calibrate(model_factory, params0, bounds, F, Ks, T, market_ivs, *, N=256, L=12.0,
              names=(), vega_floor=1e-6, **least_squares_kwargs):
    """Fit model parameters to market implied vols.

    model_factory : callable, params (1-D array) -> Model built with r = 0
    params0       : starting point (for a time series: the previous snapshot's answer)
    bounds        : (lower, upper) arrays, as in scipy.optimize.least_squares
    F, Ks, T, market_ivs : one expiry (F, T scalars; Ks, market_ivs arrays) or several
                    (sequences of the same length, one entry per expiry)
    """
    slices = _as_slices(F, Ks, T, market_ivs)
    if model_factory(np.asarray(params0, float)).rate() != 0.0:
        raise ValueError("calibrate: build the model with r = 0 (prices are in forward terms)")

    # Market side, computed once: OTM Black prices and vegas at the market vols
    targets = []
    for f, k, t, iv in slices:
        is_call = k >= f
        targets.append((black_price(f, k, t, iv, is_call),
                        np.maximum(black_vega(f, k, t, iv), vega_floor * f)))

    def residuals(p):
        model = model_factory(p)
        out = []
        for (f, k, t, _), (market_price, vega) in zip(slices, targets):
            model_price = otm_prices(model, f, k, t, N, L)
            # price error / vega  ~  implied-vol error
            out.append((model_price - market_price) / vega)
        return np.concatenate(out)

    res = least_squares(residuals, np.asarray(params0, float), bounds=bounds,
                        **least_squares_kwargs)

    # Uncertainty from the Jacobian at the optimum: cov ~ s^2 (J^T J)^{-1}
    J = res.jac
    m, n = J.shape
    JtJ = J.T @ J
    s2 = 2.0 * res.cost / max(m - n, 1)
    cov = s2 * np.linalg.pinv(JtJ)
    std = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = cov / np.outer(std, std)
    cond = np.linalg.cond(JtJ)

    model = model_factory(res.x)
    model_ivs, iv_errors = [], []
    for f, k, t, iv in slices:
        is_call = k >= f
        miv = black_implied_vol(otm_prices(model, f, k, t, N, L), f, k, t, is_call)
        model_ivs.append(miv)
        iv_errors.append(miv - iv)
    err = np.concatenate(iv_errors)
    rmse = float(np.sqrt(np.nanmean(err**2)))
    return CalibrationResult(res.x, model, model_ivs, iv_errors, rmse, std, corr, cond,
                             bool(res.success), str(res.message), int(res.nfev), tuple(names))


def calibrate_series(model_factory, params0, bounds, snapshots, **kwargs):
    """Calibrate a sequence of snapshots, each fit starting from the previous answer.

    snapshots : iterable of dicts with keys F, Ks, T, market_ivs (each as in calibrate)
    Returns a list of CalibrationResult, one per snapshot.
    """
    results, start = [], np.asarray(params0, float)
    for snap in snapshots:
        r = calibrate(model_factory, start, bounds, snap["F"], snap["Ks"], snap["T"],
                      snap["market_ivs"], **kwargs)
        results.append(r)
        start = r.params   # warm start: parameters move little between nearby snapshots
    return results


def profile(model_factory, params0, bounds, index, values, F, Ks, T, market_ivs, **kwargs):
    """Profile the fit along one parameter: fix params[index] at each value, refit the rest.

    A flat profile (the RMSE barely changes over a wide range of values) means the data do not
    pin that parameter down: other parameters compensate. Returns a list of
    (value, rmse_vol, full parameter vector) tuples.
    """
    params0 = np.asarray(params0, float)
    lower, upper = (np.asarray(b, float) for b in bounds)
    keep = np.arange(len(params0)) != index
    out = []
    start = params0[keep]
    for v in values:
        def factory(q, v=v):
            p = np.empty(len(params0))
            p[keep], p[index] = q, v
            return model_factory(p)
        r = calibrate(factory, np.clip(start, lower[keep], upper[keep]),
                      (lower[keep], upper[keep]), F, Ks, T, market_ivs, **kwargs)
        full = np.empty(len(params0))
        full[keep], full[index] = r.params, v
        out.append((float(v), r.rmse_vol, full))
        start = r.params
    return out
