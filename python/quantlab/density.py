"""Breeden-Litzenberger: the risk-neutral density implied by option quotes.

    f_{S(T)}(K) = (1 / P(0,T)) * d^2 C / dK^2        (book Ch 4.3.3)

In forward terms (undiscounted prices C = price / P(0,T)) the discount factor drops out:
f(K) = d^2 C_undiscounted / dK^2. Differentiating twice amplifies every bit of noise in the
quotes, so raw quotes are never differenced directly. The pipeline is:

  1. out-of-the-money mid prices             (otm_ivs_from_quotes)
  2. implied vols
  3. a smooth curve through the vols         (cubic spline in log-moneyness, or SABR)
  4. prices on a fine strike grid            (Black-76 with the smoothed vols)
  5. second differences                      -> density on the grid
"""
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import UnivariateSpline

from .black import black_implied_vol, black_price
from .sabr import fit_sabr


def otm_ivs_from_quotes(F, DF, Ks, call_mids, put_mids, T):
    """Steps 1-2: implied vols from out-of-the-money mids (puts below F, calls at/above F).

    DF is the discount factor P(0, T); prices are divided by it to put them in forward terms.
    Returns the implied vols (NaN where the quote has no time value).
    """
    Ks = np.asarray(Ks, float)
    is_call = Ks >= F
    mids = np.where(is_call, np.asarray(call_mids, float), np.asarray(put_mids, float)) / DF
    return black_implied_vol(mids, F, Ks, T, is_call)


@dataclass
class ImpliedDensity:
    K: np.ndarray            # strike grid
    density: np.ndarray      # f_{S(T)}(K) on the grid
    vols: np.ndarray         # smoothed implied vols on the grid
    calls: np.ndarray        # undiscounted call prices on the grid
    mass: float              # integral of the density over the grid (should be close to 1)
    mean: float              # integral of K f(K): should be close to the forward
    method: str

    def cdf(self):
        """Cumulative distribution on the grid (trapezoid rule)."""
        dK = np.diff(self.K)
        steps = 0.5 * (self.density[1:] + self.density[:-1]) * dK
        return np.concatenate([[0.0], np.cumsum(steps)])


def smooth_vols(F, Ks, ivs, T, method="spline", smoothing=0.0, beta=1.0):
    """Step 3: return a function K -> smoothed implied vol.

    spline : cubic smoothing spline of vol against log(K/F); smoothing = 0 interpolates the
             quotes exactly, larger values trade fit for smoothness (try ~ n * noise^2).
             Outside the quoted range the vol continues along the spline's end slope, so the
             vol curve has no kink there. (Holding it flat would put a kink in the vol at the
             last quote, which becomes a kink in dC/dK and a spike in the density.)
    sabr   : Hagan SABR fit with beta fixed; extrapolates with the SABR wings.
    """
    Ks, ivs = np.asarray(Ks, float), np.asarray(ivs, float)
    ok = np.isfinite(ivs)
    Ks, ivs = Ks[ok], ivs[ok]
    order = np.argsort(Ks)
    Ks, ivs = Ks[order], ivs[order]
    if method == "spline":
        k = np.log(Ks / F)
        spl = UnivariateSpline(k, ivs, k=3, s=smoothing)
        dspl = spl.derivative()
        lo, hi = k[0], k[-1]
        v_lo, v_hi, s_lo, s_hi = float(spl(lo)), float(spl(hi)), float(dspl(lo)), float(dspl(hi))

        def vol(K):
            x = np.log(np.asarray(K, float) / F)
            v = spl(np.clip(x, lo, hi))
            v = np.where(x < lo, v_lo + s_lo * (x - lo), v)    # linear, C1 extension
            v = np.where(x > hi, v_hi + s_hi * (x - hi), v)
            return np.maximum(v, 1e-4)
        return vol
    if method == "sabr":
        fit = fit_sabr(F, Ks, T, ivs, beta=beta)
        return fit.vols
    raise ValueError("method must be 'spline' or 'sabr'")


def implied_density(F, Ks, ivs, T, method="spline", K_grid=None, n_grid=2001, smoothing=0.0,
                    beta=1.0):
    """Steps 3-5: risk-neutral density of S(T) from implied vols at strikes Ks.

    K_grid : strike grid (uniform spacing). Default: 2001 points from F e^{-6 s} to F e^{+6 s},
             with s = ATM vol * sqrt(T), so the tails are covered.
    """
    vol_fn = smooth_vols(F, Ks, ivs, T, method, smoothing, beta)
    if K_grid is None:
        s = float(vol_fn(F)) * np.sqrt(T)
        K_grid = np.linspace(F * np.exp(-6.0 * s), F * np.exp(6.0 * s), n_grid)
    K = np.asarray(K_grid, float)
    h = K[1] - K[0]
    if not np.allclose(np.diff(K), h):
        raise ValueError("implied_density: K_grid must be uniformly spaced")

    vols = np.asarray(vol_fn(K), float)
    calls = black_price(F, K, T, vols, True)             # step 4: undiscounted call prices

    # Step 5: second difference d^2C/dK^2 at the interior points; ends set to 0
    dens = np.zeros_like(K)
    dens[1:-1] = (calls[2:] - 2.0 * calls[1:-1] + calls[:-2]) / (h * h)

    mass = float(np.trapezoid(dens, K))
    mean = float(np.trapezoid(K * dens, K))
    return ImpliedDensity(K, dens, vols, calls, mass, mean, method)
