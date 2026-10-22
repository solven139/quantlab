"""Fitting the SABR smile (Hagan 2002) to one expiry's implied vols.

SABR is a smile parametrisation, not a quantlab Model: there is no characteristic function,
so it is fitted directly in implied-vol space with Hagan's formula (C++: sabr_implied_vols).
beta is fixed (beta = 1 is the usual choice for an equity index, 0.5 for rates); alpha, rho
and nu are fitted. alpha sets the level, rho the skew and nu the curvature of the smile.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from ._core import sabr_implied_vols


@dataclass
class SabrFit:
    alpha: float
    beta: float
    rho: float
    nu: float
    F: float
    T: float
    rmse_vol: float      # root-mean-square implied-vol error of the fit (0.01 = 1 vol pt)
    success: bool

    def vols(self, Ks):
        return sabr_implied_vols(self.F, list(np.asarray(Ks, float)), self.T, self.alpha,
                                 self.beta, self.rho, self.nu)


def fit_sabr(F, Ks, T, market_ivs, beta=1.0, x0=None, weights=None):
    """Least-squares fit of (alpha, rho, nu) for fixed beta to one expiry's implied vols.

    x0      : starting (alpha, rho, nu); pass the previous snapshot's answer for a time series
    weights : optional weights per quote (e.g. vega or 1 / bid-ask spread)
    """
    Ks = np.asarray(Ks, float)
    ivs = np.asarray(market_ivs, float)
    w = np.ones_like(ivs) if weights is None else np.asarray(weights, float)
    if x0 is None:
        atm = np.interp(F, Ks, ivs)                    # ATM vol from the quotes
        x0 = (atm * F ** (1.0 - beta), -0.3, 0.5)      # alpha ~ sigma_ATM F^(1 - beta)

    def residuals(p):
        alpha, rho, nu = p
        return w * (sabr_implied_vols(F, list(Ks), T, alpha, beta, rho, nu) - ivs)

    lower = (1e-8, -0.9999, 1e-8)
    upper = (np.inf, 0.9999, np.inf)
    res = least_squares(residuals, np.clip(x0, lower, [1e6, 0.9999, 1e6]), bounds=(lower, upper))
    alpha, rho, nu = res.x
    err = sabr_implied_vols(F, list(Ks), T, alpha, beta, rho, nu) - ivs
    return SabrFit(float(alpha), float(beta), float(rho), float(nu), float(F), float(T),
                   float(np.sqrt(np.mean(err**2))), bool(res.success))
