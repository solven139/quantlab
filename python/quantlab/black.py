"""Black-76 helpers in forward terms (vectorised with NumPy).

All prices here are *undiscounted*: the value of the option paid at T, measured in money at T,
on a forward F. Multiply by the discount factor P(0, T) to get today's price. Working in
forward terms removes the rate and the dividend yield from every formula, which is how option
quotes on an index are usually handled.
"""
import numpy as np
from scipy.special import ndtr

from ._core import OptionType, implied_vol


def _d1(F, K, T, sigma):
    return (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))


def black_price(F, K, T, sigma, is_call):
    """Undiscounted Black-76 price. is_call: bool or boolean array (True = call)."""
    F, K, sigma = np.asarray(F, float), np.asarray(K, float), np.asarray(sigma, float)
    d1 = _d1(F, K, T, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    call = F * ndtr(d1) - K * ndtr(d2)
    put = K * ndtr(-d2) - F * ndtr(-d1)
    return np.where(is_call, call, put)


def black_vega(F, K, T, sigma):
    """d(price)/d(sigma), undiscounted. The same for calls and puts."""
    d1 = _d1(np.asarray(F, float), np.asarray(K, float), T, np.asarray(sigma, float))
    return F * np.sqrt(T) * np.exp(-0.5 * d1**2) / np.sqrt(2.0 * np.pi)


def black_implied_vol(prices, F, Ks, T, is_call):
    """Implied vols of undiscounted prices (NaN where the price carries no vol information)."""
    prices, Ks = np.atleast_1d(np.asarray(prices, float)), np.atleast_1d(np.asarray(Ks, float))
    is_call = np.broadcast_to(is_call, Ks.shape)
    out = np.full(Ks.shape, np.nan)
    for j, (p, K, c) in enumerate(zip(prices, Ks, is_call)):
        try:
            # Black-76 on F with zero rate is Black-Scholes with S0 = F and r = 0
            out[j] = implied_vol(OptionType.Call if c else OptionType.Put, p, F, K, T, 0.0)
        except ValueError:
            pass
    return out
