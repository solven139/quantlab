import numpy as np
import pytest
from quantlab import (BlackScholesModel, CirScheme, HestonModel, LocalVolSurface, OptionType,
                      cos_price, dupire_local_vol, heston_at_times, implied_vol,
                      local_vol_at_times)

# Book Ch 10.2 experiment (Fig 10.2): Heston prices play the role of the market
H = HestonModel(r=0.0, kappa=1.3, vbar=0.05, gamma=0.3, rho=-0.3, v0=0.1)
KS = list(np.linspace(0.3, 2.5, 45))
TS = list(np.linspace(0.05, 3.0, 30))


@pytest.mark.parametrize("K, T", [(80.0, 0.5), (100.0, 1.0), (130.0, 2.0)])
def test_dupire_recovers_constant_black_scholes_vol(K, T):
    # If every option is priced with one sigma, the local vol must be that sigma everywhere
    bs = BlackScholesModel(r=0.03, sigma=0.25)
    assert dupire_local_vol(bs, 100.0, K, T, 1.0, 0.01) == pytest.approx(0.25, abs=1e-4)


def test_dupire_heston_local_vol_has_a_skew():
    # rho < 0: local vol is higher for low strikes than for high strikes
    low = dupire_local_vol(H, 1.0, 0.8, 1.0, 0.01, 0.01)
    high = dupire_local_vol(H, 1.0, 1.2, 1.0, 0.01, 0.01)
    assert low > high > 0.0


def test_surface_interpolation():
    s = LocalVolSurface(BlackScholesModel(r=0.0, sigma=0.2), 1.0, KS, TS)
    assert np.allclose(s.values(), 0.2, atol=3e-3)   # flat model, flat surface (FD error in far wings)
    sh = LocalVolSurface(H, 1.0, KS, TS)
    V = sh.values()
    assert sh.sigma(KS[10], TS[5]) == pytest.approx(V[5, 10])          # exact on grid points
    mid = sh.sigma(0.5 * (KS[10] + KS[11]), TS[5])
    assert min(V[5, 10], V[5, 11]) <= mid <= max(V[5, 10], V[5, 11])  # between neighbours
    assert sh.sigma(10.0, 50.0) == pytest.approx(V[-1, -1])          # flat extrapolation


def test_local_vol_reproduces_todays_heston_smile():
    surf = LocalVolSurface(H, 1.0, KS, TS)
    ST = local_vol_at_times(surf, 1.0, 0.0, [1.0], 100, 100_000, seed=1)[:, 0]
    for k in [0.7, 0.9, 1.0, 1.1, 1.4]:
        typ = OptionType.Put if k < 1.0 else OptionType.Call
        pay = np.maximum(ST - k, 0.0) if typ == OptionType.Call else np.maximum(k - ST, 0.0)
        iv_lv = implied_vol(typ, pay.mean(), 1.0, k, 1.0, 0.0)
        iv_h = implied_vol(typ, cos_price(H, typ, 1.0, k, 1.0, N=256, L=12.0), 1.0, k, 1.0, 0.0)
        assert iv_lv == pytest.approx(iv_h, abs=0.004)


def forward_smile(paths, ks, tau):
    R = paths[:, 1] / paths[:, 0]        # return from T1 to T2
    out = []
    for k in ks:
        typ = OptionType.Put if k < 1.0 else OptionType.Call
        pay = np.maximum(R - k, 0.0) if typ == OptionType.Call else np.maximum(k - R, 0.0)
        out.append(implied_vol(typ, pay.mean(), 1.0, k, tau, 0.0))
    return np.array(out)


def test_local_vol_forward_smile_is_flatter_than_heston():
    # Same prices today, different future: the key message of book Ch 10.2
    ks = [0.7, 0.85, 1.0, 1.15, 1.3]
    surf = LocalVolSurface(H, 1.0, KS, TS)
    ph = heston_at_times(CirScheme.QE, 1.0, 0.0, 1.3, 0.05, 0.3, -0.3, 0.1, [1.0, 2.0], 64,
                         100_000, seed=2)
    pl = local_vol_at_times(surf, 1.0, 0.0, [1.0, 2.0], 64, 100_000, seed=3)
    fh, fl = forward_smile(ph, ks, 1.0), forward_smile(pl, ks, 1.0)
    assert (fh.max() - fh.min()) > 1.3 * (fl.max() - fl.min())
