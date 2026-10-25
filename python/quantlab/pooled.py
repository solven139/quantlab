"""Pooled calibration: several snapshots at once, some parameters shared by all of them.

Why
---
One expiry's smile cannot pin down every parameter of a rich model such as Bates: a
profile of the crash intensity lambda is flat (frequent small crashes fit as well as rare
large ones). Holding the hard-to-identify parameters (for example kappa and the jump size)
common to many snapshots, while the rest move per snapshot, lets the cross-section of smiles
identify them.

Parameter vector
----------------
p = [shared (n_shared), per_0 (n_per), per_1 (n_per), ..., per_{n-1} (n_per)]

The residuals of snapshot i depend only on the shared block and on per_i, so the Jacobian is
sparse. Telling least_squares that (jac_sparsity) means it perturbs every snapshot's own
parameters at the same time when estimating derivatives: the cost grows like
n_shared + n_per function evaluations instead of n_shared + n * n_per.

Conventions are those of calibration.py: forward terms (models built with r = 0, priced with
S0 = F), out-of-the-money options only, residuals = price error / Black vega (~ vol error).
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from .black import black_implied_vol, black_price, black_vega
from .calibration import otm_prices


@dataclass
class PooledResult:
    shared: np.ndarray          # fitted shared parameters, shape (n_shared,)
    per: np.ndarray             # fitted per-snapshot parameters, shape (n_snapshots, n_per)
    rmse_vol: np.ndarray        # implied-vol RMSE of each snapshot (0.01 = 1 vol pt)
    rmse_vol_all: float         # over every quote of every snapshot
    success: bool
    message: str
    nfev: int
    shared_names: tuple = field(default=())
    per_names: tuple = field(default=())

    def table(self):
        """Per-snapshot parameters and fit errors as a list of dicts (easy to turn into a DataFrame)."""
        names = self.per_names or tuple(f"p{j}" for j in range(self.per.shape[1]))
        return [dict(zip(names, row), rmse_vol=e) for row, e in zip(self.per, self.rmse_vol)]


def _broadcast_per(per0, n, n_per):
    per0 = np.asarray(per0, float)
    if per0.ndim == 1:
        return np.tile(per0, (n, 1))
    if per0.shape != (n, n_per):
        raise ValueError("calibrate_pooled: per0 must have shape (n_per,) or (n_snapshots, n_per)")
    return per0


def calibrate_pooled(model_factory, shared0, per0, shared_bounds, per_bounds, snapshots, *,
                     N=256, L=12.0, shared_names=(), per_names=(), vega_floor=1e-6,
                     **least_squares_kwargs):
    """Fit shared and per-snapshot parameters to several snapshots at once.

    model_factory : callable (shared, per, snap) -> Model built with r = 0. It receives the
                    snapshot dict too, so a snapshot-specific fixed input (e.g. v0 anchored to
                    realized variance) can be read from it.
    shared0, per0 : starting values; per0 is (n_per,) for all snapshots or (n_snapshots, n_per)
    shared_bounds, per_bounds : (lower, upper) pairs for each block
    snapshots     : list of dicts with keys F, Ks, T, market_ivs (one expiry each)
    """
    snaps = list(snapshots)
    n = len(snaps)
    shared0 = np.asarray(shared0, float)
    n_shared = len(shared0)
    per0 = _broadcast_per(per0, n, np.size(per_bounds[0]))
    n_per = per0.shape[1]

    # 1. Market side, once per snapshot: OTM Black prices and vegas at the market vols
    targets = []
    for s in snaps:
        F, K, T, iv = float(s["F"]), np.asarray(s["Ks"], float), float(s["T"]), np.asarray(s["market_ivs"], float)
        is_call = K >= F
        targets.append((F, K, T, iv, black_price(F, K, T, iv, is_call),
                        np.maximum(black_vega(F, K, T, iv), vega_floor * F)))

    # 2. Pack and unpack the parameter vector
    def unpack(p):
        return p[:n_shared], p[n_shared:].reshape(n, n_per)

    # 3. Residuals: every snapshot's vega-scaled price errors, stacked
    def residuals(p):
        shared, per = unpack(p)
        out = []
        for i, (F, K, T, _, market, vega) in enumerate(targets):
            model = model_factory(shared, per[i], snaps[i])
            out.append((otm_prices(model, F, K, T, N, L) - market) / vega)
        return np.concatenate(out)

    # 4. Jacobian sparsity: snapshot i's rows depend on the shared block and on its own block
    sizes = [len(t[1]) for t in targets]
    sparsity = lil_matrix((sum(sizes), n_shared + n * n_per), dtype=int)
    row = 0
    for i, size in enumerate(sizes):
        sparsity[row:row + size, :n_shared] = 1
        sparsity[row:row + size, n_shared + i * n_per: n_shared + (i + 1) * n_per] = 1
        row += size

    # 5. Fit
    p0 = np.concatenate([shared0, per0.ravel()])
    lower = np.concatenate([np.asarray(shared_bounds[0], float), np.tile(np.asarray(per_bounds[0], float), n)])
    upper = np.concatenate([np.asarray(shared_bounds[1], float), np.tile(np.asarray(per_bounds[1], float), n)])
    least_squares_kwargs.setdefault("x_scale", "jac")
    res = least_squares(residuals, np.clip(p0, lower, upper), bounds=(lower, upper),
                        jac_sparsity=sparsity, **least_squares_kwargs)

    # 6. Fit quality in implied-vol terms, per snapshot
    shared, per = unpack(res.x)
    errs = []
    for i, (F, K, T, iv, _, _) in enumerate(targets):
        model = model_factory(shared, per[i], snaps[i])
        errs.append(black_implied_vol(otm_prices(model, F, K, T, N, L), F, K, T, K >= F) - iv)
    rmse = np.array([np.sqrt(np.nanmean(e ** 2)) for e in errs])
    rmse_all = float(np.sqrt(np.nanmean(np.concatenate(errs) ** 2)))
    return PooledResult(shared.copy(), per.copy(), rmse, rmse_all, bool(res.success),
                        str(res.message), int(res.nfev), tuple(shared_names), tuple(per_names))