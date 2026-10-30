"""Aggregation multigrid preconditioner for 5-point conductance operators on rectilinear grids (GPU, JAX).

The operator (A v)_c = D_c v_c - sum_faces G_f v_nb, with D_c = sum of its face conductances + an extra diagonal term
(boundary conductances, the Newton term of charge-gpu, or C/dt of a transient). Coarsening joins 2 x 2 cells into
one (a last single row or column when the size is odd). With piecewise-constant interpolation P, the Galerkin
coarse operator P^T A P is again such an operator: the conductances across the aggregate boundaries add, the
internal ones cancel, the extra terms add. So the coarse levels keep the large contrasts exactly (no averaging of
coefficients).

The preconditioner is a symmetric V-cycle: damped Jacobi before and after, an over-corrected coarse correction
(aggregation multigrid needs a scale factor, here 1.6, because constant interpolation underestimates smooth
errors), and a dense solve on the coarsest level. Symmetric, so it is a valid preconditioner for CG.

Limits:
- The iteration count depends on the cells. On the heater stack (square cells) CG needs 17 to 27 iterations from 50
  to 5 nm (about 3 more per halving of the cell). Cells with a large aspect ratio need more (a review test: 143 at
  1:5, 624 at 1:25), because 2 x 2 aggregation does not follow the strong direction.
- The coarsest solve is a pseudo-inverse with the default cutoff of numpy. A near-null mode of the coarsest matrix
  (pure Neumann with only a tiny Robin or extra term, about 1e-12 of the face conductances) is dropped. Then the
  preconditioner is singular in that mode, and CG cannot reduce it; the solvers then report converged = False (they
  check the true residual). Realistic convective tops (h of 0.01 W/m^2/K and more) are far above this limit.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

OMEGA = 0.8          # Jacobi damping
SWEEPS = 2           # Jacobi sweeps before and after
OVERCORRECT = 1.6    # coarse-correction scale (aggregation multigrid)
COARSEST = 1024      # cells: a dense solve below this size


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Level:
    gx: jax.Array        # (nx - 1, ny) interior x-face conductances
    gy: jax.Array        # (nx, ny - 1)
    diag: jax.Array      # (nx, ny)


def _coarsen(gx: np.ndarray, gy: np.ndarray, extra: np.ndarray):
    """The Galerkin coarse operator of 2 x 2 aggregation (2 x 1 or 1 x 2 when an axis has one cell), on the host."""
    nx, ny = extra.shape
    ax, ay = (2 if nx > 1 else 1), (2 if ny > 1 else 1)
    cx, cy = -(-nx // ax), -(-ny // ay)

    def sum_groups(a, axis, k):
        if k == 1:
            return a
        n = a.shape[axis]
        if n % k:
            pad = [(0, 0)] * a.ndim
            pad[axis] = (0, k - n % k)
            a = np.pad(a, pad)
        s = list(a.shape)
        s[axis:axis + 1] = [s[axis] // k, k]
        return a.reshape(s).sum(axis=axis + 1)

    # faces between aggregates: the fine faces between cells k*i + k - 1 and k*i + k
    gx_c = sum_groups(gx[ax - 1::ax], 1, ay)[: cx - 1] if nx > 1 else np.zeros((0, cy))
    gy_c = sum_groups(gy[:, ay - 1::ay], 0, ax)[:, : cy - 1] if ny > 1 else np.zeros((cx, 0))
    extra_c = sum_groups(sum_groups(extra, 0, ax), 1, ay)
    return gx_c, gy_c, extra_c


def _diag(gx, gy, extra):
    d = extra.copy()
    d[:-1] += gx
    d[1:] += gx
    d[:, :-1] += gy
    d[:, 1:] += gy
    return d


def build_hierarchy(gx: np.ndarray, gy: np.ndarray, extra: np.ndarray, device=None, dtype=jnp.float64):
    """Levels (fine first) and the dense inverse of the coarsest operator."""
    put = (lambda a: jax.device_put(jnp.asarray(a, dtype), device)) if device else (lambda a: jnp.asarray(a, dtype))
    levels = []
    while True:
        levels.append(Level(put(gx), put(gy), put(_diag(gx, gy, extra))))
        if extra.size <= COARSEST:
            break
        gx, gy, extra = _coarsen(gx, gy, extra)
    # the dense coarsest matrix
    nx, ny = extra.shape
    idx = np.arange(nx * ny).reshape(nx, ny)
    a = np.diag(_diag(gx, gy, extra).ravel())
    a[idx[:-1].ravel(), idx[1:].ravel()] -= gx.ravel()
    a[idx[1:].ravel(), idx[:-1].ravel()] -= gx.ravel()
    a[idx[:, :-1].ravel(), idx[:, 1:].ravel()] -= gy.ravel()
    a[idx[:, 1:].ravel(), idx[:, :-1].ravel()] -= gy.ravel()
    return tuple(levels), put(np.linalg.pinv(a))


def _apply(lv: Level, v):
    out = lv.diag * v
    out = out.at[:-1].add(-lv.gx * v[1:]).at[1:].add(-lv.gx * v[:-1])
    return out.at[:, :-1].add(-lv.gy * v[:, 1:]).at[:, 1:].add(-lv.gy * v[:, :-1])


def _restrict(r):
    """P^T r: the sum over each aggregate (2 cells per axis that has more than one cell)."""
    nx, ny = r.shape
    ax, ay = (2 if nx > 1 else 1), (2 if ny > 1 else 1)
    r = jnp.pad(r, ((0, (-nx) % ax), (0, (-ny) % ay)))
    return r.reshape(r.shape[0] // ax, ax, r.shape[1] // ay, ay).sum(axis=(1, 3))


def _prolong(e, shape):
    """P e: the aggregate value in each of its cells."""
    ax, ay = (2 if shape[0] > 1 else 1), (2 if shape[1] > 1 else 1)
    return jnp.repeat(jnp.repeat(e, ax, axis=0), ay, axis=1)[: shape[0], : shape[1]]


def vcycle(levels: tuple, coarse_inv, r):
    """M^-1 r (symmetric)."""
    def cycle(k, r):
        lv = levels[k]
        if k == len(levels) - 1:
            return (coarse_inv @ r.ravel()).reshape(r.shape)
        x = jnp.zeros_like(r)
        for _ in range(SWEEPS):
            x = x + OMEGA * (r - _apply(lv, x)) / lv.diag
        e = cycle(k + 1, _restrict(r - _apply(lv, x)))
        x = x + OVERCORRECT * _prolong(e, r.shape)
        for _ in range(SWEEPS):
            x = x + OMEGA * (r - _apply(lv, x)) / lv.diag
        return x
    return cycle(0, r)
