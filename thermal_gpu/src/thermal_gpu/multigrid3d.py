"""Aggregation multigrid preconditioner for 7-point conductance operators on 3-D rectilinear grids (GPU, JAX).

The same method as thermal_gpu.multigrid (2-D). The operator (A v)_c = D_c v_c - sum_faces G_f v_nb, with D_c =
the sum of its face conductances + an extra diagonal term (boundary conductances). Coarsening joins 2 x 2 x 2
cells into one (a last single layer when a size is odd). With piecewise-constant interpolation P, the Galerkin
coarse operator P^T A P is again a 7-point conductance operator: the conductances across the aggregate
boundaries add, the internal ones cancel, the extra terms add. So the coarse levels keep the large contrasts
exactly.

Anisotropic cells: an axis is aggregated only when its typical face conductance (the median) is at least
THETA of that of the strongest axis (semi-coarsening). Cells that are long along z (the waveguide axis of a
phase shifter) have a weak z coupling; the coarse levels then join cells only in x and y, until the summed
z conductances are strong enough. On cells with an aspect ratio of 2 or less (and one material) this gives
the plain 2 x 2 x 2 aggregation. An axis with one cell is never aggregated.

The preconditioner is the symmetric V-cycle of the 2-D module: damped Jacobi before and after, an
over-corrected coarse correction, and a dense solve (pseudo-inverse) on the coarsest level. The constants
OMEGA, SWEEPS, OVERCORRECT and COARSEST are those of thermal_gpu.multigrid.

Eigenvalue bounds (B = this preconditioner, A nonsingular, the coarsest pseudo-inverse = the inverse):
- Each operator is diagonally dominant, so lambda(D^-1 A) is in (0, 2] (Gershgorin), and the damped Jacobi
  step G = I - OMEGA D^-1 A has eigenvalues in [1 - 2 OMEGA, 1) = [-0.6, 1). Let C be the approximate coarse
  inverse that a level uses. If lambda(C A_c) is in (0, m], the coarse step K = I - OVERCORRECT P C P^T A has
  eigenvalues in [1 - 1.6 m, 1], and the A-norm of G is at most 1. So the error operator E = G^2 K G^2 has
  eigenvalues in [min(0, 1 - 1.6 m), 1), and lambda(B A) = 1 - lambda(E) is in (0, max(1, 1.6 m)].
- V-cycle: C is the V-cycle of the next level, and the coarsest solve is exact (m = 1). By induction,
  lambda(B A) is in (0, 1.6^(L - 1)] for L levels: B is positive definite, but the largest eigenvalue can
  grow by up to 1.6 per level.
- W-cycle: C = 2 B_c - B_c A_c B_c (the two coarse cycles), so lambda(C A_c) = 1 - (1 - lambda(B_c A_c))^2. If
  lambda(B_c A_c) is in (0, 1.6], this is in (0, 1] (m = 1). By induction, lambda(B A) is in (0, 1.6] on each
  level: the W-cycle is positive definite with a bound that does not grow with the number of levels.

Limits:
- V-cycle iterations grow on hierarchies with levels that do not aggregate all axes (semi-coarsening). A mode
  that changes only along an axis that is not aggregated is exact on the coarse level, so the over-correction
  1.6 overshoots on it; this compounds over such levels. On the 3-D heater (cells long in z, all levels (2, 2, 1))
  the Lanczos estimates of lambda(B A) from CG are 0.29 to 4.2 at 100 nm and 0.18 to 7.0 at 50 nm: the condition
  number (14, then 38) grows at both ends, and CG needs 41, 62 and 88 iterations at 100, 50 and 25 nm (the 2-D
  cross-section: 17 at 50 nm). On random test hierarchies the largest eigenvalue reaches 29. An over-correction
  of 1.0 on these levels limits the largest eigenvalue to 1, but the smallest falls to 0.04 and 0.019, and CG
  needs more iterations (58 and 83); so the constant stays 1.6 on all levels.
- The option cycle="W" (two coarse cycles per level) needs 14 iterations on the heater at 50 nm and 15 at 25 nm,
  at about 0.4 of the V-cycle time on a 16-thread CPU (0.6 and 0.35 on the GPU). Its cost per iteration grows as
  2^levels on the coarse levels (many small kernels on a GPU), and the traced program is larger (the first GPU
  solve, with compilation, takes 8 to 76 s).
- The limits of the 2-D module apply (the pseudo-inverse drops a near-null mode of the coarsest matrix; then the
  bounds above do not apply to that mode).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from thermal_gpu.fv3d import apply_op, assemble_op, full_diag
from thermal_gpu.multigrid import COARSEST, OMEGA, OVERCORRECT, SWEEPS

THETA = 0.25         # aggregate an axis when its median face conductance is >= THETA x that of the strongest axis
CYCLES = ("V", "W")


@dataclass(frozen=True)
class Level3:
    gx: jax.Array        # (nx - 1, ny, nz) interior face conductances
    gy: jax.Array        # (nx, ny - 1, nz)
    gz: jax.Array        # (nx, ny, nz - 1)
    diag: jax.Array      # (nx, ny, nz)
    agg: tuple = (1, 1, 1)   # aggregation factors to the next coarser level (static)


jax.tree_util.register_dataclass(Level3, data_fields=["gx", "gy", "gz", "diag"], meta_fields=["agg"])


def choose_factors(g: tuple, shape: tuple) -> tuple:
    """The aggregation factor (1 or 2) of each axis: 2 when the axis has more than one cell and its median face
    conductance is at least THETA of the strongest axis that has more than one cell."""
    live = [ax for ax in range(3) if shape[ax] > 1]
    if not live:
        return (1, 1, 1)
    s = {ax: float(np.median(g[ax])) for ax in live}
    smax = max(s.values())
    return tuple(2 if ax in s and s[ax] >= THETA * smax else 1 for ax in range(3))


def _sum_groups(a: np.ndarray, axis: int, k: int) -> np.ndarray:
    """Sum of groups of k consecutive entries along axis (zero padding at the end)."""
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


def coarsen3(g: tuple, extra: np.ndarray, agg: tuple):
    """The Galerkin coarse operator P^T A P of the aggregation with the factors agg, on the host. Returns the
    coarse face conductances (per axis) and the coarse extra diagonal term."""
    shape = extra.shape
    cshape = tuple(-(-n // a) for n, a in zip(shape, agg))
    gc = []
    for ax in range(3):
        a = agg[ax]
        sel = [slice(None)] * 3
        sel[ax] = slice(a - 1, None, a)               # the fine faces between cells a*i + a - 1 and a*i + a
        f = g[ax][tuple(sel)]
        for other in range(3):
            if other != ax:
                f = _sum_groups(f, other, agg[other])
        cut = [slice(None)] * 3
        cut[ax] = slice(0, cshape[ax] - 1)
        gc.append(f[tuple(cut)])
    extra_c = extra
    for ax in range(3):
        extra_c = _sum_groups(extra_c, ax, agg[ax])
    return tuple(gc), extra_c


def build_hierarchy3(g: tuple, extra: np.ndarray, device=None, dtype=jnp.float64, coarsest: int = COARSEST):
    """Levels (fine first) and the dense pseudo-inverse of the coarsest operator (at most `coarsest` cells)."""
    put = (lambda a: jax.device_put(jnp.asarray(a, dtype), device)) if device else (lambda a: jnp.asarray(a, dtype))
    g = tuple(np.asarray(ga, float) for ga in g)
    extra = np.asarray(extra, float)
    levels = []
    while True:
        last = extra.size <= coarsest
        agg = (1, 1, 1) if last else choose_factors(g, extra.shape)
        levels.append(Level3(*(put(ga) for ga in g), put(full_diag(g, extra)), agg))
        if last:
            break
        g, extra = coarsen3(g, extra, agg)
    a = assemble_op(g, full_diag(g, extra)).toarray()
    return tuple(levels), put(np.linalg.pinv(a))


def _apply(lv: Level3, v):
    return apply_op((lv.gx, lv.gy, lv.gz), lv.diag, v)


def _restrict(r, agg):
    """P^T r: the sum over each aggregate."""
    r = jnp.pad(r, [(0, (-n) % a) for n, a in zip(r.shape, agg)])
    s = r.shape
    return r.reshape(s[0] // agg[0], agg[0], s[1] // agg[1], agg[1], s[2] // agg[2], agg[2]).sum(axis=(1, 3, 5))


def _prolong(e, agg, shape):
    """P e: the aggregate value in each of its cells."""
    for ax in range(3):
        if agg[ax] > 1:
            e = jnp.repeat(e, agg[ax], axis=ax)
    return e[: shape[0], : shape[1], : shape[2]]


def mg_cycle3(levels: tuple, coarse_inv, r, cycle: str = "V"):
    """M^-1 r (symmetric). cycle "V": one coarse cycle per level (the method of the 2-D module). "W": two coarse
    cycles on each level above the coarsest, e = B r_c, then e += B (r_c - A_c e)."""
    if cycle not in CYCLES:
        raise ValueError(f"cycle must be one of {CYCLES}, not {cycle!r}")

    def run(k, r):
        lv = levels[k]
        if k == len(levels) - 1:
            return (coarse_inv @ r.ravel()).reshape(r.shape)
        x = jnp.zeros_like(r)
        for _ in range(SWEEPS):
            x = x + OMEGA * (r - _apply(lv, x)) / lv.diag
        rc = _restrict(r - _apply(lv, x), lv.agg)
        e = run(k + 1, rc)
        if cycle == "W" and k + 2 < len(levels):
            e = e + run(k + 1, rc - _apply(levels[k + 1], e))
        x = x + OVERCORRECT * _prolong(e, lv.agg, r.shape)
        for _ in range(SWEEPS):
            x = x + OMEGA * (r - _apply(lv, x)) / lv.diag
        return x
    return run(0, r)


def vcycle3(levels: tuple, coarse_inv, r):
    """M^-1 r with the V-cycle."""
    return mg_cycle3(levels, coarse_inv, r, "V")
