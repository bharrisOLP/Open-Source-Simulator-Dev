"""3-D steady-state solves: conjugate gradients as one jitted loop on the device, preconditioned with 3-D
aggregation multigrid (default; thermal_gpu.multigrid3d) or Jacobi. The same method as thermal_gpu.solver (2-D).

    res = solve_steady3d(x, y, z, kappa, q, bc)        # res.T: (nx, ny, nz) temperatures at the cell centers

The loop runs on the device (jax.lax.while_loop): no host synchronization per iteration.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from thermal_gpu.fv3d import DeviceProblem3, apply3, build3, to_device3
from thermal_gpu.multigrid3d import CYCLES, Level3, build_hierarchy3, mg_cycle3
from thermal_gpu.solver import PRECONDITIONERS, SteadyResult, solver_device


def _cg(apply_fn, precond_fn, b, x0, tol, max_iter):
    """Preconditioned CG for A x = b. Stops at |r| <= tol |b| (the recursive residual) or after max_iter
    iterations. Returns x, the iteration count and the true relative residual |b - A x| / |b|."""
    r = b - apply_fn(x0)
    z = precond_fn(r)
    bnorm = jnp.linalg.norm(b)
    bnorm = jnp.where(bnorm > 0, bnorm, 1.0)          # b = 0: the absolute residual

    def cond(state):
        _, r, _, _, it = state
        return (jnp.linalg.norm(r) > tol * bnorm) & (it < max_iter)

    def body(state):
        x, r, p, rz, it = state
        ap = apply_fn(p)
        alpha = rz / jnp.vdot(p, ap)
        x = x + alpha * p
        r = r - alpha * ap
        z = precond_fn(r)
        rz_new = jnp.vdot(r, z)
        return x, r, z + (rz_new / rz) * p, rz_new, it + 1

    x, _, _, _, it = jax.lax.while_loop(cond, body, (x0, r, z, jnp.vdot(r, z), jnp.asarray(0)))
    return x, it, jnp.linalg.norm(b - apply_fn(x)) / bnorm


@jax.jit
def pcg3d(dp: DeviceProblem3, x0: jax.Array, tol: jax.Array, max_iter: jax.Array):
    """Jacobi-preconditioned CG."""
    return _cg(lambda v: apply3(dp, v), lambda r: r / dp.diag, dp.b, x0, tol, max_iter)


@partial(jax.jit, static_argnames=("cycle",))
def pcg3d_mg(dp: DeviceProblem3, levels, coarse_inv, x0: jax.Array, tol: jax.Array, max_iter: jax.Array,
             cycle: str = "V"):
    """CG preconditioned with the 3-D aggregation-multigrid cycle (thermal_gpu.multigrid3d; "V" or "W")."""
    return _cg(lambda v: apply3(dp, v), lambda r: mg_cycle3(levels, coarse_inv, r, cycle), dp.b, x0, tol, max_iter)


def prepare3(x, y, z, kappa, q, bc: dict, dtype=jnp.float64, precond: str = "mg", device=None):
    """The device problem and, for "mg", the multigrid levels and the coarsest inverse (else None, None)."""
    if precond not in PRECONDITIONERS:
        raise ValueError(f"precond must be one of {PRECONDITIONERS}, not {precond!r}")
    dev = device or solver_device()
    p = build3(x, y, z, kappa, q, bc)
    dp = to_device3(p, dtype, dev)
    if precond == "mg":
        levels, cinv = build_hierarchy3(p.g, p.diag_extra, dev, dtype)
        # the fine level holds the same arrays as dp: share them (less device memory)
        levels = (Level3(dp.gx, dp.gy, dp.gz, dp.diag, levels[0].agg),) + levels[1:]
        return dp, levels, cinv, dev
    return dp, None, None, dev


def solve_steady3d(x, y, z, kappa, q, bc: dict, tol: float = 1e-12, max_iter: int | None = None,
                   dtype=jnp.float64, t0: np.ndarray | None = None, precond: str = "mg",
                   cycle: str = "V") -> SteadyResult:
    """The 3-D steady temperature. x, y, z: cell edges; kappa, q: (nx, ny, nz); bc: see thermal_gpu.fv3d.
    precond: "mg" (aggregation multigrid, max_iter 2000 by default) or "jacobi" (max_iter 200000). cycle: the
    multigrid cycle, "V" (default) or "W" (fewer iterations on deep hierarchies; see thermal_gpu.multigrid3d).
    SteadyResult.residual is the true relative residual |b - A T| / |b| after the loop; converged is False when
    it is above tol (at max_iter, and in float32, which cannot reach 1e-12)."""
    if cycle not in CYCLES:
        raise ValueError(f"cycle must be one of {CYCLES}, not {cycle!r}")
    dp, levels, cinv, dev = prepare3(x, y, z, kappa, q, bc, dtype, precond)
    shape = dp.b.shape
    x0 = jax.device_put(jnp.zeros(shape, dtype) if t0 is None else jnp.asarray(t0, dtype), dev)
    tol_d = jnp.asarray(tol, dtype)
    if precond == "mg":
        T, it, res = pcg3d_mg(dp, levels, cinv, x0, tol_d, jnp.asarray(max_iter or 2000), cycle)
    else:
        T, it, res = pcg3d(dp, x0, tol_d, jnp.asarray(max_iter or 200000))
    return SteadyResult(np.asarray(T), int(it), float(res), str(dev), bool(float(res) <= tol))
