"""Steady-state solves: conjugate gradients as one jitted loop on the device, preconditioned with aggregation
multigrid (default; thermal_gpu.multigrid) or Jacobi.

    T = solve_steady(x, y, kappa, q, bc)        # (nx, ny) temperatures at the cell centers

The loop runs on the device (jax.lax.while_loop): no host synchronization per iteration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from thermal_gpu.fv import DeviceProblem, apply, build, to_device


def solver_device() -> jax.Device:
    """The first GPU (ROCm or CUDA), else the CPU. THERMAL_GPU_DEVICE=cpu forces the CPU."""
    if os.environ.get("THERMAL_GPU_DEVICE", "").lower() != "cpu":
        try:
            return jax.devices("gpu")[0]
        except RuntimeError:
            pass
    return jax.devices("cpu")[0]


@jax.jit
def pcg(dp: DeviceProblem, x0: jax.Array, tol: jax.Array, max_iter: jax.Array):
    """Jacobi-preconditioned CG for A x = b. Stops at |r| <= tol |b| or after max_iter iterations."""
    minv = 1.0 / dp.diag
    r = dp.b - apply(dp, x0)
    z = minv * r
    p = z
    rz = jnp.vdot(r, z)
    bnorm = jnp.linalg.norm(dp.b)

    def cond(state):
        _, r, _, _, it = state
        return (jnp.linalg.norm(r) > tol * bnorm) & (it < max_iter)

    def body(state):
        x, r, p, rz, it = state
        ap = apply(dp, p)
        alpha = rz / jnp.vdot(p, ap)
        x = x + alpha * p
        r = r - alpha * ap
        z = minv * r
        rz_new = jnp.vdot(r, z)
        p = z + (rz_new / rz) * p
        return x, r, p, rz_new, it + 1

    x, _, _, _, it = jax.lax.while_loop(cond, body, (x0, r, p, rz, jnp.asarray(0)))
    return x, it, jnp.linalg.norm(dp.b - apply(dp, x)) / bnorm


@dataclass
class SteadyResult:
    T: np.ndarray
    iterations: int
    residual: float          # the true relative residual |b - A T| / |b|, computed once after the CG loop
    device: str
    converged: bool = True   # residual <= tol (False at max_iter, and when the precision cannot reach tol)


PRECONDITIONERS = ("mg", "jacobi")


def solve_steady(x, y, kappa, q, bc: dict, tol: float = 1e-12, max_iter: int | None = None, dtype=jnp.float64,
                 t0: np.ndarray | None = None, precond: str = "mg") -> SteadyResult:
    """precond: "mg" (aggregation multigrid; 17 to 27 iterations on the heater stack from 50 to 5 nm cells, more on
    cells with a large aspect ratio) or "jacobi"."""
    if precond not in PRECONDITIONERS:
        raise ValueError(f"precond must be one of {PRECONDITIONERS}, not {precond!r}")
    if precond == "mg":
        return solve_steady_mg(x, y, kappa, q, bc, tol, max_iter or 2000, dtype, t0)
    max_iter = max_iter or 200000
    dev = solver_device()
    p = build(x, y, kappa, q, bc)
    dp = to_device(p, dtype, dev)
    x0 = jax.device_put(jnp.zeros(p.shape, dtype) if t0 is None else jnp.asarray(t0, dtype), dev)
    T, it, res = pcg(dp, x0, jnp.asarray(tol, dtype), jnp.asarray(max_iter))
    return SteadyResult(np.asarray(T), int(it), float(res), str(dev), bool(float(res) <= tol))


@dataclass
class TransientResult:
    times: np.ndarray        # (nt,)
    probes: np.ndarray       # (nt, n_probes): T at the probe cells
    T: np.ndarray            # the final field
    iterations: int          # total CG iterations
    device: str
    max_residual: float = 0.0   # the largest true relative residual of a step
    converged: bool = True      # every step reached tol


def solve_transient(x, y, kappa, q, bc: dict, rho_cp, t_end: float, dt: float, probes=(), t0_field=None,
                    tol: float = 1e-12, dtype=jnp.float64) -> TransientResult:
    """Implicit Euler: (C/dt + A) T_new = (C/dt) T_old + b, C = rho_cp * cell area (J/m/K per unit length).
    Each step: multigrid-preconditioned CG from the previous temperature.
    rho_cp: (nx, ny) volumetric heat capacity (J/m^3/K). probes: list of (i, j) cells to record. The initial field
    is t0_field, or the Dirichlet temperature of the boundary conditions (the first one found), or 0."""
    dev = solver_device()
    p = build(x, y, kappa, q, bc)
    dx, dy = np.diff(np.asarray(x, float)), np.diff(np.asarray(y, float))
    c = np.asarray(rho_cp, float) * dx[:, None] * dy[None, :] / dt
    dp = to_device(p, dtype, dev)
    dp_step = DeviceProblem(dp.gx, dp.gy, dp.diag + jax.device_put(jnp.asarray(c, dtype), dev), dp.b)
    c_dev = jax.device_put(jnp.asarray(c, dtype), dev)
    if t0_field is None:
        tb = next((s[1] for s in bc.values() if s[0] == "dirichlet"), 0.0)
        t0_field = np.full(p.shape, float(tb))
    t = jax.device_put(jnp.asarray(t0_field, dtype), dev)
    n_steps = int(round(t_end / dt))
    times, rec, total = [0.0], [np.asarray(t)[tuple(np.array(probes).T)] if probes else np.zeros(0)], 0
    worst = 0.0
    from thermal_gpu.multigrid import build_hierarchy
    levels, cinv = build_hierarchy(p.gx[1:-1], p.gy[:, 1:-1], p.diag_extra + c, dev, dtype)   # C/dt: extra diagonal
    for s in range(1, n_steps + 1):
        rhs = c_dev * t + dp.b
        t, it, res = pcg_mg(DeviceProblem(dp_step.gx, dp_step.gy, dp_step.diag, rhs), levels, cinv, t,
                            jnp.asarray(tol, dtype), jnp.asarray(2000))
        total += int(it)
        worst = max(worst, float(res))
        times.append(s * dt)
        rec.append(np.asarray(t)[tuple(np.array(probes).T)] if probes else np.zeros(0))
    return TransientResult(np.asarray(times), np.asarray(rec), np.asarray(t), total, str(dev), worst, worst <= tol)


@jax.jit
def pcg_mg(dp: DeviceProblem, levels, coarse_inv, x0: jax.Array, tol: jax.Array, max_iter: jax.Array):
    """CG preconditioned with the aggregation-multigrid V-cycle (thermal_gpu.multigrid)."""
    from thermal_gpu.multigrid import vcycle
    r = dp.b - apply(dp, x0)
    z = vcycle(levels, coarse_inv, r)
    p = z
    rz = jnp.vdot(r, z)
    bnorm = jnp.linalg.norm(dp.b)

    def cond(state):
        _, r, _, _, it = state
        return (jnp.linalg.norm(r) > tol * bnorm) & (it < max_iter)

    def body(state):
        x, r, p, rz, it = state
        ap = apply(dp, p)
        alpha = rz / jnp.vdot(p, ap)
        x = x + alpha * p
        r = r - alpha * ap
        z = vcycle(levels, coarse_inv, r)
        rz_new = jnp.vdot(r, z)
        return x, r, z + (rz_new / rz) * p, rz_new, it + 1

    x, _, _, _, it = jax.lax.while_loop(cond, body, (x0, r, p, rz, jnp.asarray(0)))
    return x, it, jnp.linalg.norm(dp.b - apply(dp, x)) / bnorm


def solve_steady_mg(x, y, kappa, q, bc: dict, tol: float = 1e-12, max_iter: int = 2000, dtype=jnp.float64,
                    t0: np.ndarray | None = None) -> SteadyResult:
    """solve_steady with the multigrid preconditioner (far fewer iterations on fine grids)."""
    from thermal_gpu.multigrid import build_hierarchy
    dev = solver_device()
    p = build(x, y, kappa, q, bc)
    dp = to_device(p, dtype, dev)
    levels, cinv = build_hierarchy(p.gx[1:-1], p.gy[:, 1:-1], p.diag_extra, dev, dtype)
    x0 = jax.device_put(jnp.zeros(p.shape, dtype) if t0 is None else jnp.asarray(t0, dtype), dev)
    T, it, res = pcg_mg(dp, levels, cinv, x0, jnp.asarray(tol, dtype), jnp.asarray(max_iter))
    return SteadyResult(np.asarray(T), int(it), float(res), str(dev), bool(float(res) <= tol))
