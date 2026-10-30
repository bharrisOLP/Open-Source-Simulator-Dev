"""Nonlinear Poisson with constant quasi-Fermi levels: carriers of a reverse-biased (or weakly biased) PN junction.

    -div(eps grad psi) = q (p - n + N_D - N_A),   n = n_i exp((psi - phi_n)/V_T),   p = n_i exp((phi_p - psi)/V_T)

With phi_n = V of the n contact and phi_p = V of the p contact everywhere (the standard approximation for reverse
bias: majority quasi-Fermi levels are flat, minority carriers are negligible), the problem has one unknown, psi.
Newton's method: the Jacobian A + diag(q (n + p) V_cell / V_T) is symmetric positive definite (A: the finite-volume
-div eps grad), so each step is a CG solve on the device, preconditioned with aggregation multigrid (thermal_gpu,
default) or Jacobi. Steps are damped
(delta -> V_T sign(delta) log(1 + |delta| / V_T)), which is the usual safeguard for the exponentials.

Discretization: cell-centered finite volumes on a 2-D rectilinear grid (x, y cell edges in m), face coefficient
L / (d1/eps1 + d2/eps2). Ohmic contacts: Dirichlet psi = V + V_T asinh(N_net / (2 n_i)) on a boundary side, on a
mask of boundary cells. SI units throughout (doping in m^-3; 1e18 cm^-3 = 1e24 m^-3).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

Q = 1.602176634e-19
KB = 1.380649e-23
EPS0 = 8.8541878128e-12


def thermal_voltage(T: float = 300.0) -> float:
    return KB * T / Q


def solver_device() -> jax.Device:
    if os.environ.get("CHARGE_GPU_DEVICE", "").lower() != "cpu":
        try:
            return jax.devices("gpu")[0]
        except RuntimeError:
            pass
    return jax.devices("cpu")[0]


@dataclass
class Contact:
    """An ohmic contact on one boundary side, on the boundary cells where ``mask`` (1-D, along the side) is True."""

    side: str            # "xmin", "xmax", "ymin", "ymax"
    mask: np.ndarray
    voltage: float


@dataclass
class Device:
    x: np.ndarray                    # cell edges (m)
    y: np.ndarray
    eps_r: np.ndarray                # (nx, ny) relative permittivity
    n_net: np.ndarray                # (nx, ny) N_D - N_A (m^-3); 0 in insulators
    semiconductor: np.ndarray        # (nx, ny) bool: carriers exist here
    contacts: list[Contact]
    n_i: float = 1.0e16              # Si at 300 K (m^-3), about 1e10 cm^-3
    T: float = 300.0


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class _Ops:
    gx: jax.Array            # interior x-face coefficients (nx - 1, ny)
    gy: jax.Array            # (nx, ny - 1)
    diag: jax.Array          # A diagonal (faces + contacts)
    bc: jax.Array            # contact terms of the right-hand side: sum G_b psi_b
    vol: jax.Array           # cell areas
    n_net: jax.Array
    semi: jax.Array          # 1.0 in semiconductor cells
    phi_n: jax.Array         # quasi-Fermi levels (V), per cell
    phi_p: jax.Array


def _operators(dev: Device, phi_n: float, phi_p: float, dtype):
    """The device operators, and the host (float64) face and contact conductances for the multigrid hierarchy."""
    x, y = np.asarray(dev.x, float), np.asarray(dev.y, float)
    eps = dev.eps_r * EPS0
    nx, ny = eps.shape
    dx, dy = np.diff(x), np.diff(y)
    hx, hy = dx / 2, dy / 2
    gx = dy[None, :] / (hx[:-1, None] / eps[:-1] + hx[1:, None] / eps[1:])
    gy = dx[:, None] / (hy[None, :-1] / eps[:, :-1] + hy[None, 1:] / eps[:, 1:])
    diag = np.zeros((nx, ny))
    diag[:-1] += gx
    diag[1:] += gx
    diag[:, :-1] += gy
    diag[:, 1:] += gy
    gc = np.zeros((nx, ny))                                     # contact conductances (the extra diagonal)
    bc = np.zeros((nx, ny))
    vt = thermal_voltage(dev.T)
    for c in dev.contacts:
        if c.side in ("xmin", "xmax"):
            i = 0 if c.side == "xmin" else nx - 1
            g = dy * eps[i, :] / hx[i]
            psi_b = c.voltage + vt * np.arcsinh(dev.n_net[i, :] / (2 * dev.n_i))
            gc[i, c.mask] += g[c.mask]
            bc[i, c.mask] += (g * psi_b)[c.mask]
        else:
            j = 0 if c.side == "ymin" else ny - 1
            g = dx * eps[:, j] / hy[j]
            psi_b = c.voltage + vt * np.arcsinh(dev.n_net[:, j] / (2 * dev.n_i))
            gc[c.mask, j] += g[c.mask]
            bc[c.mask, j] += (g * psi_b)[c.mask]
    diag += gc
    vol = dx[:, None] * dy[None, :]
    semi = dev.semiconductor.astype(float)
    d = solver_device()
    put = lambda a: jax.device_put(jnp.asarray(a, dtype), d)  # noqa: E731
    ops = _Ops(put(gx), put(gy), put(diag), put(bc), put(vol), put(dev.n_net * semi), put(semi),
               put(np.full((nx, ny), phi_n)), put(np.full((nx, ny), phi_p)))
    return ops, (gx, gy, gc)


def _apply_a(o: _Ops, v):
    out = o.diag * v
    out = out.at[:-1].add(-o.gx * v[1:]).at[1:].add(-o.gx * v[:-1])
    return out.at[:, :-1].add(-o.gy * v[:, 1:]).at[:, 1:].add(-o.gy * v[:, :-1])


def _carriers(o: _Ops, psi, n_i, vt):
    n = n_i * jnp.exp(jnp.clip((psi - o.phi_n) / vt, -200, 200)) * o.semi
    p = n_i * jnp.exp(jnp.clip((o.phi_p - psi) / vt, -200, 200)) * o.semi
    return n, p


@jax.jit
def _newton_step(o: _Ops, psi, n_i, vt, cg_tol, cg_max):
    n, p = _carriers(o, psi, n_i, vt)
    rho = Q * (p - n + o.n_net)
    f = _apply_a(o, psi) - o.bc - rho * o.vol                  # F(psi) = A psi - b(psi)
    jdiag = Q * (n + p) * o.vol / vt                            # dF/dpsi beyond A (positive)
    diag = o.diag + jdiag
    minv = 1.0 / diag

    def jv(v):
        return _apply_a(o, v) + jdiag * v

    r = -f
    x = jnp.zeros_like(psi)
    z = minv * r
    pdir = z
    rz = jnp.vdot(r, z)
    r0 = jnp.linalg.norm(r)

    def cond(s):
        _, r, _, _, it = s
        return (jnp.linalg.norm(r) > cg_tol * r0) & (it < cg_max)

    def body(s):
        x, r, pdir, rz, it = s
        ap = jv(pdir)
        a = rz / jnp.vdot(pdir, ap)
        x = x + a * pdir
        r = r - a * ap
        z = minv * r
        rz2 = jnp.vdot(r, z)
        return x, r, z + (rz2 / rz) * pdir, rz2, it + 1

    delta, _, _, _, it = jax.lax.while_loop(cond, body, (x, r, pdir, rz, jnp.asarray(0)))
    damped = vt * jnp.sign(delta) * jnp.log1p(jnp.abs(delta) / vt)
    return psi + damped, jnp.max(jnp.abs(delta)), jnp.linalg.norm(f), it


@jax.jit
def _residual_and_jdiag(o: _Ops, psi, n_i, vt):
    n, p = _carriers(o, psi, n_i, vt)
    f = _apply_a(o, psi) - o.bc - Q * (p - n + o.n_net) * o.vol
    return f, Q * (n + p) * o.vol / vt


@jax.jit
def _cg_mg(o: _Ops, jdiag, f, levels, coarse_inv, cg_tol, cg_max):
    """Solve (A + diag(jdiag)) delta = -f with multigrid-preconditioned CG (thermal_gpu.multigrid)."""
    from thermal_gpu.multigrid import vcycle

    def jv(v):
        return _apply_a(o, v) + jdiag * v

    r = -f
    x = jnp.zeros_like(f)
    z = vcycle(levels, coarse_inv, r)
    pdir, rz, r0 = z, jnp.vdot(r, z), jnp.linalg.norm(r)

    def cond(s):
        _, r, _, _, it = s
        return (jnp.linalg.norm(r) > cg_tol * r0) & (it < cg_max)

    def body(s):
        x, r, pdir, rz, it = s
        ap = jv(pdir)
        a = rz / jnp.vdot(pdir, ap)
        x, r = x + a * pdir, r - a * ap
        z = vcycle(levels, coarse_inv, r)
        rz2 = jnp.vdot(r, z)
        return x, r, z + (rz2 / rz) * pdir, rz2, it + 1

    delta, _, _, _, it = jax.lax.while_loop(cond, body, (x, r, pdir, rz, jnp.asarray(0)))
    return delta, it


@dataclass
class Solution:
    psi: np.ndarray
    n: np.ndarray
    p: np.ndarray
    newton_iterations: int
    cg_iterations: int
    converged: bool
    device: str


def solve(dev: Device, v_n: float, v_p: float, psi0: np.ndarray | None = None, tol: float = 1e-9,
          max_newton: int = 200, cg_tol: float = 1e-10, dtype=jnp.float64, precond: str = "mg") -> Solution:
    """Potential and carriers with the n contact at v_n and the p contact at v_p (reverse bias: v_n > v_p).
    Each contact takes v_n when the net doping under it is positive, else v_p (Contact.voltage is set here, so the
    bias is given once). precond: "mg" (aggregation multigrid) or "jacobi"."""
    if precond not in ("mg", "jacobi"):
        raise ValueError(f"precond must be 'mg' or 'jacobi', not {precond!r}")
    vt = thermal_voltage(dev.T)
    nx, ny = dev.n_net.shape
    for c in dev.contacts:
        under = {"xmin": dev.n_net[0, :], "xmax": dev.n_net[nx - 1, :], "ymin": dev.n_net[:, 0],
                 "ymax": dev.n_net[:, ny - 1]}[c.side][c.mask]
        c.voltage = v_n if under.size and np.mean(under) > 0 else v_p
    o, (gx_h, gy_h, extra0) = _operators(dev, v_n, v_p, dtype)
    if psi0 is None:
        psi0 = np.where(dev.semiconductor, vt * np.arcsinh(dev.n_net / (2 * dev.n_i)), 0.0)
        psi0 = psi0 + np.where(dev.n_net > 0, v_n, v_p)
    psi = jax.device_put(jnp.asarray(psi0, dtype), solver_device())
    cg_total, conv, k = 0, False, 0
    if precond == "mg":
        from thermal_gpu.multigrid import build_hierarchy
    for k in range(1, max_newton + 1):
        if precond == "mg":
            f, jdiag = _residual_and_jdiag(o, psi, dev.n_i, vt)
            # the Newton term changes the diagonal: a new hierarchy per step (host coarsening, small arrays)
            levels, cinv = build_hierarchy(gx_h, gy_h, extra0 + np.asarray(jdiag), solver_device(), dtype)
            delta, it = _cg_mg(o, jdiag, f, levels, cinv, cg_tol, 2000)
            psi = psi + vt * jnp.sign(delta) * jnp.log1p(jnp.abs(delta) / vt)
            step = jnp.max(jnp.abs(delta))
        else:
            psi, step, _, it = _newton_step(o, psi, dev.n_i, vt, cg_tol, 20000)
        cg_total += int(it)
        if float(step) < tol:
            conv = True
            break
    n, p = _carriers(o, psi, dev.n_i, vt)
    return Solution(np.asarray(psi), np.asarray(n), np.asarray(p), k, cg_total, conv, str(solver_device()))
