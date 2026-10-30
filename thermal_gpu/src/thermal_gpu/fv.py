"""Cell-centered finite volumes for -div(kappa grad T) = q on a 2-D rectilinear (non-uniform) grid.

Grid: cell edges x (nx + 1 values) and y (ny + 1 values), in meters. Unknowns: T at the cell centers.
Face conductance between cells 1 and 2 (center-to-face distances d1, d2, face length L):

    G = L / (d1 / kappa1 + d2 / kappa2)          (series resistance: the harmonic mean, exact for layered media)

Boundaries, per side ("xmin", "xmax", "ymin", "ymax"): ("dirichlet", T_b) with G = L kappa_c / d_c, or
("neumann",) for zero flux (adiabatic), or ("robin", h, T_inf) for a heat transfer coefficient h (W/m^2/K) with
G = L / (d_c / kappa_c + 1 / h). The linear system is A T = b with A symmetric positive (semi-)definite:

    (A T)_c = sum_faces G_f (T_c - T_nb) + sum_boundary G_b T_c,     b_c = q_c V_c + sum_boundary G_b T_b.

Units: q in W/m^3 (per unit length in z: the 2-D cross-section), kappa in W/m/K, T in K. V_c = cell area (m^2).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

SIDES = ("xmin", "xmax", "ymin", "ymax")


@dataclass(frozen=True)
class Problem:
    gx: np.ndarray        # (nx + 1, ny): conductance of the x faces (boundary faces included, 0 for Neumann)
    gy: np.ndarray        # (nx, ny + 1)
    diag_extra: np.ndarray  # (nx, ny): the boundary conductances on the diagonal
    b: np.ndarray         # (nx, ny): right-hand side
    shape: tuple[int, int]


def build(x: np.ndarray, y: np.ndarray, kappa: np.ndarray, q: np.ndarray, bc: dict) -> Problem:
    """kappa, q: (nx, ny) cell values. bc: {side: ("dirichlet", T) | ("neumann",) | ("robin", h, T_inf)}."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    kappa, q = np.asarray(kappa, float), np.asarray(q, float)
    nx, ny = kappa.shape
    dx, dy = np.diff(x), np.diff(y)
    hx, hy = dx / 2, dy / 2                                    # center-to-face distances
    gx = np.zeros((nx + 1, ny))
    gy = np.zeros((nx, ny + 1))
    gx[1:-1] = dy[None, :] / (hx[:-1, None] / kappa[:-1] + hx[1:, None] / kappa[1:])
    gy[:, 1:-1] = dx[:, None] / (hy[None, :-1] / kappa[:, :-1] + hy[None, 1:] / kappa[:, 1:])
    diag = np.zeros((nx, ny))
    b = q * dx[:, None] * dy[None, :]
    for side in SIDES:
        spec = bc.get(side, ("neumann",))
        kind = spec[0]
        if side in ("xmin", "xmax"):
            i = 0 if side == "xmin" else nx - 1
            L, d, kc = dy, hx[i], kappa[i, :]
        else:
            j = 0 if side == "ymin" else ny - 1
            L, d, kc = dx, hy[j], kappa[:, j]
        if kind == "neumann":
            continue
        if kind == "dirichlet":
            g, tb = L * kc / d, float(spec[1])
        elif kind == "robin":
            g, tb = L / (d / kc + 1.0 / float(spec[1])), float(spec[2])
        else:
            raise ValueError(f"unknown boundary kind {kind!r}")
        if side in ("xmin", "xmax"):
            diag[i, :] += g
            b[i, :] += g * tb
        else:
            diag[:, j] += g
            b[:, j] += g * tb
    return Problem(gx, gy, diag, b, (nx, ny))


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class DeviceProblem:
    gx: jax.Array
    gy: jax.Array
    diag: jax.Array       # the full diagonal of A
    b: jax.Array


def to_device(p: Problem, dtype=jnp.float64, device=None) -> DeviceProblem:
    diag = p.gx[:-1] + p.gx[1:] + p.gy[:, :-1] + p.gy[:, 1:] + p.diag_extra
    put = (lambda a: jax.device_put(jnp.asarray(a, dtype), device)) if device else (lambda a: jnp.asarray(a, dtype))
    return DeviceProblem(put(p.gx[1:-1]), put(p.gy[:, 1:-1]), put(diag), put(p.b))


def apply(dp: DeviceProblem, t: jax.Array) -> jax.Array:
    """A t (matrix-free): the diagonal term minus the conductance-weighted neighbors."""
    out = dp.diag * t                                               # A_cc
    out = out.at[:-1].add(-dp.gx * t[1:]).at[1:].add(-dp.gx * t[:-1])  # A_c,nb = -G (x faces)
    out = out.at[:, :-1].add(-dp.gy * t[:, 1:]).at[:, 1:].add(-dp.gy * t[:, :-1])
    return out


def assemble(p: Problem) -> sp.csr_matrix:
    """The sparse matrix A (CPU reference), row-major flattening of (nx, ny)."""
    nx, ny = p.shape
    idx = np.arange(nx * ny).reshape(nx, ny)
    diag = p.gx[:-1] + p.gx[1:] + p.gy[:, :-1] + p.gy[:, 1:] + p.diag_extra
    rows, cols, vals = [idx.ravel()], [idx.ravel()], [diag.ravel()]
    g = p.gx[1:-1]
    for a, c in ((idx[:-1], idx[1:]), (idx[1:], idx[:-1])):
        rows.append(a.ravel()); cols.append(c.ravel()); vals.append(-g.ravel())
    g = p.gy[:, 1:-1]
    for a, c in ((idx[:, :-1], idx[:, 1:]), (idx[:, 1:], idx[:, :-1])):
        rows.append(a.ravel()); cols.append(c.ravel()); vals.append(-g.ravel())
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nx * ny,) * 2)
