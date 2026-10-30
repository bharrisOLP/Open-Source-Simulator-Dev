"""Cell-centered finite volumes for -div(kappa grad T) = q on a 3-D rectilinear (non-uniform) grid.

The same method as thermal_gpu.fv (2-D), with a third axis. Grid: cell edges x, y, z (nx + 1, ny + 1, nz + 1
values), in meters. Unknowns: T at the cell centers, shape (nx, ny, nz). Face conductance between cells 1 and 2
(center-to-face distances d1, d2, face area S):

    G = S / (d1 / kappa1 + d2 / kappa2)          (series resistance: the harmonic mean, exact for layered media)

Boundaries, per side ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax"): ("dirichlet", T_b) with G = S kappa_c / d_c,
or ("neumann",) for zero flux (adiabatic, the default), or ("robin", h, T_inf) with G = S / (d_c / kappa_c + 1 / h).
The linear system A T = b is a 7-point conductance operator, symmetric positive (semi-)definite:

    (A T)_c = sum_faces G_f (T_c - T_nb) + sum_boundary G_b T_c,     b_c = q_c V_c + sum_boundary G_b T_b.

Units: q in W/m^3, kappa in W/m/K, T in K, V_c = cell volume (m^3), b in W.
In the structures of this package, y is the vertical axis (as in the 2-D cross-section) and z is the axis of
the waveguide: a 3-D problem that does not change along z gives the 2-D solution in each z cell.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

SIDES3 = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")


def _lo(ax: int, ndim: int = 3) -> tuple:
    """Index of cells 0 .. n - 2 along axis ax (the low cell of each interior face)."""
    return (slice(None),) * ax + (slice(None, -1),) + (slice(None),) * (ndim - ax - 1)


def _hi(ax: int, ndim: int = 3) -> tuple:
    """Index of cells 1 .. n - 1 along axis ax (the high cell of each interior face)."""
    return (slice(None),) * ax + (slice(1, None),) + (slice(None),) * (ndim - ax - 1)


def _along(v: np.ndarray, ax: int, ndim: int = 3) -> np.ndarray:
    """A 1-D array along axis ax, ready to broadcast."""
    s = [1] * ndim
    s[ax] = -1
    return np.reshape(v, s)


N_VALUES = {"neumann": 0, "dirichlet": 1, "robin": 2}    # the number of values after the kind


def _check_spec(side: str, spec) -> tuple:
    """The boundary spec of a side as (kind, float values...); ValueError when it is not valid."""
    if isinstance(spec, str) or not isinstance(spec, (tuple, list)) or not spec:
        raise ValueError(f"{side}: the boundary must be a tuple such as ('dirichlet', T), not {spec!r}")
    kind = spec[0]
    if kind not in N_VALUES:
        raise ValueError(f"{side}: unknown boundary kind {kind!r}; use {tuple(N_VALUES)}")
    if len(spec) != 1 + N_VALUES[kind]:
        raise ValueError(f"{side}: {kind!r} needs {N_VALUES[kind]} value(s), not {len(spec) - 1}: {spec!r}")
    try:
        values = tuple(float(v) for v in spec[1:])
    except (TypeError, ValueError):
        raise ValueError(f"{side}: the boundary values must be numbers: {spec!r}") from None
    if not all(np.isfinite(values)):
        raise ValueError(f"{side}: the boundary values must be finite: {spec!r}")
    if kind == "robin" and values[0] <= 0:
        raise ValueError(f"{side}: the heat transfer coefficient h must be positive (use 'neumann' for h = 0)")
    return (kind, *values)


@dataclass(frozen=True)
class Problem3:
    g: tuple              # interior face conductances per axis: (nx-1, ny, nz), (nx, ny-1, nz), (nx, ny, nz-1)
    diag_extra: np.ndarray  # (nx, ny, nz): the boundary conductances on the diagonal
    b: np.ndarray         # (nx, ny, nz): right-hand side (W)
    shape: tuple[int, int, int]


def build3(x: np.ndarray, y: np.ndarray, z: np.ndarray, kappa: np.ndarray, q: np.ndarray, bc: dict) -> Problem3:
    """kappa, q: (nx, ny, nz) cell values. bc: {side: ("dirichlet", T) | ("neumann",) | ("robin", h, T_inf)};
    a side that is not in bc is adiabatic. Invalid input raises ValueError: an unknown side or kind, a spec with
    the wrong number of values, a value that is not finite, h <= 0, a wrong shape, edges that are not finite or
    do not increase, kappa that is not finite and positive, q that is not finite."""
    edges = [np.asarray(e, float) for e in (x, y, z)]
    kappa, q = np.asarray(kappa, float), np.asarray(q, float)
    if any(e.ndim != 1 or e.size < 2 for e in edges):
        raise ValueError("x, y and z must be 1-D arrays of at least 2 cell edges")
    shape = tuple(len(e) - 1 for e in edges)
    if kappa.shape != shape or q.shape != shape:
        raise ValueError(f"kappa and q must have the shape {shape} of the grid, not {kappa.shape}, {q.shape}")
    if not all(np.all(np.isfinite(e)) and np.all(np.diff(e) > 0) for e in edges):
        raise ValueError("the cell edges must be finite and increase strictly along each axis")
    if not (np.all(np.isfinite(kappa)) and np.all(kappa > 0)):
        raise ValueError("kappa must be finite and positive in each cell")
    if not np.all(np.isfinite(q)):
        raise ValueError("q must be finite in each cell")
    unknown = set(bc) - set(SIDES3)
    if unknown:
        raise ValueError(f"unknown boundary sides {sorted(unknown)}; use {SIDES3}")
    specs = {side: _check_spec(side, bc.get(side, ("neumann",))) for side in SIDES3}
    d = [np.diff(e) for e in edges]
    vol = d[0][:, None, None] * d[1][None, :, None] * d[2][None, None, :]
    g = []
    for ax in range(3):
        area = vol / _along(d[ax], ax)                     # the face area normal to ax (equal on both sides)
        r = _along(d[ax] / 2, ax) / kappa                  # half-cell resistance times the area
        g.append(area[_lo(ax)] / (r[_lo(ax)] + r[_hi(ax)]))
    diag = np.zeros(shape)
    b = q * vol
    for n_side, side in enumerate(SIDES3):
        spec = specs[side]
        kind = spec[0]
        if kind == "neumann":
            continue
        ax = n_side // 2
        cell = [slice(None)] * 3
        cell[ax] = slice(0, 1) if side.endswith("min") else slice(shape[ax] - 1, shape[ax])
        cell = tuple(cell)
        area = (vol / _along(d[ax], ax))[cell]
        dist = _along(d[ax] / 2, ax)[cell]
        kc = kappa[cell]
        if kind == "dirichlet":
            gb, tb = area * kc / dist, spec[1]
        else:                                               # robin (checked in _check_spec)
            gb, tb = area / (dist / kc + 1.0 / spec[1]), spec[2]
        diag[cell] += gb
        b[cell] += gb * tb
    return Problem3(tuple(g), diag, b, shape)


def full_diag(g: tuple, extra: np.ndarray) -> np.ndarray:
    """The diagonal of A: the sum of the face conductances of each cell, plus the extra (boundary) term."""
    out = np.array(extra, float, copy=True)
    for ax, ga in enumerate(g):
        out[_lo(ax)] += ga
        out[_hi(ax)] += ga
    return out


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class DeviceProblem3:
    gx: jax.Array
    gy: jax.Array
    gz: jax.Array
    diag: jax.Array       # the full diagonal of A
    b: jax.Array


def to_device3(p: Problem3, dtype=jnp.float64, device=None) -> DeviceProblem3:
    put = (lambda a: jax.device_put(jnp.asarray(a, dtype), device)) if device else (lambda a: jnp.asarray(a, dtype))
    return DeviceProblem3(*(put(ga) for ga in p.g), put(full_diag(p.g, p.diag_extra)), put(p.b))


def apply_op(g: tuple, diag: jax.Array, t: jax.Array) -> jax.Array:
    """A t (matrix-free) for the 7-point conductance operator: diag t minus the conductance-weighted neighbors.
    Each neighbor term is padded to the full shape, so the operator is one elementwise expression for XLA."""
    out = diag * t
    for ax, ga in enumerate(g):
        pad_hi = [(0, 0)] * 3
        pad_lo = [(0, 0)] * 3
        pad_hi[ax], pad_lo[ax] = (0, 1), (1, 0)
        out = out - jnp.pad(ga * t[_hi(ax)], pad_hi) - jnp.pad(ga * t[_lo(ax)], pad_lo)
    return out


def apply3(dp: DeviceProblem3, t: jax.Array) -> jax.Array:
    return apply_op((dp.gx, dp.gy, dp.gz), dp.diag, t)


def assemble_op(g: tuple, diag: np.ndarray) -> sp.csr_matrix:
    """The sparse matrix of a 7-point conductance operator (CPU reference), row-major flattening of the shape."""
    shape = diag.shape
    n = int(np.prod(shape))
    idx = np.arange(n).reshape(shape)
    rows, cols, vals = [idx.ravel()], [idx.ravel()], [np.asarray(diag).ravel()]
    for ax, ga in enumerate(g):
        ga = np.asarray(ga).ravel()
        a, c = idx[_lo(ax)].ravel(), idx[_hi(ax)].ravel()
        rows += [a, c]
        cols += [c, a]
        vals += [-ga, -ga]
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))


def assemble3(p: Problem3) -> sp.csr_matrix:
    """The sparse matrix A of the problem (CPU reference)."""
    return assemble_op(p.g, full_diag(p.g, p.diag_extra))
