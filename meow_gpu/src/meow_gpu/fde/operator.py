"""The full-vector finite-difference mode operator, matrix-free (JAX) and assembled (SciPy, the CPU reference).

Unknowns: u = (Ex, Ey) on the Yee grid of the cross-section, each of shape (Nx, Ny). With the k0-normalized
derivatives Dxf, Dxb, Dyf, Dyb (``grid.axis_stencils``), diagonal eps and mu = 1, Maxwell's equations for a mode
exp(i k0 neff z) give the eigenproblem

    A u = lambda u,   lambda = -(neff + i keff)^2,

    c  = Dxf Ey - Dyf Ex                                   (proportional to Hz)
    hx =  eps_yy Ey + Dxb c,     hy = -eps_xx Ex + Dyb c
    d  = (Dxb(-eps_xx Ex) - Dyb(eps_yy Ey)) / eps_zz         (proportional to Ez)
    (A u)_x = hy + Dxf d,        (A u)_y = -hx + Dyf d

This is the formulation of the tidy3d local mode solver for diagonal media ("P Q" form), so both solvers have the
same discrete spectrum. Guided modes are the eigenvalues with the most negative real part.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

from meow_gpu.device import put
from meow_gpu.fde.grid import Stencil1D, axis_stencils, pml_speeds


@dataclass(frozen=True)
class FDEProblem:
    """One cross-section at one wavelength. Arrays are numpy (host); ``device()`` gives the JAX pytree."""

    eps_xx: np.ndarray
    eps_yy: np.ndarray
    eps_zz: np.ndarray
    dxf: Stencil1D
    dxb: Stencil1D
    dyf: Stencil1D
    dyb: Stencil1D
    k0: float
    dl_f: tuple[np.ndarray, np.ndarray] = field(repr=False)
    dl_b: tuple[np.ndarray, np.ndarray] = field(repr=False)

    @property
    def shape(self) -> tuple[int, int]:
        return self.eps_xx.shape

    @property
    def n_unknowns(self) -> int:
        return 2 * self.eps_xx.size

    def device(self, dtype=jnp.complex128) -> "DeviceOperator":
        c = lambda a: put(np.asarray(a), dtype)  # noqa: E731   (on the solver device, see device.py)
        return DeviceOperator(
            eps_xx=c(self.eps_xx), eps_yy=c(self.eps_yy), inv_eps_zz=c(1.0 / self.eps_zz),
            dxf=(c(self.dxf.main)[:, None], c(self.dxf.off)[:, None]),
            dxb=(c(self.dxb.main)[:, None], c(self.dxb.off)[:, None]),
            dyf=(c(self.dyf.main)[None, :], c(self.dyf.off)[None, :]),
            dyb=(c(self.dyb.main)[None, :], c(self.dyb.off)[None, :]),
        )


def make_problem(eps_xx: np.ndarray, eps_yy: np.ndarray, eps_zz: np.ndarray, x: np.ndarray, y: np.ndarray,
                 wavelength: float, num_pml: tuple[int, int] = (0, 0),
                 symmetry: tuple[int, int] = (0, 0)) -> FDEProblem:
    """Build the problem. ``x`` and ``y`` are the cell corners (Nx + 1 and Ny + 1 values, in the units of
    ``wavelength``); eps_** have the shape (Nx, Ny) at the Ex, Ey and Ez positions."""
    eps = [np.asarray(e, dtype=np.complex128) for e in (eps_xx, eps_yy, eps_zz)]
    nx, ny = eps[0].shape
    if len(x) != nx + 1 or len(y) != ny + 1:
        raise ValueError(f"coords must have Nx+1, Ny+1 values; got {len(x)}, {len(y)} for eps {eps[0].shape}")
    k0 = 2 * np.pi / wavelength
    speeds = pml_speeds(np.stack(eps), num_pml)
    dxf, dxb = axis_stencils(x, k0, num_pml[0], symmetry[0], (speeds[0], speeds[1]))
    dyf, dyb = axis_stencils(y, k0, num_pml[1], symmetry[1], (speeds[2], speeds[3]))
    dlx = (np.diff(x), np.concatenate([np.diff(x)[:1], 0.5 * (np.diff(x)[:-1] + np.diff(x)[1:])]))
    dly = (np.diff(y), np.concatenate([np.diff(y)[:1], 0.5 * (np.diff(y)[:-1] + np.diff(y)[1:])]))
    return FDEProblem(eps[0], eps[1], eps[2], dxf, dxb, dyf, dyb, k0,
                      dl_f=(dlx[0], dly[0]), dl_b=(dlx[1], dly[1]))


# ---------------------------------------------------------------------------------------------------------------
# Matrix-free operator on the device
# ---------------------------------------------------------------------------------------------------------------

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class DeviceOperator:
    eps_xx: jax.Array
    eps_yy: jax.Array
    inv_eps_zz: jax.Array
    dxf: tuple[jax.Array, jax.Array]
    dxb: tuple[jax.Array, jax.Array]
    dyf: tuple[jax.Array, jax.Array]
    dyb: tuple[jax.Array, jax.Array]


def _next(v: jax.Array, axis: int) -> jax.Array:
    """v[i + 1] along ``axis``, with 0 after the last element."""
    n = v.shape[axis]
    pad = [(0, 0)] * v.ndim
    pad[axis] = (0, 1)
    return jnp.pad(jax.lax.slice_in_dim(v, 1, n, axis=axis), pad)


def _prev(v: jax.Array, axis: int) -> jax.Array:
    """v[i - 1] along ``axis``, with 0 before the first element."""
    n = v.shape[axis]
    pad = [(0, 0)] * v.ndim
    pad[axis] = (1, 0)
    return jnp.pad(jax.lax.slice_in_dim(v, 0, n - 1, axis=axis), pad)


def _fx(s, v):
    return s[0] * v + s[1] * _next(v, -2)


def _bx(s, v):
    return s[0] * v + s[1] * _prev(v, -2)


def _fy(s, v):
    return s[0] * v + s[1] * _next(v, -1)


def _by(s, v):
    return s[0] * v + s[1] * _prev(v, -1)


def apply(op: DeviceOperator, u: jax.Array) -> jax.Array:
    """A u for u of shape (..., 2, Nx, Ny); the leading axes are a batch."""
    ex, ey = u[..., 0, :, :], u[..., 1, :, :]
    c = _fx(op.dxf, ey) - _fy(op.dyf, ex)
    hx = op.eps_yy * ey + _bx(op.dxb, c)
    hy = -op.eps_xx * ex + _by(op.dyb, c)
    d = op.inv_eps_zz * (_bx(op.dxb, -op.eps_xx * ex) - _by(op.dyb, op.eps_yy * ey))
    return jnp.stack([hy + _fx(op.dxf, d), -hx + _fy(op.dyf, d)], axis=-3)


def fields(op: DeviceOperator, u: jax.Array, neff: jax.Array) -> tuple[jax.Array, jax.Array]:
    """E and H (each (..., 3, Nx, Ny)) from the eigenvector u and the complex index neff (shape (...)), in the
    units of the tidy3d solver (H multiplied by -i / eta_0)."""
    eta0 = 376.730313412
    ex, ey = u[..., 0, :, :], u[..., 1, :, :]
    g = (1j * neff)[..., None, None]              # 1j * neff - keff, with neff complex
    c = _fx(op.dxf, ey) - _fy(op.dyf, ex)
    hx = (op.eps_yy * ey + _bx(op.dxb, c)) / g
    hy = (-op.eps_xx * ex + _by(op.dyb, c)) / g
    hz = c
    px, py = op.eps_yy * ey / g, -op.eps_xx * ex / g
    ez = op.inv_eps_zz * (_bx(op.dxb, py) - _by(op.dyb, px))
    e = jnp.stack([ex, ey, ez], axis=-3)
    h = jnp.stack([hx, hy, hz], axis=-3) * (-1j / eta0)
    return e, h


# ---------------------------------------------------------------------------------------------------------------
# Assembled operator (CPU reference)
# ---------------------------------------------------------------------------------------------------------------

def _sparse_x(s: Stencil1D, ny: int) -> sp.csr_matrix:
    return sp.kron(sp.csr_matrix(s.dense()), sp.eye(ny), format="csr")


def _sparse_y(s: Stencil1D, nx: int) -> sp.csr_matrix:
    return sp.kron(sp.eye(nx), sp.csr_matrix(s.dense()), format="csr")


def assemble(p: FDEProblem) -> sp.csr_matrix:
    """The 2N x 2N matrix of ``apply`` (row-major flattening of (Nx, Ny), Ex block first)."""
    nx, ny = p.shape
    n = nx * ny
    dxf, dxb = _sparse_x(p.dxf, ny), _sparse_x(p.dxb, ny)
    dyf, dyb = _sparse_y(p.dyf, nx), _sparse_y(p.dyb, nx)
    exx = sp.diags(p.eps_xx.ravel())
    eyy = sp.diags(p.eps_yy.ravel())
    iez = sp.diags(1.0 / p.eps_zz.ravel())
    eye = sp.eye(n)
    p_mu = sp.bmat([[None, eye], [-eye, None]])
    p_partial = sp.bmat([[-dxf @ iez @ dyb, dxf @ iez @ dxb], [-dyf @ iez @ dyb, dyf @ iez @ dxb]])
    q_ep = sp.bmat([[None, eyy], [-exx, None]])
    q_partial = sp.bmat([[-dxb @ dyf, dxb @ dxf], [-dyb @ dyf, dyb @ dxf]])
    return (p_mu @ (q_ep + q_partial) + p_partial @ q_ep).tocsr()
