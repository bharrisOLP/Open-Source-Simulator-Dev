"""The full-vector finite-difference mode operator, matrix-free (JAX) and assembled (SciPy, the CPU reference).

Unknowns: u = (Ex, Ey) on the Yee grid of the cross-section, each of shape (Nx, Ny). With the k0-normalized
derivatives Dxf, Dxb, Dyf, Dyb (``grid.axis_stencils``) and diagonal eps and mu, Maxwell's equations for a mode
exp(i k0 neff z) give the eigenproblem

    A u = lambda u,   lambda = -(neff + i keff)^2,

    c  = (Dxf Ey - Dyf Ex) / mu_zz                           (proportional to Hz)
    hx =  eps_yy Ey + Dxb c,     hy = -eps_xx Ex + Dyb c
    d  = (Dxb(-eps_xx Ex) - Dyb(eps_yy Ey)) / eps_zz         (proportional to Ez)
    (A u)_x = mu_yy hy + Dxf d,  (A u)_y = -mu_xx hx + Dyf d

This is the formulation of the tidy3d local mode solver for diagonal media ("P Q" form), so both solvers have the
same discrete spectrum. Guided modes are the eigenvalues with the most negative real part. A straight waveguide has
mu = 1 (the device operator then has no mu arrays and no multiplications by mu).

Bends (``make_problem(..., bend_radius=R, bend_axis=...)``): the radial transform of the tidy3d solver
(``tidy3d.components.mode.transforms.radial_transform``). The coordinate along the axis in the bend plane (the
"normal axis": x for bend_axis = 1, y for bend_axis = 0) moves to u = coordinate + R - plane_center, and
w = R phi is the propagation coordinate, so that neff is the effective index at the radius R. The Jacobian is
J = diag(1, 1, a) with a = dw/dz = R / u, at the E positions (u at the lower cell corner of the normal axis) and at
the H positions (u at the cell centre). The transformed media are eps' = J eps J^T / det J and the same for mu:

    eps'_xx = eps_xx / a_e,  eps'_yy = eps_yy / a_e,  eps'_zz = eps_zz a_e,   mu'_xx = mu'_yy = 1 / a_h,  mu'_zz = a_h.

The PML uses the transformed eps and mu for its average speed. The eigenvector gives the transformed fields; the
fields in the cross-section are E = J^T E' and H = J^T H' (only Ez and Hz change: Ez = a_e Ez', Hz = a_h Hz').
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
    """One cross-section at one wavelength. Arrays are numpy (host); ``device()`` gives the JAX pytree.

    eps_** (and mu_** of a bend) are the media of the eigenproblem (for a bend: after the radial transform).
    ``mu``: (mu_xx, mu_yy, mu_zz), each (Nx, Ny), or None for mu = 1. ``jac``: (a_e, a_h), each (Nx, Ny), the
    Jacobian dw/dz of a bend at the E and H positions, or None. ``eps_max``: max |eps| of the physical media (the
    default target of the eigensolver, as in tidy3d). A bend: ``bend_radius``, ``bend_axis`` and ``bend_shift`` (the
    shift R - plane_center of the normal axis: u = coordinate + bend_shift); these three define the transform."""

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
    mu: tuple[np.ndarray, np.ndarray, np.ndarray] | None = field(default=None, repr=False)
    jac: tuple[np.ndarray, np.ndarray] | None = field(default=None, repr=False)
    eps_max: float | None = None
    bend_radius: float | None = None
    bend_axis: int | None = None
    bend_shift: float | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return self.eps_xx.shape

    @property
    def n_unknowns(self) -> int:
        return 2 * self.eps_xx.size

    @property
    def bent(self) -> bool:
        return self.jac is not None

    def device(self, dtype=jnp.complex128, bend_arrays: bool | None = None) -> "DeviceOperator":
        """The device operator. ``bend_arrays``: None (default) gives the mu and Jacobian arrays only to a bend;
        True gives them to every problem (ones for a straight one), so that bends and straight problems can be
        stacked into one batch."""
        c = lambda a: put(np.asarray(a), dtype)  # noqa: E731   (on the solver device, see device.py)
        mu, jac = self.mu, self.jac
        if bend_arrays and not self.bent:
            one = np.ones(self.shape)
            mu, jac = (one, one, one), (one, one)
        return DeviceOperator(
            eps_xx=c(self.eps_xx), eps_yy=c(self.eps_yy), inv_eps_zz=c(1.0 / self.eps_zz),
            dxf=(c(self.dxf.main)[:, None], c(self.dxf.off)[:, None]),
            dxb=(c(self.dxb.main)[:, None], c(self.dxb.off)[:, None]),
            dyf=(c(self.dyf.main)[None, :], c(self.dyf.off)[None, :]),
            dyb=(c(self.dyb.main)[None, :], c(self.dyb.off)[None, :]),
            mu_xx=None if mu is None else c(mu[0]),
            mu_yy=None if mu is None else c(mu[1]),
            inv_mu_zz=None if mu is None else c(1.0 / mu[2]),
            jac_e=None if jac is None else c(jac[0]),
            jac_h=None if jac is None else c(jac[1]),
        )


def bend_error(x: np.ndarray, y: np.ndarray, radius: float, bend_axis: int | None,
               plane_center: tuple[float, float] = (0.0, 0.0)) -> str | None:
    """Why the radial transform cannot represent this bend (None: it can). The Jacobian a = R / u must be finite and
    positive at all E and H positions, so the centre of the bend must be outside the cross-section."""
    if bend_axis not in (0, 1):
        return "a bend needs bend_axis 0 or 1"
    if radius == 0 or not np.isfinite(radius):
        return f"bend_radius must be finite and not 0 (got {radius})"
    norm_axis = 0 if bend_axis == 1 else 1
    u = np.asarray((x, y)[norm_axis], dtype=np.float64) + (radius - plane_center[norm_axis])
    with np.errstate(divide="ignore", invalid="ignore"):
        a_e = radius / u[:-1]
        a_h = radius / (u[:-1] + u[1:]) * 2
    if not (np.all(np.isfinite(a_e)) and np.all(a_e > 0) and np.all(np.isfinite(a_h)) and np.all(a_h > 0)):
        return ("the centre of the bend must be outside the cross-section (|bend_radius| larger than the distance "
                "from the plane centre to the domain edge, on the side of the centre)")
    return None


def radial_transform(x: np.ndarray, y: np.ndarray, radius: float, bend_axis: int,
                     plane_center: tuple[float, float] = (0.0, 0.0)):
    """The coordinates and the Jacobian dw/dz of a bend (``tidy3d.components.mode.transforms.radial_transform``):
    ((u_x, u_y), a_e, a_h), with a_e and a_h of shape (Nx, Ny) at the E and H positions. ValueError for a bend that
    the transform cannot represent (``bend_error``)."""
    why = bend_error(x, y, radius, bend_axis, plane_center)
    if why:
        raise ValueError(why)
    norm_axis = 0 if bend_axis == 1 else 1
    coords = [np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)]
    coords[norm_axis] = coords[norm_axis] + (radius - plane_center[norm_axis])
    u = coords[norm_axis]
    a_e = radius / u[:-1]
    a_h = radius / (u[:-1] + u[1:]) * 2
    shape = (len(coords[0]) - 1, len(coords[1]) - 1)
    a_e, a_h = (np.broadcast_to(np.expand_dims(a, axis=bend_axis), shape).copy() for a in (a_e, a_h))
    return (coords[0], coords[1]), a_e, a_h


def make_problem(eps_xx: np.ndarray, eps_yy: np.ndarray, eps_zz: np.ndarray, x: np.ndarray, y: np.ndarray,
                 wavelength: float, num_pml: tuple[int, int] = (0, 0), symmetry: tuple[int, int] = (0, 0),
                 bend_radius: float | None = None, bend_axis: int | None = None,
                 plane_center: tuple[float, float] = (0.0, 0.0)) -> FDEProblem:
    """Build the problem. ``x`` and ``y`` are the cell corners (Nx + 1 and Ny + 1 values, in the units of
    ``wavelength``); eps_** have the shape (Nx, Ny) at the Ex, Ey and Ez positions. ``bend_radius`` (None: a
    straight waveguide), ``bend_axis`` (0 or 1, the axis normal to the bend plane) and ``plane_center`` have the
    meaning of the tidy3d ModeSpec (and of meow.Mesh2D)."""
    eps = [np.asarray(e, dtype=np.complex128) for e in (eps_xx, eps_yy, eps_zz)]
    nx, ny = eps[0].shape
    if len(x) != nx + 1 or len(y) != ny + 1:
        raise ValueError(f"coords must have Nx+1, Ny+1 values; got {len(x)}, {len(y)} for eps {eps[0].shape}")
    eps_max = float(np.max(np.abs(np.stack(eps))))
    mu = jac = None
    if bend_radius is not None:
        if bend_axis is None:
            raise ValueError("a bend needs bend_axis (0 or 1)")
        (x, y), a_e, a_h = radial_transform(x, y, bend_radius, bend_axis, plane_center)
        eps = [eps[0] / a_e, eps[1] / a_e, eps[2] * a_e]
        mu = (1.0 / a_h, 1.0 / a_h, a_h)
        jac = (a_e, a_h)
    k0 = 2 * np.pi / wavelength
    speeds = pml_speeds(np.stack(eps), num_pml, None if mu is None else np.stack(mu))
    dxf, dxb = axis_stencils(x, k0, num_pml[0], symmetry[0], (speeds[0], speeds[1]))
    dyf, dyb = axis_stencils(y, k0, num_pml[1], symmetry[1], (speeds[2], speeds[3]))
    dlx = (np.diff(x), np.concatenate([np.diff(x)[:1], 0.5 * (np.diff(x)[:-1] + np.diff(x)[1:])]))
    dly = (np.diff(y), np.concatenate([np.diff(y)[:1], 0.5 * (np.diff(y)[:-1] + np.diff(y)[1:])]))
    return FDEProblem(eps[0], eps[1], eps[2], dxf, dxb, dyf, dyb, k0,
                      dl_f=(dlx[0], dly[0]), dl_b=(dlx[1], dly[1]), mu=mu, jac=jac, eps_max=eps_max,
                      bend_radius=bend_radius, bend_axis=None if bend_radius is None else bend_axis,
                      bend_shift=None if bend_radius is None else
                      float(bend_radius - plane_center[0 if bend_axis == 1 else 1]))


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
    mu_xx: jax.Array | None = None         # None (a straight waveguide): mu = 1, no multiplications
    mu_yy: jax.Array | None = None
    inv_mu_zz: jax.Array | None = None
    jac_e: jax.Array | None = None         # bend: Ez = jac_e Ez', Hz = jac_h Hz' (``fields``)
    jac_h: jax.Array | None = None


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


def _curl_z(op: DeviceOperator, ex: jax.Array, ey: jax.Array) -> jax.Array:
    """c = (Dxf Ey - Dyf Ex) / mu_zz."""
    c = _fx(op.dxf, ey) - _fy(op.dyf, ex)
    return c if op.inv_mu_zz is None else op.inv_mu_zz * c


def apply(op: DeviceOperator, u: jax.Array) -> jax.Array:
    """A u for u of shape (..., 2, Nx, Ny); the leading axes are a batch."""
    ex, ey = u[..., 0, :, :], u[..., 1, :, :]
    c = _curl_z(op, ex, ey)
    hx = op.eps_yy * ey + _bx(op.dxb, c)
    hy = -op.eps_xx * ex + _by(op.dyb, c)
    d = op.inv_eps_zz * (_bx(op.dxb, -op.eps_xx * ex) - _by(op.dyb, op.eps_yy * ey))
    if op.mu_xx is not None:
        hx, hy = op.mu_xx * hx, op.mu_yy * hy
    return jnp.stack([hy + _fx(op.dxf, d), -hx + _fy(op.dyf, d)], axis=-3)


def fields(op: DeviceOperator, u: jax.Array, neff: jax.Array) -> tuple[jax.Array, jax.Array]:
    """E and H (each (..., 3, Nx, Ny)) from the eigenvector u and the complex index neff (shape (...)), in the
    units of the tidy3d solver (H multiplied by -i / eta_0). For a bend: the fields in the cross-section
    (E = J^T E', H = J^T H')."""
    eta0 = 376.730313412
    ex, ey = u[..., 0, :, :], u[..., 1, :, :]
    g = (1j * neff)[..., None, None]              # 1j * neff - keff, with neff complex
    c = _curl_z(op, ex, ey)
    hx = (op.eps_yy * ey + _bx(op.dxb, c)) / g
    hy = (-op.eps_xx * ex + _by(op.dyb, c)) / g
    hz = c
    px, py = op.eps_yy * ey / g, -op.eps_xx * ex / g
    ez = op.inv_eps_zz * (_bx(op.dxb, py) - _by(op.dyb, px))
    if op.jac_e is not None:
        ez, hz = op.jac_e * ez, op.jac_h * hz
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
    if p.mu is None:
        mxx = myy = imz = sp.eye(n)
    else:
        mxx, myy, imz = sp.diags(p.mu[0].ravel()), sp.diags(p.mu[1].ravel()), sp.diags(1.0 / p.mu[2].ravel())
    p_mu = sp.bmat([[None, myy], [-mxx, None]])
    p_partial = sp.bmat([[-dxf @ iez @ dyb, dxf @ iez @ dxb], [-dyf @ iez @ dyb, dyf @ iez @ dxb]])
    q_ep = sp.bmat([[None, eyy], [-exx, None]])
    q_partial = sp.bmat([[-dxb @ imz @ dyf, dxb @ imz @ dxf], [-dyb @ imz @ dyf, dyb @ imz @ dxf]])
    return (p_mu @ (q_ep + q_partial) + p_partial @ q_ep).tocsr()
