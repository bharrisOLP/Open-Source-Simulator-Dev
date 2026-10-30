"""The Maxwell operator Theta = curl eps^-1 curl in a transverse plane-wave basis (the formulation of MPB).

H(r) = sum_G h(G) exp(i (k + G) . r), with h(G) orthogonal to q = k + G: h = c1 u1(q) + c2 u2(q). Then

    Theta h = -q x FFT[ eps_inv(r) . IFFT[ q x h ] ],   eigenvalue (omega / c)^2,

six FFTs per product (three in each direction). Theta is Hermitian and positive semi-definite on the transverse
basis, so the lowest eigenvalues are the bands. Units: lengths in units of the lattice constant a; k in units of
2 pi / a in the input; frequencies out in units of 2 pi c / a (as MPB).

2-D cells (nz = 1, k_z = 0) decouple into two polarizations, solved apart with one coefficient per G:
    "te": H along z (u = z), E in the plane;     "tm": H in the plane (u = z x q^), E along z.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from bands_gpu.geometry import Cell


def reciprocal_vectors(cell: Cell, k_frac) -> np.ndarray:
    """q = k + G (shape (3, nx, ny, nz)) in units of 1/a (2 pi included). k_frac: k in units of the reciprocal
    lattice vectors (2 pi / a_i per axis)."""
    comps = []
    for i, (a, n) in enumerate(zip(cell.size, cell.shape)):
        g = np.fft.fftfreq(n, d=1.0 / n) if n > 1 else np.zeros(1)
        comps.append(2 * np.pi * (k_frac[i] + g) / a)
    Q = np.meshgrid(*comps, indexing="ij")
    return np.stack(Q)


def transverse_basis(q: np.ndarray, polarization: str | None) -> np.ndarray:
    """Unit vectors u (shape (m, 3, nx, ny, nz), m = 2 for 3-D, 1 for a 2-D polarization), orthogonal to q."""
    qn = np.sqrt((q**2).sum(axis=0))
    zero = qn < 1e-14
    qhat = q / np.where(zero, 1.0, qn)[None]
    qhat = np.where(zero[None], np.array([1.0, 0.0, 0.0])[:, None, None, None], qhat)
    zhat = np.zeros_like(q)
    zhat[2] = 1.0
    if polarization == "te":
        return zhat[None]
    if polarization == "tm":
        return np.cross(zhat, qhat, axis=0)[None]
    # 3-D: u1 = normalize(qhat x ref), u2 = qhat x u1; ref = z unless q is along z
    ref = np.where((np.abs(qhat[2]) > 0.9)[None], np.array([1.0, 0.0, 0.0])[:, None, None, None], zhat)
    u1 = np.cross(qhat, ref, axis=0)
    u1 /= np.sqrt((u1**2).sum(axis=0))[None]
    u2 = np.cross(qhat, u1, axis=0)
    return np.stack([u1, u2])


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Operator:
    q: jax.Array          # (3, nx, ny, nz)
    u: jax.Array          # (m, 3, nx, ny, nz)
    eps_inv: jax.Array    # (3, 3, nx, ny, nz) or (nx, ny, nz) for a scalar
    precond: jax.Array    # (nx, ny, nz): 1 / (|q|^2 <eps_inv> + shift)


def make_operator(cell: Cell, k_frac, polarization: str | None = None, eps_inv: np.ndarray | None = None,
                  dtype=jnp.complex64, device=None) -> Operator:
    if polarization not in (None, "te", "tm"):
        raise ValueError("polarization: None (3-D), 'te' or 'tm'")
    if polarization and cell.shape[2] != 1:
        raise ValueError("te/tm need a 2-D cell (nz = 1)")
    q = reciprocal_vectors(cell, np.asarray(k_frac, float))
    u = transverse_basis(q, polarization)
    if eps_inv is None:
        eps_inv = cell.eps_inverse()
    q2 = (q**2).sum(axis=0)
    mean_inv = float(np.mean(np.trace(eps_inv))) / 3 if eps_inv.ndim == 5 else float(np.mean(eps_inv))
    shift = 0.1 * mean_inv * (2 * np.pi / max(cell.size)) ** 2
    pre = 1.0 / (q2 * mean_inv + shift)
    rdt = jnp.float32 if dtype == jnp.complex64 else jnp.float64
    put = (lambda a, t: jax.device_put(jnp.asarray(a, t), device)) if device else (lambda a, t: jnp.asarray(a, t))
    return Operator(put(q, rdt), put(u, rdt), put(eps_inv, rdt), put(pre, rdt))


def _cross(a: jax.Array, b: jax.Array) -> jax.Array:
    """Cross product along axis -4 (the vector axis of (..., 3, nx, ny, nz) arrays)."""
    a0, a1, a2 = a[..., 0, :, :, :], a[..., 1, :, :, :], a[..., 2, :, :, :]
    b0, b1, b2 = b[..., 0, :, :, :], b[..., 1, :, :, :], b[..., 2, :, :, :]
    return jnp.stack([a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0], axis=-4)


def apply(op: Operator, c: jax.Array) -> jax.Array:
    """Theta c for coefficients c of shape (..., m, nx, ny, nz)."""
    h = jnp.einsum("...mxyz,mixyz->...ixyz", c, op.u.astype(c.dtype))          # (..., 3, nx, ny, nz)
    e = _cross(op.q.astype(c.dtype), h)
    e = jnp.fft.ifftn(e, axes=(-3, -2, -1))
    if op.eps_inv.ndim == 5:
        e = jnp.einsum("ijxyz,...jxyz->...ixyz", op.eps_inv.astype(c.dtype), e)
    else:
        e = op.eps_inv.astype(c.dtype) * e
    e = jnp.fft.fftn(e, axes=(-3, -2, -1))
    r = -_cross(op.q.astype(c.dtype), e)
    return jnp.einsum("...ixyz,mixyz->...mxyz", r, op.u.astype(c.dtype))


def precondition(op: Operator, r: jax.Array) -> jax.Array:
    return r * op.precond.astype(r.dtype)


def fields_h(op: Operator, c: jax.Array) -> jax.Array:
    """H(r) on the grid (..., 3, nx, ny, nz) for coefficients c (without the Bloch phase exp(i k . r))."""
    h = jnp.einsum("...mxyz,mixyz->...ixyz", c, op.u.astype(c.dtype))
    return jnp.fft.ifftn(h, axes=(-3, -2, -1))
