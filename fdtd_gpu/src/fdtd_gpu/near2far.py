"""Near-to-far-field transform by surface equivalence (`Simulation.add_near2far`, `get_farfield`, `get_farfields`).

Written from textbook formulas and from the documentation of the MIT FDTD package; no code of that package is used.

Conventions. Units: um, frequency f in 1/um, c = 1, eps0 = mu0 = 1 (the fields of the DFT monitors). The DFT is
sum f(t) exp(+i omega t) dt / sqrt(2 pi), so a phasor has the time dependence exp(-i omega t) and an outgoing wave
is exp(+i k r). In a medium (eps, mu): omega = 2 pi f, k = omega sqrt(eps mu), wave impedance sqrt(mu / eps).

Method (surface equivalence; Taflove and Hagness, Computational Electrodynamics, 3rd ed., ch. 8). The DFT fields E
and H on a closed surface with the outward normal n give the surface currents J = n x H and M = -n x E (Maxwell
equations curl E = i omega mu H - M, curl H = -i omega eps E + J). In a homogeneous medium these currents radiate
the fields of the sources inside the surface at each point outside it, and zero fields inside it. The sum over the
surface cells uses the exact Green's function, not only its far-field limit, so the result is valid at each
distance outside the surface (the sum is a quadrature: keep the points several cells away from the surface). For a
current element J (current x length) and a magnetic current element M at the distance R along the unit vector u:

    E = i omega mu [At J_t + Ap J_p + B (u . J) u] + i k C (u x M)
    H = i omega eps [At M_t + Ap M_p + B (u . M) u] - i k C (u x J)

J_t is the part of J in the plane of the cell and J_p the part along the axis of size 0 (2-D only; in 3-D At = Ap).
- 3-D, G = exp(i x) / (4 pi R), x = k R (the fields of a Hertzian dipole, Jackson, Classical Electrodynamics, 3rd
  ed., Sec. 9.2, eq. 9.18, with the current moment J = -i omega p; magnetic currents by duality):
  At = Ap = G (1 + i/x - 1/x^2), B = G (-1 - 3i/x + 3/x^2), C = -G (1 + i/x).
- 2-D (line currents per unit length along the axis of size 0; the Green's function (i/4) H0(x), x = k rho, H_n the
  Hankel functions of the first kind; E = i omega mu [g J + (1/k^2) grad (J . grad g)], H = grad g x J):
  At = (i/4) (H0 - H1/x), Ap = (i/4) H0, B = (i/4) (-H0 + 2 H1/x), C = H1 / 4.

The 3-D sum runs with JAX in float64 on the default device (the GPU when there is one), the 2-D sum with NumPy and
the Hankel functions of SciPy. Both take the points in chunks of about `BUDGET` (point, surface cell) pairs.
"""

from __future__ import annotations

import functools
import math

import numpy as np

from fdtd_gpu.geometry import Vector3, _v3

BUDGET = 1 << 21          # (point, surface cell) pairs per chunk: about 0.6 GB of temporary arrays in 3-D


class Volume:
    """A box of points: Volume(center, size). Only center and size are supported."""

    def __init__(self, center=Vector3(), size=Vector3(), **unsupported):
        if unsupported:
            raise NotImplementedError(f"Volume: unsupported arguments {sorted(unsupported)}")
        self.center, self.size = _v3(center), _v3(size)


def farfield_grid(center, size, resolution: float, axes):
    """The points of get_farfields: on each axis a of `axes`, n = floor(size_a x resolution) points from the low end to
    the high end of the volume (both included); one point at the center if n < 2. Other axes: the coordinate 0.

    Returns (points (P, 3), shape, weights (P,)): shape has the number of points of each axis in `axes` with more
    than one point (C order of the points); the weights are the trapezoidal rule over those axes (spacing
    size / (n - 1), half weight at the two ends; 1 on the other axes), for `DftNear2Far.flux`."""
    if not resolution > 0:
        raise ValueError("resolution must be positive")
    c, s = _v3(center), _v3(size)
    coords, weights, shape = [], [], []
    for a in range(3):
        n = int(math.floor(s[a] * resolution + 1e-9)) if a in axes else 1
        if n < 2:
            coords.append(np.array([c[a] if a in axes else 0.0]))
            weights.append(np.ones(1))
            continue
        coords.append(np.linspace(c[a] - s[a] / 2, c[a] + s[a] / 2, n))
        w = np.full(n, s[a] / (n - 1))
        w[[0, -1]] *= 0.5
        weights.append(w)
        shape.append(n)
    grids = np.meshgrid(*coords, indexing="ij")
    pts = np.stack([g.reshape(-1) for g in grids], axis=-1)
    w = functools.reduce(np.multiply.outer, weights).reshape(-1)
    return pts, tuple(shape), w


def _hankel01(x):
    from scipy import special
    return special.hankel1(0, x), special.hankel1(1, x)


def _chunk(xp, pts, src, J, M, omega, k, eps, mu, perp=None, two_d=False):
    """E and H (P, 3) at pts (P, 3) of the current elements J, M (N, 3) at src (N, 3), one frequency.
    perp: None (3-D) or the unit vector (3,) of the axis of size 0 (2-D)."""
    D = pts[:, None, :] - src[None, :, :]
    R = xp.sqrt(xp.sum(D * D, axis=-1))
    u = D / R[..., None]
    x = k * R
    if two_d:
        H0, H1 = _hankel01(x)
        At, Ap = 0.25j * (H0 - H1 / x), 0.25j * H0
        B, C = 0.25j * (-H0 + 2 * H1 / x), 0.25 * H1
    else:
        G = xp.exp(1j * x) / (4 * math.pi * R)
        ix = 1j / x
        At = Ap = G * (1 + ix - 1 / x**2)
        B, C = G * (-1 - 3 * ix + 3 / x**2), -G * (1 + ix)

    def direct(Q):            # sum_n At Q_t + Ap Q_p + B (u . Q) u
        s = xp.matmul(At, Q) if perp is None else xp.matmul(At, Q * (1 - perp)) + xp.matmul(Ap, Q * perp)
        uQ = xp.sum(u * Q[None], axis=-1)
        return s + xp.sum((B * uQ)[..., None] * u, axis=1)

    def cross(Q):             # sum_n C (u x Q): X[p, c, d] = sum_n C u_c Q_d
        X = xp.matmul(xp.swapaxes(C[..., None] * u, 1, 2), Q)
        return xp.stack([X[:, 1, 2] - X[:, 2, 1], X[:, 2, 0] - X[:, 0, 2], X[:, 0, 1] - X[:, 1, 0]], axis=-1)

    E = 1j * omega * mu * direct(J) + 1j * k * cross(M)
    H = 1j * omega * eps * direct(M) - 1j * k * cross(J)
    return E, H


@functools.lru_cache(maxsize=None)
def _jax_chunk():
    import jax
    import jax.numpy as jnp
    return jax.jit(functools.partial(_chunk, jnp))


def radiate(points, src, J, M, freqs, eps: float = 1.0, mu: float = 1.0, trivial_axis: int | None = None,
            backend: str = "auto", device=None, budget: int = BUDGET):
    """E and H, each (nfreq, P, 3) complex128, at `points` (P, 3) of the surface current elements J and M (nfreq, N, 3:
    current density x area; in 2-D x length) at `src` (N, 3), in a homogeneous medium (eps, mu).

    trivial_axis: None (3-D), or the axis of size 0 of a 2-D cell: the currents are line currents along it, and the
    coordinates along it are ignored. backend: "jax" (3-D only: float64, on `device` or the default device), "numpy",
    or "auto" (jax in 3-D, numpy in 2-D). The points must not be on a surface cell (R = 0 gives inf or NaN)."""
    pts = np.atleast_2d(np.asarray(points, float)).copy()
    src = np.asarray(src, float).copy()
    J, M = np.asarray(J, np.complex128), np.asarray(M, np.complex128)
    freqs = np.atleast_1d(np.asarray(freqs, float))
    if J.shape != M.shape or J.shape != (len(freqs), len(src), 3):
        raise ValueError(f"J and M need the shape (nfreq, N, 3) = {(len(freqs), len(src), 3)}, got {J.shape}")
    perp = None
    if trivial_axis is not None:
        pts[:, trivial_axis] = 0.0
        src[:, trivial_axis] = 0.0
        perp = np.zeros(3)
        perp[trivial_axis] = 1.0
    if backend == "auto":
        backend = "numpy" if trivial_axis is not None else "jax"
    if backend == "jax" and trivial_axis is not None:
        raise ValueError("the jax backend is 3-D only (the Hankel functions come from SciPy)")
    if backend not in ("jax", "numpy"):
        raise ValueError(f"unknown backend {backend!r}")
    P, N = len(pts), len(src)
    E = np.zeros((len(freqs), P, 3), np.complex128)
    H = np.zeros_like(E)
    if P == 0 or N == 0:
        return E, H
    chunk = max(1, min(P, budget // N))
    n_chunks = -(-P // chunk)
    pad = np.concatenate([pts, np.repeat(pts[-1:], n_chunks * chunk - P, axis=0)])     # one compiled shape
    if backend == "numpy":
        fn = functools.partial(_chunk, np, perp=perp, two_d=trivial_axis is not None)
        for fi, f in enumerate(freqs):
            om = 2 * math.pi * f
            k = om * math.sqrt(eps * mu)
            for c0 in range(0, P, chunk):
                e, h = fn(pad[c0:c0 + chunk], src, J[fi], M[fi], om, k, eps, mu)
                n = min(chunk, P - c0)
                E[fi, c0:c0 + n], H[fi, c0:c0 + n] = e[:n], h[:n]
        return E, H
    import jax
    import jax.numpy as jnp
    if device is None:
        device = jax.devices()[0]
    fn = _jax_chunk()
    with jax.enable_x64(True), jax.default_device(device):
        src_d = jnp.asarray(src, jnp.float64)
        for fi, f in enumerate(freqs):
            om = 2 * math.pi * f
            k = om * math.sqrt(eps * mu)
            Jd, Md = jnp.asarray(J[fi]), jnp.asarray(M[fi])
            for c0 in range(0, P, chunk):
                e, h = fn(jnp.asarray(pad[c0:c0 + chunk], jnp.float64), src_d, Jd, Md, om, k, eps, mu)
                n = min(chunk, P - c0)
                E[fi, c0:c0 + n], H[fi, c0:c0 + n] = np.asarray(e)[:n], np.asarray(h)[:n]
    return E, H
