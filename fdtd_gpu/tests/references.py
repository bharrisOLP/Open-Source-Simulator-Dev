"""Independent references for the checks (no FDTD): exact transfer matrices, the exact resonances of a dielectric
ring (Bessel functions), and the radiated power of a point dipole. Units: um, frequency in 1/um, c = 1."""

from __future__ import annotations

import warnings

import numpy as np
from scipy import optimize, special


def tmm_reflectance(f, n0: float, layers, ns: float):
    """Normal-incidence power reflectance of a layer stack (list of (n, d)) between n0 and ns (Born and Wolf,
    characteristic matrices)."""
    f = np.atleast_1d(np.asarray(f, float))
    out = np.empty(len(f))
    for i, fi in enumerate(f):
        k0 = 2 * np.pi * fi
        M = np.eye(2, dtype=complex)
        for n, d in layers:
            delta = k0 * n * d
            M = M @ np.array([[np.cos(delta), -1j * np.sin(delta) / n], [-1j * n * np.sin(delta), np.cos(delta)]])
        B, C = M @ np.array([1.0, ns])
        r = (n0 * B - C) / (n0 * B + C)
        out[i] = abs(r) ** 2
    return out


def _ring_matrix(k, m, n_in, n_ring, n_out, a, b):
    """Ez = A J_m (r < a), B J_m + C Y_m (a < r < b), D H1_m (r > b); Ez and dEz/dr continuous at a and b."""
    x1, x2, x3, x4 = n_in * k * a, n_ring * k * a, n_ring * k * b, n_out * k * b
    M = np.array([
        [special.jv(m, x1), -special.jv(m, x2), -special.yv(m, x2), 0],
        [n_in * special.jvp(m, x1), -n_ring * special.jvp(m, x2), -n_ring * special.yvp(m, x2), 0],
        [0, special.jv(m, x3), special.yv(m, x3), -special.hankel1(m, x4)],
        [0, n_ring * special.jvp(m, x3), n_ring * special.yvp(m, x3), -n_out * special.h1vp(m, x4)],
    ], dtype=complex)
    return M / np.abs(M).max(axis=0)          # column scaling: the same roots, better conditioned


def ring_resonance(m: int, f_guess: complex, n_ring: float, a: float, b: float, n_in: float = 1.0,
                   n_out: float = 1.0) -> complex:
    """Exact complex resonance frequency (1/um) of the TM (Ez) whispering-gallery mode of azimuthal order m of a
    dielectric ring a < r < b in 2-D (time dependence exp(-i omega t); Im f < 0 for radiation loss)."""
    def det(f):
        return np.linalg.det(_ring_matrix(2 * np.pi * f, m, n_in, n_ring, n_out, a, b))

    return complex(optimize.newton(det, complex(f_guess), x1=complex(f_guess) * (1 + 1e-4), tol=1e-13,
                                   maxiter=200))


def slab_te_neff(f: float, n_core: float, n_clad: float, width: float) -> float:
    """Effective index of the fundamental even mode with E parallel to the faces of a symmetric slab (exact)."""
    k0 = 2 * np.pi * f

    def g(ne):
        kx = k0 * np.sqrt(n_core**2 - ne**2)
        gam = k0 * np.sqrt(ne**2 - n_clad**2)
        return np.tan(kx * width / 2) - gam / kx

    hi = n_core - 1e-12
    lo = max(n_clad + 1e-12, np.sqrt(max(n_core**2 - (np.pi / (k0 * width)) ** 2, 0)) + 1e-12)
    return float(optimize.brentq(g, lo, hi))


def dipole_flux_3d(f, n: float, current_spectrum):
    """Flux sum Re(E x conj H) through a closed surface around a point dipole in a homogeneous medium of index n,
    in the DFT convention of the monitors: n omega^2 |I(f)|^2 / (6 pi), with I(f) the spectrum of the dipole
    moment current (I l). The time average of a harmonic dipole is n omega^2 |I|^2 / (12 pi) (Jackson 9.24 with
    eta = 1 / n, k = n omega)."""
    om = 2 * np.pi * np.asarray(f, float)
    return n * om**2 * np.abs(current_spectrum) ** 2 / (6 * np.pi)


def slab_tm_neff(f: float, n_core: float, n_clad: float, width: float) -> float:
    """Effective index of the fundamental even mode with E normal to the faces of a symmetric slab (H parallel to
    the faces; exact): tan(kx w / 2) = (n_core / n_clad)^2 gamma / kx."""
    k0 = 2 * np.pi * f
    r = (n_core / n_clad) ** 2

    def g(ne):
        kx = k0 * np.sqrt(n_core**2 - ne**2)
        gam = k0 * np.sqrt(ne**2 - n_clad**2)
        return np.tan(kx * width / 2) - r * gam / kx

    hi = n_core - 1e-12
    lo = max(n_clad + 1e-12, np.sqrt(max(n_core**2 - (np.pi / (k0 * width)) ** 2, 0)) + 1e-12)
    return float(optimize.brentq(g, lo, hi))


def slab_bloch_frequency(beta: float, n_core: float, n_clad: float, width: float, dx: float, dt: float,
                         e_normal: bool = False) -> float:
    """Frequency (1/um) of the fundamental even slab mode with the propagation constant beta along x on a Yee grid
    that is exact across the slab: the grid replaces beta by (2 / dx) sin(beta dx / 2) (central difference of
    exp(i beta x) on the staggered grid, exact for a structure that is uniform along x), and the leapfrog in time
    maps the frequency w of the space-continuous problem to (2 / dt) arcsin(w dt / 2) (Taflove and Hagness,
    Computational Electrodynamics, ch. 4). The remaining FDTD error is the discretization across the slab
    (the geometry and the derivative in y). e_normal: E normal to the faces (slab_tm_neff), else parallel."""
    neff = slab_tm_neff if e_normal else slab_te_neff
    b = 2.0 / dx * np.sin(beta * dx / 2)

    def h(f):
        return 2 * np.pi * f * neff(f, n_core, n_clad, width) - b

    fc = optimize.brentq(h, b / (2 * np.pi * n_core) * (1 + 1e-9), b / (2 * np.pi * n_clad) * (1 - 1e-9), xtol=1e-14)
    return float(2.0 / dt * np.arcsin(np.pi * fc * dt) / (2 * np.pi))


def disk_resonance(m: int, f_guess: complex, n: float, radius: float, te: bool = False, n_out: float = 1.0) -> complex:
    """Exact complex resonance frequency (1/um) of the whispering-gallery mode of azimuthal order m of a dielectric
    disk in 2-D. TM: Ez = A J_m(n k r) inside, D H1_m(n_out k r) outside, Ez and dEz/dr continuous. TE (te=True):
    the same for Hz, with Hz and (1 / eps) dHz/dr continuous. Root of p_in J_m'/J_m - p_out H_m'/H_m with p = n (TM)
    or 1 / n (TE)."""
    p_in, p_out = (1 / n, 1 / n_out) if te else (n, n_out)

    def g(f):
        k = 2 * np.pi * f
        x1, x2 = n * k * radius, n_out * k * radius
        return p_in * special.jvp(m, x1) / special.jv(m, x1) - p_out * special.h1vp(m, x2) / special.hankel1(m, x2)

    with warnings.catch_warnings():     # Newton steps near a pole of J_m or H_m: NaN steps, harmless; keep stderr clean
        warnings.simplefilter("ignore", RuntimeWarning)
        return complex(optimize.newton(g, complex(f_guess), x1=complex(f_guess) * (1 + 1e-4), tol=1e-13, maxiter=200))
