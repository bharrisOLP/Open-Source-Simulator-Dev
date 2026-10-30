"""Independent references for the checks (no FDTD): exact transfer matrices, the exact resonances of a dielectric
ring (Bessel functions), and the radiated power of a point dipole. Units: um, frequency in 1/um, c = 1."""

from __future__ import annotations

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
