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


def dipole_fields_3d(points, f, n: float, current, r0=(0.0, 0.0, 0.0), axis: int = 2):
    """E and H (nfreq, P, 3) of a point current element (current moment I l = current(f), along `axis`) at r0 in a
    homogeneous medium of index n (mu = 1), at each distance (near and far terms). Spherical components about the
    dipole axis a (Balanis, Antenna Theory, 3rd ed., Sec. 4.2, the infinitesimal dipole, with j -> -i for the time
    dependence exp(-i omega t)), eta = 1 / n:
        E_r = eta I l cos(theta) / (2 pi r^2) (1 + i / (k r)) exp(i k r)
        E_theta = -i eta k I l sin(theta) / (4 pi r) (1 + i / (k r) - 1 / (k r)^2) exp(i k r)
        H_phi = -i k I l sin(theta) / (4 pi r) (1 + i / (k r)) exp(i k r)
    with E_theta theta_hat = (E_theta / sin(theta)) (cos(theta) r_hat - a) and H_phi phi_hat = (H_phi / sin(theta))
    (a x r_hat). Units c = 1, eps0 = mu0 = 1."""
    f = np.atleast_1d(np.asarray(f, float))
    current = np.broadcast_to(np.asarray(current, complex), f.shape)
    d = np.atleast_2d(np.asarray(points, float)) - np.asarray(r0, float)
    r = np.linalg.norm(d, axis=1)
    rh = d / r[:, None]
    a = np.zeros(3)
    a[axis] = 1.0
    ct = rh @ a
    th_dir = ct[:, None] * rh - a                                  # sin(theta) theta_hat
    ph_dir = np.cross(a, rh)                                       # sin(theta) phi_hat
    eta = 1.0 / n
    E = np.zeros((len(f), len(r), 3), complex)
    H = np.zeros_like(E)
    for i, (fi, I) in enumerate(zip(f, current)):
        k = 2 * np.pi * fi * n
        kr = k * r
        ph = np.exp(1j * kr)
        er = eta * I * ct / (2 * np.pi * r**2) * (1 + 1j / kr) * ph
        eth = -1j * eta * k * I / (4 * np.pi * r) * (1 + 1j / kr - 1 / kr**2) * ph       # / sin(theta)
        hph = -1j * k * I / (4 * np.pi * r) * (1 + 1j / kr) * ph                          # / sin(theta)
        E[i] = er[:, None] * rh + eth[:, None] * th_dir
        H[i] = hph[:, None] * ph_dir
    return E, H


def line_source_fields_2d(points, f, n: float, current, r0=(0.0, 0.0), component: str = "Ez"):
    """E and H (nfreq, P, 3) of a line current I = current(f) (per unit length along z) at r0 in the x-y plane, in a
    homogeneous medium of index n (mu = 1); x = k rho, H_m the Hankel functions of the first kind (time dependence
    exp(-i omega t); the 2-D Green's function (i/4) H0(k rho), for example Chew, Waves and Fields in Inhomogeneous
    Media, ch. 2). From the vector potential A = mu g I:
    - "Ez" (current along z): E_z = -(omega mu I / 4) H0(x), H_phi = (i k I / 4) H1(x).
    - "Ex" or "Ey" (current along a in the plane, b = z x a, phi' the angle from a): H_z = (i k I / 4) H1(x) sin(phi'),
      and E = (i / (omega eps)) curl H: E_a = -(omega mu I / 4) [(H0 - H1/x) sin^2(phi') + (H1/x) cos^2(phi')],
      E_b = (omega mu I / 4) sin(phi') cos(phi') (H0 - 2 H1/x)."""
    f = np.atleast_1d(np.asarray(f, float))
    current = np.broadcast_to(np.asarray(current, complex), f.shape)
    p = np.atleast_2d(np.asarray(points, float))
    d = p[:, :2] - np.asarray(r0, float)[:2]
    rho = np.linalg.norm(d, axis=1)
    c, s = d[:, 0] / rho, d[:, 1] / rho
    E = np.zeros((len(f), len(rho), 3), complex)
    H = np.zeros_like(E)
    for i, (fi, I) in enumerate(zip(f, current)):
        om = 2 * np.pi * fi
        k = om * n
        x = k * rho
        h0, h1 = special.hankel1(0, x), special.hankel1(1, x)
        if component == "Ez":
            E[i, :, 2] = -om * I / 4 * h0
            hphi = 1j * k * I / 4 * h1
            H[i, :, 0], H[i, :, 1] = -hphi * s, hphi * c
            continue
        if component not in ("Ex", "Ey"):
            raise ValueError(component)
        a = np.array([1.0, 0.0]) if component == "Ex" else np.array([0.0, 1.0])
        b = np.array([-a[1], a[0]])                                   # z x a
        cp, sp = c * a[0] + s * a[1], c * b[0] + s * b[1]             # cos(phi'), sin(phi')
        H[i, :, 2] = 1j * k * I / 4 * h1 * sp
        ea = -om * I / 4 * ((h0 - h1 / x) * sp**2 + h1 / x * cp**2)
        eb = om * I / 4 * sp * cp * (h0 - 2 * h1 / x)
        E[i, :, :2] = ea[:, None] * a + eb[:, None] * b
    return E, H


def line_source_power_2d(f, current, in_plane: bool = False):
    """Radiated power per unit length of a line current I in the DFT convention of the monitors (twice the time
    average |I|^2 eta k / 8 = omega |I|^2 / 8 of a harmonic current; it does not depend on n): omega |I|^2 / 4 for a
    current along z, half of it for a current in the plane (the pattern sin^2)."""
    om = 2 * np.pi * np.asarray(f, float)
    p = om * np.abs(current) ** 2 / 4
    return p / 2 if in_plane else p
