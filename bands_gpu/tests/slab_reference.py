"""Exact references for a symmetric dielectric slab waveguide (core n1, thickness d, cladding n2), independent of the
band solver: the isolated slab (transcendental equations) and the periodic slab array of period L at Bloch K = 0
(the supercell of the band solver; transfer matrix). Units as the solver: omega in 2 pi c / a, kmag in 2 pi / a.

pol "E": E along z (the TE modes of the slab; the "tm" polarization of a 2-D cell); "H": H along z (TM of the slab;
"te" of a 2-D cell). The textbook equations (for example K. Okamoto, Fundamentals of Optical Waveguides, chapter 2):
gamma = p k1 tan(k1 d / 2) (even), gamma = -p k1 cot(k1 d / 2) (odd), k1 = sqrt(n1^2 k0^2 - beta^2),
gamma = sqrt(beta^2 - n2^2 k0^2), p = 1 (TE) or n2^2 / n1^2 (TM).
"""
import numpy as np
from scipy.optimize import brentq


def slab_isolated(pol: str, even: bool, n1: float, n2: float, d: float, omega: float) -> float:
    """kmag of the fundamental guided mode of the given symmetry (even or odd E_z / H_z in y)."""
    k0 = 2 * np.pi * omega
    p = 1.0 if pol == "E" else (n2 / n1) ** 2

    def g(neff):
        b = neff * k0
        k1, gam = np.sqrt(n1**2 * k0**2 - b**2), np.sqrt(b**2 - n2**2 * k0**2)
        if even:
            return gam * np.cos(k1 * d / 2) - p * k1 * np.sin(k1 * d / 2)
        return gam * np.sin(k1 * d / 2) + p * k1 * np.cos(k1 * d / 2)

    ns = np.linspace(n2 + 1e-9, n1 - 1e-9, 20001)
    v = g(ns)
    roots = [brentq(g, ns[i], ns[i + 1], xtol=1e-15) for i in range(len(ns) - 1) if v[i] * v[i + 1] < 0]
    return max(roots) * omega


def supercell_trace(pol: str, omega: float, kmag: float, n1: float, n2: float, d: float, period: float) -> float:
    """cos(K L) of the periodic slab array: half the trace of the transfer matrix of one period."""
    k0, b = 2 * np.pi * omega, 2 * np.pi * kmag
    k1, k2 = np.sqrt(complex(n1**2 * k0**2 - b**2)), np.sqrt(complex(n2**2 * k0**2 - b**2))
    p1, p2 = (1.0, 1.0) if pol == "E" else (1 / n1**2, 1 / n2**2)
    r = p1 * k1 / (p2 * k2)
    c2 = period - d
    return (np.cos(k1 * d) * np.cos(k2 * c2) - 0.5 * (r + 1 / r) * np.sin(k1 * d) * np.sin(k2 * c2)).real


def supercell_exact(pol: str, kguess: float, n1: float, n2: float, d: float, omega: float,
                    period: float) -> tuple[float, float]:
    """The supercell band at K = 0 (root of cos(K L) = 1) nearest to kguess, and its group velocity
    d omega / dk = -F_k / F_omega (central differences of the smooth function F = cos(K L) - 1)."""
    args = (n1, n2, d, period)
    f = lambda k: supercell_trace(pol, omega, k, *args) - 1.0  # noqa: E731
    ks = np.linspace(kguess - 0.02, kguess + 0.02, 40001)
    v = np.array([f(k) for k in ks])
    roots = [brentq(f, ks[i], ks[i + 1], xtol=1e-15) for i in range(len(ks) - 1) if v[i] * v[i + 1] <= 0]
    k = min(roots, key=lambda x: abs(x - kguess))
    h = 1e-6
    fk = (supercell_trace(pol, omega, k + h, *args) - supercell_trace(pol, omega, k - h, *args)) / (2 * h)
    fw = (supercell_trace(pol, omega + h, k, *args) - supercell_trace(pol, omega - h, k, *args)) / (2 * h)
    return k, -fk / fw
