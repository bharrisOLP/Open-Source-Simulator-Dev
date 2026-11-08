"""Source fields for the coupling examples: one waveguide mode with a defined polarization.

A waveguide with a square core has two degenerate fundamental modes, polarized along x and along y. A mode solver
can give any basis of that 2-D subspace: either mode first, or two mixed (rotated and not orthogonal) modes, and
the effective indices agree. "Mode 0" is then not a defined source. polarized_mode selects the field with the
largest power fraction in the chosen component. Example (review of 2026-11-07, the 0.3 x 0.3 um SiN tip of
examples/coupling_scans_on_device.py with a symmetric core): neff of modes 0 and 1 equal to 1.6e-14, x fractions
0.113 and 0.0067, overlap |P01| / sqrt(P00 P11) 0.94; polarized_mode gave the x source with fraction 0.99746.
"""

from __future__ import annotations

import numpy as np

DEGENERATE_TOL = 1e-3    # |neff_0 - neff_1| below which modes 0 and 1 are one 2-D subspace (beat length >= 1e3 lambda)
TIE_TOL = 1e-6           # fractions of modes 0 and 1 closer than this: no defined choice without the combination
COND_TOL = 1e-8          # smallest eigenvalue / largest eigenvalue of the power matrix P of a combined pair


def _fraction(E: np.ndarray, c: int) -> float:
    """The power fraction of transverse component c (0: Ex, 1: Ey) of one field E (3 or 2, ...)."""
    p = [float(np.vdot(E[i], E[i]).real) for i in (0, 1)]
    return p[c] / (p[0] + p[1])


def polarized_mode(neff, E, pol: str = "x", tol: float = DEGENERATE_TOL, min_fraction: float = 0.0):
    """The mode of the two lowest modes (neff (k,), E (k, 3, Nx, Ny) or (k, 2, Nx, Ny), k >= 1) whose power is
    most in the component pol ("x": Ex, "y": Ey). Returns (field, info).

    - Modes 0 and 1 near-degenerate (|Re neff_0 - Re neff_1| < tol): the combination a_0 E_0 + a_1 E_1 with the
      largest power fraction in pol. This is the largest eigenvalue f_max of the 2 x 2 generalized Hermitian
      problem P_c a = f P a, with P_c[i, j] = <E_i,c, E_j,c> (the chosen component) and P[i, j] = <E_i, E_j> (Ex
      and Ey). A ValueError stops it if P is near-singular (smallest / largest eigenvalue below COND_TOL: the two
      modes are the same field), or if the pair is not one x-like and one y-like field (f_min >= 0.5 or
      f_max <= 0.5: for example the even and odd supermodes of a coupler, both x-polarized; their combination is
      not a mode).
    - Else (or one mode only): the mode (0 or 1) with the larger power fraction in pol, without a change. If the two
      fractions differ by less than TIE_TOL (for example two diagonal polarizations), a ValueError stops it: the
      choice would depend on rounding.
    - A result with a power fraction in pol below min_fraction stops with a ValueError.
    The combination is a field at the source plane, not an eigenmode when neff_0 != neff_1: it changes along z with
    the beat length lambda / |dneff| (at least 1000 lambda for DEGENERATE_TOL). Its neff is the Hermitian mean
    sum_ij a_i* a_j P_ij (neff_i + neff_j) / 2 / (a^H P a): neff_0 for a degenerate pair, and the power-weighted
    mean for orthogonal parts.
    The field is normalized to a transverse power sum_(Ex, Ey) |E|^2 = 1, with the largest value of the chosen
    component real and positive. info: pol, degenerate, dneff (|Re neff_0 - Re neff_1|, inf for one mode), the power
    fractions in pol of modes 0 and 1 ("fractions"), that of the result ("fraction"), neff, the source ("0", "1"
    or "0+1"), and for a combination the overlap |P01| / sqrt(P00 P11) of the two modes and "part_powers", the power
    |a_i|^2 P_ii of each part relative to the power of the result (a power split only for orthogonal parts,
    overlap 0; else the cross term is not in it)."""
    if pol not in ("x", "y"):
        raise ValueError(f"pol {pol!r}: use 'x' or 'y'")
    c = 0 if pol == "x" else 1
    neff = np.asarray(neff)
    E = np.asarray(E)
    if E.ndim < 3 or E.shape[0] < 1 or E.shape[1] < 2 or neff.shape[0] != E.shape[0]:
        raise ValueError(f"E {E.shape} and neff {neff.shape}: give (k, 3 or 2, ...) fields and k effective indices")
    m = min(2, E.shape[0])
    fractions = [_fraction(E[i], c) for i in range(m)]
    dneff = float(abs(neff[0].real - neff[1].real)) if m == 2 else float("inf")
    degenerate = dneff < tol
    extra = {}
    if degenerate:
        V = E[:2, :2].reshape(2, 2, -1)                                    # (mode, Ex/Ey, points)
        Pc = V[:, c].conj() @ V[:, c].T                                    # <E_i,c, E_j,c>
        P = np.einsum("iqp,jqp->ij", V.conj(), V)                          # <E_i, E_j> over Ex and Ey
        p_eig = np.linalg.eigvalsh(P)
        if not p_eig[0] > COND_TOL * p_eig[1]:
            raise ValueError(f"modes 0 and 1 are nearly the same field (eigenvalues of their power matrix "
                             f"{p_eig[0]:.2e}, {p_eig[1]:.2e}): no 2-D subspace to select from")
        Li = np.linalg.inv(np.linalg.cholesky(P))                          # P = L L^H
        vals, vecs = np.linalg.eigh(Li @ Pc @ Li.conj().T)
        if not (vals[0] < 0.5 < vals[1]):
            raise ValueError(f"modes 0 and 1 are not one x-like and one y-like field (fractions in {pol} of their "
                             f"combinations {vals[0]:.4f} to {vals[1]:.4f}; for example two supermodes of one "
                             f"polarization): their combination is not a mode")
        a = Li.conj().T @ vecs[:, -1]                                      # the largest fraction
        field = a[0] * E[0] + a[1] * E[1]
        power = float(np.real(a.conj() @ P @ a))
        nbar = 0.5 * (neff[:2, None] + neff[None, :2])
        n_out = complex(a.conj() @ (P * nbar) @ a / power)
        extra = {"overlap": float(abs(P[0, 1]) / np.sqrt(P[0, 0].real * P[1, 1].real)),
                 "part_powers": [float(abs(a[i]) ** 2 * P[i, i].real / power) for i in (0, 1)]}
        source = "0+1"
    else:
        if m == 2 and abs(fractions[0] - fractions[1]) < TIE_TOL:
            raise ValueError(f"modes 0 and 1 have the same power fraction in {pol} ({fractions[0]:.6f}, "
                             f"{fractions[1]:.6f}) and differ by {dneff:.2e} in neff (above tol {tol:.0e}): "
                             f"no defined choice; a larger tol combines them")
        i = int(np.argmax(fractions))
        field, n_out, source = E[i].astype(complex), complex(neff[i]), str(i)
    peak = field[c].flat[int(np.argmax(np.abs(field[c])))]
    field = field * (np.conj(peak) / abs(peak))
    field = field / np.sqrt(sum(float(np.vdot(field[i], field[i]).real) for i in (0, 1)))
    fraction = _fraction(field, c)
    if fraction < min_fraction:
        raise ValueError(f"the source has only {fraction:.4f} of its power in {pol} (min_fraction {min_fraction})")
    info = {"pol": pol, "degenerate": bool(degenerate), "dneff": dneff, "fractions": fractions,
            "fraction": fraction, "neff": n_out, "source": source} | extra
    return field, info
