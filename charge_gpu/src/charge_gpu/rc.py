"""Small-signal junction capacitance, series resistance and the RC-limited bandwidth of a depletion modulator.

All values are per length along the waveguide (the 2-D cross-section of ``poisson.Device`` is one unit of length in
z): charge in C/m, capacitance in F/m, resistance in Ohm m. A lumped device of length L has C = C' L and
R = R' / L.

Capacitance C'(V) = dQ_n/dV. Q_n is the charge on the n side: the sum over the n-doped cells (N_D - N_A > 0) of
q (p - n + N_D - N_A) A_cell, that is the depleted donors less the mobile charge that remains there. Why this is the
small-signal capacitance in this model: the electron quasi-Fermi level is the n-contact voltage everywhere and the
hole quasi-Fermi level is the p-contact voltage everywhere (poisson.py), so no conduction current crosses the
junction. A change dQ_n of the n-side charge must then come through the n contact, the terminal current is
I = dQ_n/dt = C' dV/dt, and the p side gives the opposite charge (the total charge is the flux through the
contacts, zero in neutral contact regions; the code reports both sides as a check). The model is quasi-static: valid
far below the dielectric relaxation frequency q mu N / (2 pi eps): at 1e17 cm^-3 about 2e12 Hz for electrons and
0.8e12 Hz for holes. C' includes the fringe field through the oxide between the p and n silicon inside the grid; it
does not include the capacitance of the metal electrodes and pads (not in the model). Grid: cells larger than about
two Debye lengths give an error of C' that oscillates with the bias (the depletion edges cross the cells; up to
1.9 % at 1e18 cm^-3 with 10 nm cells). Keep the cells below about 1.5 Debye lengths (Debye length: 13, 7.5 and
4.1 nm at 1e17, 3e17 and 1e18 cm^-3).

Central differences with continuation: C'(V) = (Q_n(V + h) - Q_n(V - h)) / (2 h). Each solve starts from the
converged potential of the previous bias. The relative truncation error is h^2 Q'''/(6 Q'). In the depletion
approximation with the Debye term (Q ~ u^(1/2), u = V_bi + V - 2 V_T) it is h^2 / (8 u^2): below 1e-4 for
h = 20 mV when u > 0.71 V.

Series resistance R'(V) of the doped silicon between the contacts and the depletion edges: a 1-D path model along x
(one lateral junction, contacts on the xmin and xmax sides). Assumptions:
- Each grid column (one x cell, all y cells) is equipotential in y; the current flows in x. The column is one
  resistor dx / G(x), G(x) = sum_y q (mu_n n + mu_p p) dy, with the carriers of the solution at the bias. The
  current spreading between the thin slab and the rib is not modeled.
- Column x carries the conduction current f(x) I. f(x) is the part of the small-signal charge (the change between
  V - h and V + h) that lies between the junction and the column: 1 in the neutral silicon, 0 deep in the depletion
  region (there the current is displacement current). R' = sum_x dx f^2 / G (the dissipation, so Re Z for
  frequency -> 0, of this distributed RC path). So the undepleted region at each bias sets R', and the depletion
  edge needs no threshold. R'_n and R'_p come from the n and the p columns; R' = R'_n + R'_p.
- Mobility: Masetti et al. (1983), Si at 300 K, phosphorus (electrons) and boron (holes), as a function of
  |N_D - N_A| (the model knows only the net doping, so compensation is not included).
- Not included: the contact and metal resistance, velocity saturation (no effect on the small signal), the
  frequency dependence of R' (the skin depth in silicon at these frequencies is much larger than the slab), a PIN
  junction (each column must have net doping).

RC bandwidth of a lumped device of length L, driven by a source resistance R_s:
f_3dB = 1 / (2 pi (R'/L + R_s) C' L). With R_s = 0 it is 1 / (2 pi R' C'), the same for all L. The lumped model is
valid only if L is short compared with the RF wavelength on the electrodes (else use a traveling-wave model).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from charge_gpu.poisson import Q, Device, Solution, solve

# Masetti, Severi, Solmi, IEEE Trans. Electron Devices ED-30, 764 (1983), Table I (300 K, N in cm^-3, mu in
# cm^2/(V s)): mu = mu_min1 exp(-P_c/N) + (mu_max - mu_min2) / (1 + (N/C_r)^alpha) - mu_1 / (1 + (C_s/N)^beta)
MASETTI = {
    "n": dict(mu_min1=68.5, mu_min2=68.5, mu_1=56.1, mu_max=1414.0, p_c=0.0, c_r=9.20e16, c_s=3.41e20,
              alpha=0.711, beta=1.98),                                   # phosphorus-doped Si (electrons)
    "p": dict(mu_min1=44.9, mu_min2=0.0, mu_1=29.0, mu_max=470.5, p_c=9.23e16, c_r=2.23e17, c_s=6.10e20,
              alpha=0.719, beta=2.00),                                   # boron-doped Si (holes)
}


def masetti_mobility(doping_m3, carrier: str) -> np.ndarray:
    """Low-field mobility (m^2/(V s)) of electrons (carrier "n") or holes ("p") in Si at 300 K, as a function of
    the dopant concentration (m^-3), Masetti et al. (1983). Zero doping gives mu_max (1414 and 470.5 cm^2/(V s))."""
    c = MASETTI[carrier]
    n_cm = np.maximum(np.abs(np.asarray(doping_m3, float)) * 1e-6, 1.0)     # 1 cm^-3 floor: no division by zero
    mu = (c["mu_min1"] * np.exp(-c["p_c"] / n_cm) + (c["mu_max"] - c["mu_min2"]) / (1 + (n_cm / c["c_r"]) ** c["alpha"])
          - c["mu_1"] / (1 + (c["c_s"] / n_cm) ** c["beta"]))
    return mu * 1e-4


def side_charges(dev: Device, sol: Solution) -> tuple[float, float]:
    """(Q_n, Q_p) in C/m: the sum of q (p - n + N_D - N_A) A_cell over the n-doped and over the p-doped cells."""
    vol = np.diff(dev.x)[:, None] * np.diff(dev.y)[None, :]
    rho = Q * (sol.p - sol.n + dev.n_net) * vol * dev.semiconductor
    return float(np.sum(rho[dev.n_net > 0])), float(np.sum(rho[dev.n_net < 0]))


def column_conductance(dev: Device, sol: Solution) -> np.ndarray:
    """G(x) in S (per unit length in z) of each grid column: sum_y q (mu_n n + mu_p p) dy over the silicon cells."""
    if abs(dev.T - 300.0) > 1.0:
        raise ValueError("the Masetti mobility is for 300 K")
    doping = np.abs(dev.n_net)
    sigma = Q * (masetti_mobility(doping, "n") * sol.n + masetti_mobility(doping, "p") * sol.p) * dev.semiconductor
    return np.sum(sigma * np.diff(dev.y)[None, :], axis=1)


def path_resistance(dev: Device, sol: Solution, fraction: np.ndarray | None = None) -> float:
    """sum_x dx f(x)^2 / G(x) in Ohm m over all columns: the resistance between the xmin and xmax sides when the
    fraction f(x) of the current flows as conduction current in column x (None: f = 1, all of it, as in a
    resistor)."""
    g = column_conductance(dev, sol)
    if np.any(g <= 0):
        raise ValueError("each column between the contacts needs silicon (the path is broken)")
    f = np.ones_like(g) if fraction is None else np.asarray(fraction, float)
    return float(np.sum(np.diff(dev.x) * f**2 / g))


def series_resistance(dev: Device, sol: Solution, sol_lo: Solution, sol_hi: Solution) -> tuple[float, float]:
    """(R'_n, R'_p) in Ohm m at the bias of ``sol``; sol_lo and sol_hi: the solves at V - dv and V + dv.

    Dissipation form of the series resistance of a distributed RC path (the low-frequency limit of Re Z): column x
    carries the conduction current f(x) I, where f(x) is the part of the small-signal charge dQ (sol_hi - sol_lo)
    that lies between the junction and the column (half of the column itself included). R' = sum dx f^2 / G. In the
    neutral silicon f = 1 (all of the current), in the depletion region f -> 0 (displacement current), so the
    undepleted region at the bias sets R', with no threshold for the depletion edge. A uniform RC line gives the
    known R/3. dQ comes from the carriers only, q (dp - dn), so the doping cancels exactly (no round-off)."""
    if any(c.side not in ("xmin", "xmax") for c in dev.contacts):
        raise ValueError("series_resistance needs a lateral path: contacts on the xmin and xmax sides only")
    dy = np.diff(dev.y)
    net = np.sum(dev.n_net * dev.semiconductor * dy[None, :], axis=1)
    n_cols, p_cols = np.flatnonzero(net > 0), np.flatnonzero(net < 0)
    if (n_cols.size + p_cols.size != net.size or not n_cols.size or not p_cols.size
            or n_cols.size != n_cols[-1] - n_cols[0] + 1 or p_cols.size != p_cols[-1] - p_cols[0] + 1):
        raise ValueError("series_resistance needs one lateral junction: a block of n columns and a block of p columns")
    doped = dev.semiconductor & (dev.n_net != 0)
    if np.any(doped[n_cols] & (dev.n_net[n_cols] < 0)) or np.any(doped[p_cols] & (dev.n_net[p_cols] > 0)):
        # a column with n and p cells (a vertical or L-shaped junction): the 1-D path model does not apply
        raise ValueError("series_resistance needs one lateral junction: a column has n and p cells")
    vol = np.diff(dev.x)[:, None] * dy[None, :]
    dq = np.sum(Q * ((sol_hi.p - sol_lo.p) - (sol_hi.n - sol_lo.n)) * vol * dev.semiconductor, axis=1)
    frac = np.zeros_like(net)
    for cols, other in ((n_cols, p_cols), (p_cols, n_cols)):
        order = cols if cols[0] > other[0] else cols[::-1]                  # from the junction to the contact
        part = dq[order]
        if not np.sum(part) != 0:                                            # zero or NaN
            raise ValueError("no small-signal charge: sol_lo and sol_hi must be at two different biases")
        frac[order] = (np.cumsum(part) - 0.5 * part) / np.sum(part)
    g = column_conductance(dev, sol)
    r = np.diff(dev.x) * frac**2 / g
    return float(np.sum(r[n_cols])), float(np.sum(r[p_cols]))


def rc_bandwidth(r_per_length: float, c_per_length: float, length: float, r_source: float = 0.0) -> float:
    """f_3dB (Hz) = 1 / (2 pi (R'/L + R_s) C' L) of a lumped device of length L (m), R' in Ohm m, C' in F/m."""
    return 1.0 / (2 * np.pi * (r_per_length / length + r_source) * c_per_length * length)


@dataclass
class SmallSignal:
    bias: float              # reverse bias V = v_n - v_p (V)
    c: float                 # C' = dQ_n/dV (F/m)
    c_p: float               # -dQ_p/dV (F/m): the same as c when the total charge is neutral
    q_n: float               # Q_n and Q_p at the bias (C/m)
    q_p: float
    r_n: float               # R'_n, R'_p (Ohm m) at the bias
    r_p: float
    converged: bool          # all three solves converged
    solution: Solution       # the solve at the bias

    @property
    def r(self) -> float:
        return self.r_n + self.r_p

    def f3db(self, length: float | None = None, r_source: float = 0.0) -> float:
        """f_3dB (Hz). With no source resistance the length has no effect; with r_source the length is required."""
        if length is None:
            if r_source != 0.0:
                raise ValueError("f3db with a source resistance needs the device length")
            length = 1.0
        return rc_bandwidth(self.r, self.c, length, r_source)


def small_signal(dev: Device, biases, dv: float = 0.02, v_p: float = 0.0, psi0: np.ndarray | None = None,
                 **solve_kw) -> list[SmallSignal]:
    """C'(V), Q and R'(V) at each reverse bias (in the given order, best increasing). Three solves per bias
    (V - dv, V, V + dv), each started from the previous converged potential (continuation)."""
    if dv <= 0:
        raise ValueError("dv must be positive")
    out, psi = [], psi0
    for v in biases:
        sols = []
        for vv in (v - dv, v, v + dv):
            s = solve(dev, v_p + vv, v_p, psi0=psi, **solve_kw)
            psi = s.psi
            sols.append(s)
        (qn_lo, qp_lo), (qn, qp), (qn_hi, qp_hi) = (side_charges(dev, s) for s in sols)
        r_n, r_p = series_resistance(dev, sols[1], sols[0], sols[2])
        out.append(SmallSignal(bias=float(v), c=(qn_hi - qn_lo) / (2 * dv), c_p=-(qp_hi - qp_lo) / (2 * dv),
                               q_n=qn, q_p=qp, r_n=r_n, r_p=r_p, converged=all(s.converged for s in sols),
                               solution=sols[1]))
    return out
