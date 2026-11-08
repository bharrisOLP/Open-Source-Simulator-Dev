"""The switching time of a thermo-optic phase shifter: the phase of the waveguide mode against time after a step of
the heater power (on or off), and its 10 % to 90 % rise or fall time.

1. Heat (thermal-gpu): the transient of the thermal window (the problem of physics.temperature), BDF2 with time steps
   that double every per_stage steps. A constant kappa uses transient3d.solve_transient2d; kappa(T) and rho c_p(T)
   laws per material use transient_kappa.solve_transient_kappa2d. "on": from t_ambient, the heater on from t = 0.
   "off": from the steady state with the heater on (the steady solve of the same model), the heater off from t = 0.
   The monitors are the thermal cells around the optics window (one monitor per cell, so the temperature there is
   known at each step) and the core.
2. Modes (meow-gpu): one mode solve of the cross-section without heat. The first-order change of the mode index for
   a real index change dn (the perturbation formula of the propagation constant, for example A. W. Snyder and
   J. D. Love, "Optical Waveguide Theory", Chapman and Hall (1983), the chapter on perturbation theory):

       dneff = sum n dn |E|^2 dA / (eta0 Re sum (E x H*)_z dA)

   E and H are the fields of the solver (H in the units of tidy3d: E x H* is the power density); each E component
   is at its own Yee point, where the solver samples its permittivity (Ex and Hy, Ey and Hx are at the same points).
   A plane wave in a uniform medium gives dneff = dn. With dn = dn_dT dT and the bilinear interpolation of the
   thermal cells onto the Yee points (as in physics.phase_shifter_ppi), dneff = sum_j S_j dT_j over the thermal
   cells j: S (1/K) is the sensitivity map, and dneff at each time step is one dot product.
3. Checks by full mode solves (the index n + dn_dT dT on the Yee points, as in phase_shifter_ppi) at the final time
   and at selected times (full_times).
4. The times at which dneff has made 10 % and 90 % of its step (thermal_gpu.transient3d.rise_time): for "on" the
   step to the steady value, for "off" the step to 0.

slab_switch_time gives the heat-diffusion estimate of a layer stack: a 1-D slab (thickness L, diffusivity alpha)
on a heat sink, heated by a constant flux at its top. The rise at the height z above the sink, as a fraction of its
steady value (Carslaw and Jaeger, "Conduction of Heat in Solids", 2nd ed., Oxford (1959), the slab with one face at
a constant temperature and a constant flux into the other face):

    f(z, t) = 1 - sum_n 2 (-1)^n sin(l_n z) / (L l_n^2 z) exp(-alpha l_n^2 t),    l_n = (2 n + 1) pi / (2 L)

The slowest time constant is tau = 4 L^2 / (pi^2 alpha); a single exponential has the 10-90 % time tau ln 9.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import brentq

from photonsuite.geometry import CrossSection
from photonsuite.physics import (ConvergenceError, Window, _centers, _require_x64, mode_solve, temperature,
                                 thermal_problem, yee_points)

ETA0 = 376.730313412            # ohm, the impedance of free space (as in meow_gpu.fde.operator.fields)
SWITCHES = ("on", "off")


@dataclass
class PhaseTransient:
    switch: str                     # "on" or "off"
    times: np.ndarray               # (n,) s after the switch
    dneff: np.ndarray               # (n,) first-order change of the mode index against the cold cross-section
    dneff_steady: float             # first-order change with the steady field of the heater on
    core_rise: np.ndarray | None    # (n,) mean temperature rise of the core shapes (K); None without core_tag
    core_steady: float | None       # the same in the steady state with the heater on
    t10: float                      # s: dneff has made 10 % of its step
    t90: float                      # s: 90 %
    switch_time: float              # t90 - t10 (s): the rise time ("on") or the fall time ("off")
    core_switch_time: float | None  # the same for the mean core temperature
    full: dict                      # {time (s): dneff of a full mode solve}; always holds the final time
    neff0: complex                  # the mode index without heat
    sensitivity: np.ndarray         # (ni, nj) d neff / dT of each thermal cell of the box around the optics window
    x: np.ndarray                   # (ni,) cell centers of that box (um)
    y: np.ndarray                   # (nj,)
    wavelength: float               # um
    thermal: object                 # the result of the thermal-gpu transient

    def phase(self, length: float) -> np.ndarray:
        """The phase change (rad) of a heater of this length (um) at each time: 2 pi length dneff / wavelength."""
        return 2 * np.pi * length * self.dneff / self.wavelength


def _areas(win: Window):
    """The areas (um^2) of the Ex, Ey and Ez points: the forward and the dual cell sizes of the mode solver
    (meow_gpu.fde.operator.make_problem: dl_f, dl_b)."""
    out = []
    for e in win.edges():
        d = np.diff(e)
        out.append((d, np.concatenate([d[:1], 0.5 * (d[:-1] + d[1:])])))
    (fx, bx), (fy, by) = out
    return np.outer(fx, by), np.outer(bx, fy), np.outer(bx, by)


def index_weights(win: Window, n_points, e, h):
    """The first-order weights of one mode: dneff = sum_c sum w_c dn_c, for a real index change dn_c at the Ex, Ey
    and Ez points (c = 0, 1, 2). n_points: the index at these points (3 arrays); e, h: the fields (3, Nx, Ny) of
    the solver. w_c = n_c |E_c|^2 dA_c / (eta0 P), P = Re sum (Ex Hy* dA_x - Ey Hx* dA_y)."""
    dax, day, daz = _areas(win)
    power = float(np.real(np.sum(e[0] * np.conj(h[1]) * dax - e[1] * np.conj(h[0]) * day)))
    return [np.real(n) * np.abs(ec) ** 2 * da / (ETA0 * power) for n, ec, da in zip(n_points, e, (dax, day, daz))]


def _linear_index(grid, p):
    """The interval and the fraction of linear interpolation (and extrapolation) on the grid, as scipy's
    RegularGridInterpolator."""
    i = np.clip(np.searchsorted(grid, p) - 1, 0, len(grid) - 2)
    return i, (p - grid[i]) / (grid[i + 1] - grid[i])


def to_cells(weights, points, xg, yg):
    """The weights of points (lists of arrays, the same shapes) moved to the cells of the grid (xg, yg) by the
    transpose of bilinear interpolation: sum_p w_p f(p) = sum_ij S_ij f_ij for each field f_ij on the grid."""
    s = np.zeros((len(xg), len(yg)))
    for w, (X, Y) in zip(weights, points):
        i, fx = _linear_index(xg, np.ravel(X))
        j, fy = _linear_index(yg, np.ravel(Y))
        w = np.ravel(w)
        for di, wx in ((0, 1 - fx), (1, fx)):
            for dj, wy in ((0, 1 - fy), (1, fy)):
                np.add.at(s, (i + di, j + dj), w * wx * wy)
    return s


def _cover(c, a, b):
    """The cells [i0, i1) of the centers c whose linear interpolation covers [a, b] as on the whole grid."""
    return max(int(np.searchsorted(c, a)) - 2, 0), min(int(np.searchsorted(c, b)) + 2, len(c))


def _laws(xs: CrossSection, idx, table: dict, attr: str):
    """material_kappa of the laws per material name (thermal_gpu.nonlinear); other materials: Constant."""
    from thermal_gpu.nonlinear import Constant, material_kappa

    mats = [xs.background] + [s.material for s in xs.shapes]
    unknown = set(table) - {m.name for m in mats}
    if unknown:
        raise ValueError(f"no material named {sorted(unknown)} in the cross-section")
    return material_kappa(idx, [table.get(m.name, Constant(getattr(m, attr))) for m in mats])


def phase_transient(xs: CrossSection, optics: Window, thermal: Window, heater_tag: str, wavelength: float,
                    power_per_length: float = 100.0, switch: str = "on", core_tag: str | None = None,
                    kappa_laws: dict | None = None, rho_cp_laws: dict | None = None, t_end: float = 200e-6,
                    dt0: float = 1e-8, per_stage: int = 20, full_times=(), t_ambient: float = 300.0,
                    top_h: float = 10.0, solver=None) -> PhaseTransient:
    """The change of the fundamental mode index against time after the heater switches on or off (module docstring).

    kappa_laws, rho_cp_laws: {material name: law of thermal_gpu.nonlinear (Constant, PowerLaw, LinearLaw)}; the
    other materials keep their constant kappa and rho_cp. Both None: the constant-kappa transient. core_tag: the tag
    of the waveguide core (the mean core temperature). full_times (s): full mode solves at the time steps nearest
    to these times. Time steps: geometric stages from dt0, doubled every per_stage steps, to t_end (BDF2).
    Raises ConvergenceError when a thermal or a mode solve does not converge."""
    from meow_gpu.meow_backend import GPUModeSolver
    from thermal_gpu.transient3d import geometric_stages, rise_time, solve_transient2d

    if switch not in SWITCHES:
        raise ValueError(f"switch must be one of {SWITCHES}, not {switch!r}")
    _require_x64()
    x, y, idx, prop, q, bc = thermal_problem(xs, thermal, heater_tag, power_per_length, t_ambient, top_h)
    xc, yc = _centers(x), _centers(y)
    vol = np.outer(np.diff(x), np.diff(y))
    i0, i1 = _cover(xc, optics.x0, optics.x1)
    j0, j1 = _cover(yc, optics.y0, optics.y1)
    xb, yb = xc[i0:i1], yc[j0:j1]
    monitors = [(i, j) for i in range(i0, i1) for j in range(j0, j1)]
    core = None
    if core_tag is not None:
        core = xs.mask(core_tag, *np.meshgrid(xc, yc, indexing="ij"))
        if not core.any():
            raise ValueError(f"no thermal cell inside the core {core_tag!r}: use a smaller step")
        monitors.append(core)

    # the steady state with the heater on: the final state of "on", the initial state of "off"
    nonlinear = kappa_laws is not None or rho_cp_laws is not None
    if nonlinear:
        from thermal_gpu.nonlinear import solve_steady_kappa2d
        from thermal_gpu.transient_kappa import solve_transient_kappa2d

        kappa_fn = _laws(xs, idx, kappa_laws or {}, "kappa")
        rho = _laws(xs, idx, rho_cp_laws, "rho_cp") if rho_cp_laws is not None else prop["rho_cp"]
        st = solve_steady_kappa2d(x * 1e-6, y * 1e-6, kappa_fn, q, bc)
        if not st.converged:
            raise ConvergenceError(f"steady kappa(T) solve: residual {st.residual:.1e}")
        t_on = st.T
    else:
        t_on = temperature(xs, thermal, heater_tag, power_per_length, t_ambient, top_h).T
    stages = geometric_stages(dt0, t_end, per_stage)
    q_run, t0 = (q, None) if switch == "on" else (np.zeros_like(q), t_on)
    if nonlinear:
        tr = solve_transient_kappa2d(x * 1e-6, y * 1e-6, kappa_fn, q_run, bc, rho, stages, scheme="bdf2",
                                     monitors=monitors, t0_field=t0)
    else:
        tr = solve_transient2d(x * 1e-6, y * 1e-6, prop["kappa"], q_run, bc, prop["rho_cp"], stages, scheme="bdf2",
                               monitors=monitors, t0_field=t0)
    if not tr.converged:
        raise ConvergenceError(f"thermal transient: step residual {tr.max_residual:.1e}")
    times = tr.times
    box = tr.monitors[:, :(i1 - i0) * (j1 - j0)].reshape(len(times), i1 - i0, j1 - j0) - t_ambient
    box_on = t_on[i0:i1, j0:j1] - t_ambient

    # one mode solve without heat, and the sensitivity map of the thermal cells
    solver = solver or GPUModeSolver(precision="mixed", extra=4)
    neff, e, h = mode_solve(xs, optics, wavelength, 1, solver=solver)
    points = yee_points(optics)
    props = [xs.sample(X, Y)[2] for X, Y in points]
    w = index_weights(optics, [p["n"] for p in props], e[0], h[0])
    sens = to_cells([wc * p["dn_dT"] for wc, p in zip(w, props)], points, xb, yb)
    dneff = np.einsum("tij,ij->t", box, sens)
    dneff_steady = float(np.sum(sens * box_on))

    def full_dneff(rise_box):
        f = RegularGridInterpolator((xb, yb), rise_box, bounds_error=False, fill_value=None)

        def dn(X, Y):
            return xs.sample(X, Y)[2]["dn_dT"] * f(np.stack([X.ravel(), Y.ravel()], -1)).reshape(X.shape)

        return float(np.real(mode_solve(xs, optics, wavelength, 1, dn=dn, solver=solver)[0][0] - neff[0]))

    ks = sorted({int(np.argmin(np.abs(times - t))) for t in full_times} | {len(times) - 1})
    full = {float(times[k]): full_dneff(box[k]) for k in ks}

    final = dneff_steady if switch == "on" else 0.0
    t10, t90, sw = rise_time(times, dneff, final)
    core_rise = core_steady = core_sw = None
    if core is not None:
        core_rise = tr.monitors[:, -1] - t_ambient
        core_steady = float(np.sum(vol * core * (t_on - t_ambient)) / np.sum(vol * core))
        core_sw = rise_time(times, core_rise, core_steady if switch == "on" else 0.0)[2]
    return PhaseTransient(switch, times, dneff, dneff_steady, core_rise, core_steady, t10, t90, sw, core_sw, full,
                          complex(neff[0]), sens, xb, yb, wavelength, tr)


def slab_rise(thickness: float, diffusivity: float, height: float, t, terms: int = 400) -> np.ndarray:
    """The rise at the height above the sink of a 1-D slab heated by a constant flux at its top, as a fraction of
    its steady value, at the times t (module docstring). SI units or any consistent units."""
    n = np.arange(terms)
    lam = (2 * n + 1) * np.pi / (2 * thickness)
    c = 2 * (-1.0) ** n * np.sin(lam * height) / (thickness * lam**2 * height)
    t = np.asarray(t, float)
    return 1 - np.sum(c * np.exp(-diffusivity * lam**2 * t[..., None]), axis=-1)


def slab_switch_time(thickness: float, diffusivity: float, height: float | None = None) -> tuple:
    """The heat-diffusion estimate of the 10 % and 90 % times: (t10, t90, t90 - t10). height None: the single
    exponential of the slowest time constant tau = 4 L^2 / (pi^2 alpha) (t10 = tau ln(10/9), t90 = tau ln 10);
    else the 1-D series at that height."""
    tau = 4 * thickness**2 / (np.pi**2 * diffusivity)
    if height is None:
        t10, t90 = tau * np.log(10 / 9), tau * np.log(10)
    else:
        if not 0 < height <= thickness:
            raise ValueError("the height must be in (0, thickness]")
        t10, t90 = (brentq(lambda t, lv=lv: slab_rise(thickness, diffusivity, height, t) - lv, 1e-9 * tau, 50 * tau,
                           xtol=1e-12 * tau) for lv in (0.1, 0.9))
    return t10, t90, t90 - t10
