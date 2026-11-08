"""Solvers on a CrossSection: modes (meow-gpu), temperature (thermal-gpu), carriers (charge-gpu), and the two
multiphysics workflows of the examples: the thermo-optic phase shifter (P_pi) and the depletion modulator (V_pi L).

A Window is a rectangle (um) and a grid step (um); each solver samples the cross-section on its own window.
Perturbations of the index (temperature, carriers) are functions f(X, Y) -> dn on the optics grid.

Every solve is checked: a solve that does not converge raises ConvergenceError (no silent results). The solvers need
float64: ``import photonsuite`` enables jax_enable_x64, and each function raises RuntimeError if it is off.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from photonsuite.geometry import CrossSection


class ConvergenceError(RuntimeError):
    """A solver of the suite did not reach its tolerance."""


def _require_x64():
    import jax
    if not jax.config.jax_enable_x64:
        raise RuntimeError("photonsuite needs float64: jax.config.update('jax_enable_x64', True) "
                           "(float32 cannot reach the tolerances of the thermal and charge solvers)")


@dataclass(frozen=True)
class Window:
    x0: float
    x1: float
    y0: float
    y1: float
    step: float

    def edges(self):
        x = np.arange(self.x0, self.x1 + self.step / 2, self.step)
        y = np.arange(self.y0, self.y1 + self.step / 2, self.step)
        return x, y


def _centers(e):
    return 0.5 * (e[1:] + e[:-1])


def yee_points(win: Window):
    """The (X, Y) sample points (um) of Ex, Ey and Ez on the Yee grid of the mode solver: the solver samples each
    permittivity component, and gives each field component, at these points."""
    x, y = win.edges()
    xc, yc = _centers(x), _centers(y)
    return [np.meshgrid(xx, yy, indexing="ij") for xx, yy in ((xc, y[:-1]), (x[:-1], yc), (x[:-1], y[:-1]))]


def mode_problem(xs: CrossSection, win: Window, wavelength: float, dn=None):
    """The meow-gpu FDEProblem of the cross-section on the window, with an optional index change dn(X, Y)."""
    from meow_gpu.fde.operator import make_problem

    x, y = win.edges()
    eps = []
    for X, Y in yee_points(win):
        _, _, prop = xs.sample(X, Y)
        n = prop["n"].astype(complex)
        if dn is not None:
            n = n + dn(X, Y)
        eps.append(n**2)
    return make_problem(*eps, x, y, wavelength)


def mode_solve(xs: CrossSection, win: Window, wavelength: float, num_modes: int = 1, dn=None,
               precision: str = "mixed", solver=None):
    """(neff, E, H) of the cross-section (see mode_index); E and H: (num_modes, 3, Nx, Ny) at the Yee points."""
    from meow_gpu.meow_backend import GPUModeSolver

    _require_x64()
    solver = solver or GPUModeSolver(precision=precision, extra=4)
    neff, e, h = solver.solve(mode_problem(xs, win, wavelength, dn), num_modes)
    info = getattr(solver, "last_info", None)
    if info is not None and not info.converged:
        raise ConvergenceError(f"mode solve: residual {info.residual:.1e} above the tolerance")
    return np.asarray(neff), e, h


def mode_index(xs: CrossSection, win: Window, wavelength: float, num_modes: int = 1, dn=None,
               precision: str = "mixed", solver=None):
    """Complex mode indices (sorted by real part) of the cross-section, with an optional index change dn(X, Y)
    (complex: dn + i dk). Uses the meow-gpu FDE solver (the tidy3d mode operator)."""
    return mode_solve(xs, win, wavelength, num_modes, dn, precision, solver)[0]


@dataclass
class Temperature:
    rise: RegularGridInterpolator       # dT (K) at (x, y) in um
    T: np.ndarray
    x: np.ndarray                       # cell centers (um)
    y: np.ndarray
    iterations: int
    residual: float                     # true relative residual of the steady solve


def temperature(xs: CrossSection, win: Window, heater_tag: str, power_per_length: float, t_ambient=300.0,
                top_h: float = 10.0) -> Temperature:
    """Steady temperature with the heater shapes (tag) dissipating power_per_length (W/m) uniformly; the bottom of
    the window at t_ambient (heat sink), convection h = top_h W/m^2/K at the top, adiabatic sides."""
    from thermal_gpu.solver import solve_steady

    _require_x64()
    x, y, idx, prop, q, bc = thermal_problem(xs, win, heater_tag, power_per_length, t_ambient, top_h)
    xc, yc = _centers(x), _centers(y)
    res = solve_steady(x * 1e-6, y * 1e-6, prop["kappa"], q, bc)
    if not res.converged:
        raise ConvergenceError(f"thermal solve: residual {res.residual:.1e} after {res.iterations} iterations")
    rise = RegularGridInterpolator((xc, yc), res.T - t_ambient, bounds_error=False, fill_value=None)
    return Temperature(rise, res.T, xc, yc, res.iterations, res.residual)


def thermal_problem(xs: CrossSection, win: Window, heater_tag: str, power_per_length: float, t_ambient=300.0,
                    top_h: float = 10.0):
    """The thermal problem of `temperature` on the window: cell edges x, y (um), the shape index of each cell
    (0: background, k: shape k - 1), the properties of each cell, the heat source q (W/m^3: power_per_length
    uniformly in the heater cells) and the boundary conditions."""
    x, y = win.edges()
    X, Y = np.meshgrid(_centers(x), _centers(y), indexing="ij")
    idx, _, prop = xs.sample(X, Y)
    heater = xs.mask(heater_tag, X, Y)
    if not heater.any():
        raise ValueError(f"no grid cell inside the heater {heater_tag!r}: use a smaller step")
    area = heater.sum() * (win.step * 1e-6) ** 2
    q = np.where(heater, power_per_length / area, 0.0)
    bc = {"ymin": ("dirichlet", t_ambient), "ymax": ("robin", top_h, t_ambient)}
    return x, y, idx, prop, q, bc


def phase_shifter_ppi(xs: CrossSection, optics: Window, thermal: Window, heater_tag: str, wavelength: float,
                      power_per_length: float = 100.0):
    """P_pi (W) of a thermo-optic phase shifter: lambda / (2 dneff/dP'), P' the heater power per length."""
    th = temperature(xs, thermal, heater_tag, power_per_length)

    def dn(X, Y):
        _, _, prop = xs.sample(X, Y)
        return prop["dn_dT"] * th.rise(np.stack([X.ravel(), Y.ravel()], -1)).reshape(X.shape)

    n0 = mode_index(xs, optics, wavelength)[0]
    n1 = mode_index(xs, optics, wavelength, dn=dn)[0]
    dneff = float(np.real(n1 - n0))
    return wavelength * 1e-6 / (2 * dneff / power_per_length), dneff, th


def charge_device(xs: CrossSection, win: Window):
    """The charge-gpu Device of the cross-section on the window (sampled at the cell centers; SI units), with the
    contacts of `carriers`: an ohmic contact on the left and on the right window side, on every semiconductor cell
    that touches that side. Each solve of charge-gpu sets the contact voltages."""
    from charge_gpu.poisson import Contact, Device

    x, y = win.edges()
    X, Y = np.meshgrid(_centers(x), _centers(y), indexing="ij")
    _, doping, prop = xs.sample(X, Y)
    semi = prop["semiconductor"].astype(bool)
    n_net = np.where(semi, doping * 1e6, 0.0)                     # cm^-3 -> m^-3
    contacts = [Contact("xmin", semi[0, :], 0.0), Contact("xmax", semi[-1, :], 0.0)]    # voltages: set by solve
    return Device(x=x * 1e-6, y=y * 1e-6, eps_r=prop["eps_r"], n_net=n_net, semiconductor=semi, contacts=contacts)


def carriers(xs: CrossSection, win: Window, v_n: float, v_p: float):
    """Carriers (m^-3) of the semiconductor shapes, with constant quasi-Fermi levels (reverse bias: v_n > v_p).

    Contacts: an ohmic contact on the left and on the right window side, on every semiconductor cell that touches
    that side (also a substrate or any other semiconductor shape). Each contact takes v_n if the mean net doping of
    its cells is positive (n type), else v_p (charge_gpu.solve sets it). So a side with mixed doping gets one voltage,
    and the side of the n and p regions does not matter. Returns (Solution, x centers, y centers)."""
    from charge_gpu.poisson import solve

    _require_x64()
    x, y = win.edges()
    xc, yc = _centers(x), _centers(y)
    dev = charge_device(xs, win)
    sol = solve(dev, v_n, v_p)
    if not sol.converged:
        raise ConvergenceError(f"charge solve at v_n={v_n}, v_p={v_p}: no convergence in {sol.newton_iterations} "
                               "Newton steps")
    return sol, xc, yc


def modulator_vpil(xs: CrossSection, optics: Window, charge: Window, wavelength: float, biases=(1.0, 2.0, 4.0)):
    """V_pi L (V cm) at each reverse bias, from the carrier change against 0 V (Soref-Bennett, 1.55 um)."""
    from charge_gpu.modulator import plasma_dispersion

    sol0, xc, yc = carriers(xs, charge, 0.0, 0.0)
    n_ref = mode_index(xs, optics, wavelength)[0]
    out = {}

    for v in biases:
        sol, _, _ = carriers(xs, charge, v, 0.0)
        d_index, d_alpha = plasma_dispersion(sol0.n, sol0.p, sol.n, sol.p)        # absolute concentrations
        nk = d_index + 1j * d_alpha * (wavelength * 1e-6) / (4 * np.pi)
        f = RegularGridInterpolator((xc, yc), nk, bounds_error=False, fill_value=0.0)
        dn = lambda X, Y, f=f: f(np.stack([X.ravel(), Y.ravel()], -1)).reshape(X.shape)  # noqa: E731
        d = mode_index(xs, optics, wavelength, dn=dn)[0] - n_ref
        out[v] = {"dneff": float(np.real(d)), "VpiL_Vcm": v * wavelength * 1e-4 / (2 * abs(np.real(d))),
                  # negative: reverse bias removes free carriers
                  "dloss_dB_per_cm": float(10 * np.log10(np.e) * 4 * np.pi * np.imag(d) / (wavelength * 1e-4))}
    return out
