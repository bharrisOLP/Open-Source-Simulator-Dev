"""Thermo-optic phase shifter on the GPU: heat (thermal_gpu) -> index change -> mode (meow_gpu) -> P_pi.

Synthetic SOI stack (textbook values, no process data): 0.5 x 0.22 um Si waveguide on 2 um oxide, metal heater
2 um wide, 2.2 um above the waveguide. dn/dT: Si 1.86e-4 /K, SiO2 0.95e-5 /K, at 1.55 um.
P_pi = lambda / (2 dneff/dP'), with P' the heater power per unit length: independent of the heater length.

    python examples/thermo_optic_phase_shifter.py [--dx 0.02e-6]
"""
import argparse
import json
import time

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from scipy.interpolate import RegularGridInterpolator  # noqa: E402

from meow_gpu.meow_backend import GPUModeSolver  # noqa: E402
from meow_gpu.fde.operator import make_problem  # noqa: E402
from thermal_gpu.solver import solve_steady  # noqa: E402
from thermal_gpu.structures import heater_stack  # noqa: E402

N_SI, N_SIO2, DNDT_SI, DNDT_SIO2, WL = 3.476, 1.444, 1.86e-4, 0.95e-5, 1.55

ap = argparse.ArgumentParser()
ap.add_argument("--dx", type=float, default=0.02e-6, help="thermal grid step (m)")
ap.add_argument("--power", type=float, default=100.0, help="heater power per length (W/m)")
a = ap.parse_args()

t0 = time.perf_counter()
x, y, kappa, q, bc, masks = heater_stack(dx=a.dx, power_per_length=a.power)
th = solve_steady(x, y, kappa, q, bc)
t_thermal = time.perf_counter() - t0
interp = RegularGridInterpolator((0.5 * (x[1:] + x[:-1]) * 1e6, 0.5 * (y[1:] + y[:-1]) * 1e6), th.T - 300.0,
                                 bounds_error=False, fill_value=None)

# mode grid (um): 3 x 2 um around the waveguide, 20 nm
mx = np.arange(-1.5, 1.5 + 1e-9, 0.02)
my = np.arange(-0.9, 1.1 + 1e-9, 0.02)


def eps_at(X, Y, dT):
    core = (np.abs(X) < 0.25) & (Y > 0) & (Y < 0.22)
    n = np.where(core, N_SI + DNDT_SI * dT, N_SIO2 + DNDT_SIO2 * dT)
    return (n**2).astype(complex)


def eps_grids(scale):
    """eps at the Ex, Ey, Ez positions, with the temperature rise times ``scale``."""
    xc, yc, xe, ye = 0.5 * (mx[1:] + mx[:-1]), 0.5 * (my[1:] + my[:-1]), mx[:-1], my[:-1]
    out = []
    for xx, yy in ((xc, ye), (xe, yc), (xe, ye)):
        X, Y = np.meshgrid(xx, yy, indexing="ij")
        dT = scale * interp(np.stack([X.ravel(), Y.ravel()], -1)).reshape(X.shape)
        out.append(eps_at(X, Y, dT))
    return out


solver = GPUModeSolver(precision="mixed", extra=4)
t0 = time.perf_counter()
neff = []
for scale in (0.0, 1.0):
    p = make_problem(*eps_grids(scale), mx, my, WL)
    n, _, _ = solver.solve(p, 2)
    neff.append(float(np.real(n[0])))
t_modes = time.perf_counter() - t0
dneff_per_w = (neff[1] - neff[0]) / a.power            # per (W/m)
p_pi_w = WL * 1e-6 / (2 * dneff_per_w)                 # W (length-independent)
print(json.dumps({"thermal_device": th.device, "thermal_cells": int(kappa.size), "thermal_iterations": th.iterations,
                  "thermal_s": round(t_thermal, 3), "modes_s": round(t_modes, 3),
                  "dT_waveguide_K": round(float(th.T[masks["waveguide"]].mean() - 300.0), 3),
                  "neff_cold": round(neff[0], 6), "dneff": neff[1] - neff[0], "P_pi_mW": round(p_pi_w * 1e3, 2)}))
