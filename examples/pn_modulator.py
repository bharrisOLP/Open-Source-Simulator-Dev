"""Depletion modulator on the GPU: carriers (charge-gpu) -> plasma dispersion -> mode index (meow-gpu) -> V_pi L.

Synthetic SOI rib (textbook values, no process data): 500 x 220 nm rib on a 90 nm slab, lateral PN junction at the
rib center, 3e17 cm^-3 on both sides, contacts at the slab ends (3 um away). Reverse bias 0 to 4 V.
V_pi L = V lambda / (2 |dneff(V)|). Published depletion modulators of this type: typically 1.5 to 3 V cm.
The loss change is negative: reverse bias removes free carriers.

    uv run --project meow_gpu --with-editable ./charge_gpu python examples/pn_modulator.py
"""
import json
import time

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np  # noqa: E402
from scipy.interpolate import RegularGridInterpolator  # noqa: E402

from charge_gpu.modulator import plasma_dispersion, rib_pn  # noqa: E402
from charge_gpu.poisson import solve  # noqa: E402
from meow_gpu.fde.operator import make_problem  # noqa: E402
from meow_gpu.meow_backend import GPUModeSolver  # noqa: E402

N_SI, N_SIO2, WL = 3.476, 1.444, 1.55
dev = rib_pn()
xc, yc = 0.5 * (dev.x[1:] + dev.x[:-1]) * 1e6, 0.5 * (dev.y[1:] + dev.y[:-1]) * 1e6    # um

# carriers at each bias (the n contact goes up: reverse bias)
t0 = time.perf_counter()
biases = [0.0, 1.0, 2.0, 4.0]
carriers, psi, info = {}, None, []
for v in biases:
    sol = solve(dev, v, 0.0, psi0=psi)
    psi = sol.psi
    carriers[v] = (sol.n, sol.p)
    info.append((sol.newton_iterations, sol.cg_iterations, sol.converged))
t_charge = time.perf_counter() - t0

# mode grid (um): 3 x 1.4 um around the rib, 20 nm
mx = np.arange(-1.5, 1.5 + 1e-9, 0.02)
my = np.arange(-0.5, 0.9 + 1e-9, 0.02)
n0, p0 = carriers[0.0]


def eps_grids(v):
    dn_idx, dalpha = plasma_dispersion(n0, p0, carriers[v][0], carriers[v][1])
    n_complex = dn_idx + 1j * dalpha * (WL * 1e-6) / (4 * np.pi)        # alpha = 4 pi k / lambda
    f = RegularGridInterpolator((xc, yc), n_complex, bounds_error=False, fill_value=0.0)
    out = []
    xce, yce = 0.5 * (mx[1:] + mx[:-1]), 0.5 * (my[1:] + my[:-1])
    for xx, yy in ((xce, my[:-1]), (mx[:-1], yce), (mx[:-1], my[:-1])):
        X, Y = np.meshgrid(xx, yy, indexing="ij")
        core = ((Y > 0) & (Y < 0.09) | (np.abs(X) < 0.25) & (Y > 0) & (Y < 0.22))
        n = np.where(core, N_SI, N_SIO2) + f(np.stack([X.ravel(), Y.ravel()], -1)).reshape(X.shape)
        out.append(n**2)
    return out


solver = GPUModeSolver(precision="mixed", extra=4)
t0 = time.perf_counter()
neff = {}
for v in biases:
    nv, _, _ = solver.solve(make_problem(*eps_grids(v), mx, my, WL), 1)
    neff[v] = complex(nv[0])
t_modes = time.perf_counter() - t0
rows = {}
for v in biases[1:]:
    d = neff[v] - neff[0.0]
    rows[v] = {"dneff": d.real, "VpiL_Vcm": v * WL * 1e-4 / (2 * abs(d.real)),
               # loss change: alpha = 4 pi Im(neff) / lambda; reverse bias removes carriers, so it is negative
               "dloss_dB_per_cm": 10 * np.log10(np.e) * 4 * np.pi * d.imag / (WL * 1e-4)}
print(json.dumps({"charge_device": sol.device, "charge_cells": int(dev.n_net.size), "charge_s": round(t_charge, 2),
                  "newton_cg_converged": info, "modes_s": round(t_modes, 2), "neff0": neff[0.0].real,
                  "bias": rows}, default=float))
