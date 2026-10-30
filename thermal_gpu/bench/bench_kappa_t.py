"""Benchmark of the steady solve with kappa(T) (nonlinear.solve_steady_kappa3d, Picard iteration): the waveguide rise
of the heater with a constant conductivity, with Si kappa(T) (power law), and with Si kappa(T) and SiO2 kappa(T)
(linear). One JSON line per grid step, power and model.

    --case cross2d   the 2-D cross-section of structures.heater_stack (--power: W/m)
    --case heater3d  the 3-D heater of structures.heater_3d (100 um; --power: W/m, the total power is 100 um x P')

THERMAL_GPU_DEVICE=cpu runs the same code on the CPU. solve_s includes the setup and the compilation."""
import argparse
import json
import time

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np  # noqa: E402

from thermal_gpu.nonlinear import (SI_POWER_LAW, SIO2_CONSTANT, SIO2_LINEAR, Constant, material_kappa,  # noqa: E402
                                   solve_steady_kappa2d, solve_steady_kappa3d)
from thermal_gpu.structures import K_METAL, K_SI, heater_3d, heater_stack, material_labels  # noqa: E402

MODELS = {"constant": [Constant(K_SI), SIO2_CONSTANT, Constant(K_METAL)],
          "si_T": [SI_POWER_LAW, SIO2_CONSTANT, Constant(K_METAL)],
          "si_T+sio2_T": [SI_POWER_LAW, SIO2_LINEAR, Constant(K_METAL)]}

ap = argparse.ArgumentParser()
ap.add_argument("--case", choices=("cross2d", "heater3d"), default="cross2d")
ap.add_argument("--dx", type=float, nargs="+", default=[0.05e-6])
ap.add_argument("--power", type=float, nargs="+", default=[100.0])
ap.add_argument("--cycle", choices=("V", "W"), default="V")
ap.add_argument("--tol", type=float, default=1e-10)
a = ap.parse_args()

for dx in a.dx:
    for pw in a.power:
        if a.case == "cross2d":
            x, y, k, q, bc, m = heater_stack(dx=dx, power_per_length=pw)
            wg = m["waveguide"]
            solve = lambda kf: solve_steady_kappa2d(x, y, kf, q, bc, tol=a.tol, cycle=a.cycle)  # noqa: E731
        else:
            x, y, z, k, q, bc, m = heater_3d(dx=dx, length=100e-6, power=pw * 100e-6)
            zc = 0.5 * (z[1:] + z[:-1])
            wg = m["waveguide"] & (np.abs(zc) <= np.abs(zc).min() * (1 + 1e-9) + 1e-15)[None, None, :]
            solve = lambda kf: solve_steady_kappa3d(x, y, z, kf, q, bc, tol=a.tol, cycle=a.cycle)  # noqa: E731
        labels = material_labels(k)
        base = None
        for name, laws in MODELS.items():
            t0 = time.perf_counter()
            r = solve(material_kappa(labels, laws))
            rise = float(r.T[wg].mean() - 300.0)
            base = rise if base is None else base
            print(json.dumps({"case": a.case, "dx_um": round(dx * 1e6, 6), "cells": int(k.size), "power_W_per_m": pw,
                              "model": name, "device": r.device, "solve_s": round(time.perf_counter() - t0, 3),
                              "picard": r.picard_iterations, "cg_iterations": r.cg_iterations,
                              "residual": r.residual, "converged": r.converged, "rise_wg_K": rise,
                              "rise_vs_constant": rise / base, "T_max": float(r.T.max())}), flush=True)
