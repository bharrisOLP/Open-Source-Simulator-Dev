"""Benchmark of the transient with kappa(T) (transient_kappa) against the constant-kappa transient (transient3d): the
thermal rise of the 3-D heater of structures.heater_3d (100 um, 10 mW) or of its 2-D cross-section (100 W/m). One
JSON line per grid step and model.

Models (--models):
    constant        transient3d.solve_transient3d, constant kappa and rho c_p (the reference of the timing)
    constant_new    transient_kappa with constant laws (one Picard iteration per step: the cost of the new path)
    si_T            Si kappa(T) (power law), constant SiO2 and metal
    si_T+sio2_T     Si kappa(T) and SiO2 kappa(T) (linear law)
    all_T+cp_T      Si and SiO2 kappa(T) and Si rho c_p(T) (linear law, valid to 400 K)

The heater switches on at t = 0 (the structure at 300 K); the monitor is the waveguide at mid-length (3-D) or the
whole waveguide (2-D). steady_rise_K: the steady rise of the same model (solver.solve_steady / solver3d.solve_steady3d
for the constant models, nonlinear.solve_steady_kappa2d / 3d for the others); the rise time is from 10 % to 90 % of
it. Time steps: geometric stages (--dt0, --per-stage, the step doubles per stage) to --t-end, scheme --scheme.
THERMAL_GPU_DEVICE=cpu runs the same code on the CPU. run_s: the transient with compilation; --repeat adds a second
run with the same shapes (run2_s)."""
import argparse
import json
import time

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np  # noqa: E402

from thermal_gpu.nonlinear import (SI_POWER_LAW, SIO2_CONSTANT, SIO2_LINEAR, Constant, material_kappa,  # noqa: E402
                                   solve_steady_kappa2d, solve_steady_kappa3d)
from thermal_gpu.solver import solve_steady  # noqa: E402
from thermal_gpu.solver3d import solve_steady3d  # noqa: E402
from thermal_gpu.structures import (K_METAL, K_SI, RHO_CP_METAL, RHO_CP_SIO2, heat_capacity, heater_3d,  # noqa: E402
                                    heater_stack, material_labels)
from thermal_gpu.transient3d import geometric_stages, rise_time, solve_transient2d, solve_transient3d  # noqa: E402
from thermal_gpu.transient_kappa import (SI_RHO_CP_LINEAR, solve_transient_kappa2d,  # noqa: E402
                                         solve_transient_kappa3d)

MODELS = {"constant": None,
          "constant_new": ([Constant(K_SI), SIO2_CONSTANT, Constant(K_METAL)], None),
          "si_T": ([SI_POWER_LAW, SIO2_CONSTANT, Constant(K_METAL)], None),
          "si_T+sio2_T": ([SI_POWER_LAW, SIO2_LINEAR, Constant(K_METAL)], None),
          "all_T+cp_T": ([SI_POWER_LAW, SIO2_LINEAR, Constant(K_METAL)],
                         [SI_RHO_CP_LINEAR, Constant(RHO_CP_SIO2), Constant(RHO_CP_METAL)])}

ap = argparse.ArgumentParser()
ap.add_argument("--case", choices=("heater3d", "cross2d"), default="cross2d")
ap.add_argument("--dx", type=float, nargs="+", default=[0.1e-6])
ap.add_argument("--models", nargs="+", choices=tuple(MODELS), default=list(MODELS))
ap.add_argument("--width", type=float, default=10e-6)
ap.add_argument("--dz-max", type=float, default=2e-6)
ap.add_argument("--scheme", choices=("euler", "bdf2", "cn"), default="bdf2")
ap.add_argument("--cycle", choices=("V", "W"), default="V")
ap.add_argument("--dt0", type=float, default=1e-8)
ap.add_argument("--per-stage", type=int, default=20)
ap.add_argument("--t-end", type=float, default=200e-6)
ap.add_argument("--tol", type=float, default=1e-10)
ap.add_argument("--tol-picard", type=float, default=1e-10)
ap.add_argument("--repeat", action="store_true")
a = ap.parse_args()
stages = geometric_stages(a.dt0, a.t_end, per_stage=a.per_stage)


def timed(fn):
    t0 = time.perf_counter()
    r = fn()
    return r, round(time.perf_counter() - t0, 3)


for dx in a.dx:
    if a.case == "heater3d":
        x, y, z, k, q, bc, m = heater_3d(dx=dx, width=a.width, dz_max=a.dz_max, length=100e-6, power=10e-3)
        zc = 0.5 * (z[1:] + z[:-1])
        mon = m["waveguide"] & (np.abs(zc) <= np.abs(zc).min() * (1 + 1e-9) + 1e-15)[None, None, :]
        grid = (x, y, z)
        steady_lin = lambda: solve_steady3d(x, y, z, k, q, bc, cycle=a.cycle)                          # noqa: E731
        steady_nl = lambda kf: solve_steady_kappa3d(x, y, z, kf, q, bc, cycle=a.cycle)               # noqa: E731
        lin = solve_transient3d
        new = solve_transient_kappa3d
    else:
        x, y, k, q, bc, m = heater_stack(dx=dx, width=a.width, power_per_length=100.0)
        mon = m["waveguide"]
        grid = (x, y)
        steady_lin = lambda: solve_steady(x, y, k, q, bc)                                               # noqa: E731
        steady_nl = lambda kf: solve_steady_kappa2d(x, y, kf, q, bc, cycle=a.cycle)                  # noqa: E731
        lin = solve_transient2d
        new = solve_transient_kappa2d
    labels = material_labels(k)
    common = dict(scheme=a.scheme, tol=a.tol, cycle=a.cycle, monitors=[mon])
    base_rt = None
    for name in a.models:
        if MODELS[name] is None:
            steady, steady_s = timed(steady_lin)
            run = lambda: lin(*grid, k, q, bc, heat_capacity(k), stages, **common)                      # noqa: E731
        else:
            kl, cl = MODELS[name]
            kf = material_kappa(labels, kl)
            cf = heat_capacity(k) if cl is None else material_kappa(labels, cl)
            steady, steady_s = timed(lambda: steady_nl(kf))
            run = lambda: new(*grid, kf, q, bc, cf, stages, tol_picard=a.tol_picard, **common)          # noqa: E731
        final = float(steady.T[mon].mean())
        r, run_s = timed(run)
        t10, t90, rt = rise_time(r.times, r.monitors[:, 0], final)
        base_rt = rt if name == "constant" else base_rt
        row = {"case": a.case, "dx_um": round(dx * 1e6, 6), "cells": int(k.size), "shape": list(k.shape),
               "device": r.device, "model": name, "scheme": a.scheme, "cycle": a.cycle, "steps": len(r.times) - 1,
               "stages": len(stages), "dt0": a.dt0, "per_stage": a.per_stage, "t_end": a.t_end,
               "steady_s": steady_s, "run_s": run_s, "iterations": r.iterations,
               "max_step_iterations": r.max_step_iterations, "picard": getattr(r, "picard_iterations", None),
               "max_step_picard": getattr(r, "max_step_picard", None), "max_residual": r.max_residual,
               "converged": r.converged, "out_of_range": getattr(r, "out_of_range", None),
               "steady_rise_K": final - 300.0, "end_rise_K": float(r.monitors[-1, 0] - 300.0),
               "t10_us": t10 * 1e6, "t90_us": t90 * 1e6, "rise_10_90_us": rt * 1e6,
               "rise_vs_constant": None if base_rt is None else rt / base_rt}
        if a.repeat:
            row["run2_s"] = timed(run)[1]
        print(json.dumps(row), flush=True)
