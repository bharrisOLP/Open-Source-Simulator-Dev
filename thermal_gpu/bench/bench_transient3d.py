"""Benchmark of the transient solver (transient3d): the thermal rise time of the 3-D heater of structures.heater_3d
(100 um, 10 mW) or of its 2-D cross-section (100 W/m). One JSON line per grid step (and per time-step case of
--dt-scan: n runs with --per-stage x 2^k and --dt0 / 2^k, k = 0 .. n - 1, a check of the time-step error).

The heater switches on at t = 0 (the structure at 300 K). The monitor is the waveguide at mid-length (the z cells
nearest to z = 0; the 2-D case: the whole waveguide). The rise time is the time from 10 % to 90 % of the steady rise
(solve_steady3d or solve_steady). Time steps: geometric stages (--dt0, --per-stage steps per stage, the step doubles
from stage to stage) up to --t-end. --compare2d (3-D case) adds the 2-D cross-section with the same steps.
THERMAL_GPU_DEVICE=cpu runs the same code on the CPU. run_s: the transient (with compilation; --repeat adds a second
run with the same shapes, run2_s)."""
import argparse
import itertools
import json
import time

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np  # noqa: E402

from thermal_gpu.solver import solve_steady  # noqa: E402
from thermal_gpu.solver3d import solve_steady3d  # noqa: E402
from thermal_gpu.structures import heat_capacity, heater_3d, heater_stack  # noqa: E402
from thermal_gpu.transient3d import geometric_stages, rise_time, solve_transient2d, solve_transient3d  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--case", choices=("heater3d", "cross2d"), default="heater3d")
ap.add_argument("--dx", type=float, nargs="+", default=[0.1e-6])
ap.add_argument("--width", type=float, default=10e-6)
ap.add_argument("--dz-max", type=float, default=2e-6)
ap.add_argument("--scheme", choices=("euler", "bdf2", "cn"), default="bdf2")
ap.add_argument("--cycle", choices=("V", "W"), default="V")
ap.add_argument("--dt0", type=float, default=1e-8)
ap.add_argument("--per-stage", type=int, default=20)
ap.add_argument("--t-end", type=float, default=200e-6)
ap.add_argument("--tol", type=float, default=1e-10)
ap.add_argument("--compare2d", action="store_true")
ap.add_argument("--repeat", action="store_true")
ap.add_argument("--dt-scan", type=int, default=1,
                help="time-step check: n runs with per-stage x 2^k and dt0 / 2^k, k = 0 .. n - 1 (one row each)")
a = ap.parse_args()
scan = [(a.dt0 / 2**k, a.per_stage * 2**k) for k in range(a.dt_scan)]
common = dict(scheme=a.scheme, tol=a.tol, cycle=a.cycle)


def timed(fn):
    t0 = time.perf_counter()
    r = fn()
    return r, round(time.perf_counter() - t0, 3)


def cross2d(dx):
    x, y, k, q, bc, m = heater_stack(dx=dx, width=a.width, power_per_length=100.0)
    final = solve_steady(x, y, k, q, bc).T[m["waveguide"]].mean()
    def run():
        return solve_transient2d(x, y, k, q, bc, heat_capacity(k), stages, monitors=[m["waveguide"]], **common)
    return k, final, run


for (dt0, per), dx in itertools.product(scan, a.dx):
    stages = geometric_stages(dt0, a.t_end, per_stage=per)
    if a.case == "heater3d":
        x, y, z, k, q, bc, m = heater_3d(dx=dx, width=a.width, dz_max=a.dz_max, length=100e-6, power=10e-3)
        zc = 0.5 * (z[1:] + z[:-1])
        mid = (np.abs(zc) <= np.abs(zc).min() * (1 + 1e-9) + 1e-15)[None, None, :]
        steady, steady_s = timed(lambda: solve_steady3d(x, y, z, k, q, bc, cycle=a.cycle))
        final = steady.T[m["waveguide"] & mid].mean()
        run = lambda: solve_transient3d(x, y, z, k, q, bc, heat_capacity(k), stages,  # noqa: E731
                                        monitors=[m["waveguide"] & mid], **common)
        shape = list(k.shape)
    else:
        k, final, run = cross2d(dx)
        steady_s, shape = None, list(k.shape)
    r, run_s = timed(run)
    t10, t90, rt = rise_time(r.times, r.monitors[:, 0], final)
    row = {"case": a.case, "dx_um": round(dx * 1e6, 6), "width_um": round(a.width * 1e6, 6), "cells": int(k.size),
           "shape": shape, "device": r.device, "scheme": a.scheme, "cycle": a.cycle, "steps": len(r.times) - 1,
           "stages": len(stages), "dt0": dt0, "per_stage": per, "t_end": a.t_end, "steady_s": steady_s,
           "run_s": run_s, "iterations": r.iterations, "max_step_iterations": r.max_step_iterations,
           "max_residual": r.max_residual, "converged": r.converged, "steady_rise_K": float(final - 300.0),
           "end_rise_K": float(r.monitors[-1, 0] - 300.0), "t10_us": t10 * 1e6, "t90_us": t90 * 1e6,
           "rise_10_90_us": rt * 1e6}
    if a.repeat:
        row["run2_s"] = timed(run)[1]
    if a.compare2d and a.case == "heater3d":
        k2, f2, run2 = cross2d(dx)
        r2 = run2()
        t10b, t90b, rtb = rise_time(r2.times, r2.monitors[:, 0], f2)
        row.update(rise_10_90_us_2d=rtb * 1e6, t10_us_2d=t10b * 1e6, t90_us_2d=t90b * 1e6,
                   steady_rise_K_2d=float(f2 - 300.0), converged_2d=r2.converged)
    print(json.dumps(row), flush=True)
