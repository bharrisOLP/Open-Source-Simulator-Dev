"""Benchmark of solve_steady on the heater stack: GPU, the same CG on the CPU (THERMAL_GPU_DEVICE=cpu), and
scipy's sparse direct solver. One JSON line per grid step. --precond jacobi gives the Jacobi CG timings."""
import argparse
import json
import time

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np  # noqa: E402
import scipy.sparse.linalg as spl  # noqa: E402

from thermal_gpu.fv import assemble, build  # noqa: E402
from thermal_gpu.solver import solve_steady  # noqa: E402
from thermal_gpu.structures import heater_stack  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dx", type=float, default=0.02e-6)
ap.add_argument("--direct", action="store_true")
ap.add_argument("--tol", type=float, default=1e-12)
ap.add_argument("--precond", choices=("mg", "jacobi"), default="mg")
a = ap.parse_args()
x, y, kappa, q, bc, masks = heater_stack(dx=a.dx)
solve_steady(x, y, kappa, q, bc, max_iter=10, precond=a.precond)  # compile
t0 = time.perf_counter()
res = solve_steady(x, y, kappa, q, bc, tol=a.tol, precond=a.precond)
row = {"dx_um": a.dx * 1e6, "cells": int(kappa.size), "device": res.device, "precond": a.precond,
       "cg_s": round(time.perf_counter() - t0, 3), "iterations": res.iterations, "residual": res.residual,
       "converged": res.converged,
       "T_wg_mean": float(res.T[masks["waveguide"]].mean()), "T_max": float(res.T.max())}
if a.direct:
    p = build(x, y, kappa, q, bc)
    t0 = time.perf_counter()
    ref = spl.spsolve(assemble(p).tocsc(), p.b.ravel()).reshape(p.shape)
    row["direct_s"] = round(time.perf_counter() - t0, 3)
    row["max_dT_vs_direct"] = float(np.abs(res.T - ref).max())
print(json.dumps(row), flush=True)
