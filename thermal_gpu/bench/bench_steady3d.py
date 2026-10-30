"""Benchmark of the 3-D steady solver (solver3d): one JSON line per grid (cells, iterations, seconds, residual,
device).

    --case heater   the 3-D phase-shifter heater (structures.heater_3d: 100 um, 10 mW); --dx: the x-y grid steps
    --case cube     a heated block in a cube of n^3 cells (--n: one or more n)

THERMAL_GPU_DEVICE=cpu runs the same CG on the CPU. first_s: the first run (setup and compilation, one
iteration). The other times are from a second run with the same shapes: setup_s (host assembly, multigrid
hierarchy, transfer to the device), cg_s (the jitted CG loop, with the result on the host). --compare2d adds the
2-D cross-section rise (heater case) for the comparison at mid-length."""
import argparse
import json
import time

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from thermal_gpu.solver3d import pcg3d, pcg3d_mg, prepare3  # noqa: E402
from thermal_gpu.structures import heater_3d, heater_stack  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--case", choices=("heater", "cube"), default="heater")
ap.add_argument("--dx", type=float, nargs="+", default=[0.1e-6, 0.05e-6])
ap.add_argument("--n", type=int, nargs="+", default=[64, 128])
ap.add_argument("--tol", type=float, default=1e-12)
ap.add_argument("--precond", choices=("mg", "jacobi"), default="mg")
ap.add_argument("--cycle", choices=("V", "W"), default="V")
ap.add_argument("--max-iter", type=int, default=0, help="0: 2000 (mg) or 200000 (jacobi)")
ap.add_argument("--compare2d", action="store_true")
a = ap.parse_args()


def cube(n):
    e = np.linspace(0, 1e-6 * n / 32, n + 1)
    c = 0.5 * (e[1:] + e[:-1])
    X, Y, Z = np.meshgrid(c, c, c, indexing="ij")
    L = e[-1]
    kappa = np.where(Y < 0.3 * L, 148.0, 1.4)
    blk = (np.abs(X - L / 2) < 0.15 * L) & (np.abs(Y - 0.7 * L) < 0.05 * L) & (np.abs(Z - L / 2) < 0.2 * L)
    kappa = np.where(blk, 100.0, kappa)
    return (e, e, e, kappa, np.where(blk, 1e15, 0.0), {"ymin": ("dirichlet", 300.0), "ymax": ("robin", 10.0, 300.0)})


def run(args):
    def solve(max_iter):
        dp, levels, cinv, dev = prepare3(*args, precond=a.precond)
        x0 = jax.device_put(jnp.zeros(dp.b.shape), dev)
        tol, mi = jnp.asarray(a.tol), jnp.asarray(max_iter)
        t1 = time.perf_counter()
        if a.precond == "mg":
            T, it, res = pcg3d_mg(dp, levels, cinv, x0, tol, mi, a.cycle)
        else:
            T, it, res = pcg3d(dp, x0, tol, mi)
        T = np.asarray(T)
        return T, int(it), float(res), str(dev), len(levels or ()), t1

    t0 = time.perf_counter()
    solve(1)                                                     # compile (the loop bound is a traced value)
    first_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    T, it, res, dev, nlev, t1 = solve(a.max_iter or (2000 if a.precond == "mg" else 200000))
    t2 = time.perf_counter()
    return T, {"device": dev, "precond": a.precond, "cycle": a.cycle if a.precond == "mg" else None,
               "levels": nlev, "first_s": round(first_s, 3), "setup_s": round(t1 - t0, 3),
               "cg_s": round(t2 - t1, 3), "iterations": it, "residual": res, "converged": res <= a.tol}


if a.case == "cube":
    for n in a.n:
        args = cube(n)
        T, row = run(args)
        print(json.dumps({"case": "cube", "n": n, "cells": int(args[3].size), **row, "T_max": float(T.max())}),
              flush=True)
else:
    for dx in a.dx:
        x, y, z, kappa, q, bc, masks = heater_3d(dx=dx)
        T, row = run((x, y, z, kappa, q, bc))
        zc = 0.5 * (z[1:] + z[:-1])
        wg = masks["waveguide"][:, :, 0]
        prof = np.array([T[:, :, i][wg].mean() - 300.0 for i in range(zc.size)])
        out = {"case": "heater", "dx_um": round(dx * 1e6, 6), "cells": int(kappa.size), "shape": list(kappa.shape),
               **row,
               "rise_wg_mid": float(np.interp(0.0, zc, prof)), "rise_wg_end": float(np.interp(50e-6, zc, prof)),
               "T_max": float(T.max())}
        if a.compare2d:
            from thermal_gpu.solver import solve_steady
            x2, y2, k2, q2, bc2, m2 = heater_stack(dx=dx, power_per_length=100.0)
            out["rise_wg_2d"] = float(solve_steady(x2, y2, k2, q2, bc2).T[m2["waveguide"]].mean() - 300.0)
        print(json.dumps(out), flush=True)
