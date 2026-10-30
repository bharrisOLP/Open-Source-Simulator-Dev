"""Benchmark of solve_bands: 2-D rods (TM) and a 3-D simple cubic lattice of spheres (both polarizations).

    python bench/bench_bands.py --case rods --n 128 --precision single
    BANDS_GPU_DEVICE=cpu python bench/bench_bands.py ...      # the same algorithm on the CPU (XLA threads)
Prints one JSON line. The time covers the whole band path (compile included in "first", not in "time").
"""
import argparse
import json
import time

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from bands_gpu.geometry import Cell, Cylinder, Sphere  # noqa: E402
from bands_gpu.solver import interpolate, solve_bands  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--case", choices=["rods", "spheres"], default="rods")
ap.add_argument("--n", type=int, default=64)
ap.add_argument("--bands", type=int, default=8)
ap.add_argument("--precision", choices=["single", "double"], default="single")
ap.add_argument("--kinterp", type=int, default=4)
a = ap.parse_args()
dtype = jnp.complex64 if a.precision == "single" else jnp.complex128
if a.case == "rods":
    cell = Cell(size=(1, 1, 1), shape=(a.n, a.n, 1), objects=(Cylinder(radius=0.2, eps=12.0),))
    ks, pol = interpolate([[0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0], [0, 0, 0]], a.kinterp), "tm"
else:
    cell = Cell(size=(1, 1, 1), shape=(a.n, a.n, a.n), objects=(Sphere(radius=0.3, eps=12.0),))
    ks, pol = interpolate([[0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0], [0.5, 0.5, 0.5], [0, 0, 0]], a.kinterp), None
tol = 1e-4 if a.precision == "single" else 1e-6
t0 = time.perf_counter()
solve_bands(cell, ks[:1], a.bands, pol, dtype=dtype, tol=tol)
first = time.perf_counter() - t0
t0 = time.perf_counter()
res = solve_bands(cell, ks, a.bands, pol, dtype=dtype, tol=tol)
row = {"case": a.case, "n": a.n, "kpoints": len(ks), "bands": a.bands, "precision": a.precision,
       "device": res.device, "first_s": round(first, 3), "time_s": round(time.perf_counter() - t0, 3),
       "mean_iterations": float(np.mean(res.iterations)), "all_converged": bool(all(res.converged)),
       "band1_max": float(res.freqs[:, 0].max()), "band2_min": float(res.freqs[:, 1].min())}
print(json.dumps(row), flush=True)
