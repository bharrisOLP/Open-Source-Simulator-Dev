"""Benchmark of the eps sampling and averaging: the JAX path (bands_gpu/sampling.py) on the solver device against the
NumPy reference (Cell.voxel_averages, on the CPU).

    python bench/bench_sampling.py --case rods --n 256
    python bench/bench_sampling.py --case diamond --n 64 --no-numpy      # the JAX path only
    python bench/bench_sampling.py --case rods-supercell --n 80 --m 10   # m x m rods (one run of 100 objects)
Cases: "rods" (2-D square lattice, rod r 0.2 a), "tri-holes" (2-D triangular lattice, air holes r 0.3 a), "spheres"
(3-D simple cubic, r 0.3 a), "diamond" (3-D fcc primitive cell, two spheres r 0.25 a: the diamond lattice of the
3-D check), "rods-supercell" (an m x m supercell of the rod lattice, n x n points: m^2 objects of one kind),
"mixed" (2-D, m blocks and m cylinders that alternate: 2 m runs of one object). Prints one JSON line: the time of
the NumPy path, of the first JAX call (with the compile) and the median of ``--repeat`` further JAX calls (seconds,
1e-6 s steps), the speed-up (from the unrounded times), and the largest differences of the JAX path against NumPy.
"""
import argparse
import json
import statistics
import time

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np  # noqa: E402

from bands_gpu import sampling  # noqa: E402
from bands_gpu.geometry import TRIANGULAR, Block, Cell, Cylinder, Sphere  # noqa: E402
from bands_gpu.solver import solver_device  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--case", choices=["rods", "tri-holes", "spheres", "diamond", "rods-supercell", "mixed"],
                default="rods")
ap.add_argument("--n", type=int, default=128)
ap.add_argument("--m", type=int, default=10, help="rods per side (rods-supercell), pairs of objects (mixed)")
ap.add_argument("--sub", type=int, default=8)
ap.add_argument("--repeat", type=int, default=5)
ap.add_argument("--no-numpy", action="store_true", help="skip the NumPy reference (long for large 3-D cells)")
a = ap.parse_args()
n = a.n
if a.case == "rods":
    cell = Cell(size=(1, 1, 1), shape=(n, n, 1), objects=(Cylinder(radius=0.2, eps=12.0),))
elif a.case == "tri-holes":
    cell = Cell(size=(1, 1, 1), shape=(n, n, 1), basis=TRIANGULAR, eps_background=12.0,
                objects=(Cylinder(radius=0.3, eps=1.0),))
elif a.case == "spheres":
    cell = Cell(size=(1, 1, 1), shape=(n, n, n), objects=(Sphere(radius=0.3, eps=12.0),))
elif a.case == "diamond":
    cell = Cell(size=(np.sqrt(0.5),) * 3, shape=(n, n, n), basis=((0, 1, 1), (1, 0, 1), (1, 1, 0)),
                objects=(Sphere(radius=0.25, eps=11.56, center=(0.125,) * 3),
                         Sphere(radius=0.25, eps=11.56, center=(-0.125,) * 3)))
elif a.case == "rods-supercell":
    c = np.arange(a.m) - (a.m - 1) / 2
    cell = Cell(size=(a.m, a.m, 1), shape=(n, n, 1),
                objects=tuple(Cylinder(radius=0.2, eps=12.0, center=(x, y, 0)) for x in c for y in c))
else:
    x = (np.arange(a.m) + 0.5) / a.m - 0.5
    cell = Cell(size=(1, 1, 1), shape=(n, n, 1), objects=tuple(
        o for xi in x for o in (Block(size=(0.3 / a.m, 0.6, np.inf), eps=4.0, center=(xi, 0.1, 0)),
                                Cylinder(radius=0.1 / a.m, eps=9.0, center=(xi, -0.3, 0)))))
device = solver_device()
row = {"case": a.case, "n": n, "shape": list(cell.shape), "sub": a.sub, "objects": len(cell.objects),
       "runs": sampling.runs(cell), "samples": int(np.prod([k * (a.sub if k > 1 else 1) for k in cell.shape])),
       "images": [len(cell.image_shifts(o)) for o in cell.objects][:4], "device": str(device)}
t0 = time.perf_counter()
got = sampling.voxel_averages(cell, a.sub, device=device)
first = time.perf_counter() - t0
times = []
for _ in range(a.repeat):
    t0 = time.perf_counter()
    got = sampling.voxel_averages(cell, a.sub, device=device)
    times.append(time.perf_counter() - t0)
jax_s = statistics.median(times)
row.update({"jax_first_s": round(first, 6), "jax_s": round(jax_s, 6), "repeat": a.repeat})
if not a.no_numpy:
    t0 = time.perf_counter()
    ref = cell.voxel_averages(a.sub)
    numpy_s = time.perf_counter() - t0
    row["numpy_s"] = round(numpy_s, 6)
    row["speedup"] = round(numpy_s / jax_s, 1)
    row["auto_uses_jax"] = sampling.use_jax(cell, a.sub, "auto")
    row["count_equal"] = bool(np.array_equal(got["count"], ref["count"]))
    row["analytic_equal"] = bool(np.array_equal(got["analytic"], ref["analytic"]))
    for key in ("eps_inv", "mean_eps", "normal"):
        row[f"max_diff_{key}"] = float(np.abs(got[key] - ref[key]).max())
print(json.dumps(row), flush=True)
