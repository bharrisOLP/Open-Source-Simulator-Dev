"""The even (TE-like) gap of the photonic-crystal slab of the MPB example hole-slab against the resolution: a triangular
lattice of air holes (r 0.3 a) in a slab (eps 12, thickness 0.5 a), supercell height 4 a; band 1 at K and band 2 at
M of the z-even bands (tests/test_slab3d.py checks the triangular cell against the rectangular supercell at n 16).

    python bench/bench_hole_slab.py --n 16          # grid n x n x (4 n + 1) (an odd grid along z for the parity)
Prints one JSON line: the gap edges, the time of the band solves (the eps sampling included) and the device.
"""
import argparse
import json
import time

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np  # noqa: E402

from bands_gpu.findk import BandEvaluator  # noqa: E402
from bands_gpu.geometry import TRIANGULAR, Block, Cell, Cylinder  # noqa: E402
from bands_gpu.solver import TRIANGULAR_POINTS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=16)
ap.add_argument("--sub", type=int, default=8)
ap.add_argument("--tol", type=float, default=1e-8)
a = ap.parse_args()
cell = Cell(size=(1, 1, 4), shape=(a.n, a.n, 4 * a.n + 1), basis=TRIANGULAR,
            objects=(Block(size=(np.inf, np.inf, 0.5), eps=12.0), Cylinder(radius=0.3, height=0.5, eps=1.0)))
t0 = time.perf_counter()
ev = BandEvaluator(cell, 2, None, parity="even", mirror_axis=2, tol=a.tol, sub=a.sub)
setup = time.perf_counter() - t0
fk, _, ok_k = ev.solve(TRIANGULAR_POINTS["K"])
fm, _, ok_m = ev.solve(TRIANGULAR_POINTS["M"])
row = {"case": "hole-slab", "n": a.n, "shape": list(cell.shape), "sub": a.sub, "tol": a.tol,
       "band1_K": float(fk[0]), "band2_M": float(fm[1]), "converged": bool(ok_k and ok_m),
       "setup_s": round(setup, 3), "time_s": round(time.perf_counter() - t0, 3), "iterations": ev.iterations,
       "device": str(ev.device)}
print(json.dumps(row), flush=True)
