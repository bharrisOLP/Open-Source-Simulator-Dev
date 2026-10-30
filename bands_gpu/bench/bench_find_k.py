"""Benchmark of find_k: guided modes at a fixed frequency.

    python bench/bench_find_k.py --case slab --n 1025                  # 2-D slab, 4 modes, against the exact values
    python bench/bench_find_k.py --case pcw --n 31 --precision single  # line-defect photonic-crystal waveguide, TM
    python bench/bench_find_k.py --case strip --n 32                   # 3-D strip waveguide cross-section
    BANDS_GPU_DEVICE=cpu python bench/bench_find_k.py ...              # the same algorithm on the CPU (XLA threads)

Prints one JSON line per case. "first_s" is a first find_k with compile; "time_s" a second, the same call. Cases:

- slab: a symmetric slab (n 3.5 / 1.45, thickness 0.5 a) in a supercell of period 5 a, omega 0.5; the even and odd
  modes of E_z ("tm") and H_z ("te"), with the parity; --n points along the supercell (odd). Errors of kmag against
  the exact supercell (discretization) and the isolated slab (tests/slab_reference.py); also the position of the
  face in its voxel, the error of the slab thickness that the subpixel samples see (--sub), and the error against
  the exact supercell with that sampled thickness.
- pcw: a square lattice of rods (eps 12, r = 0.2 a) with one row removed, a 1 x 7 supercell, TM, omega 0.34, the
  guided band (band 4 of the even parity); --n points per a (n * 7 must be odd).
- strip: a strip of n 3.48 (0.5 x 0.22) in n 1.444 (a = 1 um, omega = 1 / 1.55, the wavelength 1.55 um), a cell of
  3 x 2, 3-D vectors; the fundamental TE-like mode (y parity odd); --n points per a (3 n + 1 along y, 2 n along z).
  Indices: silicon and fused silica near 1.55 um (textbook values). No reference value: a timing case.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from bands_gpu.findk import find_k  # noqa: E402
from bands_gpu.geometry import Block, Cell, Cylinder  # noqa: E402
from slab_reference import slab_isolated, supercell_exact  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--case", choices=["slab", "pcw", "strip"], default="slab")
ap.add_argument("--n", type=int, default=None)
ap.add_argument("--precision", choices=["single", "double"], default="double")
ap.add_argument("--sub", type=int, default=8)
ap.add_argument("--cold", action="store_true", help="no warm start between the band solves")
a = ap.parse_args()
dtype = jnp.complex64 if a.precision == "single" else jnp.complex128
tol = 1e-5 if a.precision == "single" else 1e-8


def timed(run):
    t0 = time.perf_counter()
    run()
    first = time.perf_counter() - t0
    t0 = time.perf_counter()
    res = run()
    return res, first, time.perf_counter() - t0


row = {"case": a.case, "precision": a.precision, "warm_start": not a.cold}
if a.case == "slab":
    n1, n2, d, period, omega = 3.5, 1.45, 0.5, 5.0, 0.5
    n = a.n or 1025
    cell = Cell(size=(1, period, 1), shape=(1, n, 1), eps_background=n2**2,
                objects=(Block(size=(np.inf, d, np.inf), eps=n1**2),))
    modes = [("E", True, "tm", "even"), ("E", False, "tm", "odd"), ("H", True, "te", "odd"), ("H", False, "te", "even")]

    def run():
        return [find_k(cell, omega, 1, 1, kdir=(1, 0, 0), polarization=sp, parity=par, tol=tol, dtype=dtype,
                       sub=a.sub, warm_start=not a.cold) for _, _, sp, par in modes]

    res, first, total = timed(run)
    # the slab thickness that the subpixel samples see: (samples inside) x (sample length) - d. The fill fraction of
    # a face voxel is a multiple of 1/sub, so a face at a general position moves by up to dx / (2 sub).
    d_sampled = float(np.count_nonzero(cell.eps_samples(a.sub) == n1**2)) * period / (n * a.sub)
    kmag, disc, iso, vg_err, disc_sampled = [], [], [], [], []
    for (pol, even, _, _), r in zip(modes, res):
        k_iso = slab_isolated(pol, even, n1, n2, d, omega)
        k_sc, vg_sc = supercell_exact(pol, k_iso, n1, n2, d, omega, period)
        k_sc_s, _ = supercell_exact(pol, k_sc, n1, n2, d_sampled, omega, period)
        kmag.append(float(r.kmag[0]))
        disc.append(float((r.kmag[0] - k_sc) / k_sc))
        iso.append(float((r.kmag[0] - k_iso) / k_iso))
        vg_err.append(float((r.group_velocity[0] - vg_sc) / vg_sc))
        disc_sampled.append(float((r.kmag[0] - k_sc_s) / k_sc_s))
    face = ((d / 2 + period / 2) * n / period + 0.5) % 1.0      # position of the face in its voxel (0 to 1)
    row.update({"n": n, "sub": a.sub, "modes": ["TE0", "TE1", "TM0", "TM1"], "kmag": kmag,
                "rel_err_supercell": disc, "rel_err_isolated": iso, "rel_err_vg": vg_err,
                "face_in_voxel": round(face, 4), "sampled_thickness_error": d_sampled - d,
                "rel_err_supercell_sampled_thickness": disc_sampled})
elif a.case == "pcw":
    n = a.n or 15
    rods = tuple(Cylinder(radius=0.2, eps=12.0, center=(0.0, float(j), 0.0)) for j in (-3, -2, -1, 1, 2, 3))
    cell = Cell(size=(1, 7, 1), shape=(n, 7 * n, 1), objects=rods)

    def run():
        return [find_k(cell, 0.34, 4, 4, kdir=(1, 0, 0), polarization="tm", parity="even", kmag_max=0.5,
                       kmag_guess=0.2, tol=tol, dtype=dtype, sub=a.sub, warm_start=not a.cold)]

    res, first, total = timed(run)
    row.update({"n": n, "kmag": [float(res[0].kmag[0])], "group_velocity": [float(res[0].group_velocity[0])]})
else:
    n = a.n or 16
    n_si, n_ox, omega = 3.48, 1.444, 1 / 1.55
    cell = Cell(size=(1, 3, 2), shape=(1, 3 * n + 1, 2 * n), eps_background=n_ox**2,
                objects=(Block(size=(np.inf, 0.5, 0.22), eps=n_si**2),))

    def run():
        return [find_k(cell, omega, 1, 1, kdir=(1, 0, 0), parity="odd", mirror_axis=1, tol=tol, dtype=dtype,
                       sub=a.sub, warm_start=not a.cold)]

    res, first, total = timed(run)
    row.update({"n": n, "kmag": [float(res[0].kmag[0])], "n_eff": [float(res[0].kmag[0] / omega)],
                "group_index": [float(1 / res[0].group_velocity[0])]})
row.update({"device": res[0].device, "first_s": round(first, 3), "time_s": round(total, 3),
            "solves": [r.solves for r in res], "iterations": [r.iterations for r in res],
            "converged": bool(all(all(r.converged) and r.unconverged_solves == 0 for r in res))})
print(json.dumps(row), flush=True)
