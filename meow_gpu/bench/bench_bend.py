"""Bend modes: the GPU solver (``compute_modes_gpu`` with a bend radius) against tidy3d's local mode solver on the CPU
(``meow.compute_modes_tidy3d``), on the same grid and the same radial transform.

    python bench/bench_bend.py                                         # radii 2, 5, 10, 20, 50, 100 um, 20 nm grid
    python bench/bench_bend.py --radii 5 --dx 0.04 --repeat 1          # a small case
    python bench/bench_bend.py --radii 1.5 2 3 --pml 10 --x-min -1.0 --x-max 2.4 --target 2.45 --modes 1

Synthetic structure (textbook indices): a 0.5 x 0.22 um Si strip (n 3.476) in SiO2 (n 1.444), 1.55 um, the bend in
the x-z plane (bend_axis 1, the centre of the bend on the -x side for a positive radius). One JSON line per radius:
the tidy3d time, the GPU times (the first solve includes the JIT compile and the preconditioner setup; "gpu_s" is
the median of --repeat solves with a new solver each, so no warm start), the iterations, the neff of both solvers,
max |dneff| of the tidy3d modes that the GPU solver also gives (``matched``: their number; a GPU mode within
``match_tol`` of a tidy3d mode: 1e-5 in single, 1e-6 in mixed and double precision), |dneff| and
1 - |<E_gpu, E_tidy3d>| of the fundamental mode, Im(neff) of the fundamental mode of both solvers, and |dneff| of the
fundamental mode against the straight waveguide (tidy3d, the same grid). Both
sides skip meow's post-processing (the times are the mode solves). tidy3d can miss eigenvalues nearer the target
than its last mode (box modes at the outer wall of a bend, tests/test_bend.py); the GPU solver then gives other
modes, so ``matched`` can be less than --modes.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings

import jax

jax.config.update("jax_enable_x64", True)
import meow as mw  # noqa: E402
import numpy as np  # noqa: E402
from meow.fde.post_process import normalize_modes  # noqa: E402

from meow_gpu.meow_backend import GPUModeSolver, compute_modes_gpu, device_summary  # noqa: E402


MATCH_TOL = {"single": 1e-5, "mixed": 1e-6, "double": 1e-6}   # single: TE0 errors up to about 2e-6


def cross_section(radius, a):
    oxide = mw.IndexMaterial(name="sio2", n=1.444)
    si = mw.IndexMaterial(name="si", n=3.476)
    clad = mw.Structure(material=oxide, geometry=mw.Box(x_min=-9, x_max=9, y_min=-9, y_max=9, z_min=-1, z_max=2))
    core = mw.Structure(material=si, mesh_order=1, geometry=mw.Box(x_min=-a.width / 2, x_max=a.width / 2, y_min=0,
                                                                    y_max=0.22, z_min=-1, z_max=2))
    bend = {} if radius is None else {"bend_radius": radius, "bend_axis": 1}
    mesh = mw.Mesh2D(x=np.arange(a.x_min, a.x_max + a.dx / 2, a.dx), y=np.arange(-0.8, 1.02 + a.dx / 2, a.dx),
                     num_pml=(a.pml, 0), **bend)
    cell = mw.Cell(structures=[clad, core], mesh=mesh, z_min=0.0, z_max=1.0)
    return mw.CrossSection.from_cell(cell=cell, env=mw.Environment(wl=1.55, T=25.0))


def _raw(modes):
    return modes


def _gpu(cs, a):
    s = GPUModeSolver(precision=a.precision, extra=a.extra, max_iter=a.max_iter)
    t0 = time.perf_counter()
    modes = compute_modes_gpu(cs, num_modes=a.modes, target_neff=a.target, solver=s, post_process=_raw)
    return modes, time.perf_counter() - t0, s.last_info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radii", type=float, nargs="+", default=[2.0, 5.0, 10.0, 20.0, 50.0, 100.0])
    ap.add_argument("--dx", type=float, default=0.02)
    ap.add_argument("--width", type=float, default=0.5)
    ap.add_argument("--x-min", type=float, default=-1.5)
    ap.add_argument("--x-max", type=float, default=1.5)
    ap.add_argument("--pml", type=int, default=0, help="PML cells on both x sides")
    ap.add_argument("--modes", type=int, default=4)
    ap.add_argument("--target", type=float, default=None, help="target neff (default: the largest index)")
    ap.add_argument("--precision", default="mixed", choices=["single", "mixed", "double"])
    ap.add_argument("--extra", type=int, default=4)
    ap.add_argument("--max-iter", type=int, default=500)
    ap.add_argument("--repeat", type=int, default=3)
    a = ap.parse_args()
    warnings.simplefilter("ignore")
    straight = mw.compute_modes_tidy3d(cross_section(None, a), num_modes=1, target_neff=a.target, post_process=_raw)
    for radius in a.radii:
        cs = cross_section(radius, a)
        row = {"radius": radius, "dx": a.dx, "grid": list(np.shape(cs.nx)), "unknowns": 2 * int(np.size(cs.nx)),
               "modes": a.modes, "pml": a.pml, "target": a.target, "precision": a.precision,
               "device": device_summary()}
        t0 = time.perf_counter()
        ref = mw.compute_modes_tidy3d(cs, num_modes=a.modes, target_neff=a.target, post_process=_raw)
        row["tidy3d_s"] = time.perf_counter() - t0
        got, row["gpu_first_s"], _ = _gpu(cs, a)
        times = []
        for _ in range(a.repeat):
            got, t, info = _gpu(cs, a)
            times.append(t)
        row.update(gpu_s=float(np.median(times)), iterations=info.iterations, converged=info.converged,
                   residual=info.residual)
        n_ref = np.array([complex(m.neff) for m in ref])
        n_gpu = np.array([complex(m.neff) for m in got])
        row["neff_tidy3d"] = [[v.real, v.imag] for v in n_ref]
        row["neff_gpu"] = [[v.real, v.imag] for v in n_gpu]
        d = np.array([np.min(np.abs(n_gpu - v)) for v in n_ref])
        row["match_tol"] = tol = MATCH_TOL[a.precision]
        row["matched"] = int(np.sum(d < tol))
        row["max_dneff_matched"] = float(np.max(d[d < tol])) if row["matched"] else None
        g0, r0 = normalize_modes([got[0], ref[0]], mw.inner_product)
        row["dneff_fundamental"] = float(abs(n_gpu[0] - n_ref[0]))
        row["field_1_minus_overlap"] = float(abs(1 - abs(mw.inner_product(g0, r0))))
        row["im_neff_fundamental"] = [float(n_gpu[0].imag), float(n_ref[0].imag)]
        row["dneff_fundamental_vs_straight"] = float(abs(n_gpu[0] - complex(straight[0].neff)))
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
