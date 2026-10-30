"""The complex128 polish of the "mixed" precision: the earlier Davidson polish against the mixed-precision refinement.

    python bench/bench_refine.py                                  # the 30-cell taper of bench_eme_taper.py
    python bench/bench_refine.py --cells 6 --dx 0.04 --repeat 1   # a small case

Synthetic structure (textbook indices, from ``bench_eme_taper.build``): Si strip 0.5 -> 2.0 um over 10 um, SiO2
cladding, 1.55 um. The cells are solved one after the other (``compute_modes_gpu``, warm start from the previous cell)
in four cases: "double" (the reference), "mixed" with ``polish="davidson"`` (the earlier method: complex64 Davidson,
then a complex128 Davidson with a subspace of 3 blocks), "mixed" with ``polish="refine"`` (complex64 Davidson,
then complex128 residuals and Rayleigh-Ritz with complex64 GMRES correction solves) and "single" (for a
comparison of the time; its tolerance follows the complex64 floor). Each case prints one JSON line:
the mode-solve time (with meow's post-processing, as in bench_eme_taper.py), the complex128 operator applies (vectors,
all cells), all operator applies, the iterations (for "refine": complex64 Davidson iterations + outer refinement
steps), the largest final residual, |S21| of the fundamental mode, and max |dneff| of the kept modes against the
first case of --cases (default: double). Each run uses a new solver (no warm-start memory). With --repeat 2
(default), "modes_s" is the time of the last run (compiled kernels) and "modes_s_first" includes the JIT compile.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_eme_taper import _post_process, build  # noqa: E402

CASES = {"double": ("double", "refine"), "mixed-davidson": ("mixed", "davidson"), "mixed-refine": ("mixed", "refine"),
         "single": ("single", "refine")}


def _solve(css, a, precision: str, polish: str, pp):
    """The modes of all cells, the time, and the SolveInfo of each cell."""
    from meow_gpu.meow_backend import GPUModeSolver, compute_modes_gpu

    solver = GPUModeSolver(precision=precision, polish=polish, tol=a.tol, extra=a.extra, switch_tol=a.switch_tol,
                           refine_inner=a.inner)
    modes, infos = [], []
    t0 = time.perf_counter()
    for cs in css:
        modes.append(compute_modes_gpu(cs, num_modes=a.modes, solver=solver, post_process=pp))
        infos.append(solver.last_info)
    return modes, time.perf_counter() - t0, infos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=int, default=30)
    ap.add_argument("--modes", type=int, default=10)
    ap.add_argument("--dx", type=float, default=0.02)
    ap.add_argument("--tol", type=float, default=None)
    ap.add_argument("--extra", type=int, default=8)
    ap.add_argument("--switch-tol", type=float, default=None, help="complex64 stop of mixed (default: by polish)")
    ap.add_argument("--inner", type=int, default=None, help="GMRES steps of the refinement (default: REFINE_INNER)")
    ap.add_argument("--cases", default="double,mixed-davidson,mixed-refine,single",
                    help="comma-separated; the first case is the reference of max_dneff")
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--post-process", choices=["meow_gpu", "meow"], default="meow_gpu")
    a = ap.parse_args()
    pp = _post_process(a.post_process)
    from meow_gpu.fde.eigensolver import REFINE_INNER
    a.inner = a.inner or REFINE_INNER

    import jax
    jax.config.update("jax_enable_x64", True)
    import meow as mw

    from meow_gpu.meow_backend import device_summary

    cells, css = build(a.cells, a.dx)
    ref = None
    for case in [c.strip() for c in a.cases.split(",") if c.strip()]:
        precision, polish = CASES[case]
        times = []
        for _ in range(max(1, a.repeat)):
            modes, seconds, infos = _solve(css, a, precision, polish, pp)
            times.append(seconds)
        its = [i.iterations for i in infos]
        row = {"case": case, "precision": precision, "polish": polish if precision == "mixed" else None,
               "tol": a.tol, "switch_tol": a.switch_tol if precision == "mixed" else None,
               "inner": a.inner if precision == "mixed" and polish == "refine" else None,
               "extra": a.extra, "cells": a.cells, "modes": a.modes, "grid": list(css[0].nx.shape),
               "solver_device": device_summary(), "modes_s": times[-1], "modes_s_first": times[0],
               "matvecs_c128": int(sum(i.matvecs_hi for i in infos)),
               "matvecs_all": int(sum(i.matvecs for i in infos)),
               "iterations": its, "iterations_total": int(sum(its)),
               "max_residual": float(max(i.residual for i in infos)),
               "all_converged": bool(all(i.converged for i in infos)),
               # cells of each polish ("refine+davidson": the fallback ran; "": single, double)
               "polish_used": {u: sum(i.polish_used == u for i in infos)
                               for u in sorted({i.polish_used for i in infos})}}
        t0 = time.perf_counter()
        with jax.default_device(jax.devices("cpu")[0]):
            S, port_map = mw.compute_s_matrix(modes, cells=cells)
        row["smatrix_s"] = time.perf_counter() - t0
        S = np.asarray(S)
        row["S21_00"] = float(abs(S[port_map["left@0"], port_map["right@0"]]))
        row["n_modes_kept"] = [len(m) for m in modes]
        if ref is None:
            ref = modes
            row["max_dneff"] = 0.0
        else:
            d = [max((abs(complex(x.neff) - complex(y.neff)) for x, y in zip(ms, rs)), default=0.0)
                 for ms, rs in zip(modes, ref)]
            row["max_dneff"] = float(max(d))
            row["cells_with_other_mode_count"] = int(sum(len(ms) != len(rs) for ms, rs in zip(modes, ref)))
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
