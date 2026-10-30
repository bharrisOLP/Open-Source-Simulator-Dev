"""EME benchmark of the batched mode solves: the taper of bench_eme_taper.py, per-cell against batched.

    python bench/bench_eme_batch.py --precision single --batches 4,8,16

Synthetic structure (textbook indices, from ``bench_eme_taper.build``). The first case solves the cells one after the
other (``compute_modes_gpu``, warm start from the previous cell); each next case solves them with
``compute_modes_gpu_batch(batch=B)``. Each case prints one JSON line: the mode-solve time (with meow's
post-processing, as in bench_eme_taper.py), |S21| of the fundamental mode, max |dneff| of the kept modes against the
per-cell run, and the iterations. "batch" is the requested batch size, "batch_used" the size that
``compute_modes_gpu_batch`` used (30 cells with batch 16: 2 batches of 15). Each run uses a new solver (no
warm-start memory). With --repeat 2 (default), "modes_s" is the time of the last run (compiled kernels) and
"modes_s_first" includes the JIT compile.
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


def _solve(css, a, batch: int, pp):
    """The modes of all cells, the time, and the iterations (per cell; for a batch: of each member, per batch)."""
    from meow_gpu.meow_backend import GPUModeSolver, compute_modes_gpu, compute_modes_gpu_batch

    solver = GPUModeSolver(precision=a.precision, tol=a.tol, extra=a.extra)
    its: list = []
    t0 = time.perf_counter()
    if batch == 0:
        modes = []
        for cs in css:
            modes.append(compute_modes_gpu(cs, num_modes=a.modes, solver=solver, post_process=pp))
            its.append(solver.last_info.iterations)
    else:
        solve_batch = solver.solve_batch

        def logged(*args, **kwargs):
            out = solve_batch(*args, **kwargs)
            its.append([i.iterations for i in solver.last_batch_info])
            return out

        solver.solve_batch = logged
        modes = compute_modes_gpu_batch(css, num_modes=a.modes, solver=solver, post_process=pp, batch=batch)
    return modes, time.perf_counter() - t0, its


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", default="single")
    ap.add_argument("--cells", type=int, default=30)
    ap.add_argument("--modes", type=int, default=10)
    ap.add_argument("--dx", type=float, default=0.02)
    ap.add_argument("--tol", type=float, default=None)
    ap.add_argument("--extra", type=int, default=8)
    ap.add_argument("--batches", default="4,8,16", help="batch sizes, comma-separated")
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--post-process", choices=["meow_gpu", "meow"], default="meow_gpu")
    a = ap.parse_args()
    pp = _post_process(a.post_process)

    import jax
    jax.config.update("jax_enable_x64", True)
    import meow as mw

    from meow_gpu.meow_backend import device_summary

    cells, css = build(a.cells, a.dx)
    ref = None
    for batch in [0] + [int(s) for s in a.batches.split(",") if s.strip()]:
        times = []
        for _ in range(max(1, a.repeat)):
            modes, seconds, its = _solve(css, a, batch, pp)
            times.append(seconds)
        row = {"case": "per_cell" if batch == 0 else "batch", "batch": batch or 1, "precision": a.precision,
               "tol": a.tol, "extra": a.extra, "cells": a.cells, "modes": a.modes, "grid": list(css[0].nx.shape),
               "solver_device": device_summary(), "modes_s": times[-1], "modes_s_first": times[0]}
        if batch == 0:
            row["iterations"] = its
            row["iterations_total"] = int(sum(its))
            # mean_iterations leaves out the cold first cell (as bench_eme_taper.py); mean_iterations_all has it
            row["mean_iterations"] = float(np.mean(its[1:])) if len(its) > 1 else float(its[0])
            row["mean_iterations_all"] = float(np.mean(its))
        else:
            row["batch_used"] = len(its[0])                         # the members of each batch (<= batch)
            row["iterations_max"] = [max(i) for i in its]          # the iterations of each batch
            row["batch_iterations"] = int(sum(row["iterations_max"]))
            # all members, with the cold first batch and the copies of the last cell of a short last run
            row["mean_member_iterations"] = float(np.mean([n for i in its for n in i]))
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
