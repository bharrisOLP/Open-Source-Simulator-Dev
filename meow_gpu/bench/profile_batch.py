"""Profile of the mode solves of the EME taper: where the time of one solve goes, per cell or batched (development
tool).

    python bench/profile_batch.py --precision single --batch 8            # batched, B = 8
    python bench/profile_batch.py --precision mixed --batch 0             # per cell
    python bench/profile_batch.py --precision single --batch 8 --cells 8 --dx 0.04   # a small case

Synthetic structure: the taper of bench_eme_taper.py (textbook indices). Each run uses a new solver (no warm-start
memory). Run 1 compiles. Run 2 is not instrumented: its time ("wall_s") and its JIT compiles ("compiles_run2") show
the cost without the profile and the recompiles after the first run. Run 3 is instrumented: each wrapped function
waits for its device results (``jax.block_until_ready``), and its time is exclusive (the time of the wrapped
functions that it calls is not in it). So a device kernel gets its launch and its run time; the Python, numpy and
device-to-host copy time of a solver loop stays in that loop function ("_davidson_batch_fixed (Python)" and so on).
The host steps in the thread pool are timed as a whole (``_host_map: <function>``, wall time). The instrumented run
is slower than run 2 (no overlap of host and device work): use the shares, not the sums.

One JSON line: the times of run 2 and run 3, the iterations, and for each part the exclusive seconds, the calls and
the share of run 3.
"""
from __future__ import annotations

import argparse
import functools
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")   # before numpy: OpenBLAS reads it when it loads
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from collections import defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_eme_taper import build  # noqa: E402

T: dict = defaultdict(float)
N: dict = defaultdict(int)
ON = [False]
_LOCAL = threading.local()
COMPILES: list = []


def _block(out):
    import jax
    leaves = [x for x in jax.tree.leaves(out) if isinstance(x, jax.Array) and not isinstance(x, jax.core.Tracer)]
    if leaves:
        jax.block_until_ready(leaves)


def timed(name, f):
    """f with an exclusive timer (main thread only; other threads call f directly)."""
    @functools.wraps(f)
    def g(*a, **k):
        if not ON[0] or threading.current_thread() is not threading.main_thread():
            return f(*a, **k)
        stack = getattr(_LOCAL, "stack", None)
        if stack is None:
            stack = _LOCAL.stack = []
        stack.append(0.0)
        t0 = time.perf_counter()
        try:
            out = f(*a, **k)
            _block(out)
        finally:
            dt = time.perf_counter() - t0
            child = stack.pop()
            T[name] += dt - child
            N[name] += 1
            if stack:
                stack[-1] += dt
        return out
    return g


def instrument():
    import meow_gpu.fde.eigensolver as E
    import meow_gpu.meow_backend as B

    kernels = ["_ritz_b", "_expand_b", "_project_out_b", "_apply_b", "_insert_b", "_reset_b", "_select_b", "_h_new_b",
               "_gram_b", "_combine_b", "precondition_batch", "_ritz", "precondition", "_project_out", "_apply_block",
               "_insert", "_gram", "_combine", "_take_rows", "_correction_krylov", "_correction_solution", "put"]
    host = ["_gmres_coef", "_svqb_coef", "_svqb", "_svqb_b", "_track", "_selection"]
    loops = ["refine", "_refine_with_fallback", "_davidson_fixed", "_davidson_batch_fixed", "_refine_members",
             "davidson", "davidson_batch"]
    for name in kernels + host + loops + [n for n in ("_refine_batch", "_ritz_coef_b", "_svqb_coef_b")
                                          if hasattr(E, n)]:
        if hasattr(E, name):
            label = name if name not in loops else f"{name} (Python)"
            setattr(E, name, timed(label, getattr(E, name)))
    host_map = E._host_map

    def mapped(f, args):
        return timed(f"_host_map: {getattr(f, '__name__', f)}", host_map)(f, args)
    E._host_map = mapped
    eig = np.linalg.eig
    np.linalg.eig = timed("np.linalg.eig (main thread)", eig)
    for name in ["make_problem", "make_preconditioner", "stack", "_fields_batch", "fields", "_orient", "put"]:
        setattr(B, name, timed(f"backend.{name}", getattr(B, name)))
    for name in ["_problem", "_modes"]:
        setattr(B, name, timed(f"backend.{name}", getattr(B, name)))
    B.GPUModeSolver.solve = timed("solver.solve (Python)", B.GPUModeSolver.solve)
    B.GPUModeSolver.solve_batch = timed("solver.solve_batch (Python)", B.GPUModeSolver.solve_batch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", default="single")
    ap.add_argument("--batch", type=int, default=8, help="0: per cell (compute_modes_gpu)")
    ap.add_argument("--cells", type=int, default=30)
    ap.add_argument("--modes", type=int, default=10)
    ap.add_argument("--dx", type=float, default=0.02)
    ap.add_argument("--extra", type=int, default=8)
    a = ap.parse_args()

    import jax
    jax.config.update("jax_enable_x64", True)

    def on_event(event, duration, **kw):
        if "compile" in event:
            COMPILES.append((event, duration))
    try:
        jax.monitoring.register_event_duration_secs_listener(on_event)
    except AttributeError:
        pass

    import meow_gpu.meow_backend as B
    instrument()
    _, css = build(a.cells, a.dx)
    t0 = time.perf_counter()
    _ = [cs.nx for cs in css]                     # meow's index sampling (cached), not part of the profile
    t_index = time.perf_counter() - t0

    def run():
        solver = B.GPUModeSolver(precision=a.precision, extra=a.extra)
        its = []
        t0 = time.perf_counter()
        if a.batch == 0:
            for cs in css:
                B.compute_modes_gpu(cs, num_modes=a.modes, solver=solver)
                its.append(solver.last_info.iterations)
        else:
            solve_batch = solver.solve_batch

            def logged(*args, **kwargs):
                out = solve_batch(*args, **kwargs)
                its.append([i.iterations for i in solver.last_batch_info])
                return out
            solver.solve_batch = logged
            B.compute_modes_gpu_batch(css, num_modes=a.modes, solver=solver, batch=a.batch)
        return time.perf_counter() - t0, its

    t_first, _ = run()
    n0 = len(COMPILES)
    t_plain, its = run()
    compiles_run2 = COMPILES[n0:]
    T.clear()
    N.clear()
    ON[0] = True
    n1 = len(COMPILES)
    t_prof, _ = run()
    ON[0] = False
    compiles_run3 = COMPILES[n1:]
    parts = {k: {"s": round(T[k], 4), "calls": N[k], "share": round(T[k] / t_prof, 4)}
             for k in sorted(T, key=T.get, reverse=True)}
    other = t_prof - sum(T.values())
    row = {"precision": a.precision, "batch": a.batch, "cells": a.cells, "modes": a.modes, "extra": a.extra,
           "grid": list(css[0].nx.shape), "device": B.device_summary(), "index_sampling_s": round(t_index, 3),
           "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
           "first_s": round(t_first, 3), "wall_s": round(t_plain, 3), "profiled_s": round(t_prof, 3),
           "compiles_run2": len(compiles_run2), "compile_s_run2": round(sum(d for _, d in compiles_run2), 3),
           "compiles_run3": len(compiles_run3), "iterations": its,
           "outside_wrapped_s": round(other, 4), "parts": parts}
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
