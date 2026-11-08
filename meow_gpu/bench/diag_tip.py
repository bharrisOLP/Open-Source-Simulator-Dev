"""Diagnosis of tests/test_refine.py::test_example_grid_gives_the_modes_of_the_earlier_polish (development tool): the
SiN tip of ``tests/structures.sin_tip`` (the structure of examples/chip_to_fiber_coupling.py before 2026-11-07) on
the 50 nm grid, "mixed" with and without the admission rule of ``_track``, against the earlier polish and the
double solve, at the BLAS thread counts of --threads (threadpoolctl, changed while the process runs).

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python bench/diag_tip.py --threads 1 --repeat 1 --switch-tols 1e-3
    OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python bench/diag_tip.py --threads 4,1 --repeat 2

A reproduction of the defect of 2026-11-07 needs the thread count in the environment when the process starts:
OPENBLAS_NUM_THREADS=1 gave the wrong pair with the admission rule (with --switch-tols 1e-3, the stop before the
cold-start rule), 4 threads only without it. A thread count changed while the process runs (--threads) gave the
results of the count at the start (RX 7900 XT computer).

One JSON line for each solve: the BLAS threads, the case, the polish that was used, convergence, the residual,
the iterations, the neff and the differences against the references of the same thread count.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np  # noqa: E402
from threadpoolctl import threadpool_info, threadpool_limits  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
import meow_gpu.fde.eigensolver as eigensolver  # noqa: E402
from meow_gpu.fde.operator import make_problem  # noqa: E402
from meow_gpu.meow_backend import GPUModeSolver  # noqa: E402
from structures import sin_tip  # noqa: E402

K = 4


def solve(p, **kw):
    s = GPUModeSolver(extra=8, warm_start=False, **kw)
    t0 = time.perf_counter()
    n, _, _ = s.solve(p, K)
    i = s.last_info
    return n, {"polish_used": i.polish_used, "converged": bool(i.converged), "residual": float(i.residual),
               "iterations": i.iterations, "matvecs_hi": i.matvecs_hi, "s": round(time.perf_counter() - t0, 2),
               "neff": [round(float(np.real(x)), 10) for x in n]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", default="4,1")
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--dx", type=float, default=0.05)
    ap.add_argument("--switch-tols", default="", help="comma-separated switch_tol values of mixed (default: None)")
    ap.add_argument("--no-admission-off", action="store_true", help="only the admission rule on")
    a = ap.parse_args()
    print(json.dumps({"meow_gpu": eigensolver.__file__,
                      "blas": [(i["internal_api"], i["num_threads"]) for i in threadpool_info()
                               if i["user_api"] == "blas"]}), flush=True)
    p = make_problem(*sin_tip(dx=a.dx), 1.55)
    track = eigensolver._track
    for k in [int(t) for t in a.threads.split(",")]:
        with threadpool_limits(limits=k, user_api="blas"):
            n_old, info = solve(p, precision="mixed", polish="davidson")
            print(json.dumps({"threads": k, "case": "earlier polish", **info}), flush=True)
            n_dbl, info = solve(p, precision="double")
            print(json.dumps({"threads": k, "case": "double", **info,
                              "d_old": float(np.max(np.abs(n_dbl - n_old)))}), flush=True)
            switch = [float(x) for x in a.switch_tols.split(",") if x] or [None]
            for admission in (True,) if a.no_admission_off else (True, False):
                eigensolver._track = track if admission else \
                    (lambda vals, prev, rn_vals=None, target=0.0: track(vals, prev))
                for sw in switch:
                    for r in range(a.repeat):
                        n, info = solve(p, precision="mixed", switch_tol=sw)
                        print(json.dumps({"threads": k, "case": f"mixed, admission {admission}", "switch_tol": sw,
                                          "repeat": r, **info, "d_old": float(np.max(np.abs(n - n_old))),
                                          "d_dbl": float(np.max(np.abs(n - n_dbl)))}), flush=True)
            eigensolver._track = track


if __name__ == "__main__":
    main()
