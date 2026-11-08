# SPDX-License-Identifier: GPL-2.0-or-later
"""Bench: the GPU time stepping of the add-on on the AMD GPU against the CPU solver. One JSON line per leg.

    bash tools/gpu_run.sh addon_gpu python bench/bench_addon.py                       the full run (a few minutes)
    bash tools/gpu_run.sh addon_gpu python bench/bench_addon.py --small               a short check of the script

Cases (addon_cases.py; ``name:resolution:scale:until``):

* 2-D: the slab waveguide at scale 4 (cell 40 x 24) and resolution 40: 1600 x 960 = 1,536,000 cells, to time 30
  (2,400 steps);
* 3-D: the dielectric sphere (cell 4 x 4 x 4) at resolution 32: 128^3 = 2,097,152 cells, to time 20 (1,280 steps).

Legs, each in its own process (run_case.py): ``cpu`` (the CPU solver, its own time stepping, double precision, a
serial build; OMP_NUM_THREADS from ``--threads``, one leg per value), ``triton`` and ``cuda`` (the two kernel
tables of the add-on, MEEP_GPU_HARDWARE=any), ``array`` (the CuPy array path, dispatch off). A GPU leg first runs
the case to ``--warmup`` in the same process, so that kernel compilation is not timed.

Each line: case, leg, table, threads, cells, steps, time_s (the step loop), mcells_per_s (cells x steps / time_s),
steady_mcells_per_s (GPU: between progress callbacks, from the second one on), setup_s (CPU: init_sim; GPU: lift
and the first freeze), agreement (relative L2 of the final Ez field and of the first flux spectrum against the
first CPU leg), differing_words_vs_array (kernel legs: the float32 words of the final Ez field that differ from
the array leg, which runs under keep, the policy that dispatch installs for the kernel legs), versions,
driver.gpu, active_step_path, certified, the dispatched slots, and speedup (time of the first CPU leg / time_s).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]

FULL = ("slab2d:40:4:30", "sphere3d:32:1:20")
SMALL = ("slab2d:20:1:10", "sphere3d:16:1:5")


def run_leg(case: str, res: int, scale: float, until: float, leg: str, *, table: str | None, threads: int | None,
            warmup: float, save: Path) -> dict:
    args = [sys.executable, "run_case.py", "--case", case, "--leg", "gpu" if leg in ("triton", "cuda") else leg,
            "--resolution", str(res), "--scale", str(scale), "--until", str(until), "--no-series",
            "--save", str(save), "--progress-interval", "50"]
    if table:
        args += ["--table", table]
    if leg == "array":
        args += ["--policy", "keep"]          # the policy that dispatch installs for the kernel legs
    if leg != "cpu" and warmup > 0:
        args += ["--warmup", str(warmup)]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    for key in [k for k in env if k.startswith("MEEP_GPU_")]:
        env.pop(key)
    if threads:
        env["OMP_NUM_THREADS"] = str(threads)
    clone = os.environ.get("ADDON_GPU_CLONE")
    env["PYTHONPATH"] = os.pathsep.join([p for p in (clone, str(PACKAGE), env.get("PYTHONPATH")) if p])
    proc = subprocess.run(args, cwd=PACKAGE, env=env, capture_output=True, text=True)
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("ADDON_JSON "):
            return json.loads(line[len("ADDON_JSON "):])
    raise RuntimeError(f"{case} {leg}: exit {proc.returncode}\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")


def repo_relative(path: str | None) -> str | None:
    """A path below the repository folder, relative to it (the rows go into a public repository)."""
    if not path:
        return path
    try:
        return str(Path(path).resolve().relative_to(PACKAGE.parent))
    except ValueError:
        return Path(path).name


def rel_l2(a, b) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--small", action="store_true", help="small cases, to check the script")
    p.add_argument("--cases", nargs="*", default=None, help="name:resolution:scale:until")
    p.add_argument("--legs", default="cpu,triton,cuda,array")
    p.add_argument("--threads", default="1,4", help="OMP_NUM_THREADS of each CPU leg")
    p.add_argument("--warmup", type=float, default=2.0)
    args = p.parse_args(argv)
    cases = args.cases or (SMALL if args.small else FULL)
    legs = [leg.strip() for leg in args.legs.split(",") if leg.strip()]
    threads = [int(t) for t in args.threads.split(",") if t.strip()]
    with tempfile.TemporaryDirectory() as tmp:
        for spec in cases:
            case, res, scale, until = spec.split(":")
            res, scale, until = int(res), float(scale), float(until)
            ref_field = ref_time = ref_flux = array_field = None
            plan = [("cpu", None, t) for t in threads] if "cpu" in legs else []
            order = [leg for leg in ("array", "triton", "cuda") if leg in legs]   # the array path first
            plan += [(leg, leg if leg in ("triton", "cuda") else None, None) for leg in order]
            for leg, table, nthreads in plan:
                save = Path(tmp) / f"{case}-{leg}-{nthreads}.npz"
                rec = run_leg(case, res, scale, until, leg, table=table, threads=nthreads, warmup=args.warmup,
                              save=save)
                with np.load(save) as data:
                    field = np.array(data["field"])
                    flux = np.array(data["flux_0"]) if "flux_0" in data.files else None
                time_s = rec.get("run_s") if leg == "cpu" else rec.get("loop_s")
                if ref_field is None:
                    ref_field, ref_time, ref_flux = field, time_s, flux
                if leg == "array":
                    array_field = field
                words = None
                if leg in ("triton", "cuda") and array_field is not None:
                    a = np.ascontiguousarray(field, dtype=np.float32).view(np.uint32)
                    b = np.ascontiguousarray(array_field, dtype=np.float32).view(np.uint32)
                    words = int(np.count_nonzero(a != b))
                dispatch = rec.get("dispatch") or {}
                out = {
                    "case": case, "leg": leg, "table": table, "threads": nthreads,
                    "resolution": res, "scale": scale, "until": until,
                    "cells": rec["cells"], "shape": rec["shape"], "steps": rec["steps"],
                    "time_s": time_s, "mcells_per_s": rec["cells"] * rec["steps"] / time_s / 1e6,
                    "steady_mcells_per_s": rec.get("mcells_per_s") if leg != "cpu" else None,
                    "setup_s": rec.get("setup_s") if leg == "cpu" else (rec.get("total_s", 0) - time_s),
                    "agreement_rel_l2_vs_cpu": rel_l2(field, ref_field),
                    "flux_rel_l2_vs_cpu": None if flux is None or ref_flux is None else rel_l2(flux, ref_flux),
                    "differing_words_vs_array": words,
                    "speedup_vs_first_cpu_leg": ref_time / time_s,
                    "gpu": rec.get("gpu"), "active_step_path": rec.get("active_step_path"),
                    "certified": rec.get("certified"), "dispatched_slots": dispatch.get("dispatched_slots"),
                    "dispatch_table": dispatch.get("table"), "subnormal_policy": rec.get("subnormal_policy"),
                    "versions": rec["versions"], "add_on_file": repo_relative(rec.get("meep_gpu_file")),
                }
                print(json.dumps(out), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
