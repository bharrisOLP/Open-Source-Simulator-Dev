"""Validation cases of fdtd_gpu on the GPU, one JSON line per case (results, references and run time).

    python bench/bench_fdtd.py --case bragg --resolution 200
    python bench/bench_fdtd.py --case waveguide --resolution 20 [--three-d]
    python bench/bench_fdtd.py --case ring --resolution 40
    python bench/bench_fdtd.py --case dipole --resolution 20 --index 1.5 [--half 0.5]
    python bench/bench_fdtd.py --case waveguide --three-d --src-freq 0.5701613   # mode source solved at a band edge
    python bench/bench_fdtd.py --case sheet --resolution 40 [--index 1.5]
    python bench/bench_fdtd.py --case all          # all five at the default resolutions
Each row has the resolution and the cells.
The cases are in tests/cases.py; the references (transfer matrices, Bessel-function ring resonances, the analytic
dipole power, the flux of a current sheet) are in tests/references.py and tests/cases.py. "seconds" is the time
of the FDTD loops (compile included).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))
import cases  # noqa: E402
import numpy as np  # noqa: E402


def _clean(v):
    if isinstance(v, np.ndarray):
        return [_clean(x) for x in v.tolist()]
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, complex):
        return [v.real, v.imag]
    if isinstance(v, (float, np.floating)):
        return float(f"{float(v):.7g}")
    if isinstance(v, np.integer):
        return int(v)
    return v


ap = argparse.ArgumentParser()
ap.add_argument("--case", choices=["bragg", "waveguide", "ring", "dipole", "sheet", "all"], default="all")
ap.add_argument("--resolution", type=int, default=None)
ap.add_argument("--index", type=float, default=1.0, help="medium of the dipole and sheet cases")
ap.add_argument("--half", type=float, default=1.0, help="half width of the flux box of the dipole case (um)")
ap.add_argument("--three-d", action="store_true", help="3-D strip for the waveguide case")
ap.add_argument("--src-freq", type=float, default=None, help="source center frequency of the waveguide case")
a = ap.parse_args()
todo = ["bragg", "waveguide", "ring", "dipole", "sheet"] if a.case == "all" else [a.case]
for case in todo:
    kw = {} if a.resolution is None else {"resolution": a.resolution}
    if case == "dipole":
        kw["n"] = a.index
        kw["half"] = a.half
    if case == "sheet":
        kw["n"] = a.index
    if case == "waveguide":
        kw["three_d"] = a.three_d
        kw["src_freq"] = a.src_freq
    res = getattr(cases, case)(**kw)
    row = {"case": case, **{k: v for k, v in kw.items()}, **res}
    print(json.dumps(_clean(row)), flush=True)
