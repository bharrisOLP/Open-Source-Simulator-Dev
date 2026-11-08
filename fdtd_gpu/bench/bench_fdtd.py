"""Validation cases of fdtd_gpu on the GPU, one JSON line per case (results, references and run time).

    python bench/bench_fdtd.py --case bragg --resolution 200 [--lam0 1.03] [--no-avg]
    python bench/bench_fdtd.py --case waveguide --resolution 20 [--three-d]
    python bench/bench_fdtd.py --case ring --resolution 40 [--no-avg]
    python bench/bench_fdtd.py --case disk --resolution 40 [--tm] [--no-avg] [--forward-diagonal]
    python bench/bench_fdtd.py --case slab --resolution 20 [--e-normal] [--no-avg]
    python bench/bench_fdtd.py --case dipole --resolution 20 --index 1.5 [--half 0.5]
    python bench/bench_fdtd.py --case waveguide --three-d --src-freq 0.5701613   # mode source solved at a band edge
    python bench/bench_fdtd.py --case sheet --resolution 40 [--index 1.5]
    python bench/bench_fdtd.py --case near2far --resolution 20 [--index 1.5] [--component Ex]   # 3-D dipole
    python bench/bench_fdtd.py --case near2far2d --resolution 20 [--index 1.5] [--component Ex]  # 2-D line source
    python bench/bench_fdtd.py --case transform     # transform of exact fields (no FDTD), h = 0.1, 0.05, 0.025
    python bench/bench_fdtd.py --case setup --resolution 50     # time of the subpixel average only, 3-D strip
    python bench/bench_fdtd.py --case all          # bragg, waveguide, ring, dipole, sheet at the default resolutions
    python bench/bench_fdtd.py --case ring --resolution 40,60,80 --both    # a list of resolutions; averaged and not
Each row has the resolution and the cells. --no-avg: the staircase (eps_averaging=False); default: averaged.
The cases are in tests/cases.py; the references (transfer matrices, Bessel-function ring and disk resonances, the
exact slab mode with the Yee dispersion along x and in time, the analytic dipole power, the flux of a current
sheet, the fields of a dipole and of a line current) are in tests/references.py and tests/cases.py. "seconds" is
the time of the FDTD loops (compile included); "setup_seconds" the time of the rasterization and the subpixel
average; "farfield_seconds" the time of the far fields of one box (near2far cases). NaN values (T outside the
source band of the waveguide case) are written as null.
"""
import argparse
import itertools
import json
import math
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
        return None if math.isnan(float(v)) else float(f"{float(v):.7g}")
    if isinstance(v, np.integer):
        return int(v)
    return v


def setup(resolution: int = 50):
    """Time of the subpixel average of a 3-D Si strip (0.4 x 0.22 um, n 3.45 in 1.45) in a 8 x 4 x 4 um cell, and
    the number of sampled voxels (no FDTD run)."""
    import time

    import fdtd_gpu as fg
    core = fg.Block(size=fg.Vector3(fg.inf, 0.4, 0.22), material=fg.Medium(index=3.45))
    ring = [fg.Cylinder(radius=1.2, height=0.22, material=fg.Medium(index=3.45), center=fg.Vector3(1.5, 0)),
            fg.Cylinder(radius=0.8, height=0.22, material=fg.Medium(index=1.45), center=fg.Vector3(1.5, 0))]
    sim = fg.Simulation(cell_size=fg.Vector3(8, 4, 4), resolution=resolution, geometry=[core] + ring,
                        default_material=fg.Medium(index=1.45))
    t0 = time.perf_counter()
    eps = sim.get_epsilon()
    return dict(setup_seconds=sim.setup_seconds, total_seconds=time.perf_counter() - t0,
                n_averaged=sim.n_averaged, cells=int(np.prod(sim.shape)), eps_mean=float(eps.mean()))


ap = argparse.ArgumentParser()
ap.add_argument("--case", choices=["bragg", "waveguide", "ring", "disk", "slab", "dipole", "sheet", "near2far",
                                   "near2far2d", "transform", "setup", "all"], default="all")
ap.add_argument("--component", type=str, default="Ez", help="source of the near2far cases: Ez, Ex or Ey")
ap.add_argument("--resolution", type=str, default=None, help="one value, or a list: 20,40,80")
ap.add_argument("--index", type=float, default=1.0, help="medium of the dipole and sheet cases")
ap.add_argument("--half", type=float, default=1.0, help="half width of the flux box of the dipole case (um)")
ap.add_argument("--three-d", action="store_true", help="3-D strip for the waveguide case")
ap.add_argument("--src-freq", type=float, default=None, help="source center frequency of the waveguide case")
ap.add_argument("--lam0", type=float, default=1.0, help="design wavelength of the Bragg stack (um)")
ap.add_argument("--tm", action="store_true", help="disk case: Ez (default: Hz, in-plane E)")
ap.add_argument("--e-normal", action="store_true", help="slab case: E normal to the faces (Hz polarization)")
ap.add_argument("--no-avg", action="store_true", help="staircase geometry (eps_averaging=False)")
ap.add_argument("--both", action="store_true", help="each case averaged and with the staircase")
ap.add_argument("--forward-diagonal", action="store_true",
                help="keep the diagonal of the forward tensor (fdtd_gpu._averaging.DIAGONAL = \"forward\")")
a = ap.parse_args()
if a.forward_diagonal:
    import fdtd_gpu._averaging
    fdtd_gpu._averaging.DIAGONAL = "forward"
if a.case == "transform":
    for (dims, comp), h in itertools.product([(3, "Ez"), (3, "Ex"), (2, "Ez"), (2, "Ex")], (0.1, 0.05, 0.025)):
        print(json.dumps(_clean({"case": "transform", **cases.transform_exact(dims, comp, h, compare_backends=True)})),
              flush=True)
    sys.exit(0)
todo = ["bragg", "waveguide", "ring", "dipole", "sheet"] if a.case == "all" else [a.case]
resolutions = [None] if a.resolution is None else [int(r) for r in a.resolution.split(",")]
averaging = [True, False] if a.both else [not a.no_avg]
for case, resolution, avg in itertools.product(todo, resolutions, averaging):
    kw = {} if resolution is None else {"resolution": resolution}
    if case in ("bragg", "waveguide", "ring", "disk", "slab"):
        kw["eps_averaging"] = avg
    if case == "bragg":
        kw["lam0"] = a.lam0
    if case == "disk":
        kw["te"] = not a.tm
    if case == "slab":
        kw["e_normal"] = a.e_normal
    if case == "dipole":
        kw["n"] = a.index
        kw["half"] = a.half
    if case == "sheet":
        kw["n"] = a.index
    if case in ("near2far", "near2far2d"):
        kw["n"] = a.index
        kw["component"] = a.component
    if case == "waveguide":
        kw["three_d"] = a.three_d
        kw["src_freq"] = a.src_freq
    fn = {"near2far": "near2far_dipole", "near2far2d": "near2far_line"}.get(case, case)
    res = setup(**kw) if case == "setup" else getattr(cases, fn)(**kw)
    row = {"case": case, **{k: v for k, v in kw.items()}, **res}
    if a.forward_diagonal:
        row["diagonal"] = "forward"
    print(json.dumps(_clean(row)), flush=True)
