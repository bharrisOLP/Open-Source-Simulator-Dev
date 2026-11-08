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
    python bench/bench_fdtd.py --case near2far2d --resolution 20 --fcen 0.75 --df 0.3   # other frequencies (both cases)
    python bench/bench_fdtd.py --case transform     # transform of exact fields (no FDTD), h = 0.1, 0.05, 0.025
    python bench/bench_fdtd.py --case distance --resolution 20   # near2far error on shells 1 to 10 cells out
    python bench/bench_fdtd.py --case subtract --resolution 20   # near2far with subtracted incident fields (2-D)
    python bench/bench_fdtd.py --case setup --resolution 50     # time of the subpixel average only, 3-D strip
    python bench/bench_fdtd.py --case film --kind drude --resolution 100,200,400   # dispersive film: R, T (1-D)
    python bench/bench_fdtd.py --case film --kind lorentz --thickness 0.504 --face-rule fill,average,staircase
    python bench/bench_fdtd.py --case uniform --kind conductive --resolution 20   # k = 0: decay or oscillation
    python bench/bench_fdtd.py --case dipole --sigma-d 0.5 --resolution 10,20 [--no-source-fix]   # conductive
    python bench/bench_fdtd.py --case halfspace --kind drude --resolution 50 [--pair 0] [--r-asymptotic 1e-15]   # 1-D
    python bench/bench_fdtd.py --case near2far-medium --kind lorentz --resolution 20,40   # near2far in a medium
    python bench/bench_fdtd.py --case transform --index-imag 0.05   # transform of exact fields, lossy medium
    python bench/bench_fdtd.py --case all          # bragg, waveguide, ring, dipole, sheet at the default resolutions
    python bench/bench_fdtd.py --case ring --resolution 40,60,80 --both    # a list of resolutions; averaged and not
Each row has the resolution and the cells. --no-avg: the staircase (eps_averaging=False); default: averaged.
Film and uniform rows: the arrays of the 41 frequencies (film) are in the row; the references are the transfer
matrix with the exact eps(f), the exact 1-D Yee grid with the discrete chi of the ADE, and the roots of eps = 0.
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
                                   "near2far2d", "transform", "distance", "subtract", "setup", "film", "uniform",
                                   "halfspace", "near2far-medium", "all"],
                default="all")
ap.add_argument("--kind", type=str, default=None,
                help="medium of the film case (drude, lorentz), of the uniform and halfspace cases (drude, lorentz, "
                     "conductive) or of the near2far-medium case (lorentz, conductive)")
ap.add_argument("--sigma-d", type=float, default=None, help="dipole case: D_conductivity of the medium (1/um)")
ap.add_argument("--no-source-fix", action="store_true",
                help="dipole case: the source current of fdtdx alone (fdtd_gpu._dispersion.CORRECT_SOURCES = False)")
ap.add_argument("--pair", type=float, default=None,
                help="halfspace case: the distance of the two opposite source sheets (um; default 0.1; 0: one sheet)")
ap.add_argument("--r-asymptotic", type=float, default=None,
                help="halfspace case: PML(R_asymptotic) (the round trip in vacuum; default: 1e-6 of fdtdx)")
ap.add_argument("--index-imag", type=float, default=None,
                help="transform case: Im n of the medium (n = 1.5 + i x; the far points are not checked then)")
ap.add_argument("--thickness", type=float, default=None,
                help="film case: the film thickness (um; default 0.3 Drude, 0.5 Lorentz)")
ap.add_argument("--fwidth", type=float, default=None, help="film case: the source width (1/um; default 1.6)")
ap.add_argument("--face-rule", type=str, default=None,
                help="film case: fill, average, staircase or a list (fdtd_gpu._dispersion.FACE_RULE; default fill)")
ap.add_argument("--component", type=str, default=None, help="source of the near2far cases: Ez (default), Ex or Ey")
ap.add_argument("--fcen", type=float, default=None, help="center frequency of the near2far cases (default 0.5 / um)")
ap.add_argument("--df", type=float, default=None, help="frequency width of the near2far cases (default 0.2 / um)")
ap.add_argument("--resolution", type=str, default=None, help="one value, or a list: 20,40,80")
ap.add_argument("--index", type=float, default=None, help="medium of the dipole, sheet and near2far cases (default 1)")
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
_USED_BY = {"index": ("dipole", "sheet", "near2far", "near2far2d", "all"), "component": ("near2far", "near2far2d"),
            "fcen": ("near2far", "near2far2d"), "df": ("near2far", "near2far2d"),
            "kind": ("film", "uniform", "halfspace", "near2far-medium"), "thickness": ("film",), "fwidth": ("film",),
            "face_rule": ("film",), "sigma_d": ("dipole",), "index_imag": ("transform",), "pair": ("halfspace",),
            "r_asymptotic": ("halfspace",)}
if a.no_source_fix and a.case != "dipole":
    ap.error("--no-source-fix is used only by --case dipole")
for opt, used_by in _USED_BY.items():          # an option that the case does not use is an error, not ignored
    if getattr(a, opt) is not None and a.case not in used_by:
        ap.error(f"--{opt} is not used by --case {a.case}")
index = 1.0 if a.index is None else a.index
if a.forward_diagonal:
    import fdtd_gpu._averaging
    fdtd_gpu._averaging.DIAGONAL = "forward"
import fdtd_gpu._dispersion  # noqa: E402

rules = [None] if a.face_rule is None else a.face_rule.split(",")
if any(r not in fdtd_gpu._dispersion.FACE_RULES for r in rules if r is not None):
    ap.error(f"--face-rule: one or more of {fdtd_gpu._dispersion.FACE_RULES}")
if a.case == "transform":
    kw = {} if a.index_imag is None else {"n": complex(1.5, a.index_imag)}
    for (dims, comp), h in itertools.product([(3, "Ez"), (3, "Ex"), (2, "Ez"), (2, "Ex")], (0.1, 0.05, 0.025)):
        print(json.dumps(_clean({"case": "transform", **cases.transform_exact(dims, comp, h, compare_backends=True,
                                                                              **kw)})), flush=True)
    sys.exit(0)
todo = ["bragg", "waveguide", "ring", "dipole", "sheet"] if a.case == "all" else [a.case]
resolutions = [None] if a.resolution is None else [int(r) for r in a.resolution.split(",")]
averaging = [True, False] if a.both else [not a.no_avg]
for case, resolution, avg, rule in itertools.product(todo, resolutions, averaging, rules):
    kw = {} if resolution is None else {"resolution": resolution}
    fdtd_gpu._dispersion.FACE_RULE = rule or "fill"
    if case in ("bragg", "waveguide", "ring", "disk", "slab", "film"):
        kw["eps_averaging"] = avg
    if case in ("film", "uniform"):
        kw["kind"] = a.kind or ("drude" if case == "film" else "conductive")
    if case == "film":
        kw.update({k: getattr(a, k) for k in ("thickness", "fwidth") if getattr(a, k) is not None})
    if case == "bragg":
        kw["lam0"] = a.lam0
    if case == "disk":
        kw["te"] = not a.tm
    if case == "slab":
        kw["e_normal"] = a.e_normal
    if case == "dipole":
        kw["n"] = index
        kw["half"] = a.half
        if a.sigma_d is not None:
            kw["sigma_d"] = a.sigma_d
        if a.no_source_fix:
            kw["correct_sources"] = False
    if case in ("halfspace", "near2far-medium"):
        kw["kind"] = a.kind or "lorentz"
    if case == "halfspace" and a.pair is not None:
        kw["pair"] = a.pair
    if case == "halfspace" and a.r_asymptotic is not None:
        kw["r_asymptotic"] = a.r_asymptotic
    if case == "sheet":
        kw["n"] = index
    if case in ("near2far", "near2far2d"):
        kw["n"] = index
        kw["component"] = a.component or "Ez"
        kw.update({k: getattr(a, k) for k in ("fcen", "df") if getattr(a, k) is not None})
    if case == "waveguide":
        kw["three_d"] = a.three_d
        kw["src_freq"] = a.src_freq
    fn = {"near2far": "near2far_dipole", "near2far2d": "near2far_line", "distance": "near2far_distance",
          "subtract": "near2far_subtract", "near2far-medium": "near2far_medium"}.get(case, case)
    res = setup(**kw) if case == "setup" else getattr(cases, fn)(**kw)
    row = {"case": case, **{k: v for k, v in kw.items()}, **res}
    if a.forward_diagonal:
        row["diagonal"] = "forward"
    print(json.dumps(_clean(row)), flush=True)
