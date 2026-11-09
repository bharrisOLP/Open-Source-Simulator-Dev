"""The even (TE-like) gap of the hole slab (bench/bench_hole_slab.py) against 3-D FDTD (fdtd_gpu): band 1 at K and
band 2 at M. The method and the 7/4 crystal: tests/hole_slab_fdtd.py.

    python bench/bench_hole_slab_fdtd.py fdtd --res 16,24 --point K,M      # fdtd_gpu, one JSON line per run
    python bench/bench_hole_slab_fdtd.py fdtd --res 24 --diagonal forward   # the diagonal of the forward tensor
    python bench/bench_hole_slab_fdtd.py bands --n 16,24,32                 # bands_gpu, the 7/4 crystal
    python bench/bench_hole_slab_fdtd.py bands --n 16,24 --triangular       # bands_gpu, the triangular crystal
    python bench/bench_hole_slab_fdtd.py summary rows.jsonl                 # fits and the comparison of logged rows

fdtd rows: the Harminv modes of the weighted probe sum (freq, Q, amplitude over the largest), the picked band
("band": band 1 at K or band 2 at M: the first or second mode with |Q| >= 1e3 and an amplitude of at least 1e-2),
"converged" (that mode was found), the cells, steps and times, max_rss_gb (the peak host memory of the process,
GiB).
bands rows: 4 z-even bands at K and M, "converged" (the eigensolver at both points), the time.
summary: per series (fdtd: diagonal, air, t_after, point; bands: crystal, sub, height) the values against the
resolution; for fdtd (no shift, and the mean of the 4 shifts where a resolution has all 4) the least-squares fits
f = f0 + b h + c h^2 (h = 1 / res) over all resolutions and over all but the coarsest, and f = f0 + b h over the
finest 3 and 4; the strong modes of each fdtd row against the bands_gpu bands by index (mode i is band i; the
relative difference); bands_gpu triangular - 7/4 at the largest n of both. (The summary of 2026-11-08 17:33 to 18:29
in benchmarks/ has an older identification: the mode nearest to each band, which is not one-to-one for the near pair
of bands 2 and 3 at K.)
"""
import argparse
import itertools
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
import hole_slab_fdtd as hs  # noqa: E402

PICK = {"K": 0, "M": 1, "G": 1}          # band 1 at K, band 2 at M (band 2 at Gamma: the first mode above 0)
SHIFTS = [(0.0, 0.0), (0.5, 0.25), (0.25, 0.75), (0.75, 0.5)]   # crystal shifts (voxels): a 4-point lattice


def rss_gb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 3)      # Linux: kB


def ints(s: str) -> list[int]:
    return [int(v) for v in s.split(",")]


def cmd_fdtd(a):
    shifts = SHIFTS[:a.shifts]
    for res, point, shift in itertools.product(ints(a.res), a.point.split(","), shifts):
        r = hs.run_fdtd(point, res, air=a.air, pml=a.pml, t_after=a.t_after, diagonal=a.diagonal, shift=shift)
        strong = hs.strong_modes(r["modes"])
        i = PICK[point]
        row = {"case": "hole-slab-fdtd", "method": "fdtd_gpu", "point": point, "res": res, "ly": hs.LY,
               "shift": list(shift), "diagonal": a.diagonal, "air": a.air, "pml": a.pml, "t_after": a.t_after,
               "band": strong[i] if len(strong) > i else None, "converged": len(strong) > i,
               "strong": strong, "modes": r["modes"], "fit_err": r["fit_err"]}
        row.update({k: r[k] for k in ("shape", "cells", "steps", "dt", "run_s", "setup_s", "wall_s", "n_averaged",
                                      "device")})
        row["max_rss_gb"] = rss_gb()
        print(json.dumps(row), flush=True)


def cmd_bands(a):
    import jax

    jax.config.update("jax_enable_x64", True)
    for n in ints(a.n):
        t0 = time.perf_counter()
        cell = hs.bands_cell(n, None if a.triangular else hs.LY, a.height, a.layers)
        b = hs.even_bands(cell, ("K", "M"), num=4, sub=a.sub, tol=a.tol)
        row = {"case": "hole-slab-bands", "method": "bands_gpu", "crystal": "triangular" if a.triangular else "7/4",
               "n": n, "shape": list(cell.shape), "height": cell.size[2], "sub": a.sub, "tol": a.tol,
               "K": b["K"][0].tolist(), "M": b["M"][0].tolist(), "band1_K": float(b["K"][0][0]),
               "band2_M": float(b["M"][0][1]), "converged": b["K"][1] and b["M"][1], "iterations": b["iterations"],
               "time_s": round(time.perf_counter() - t0, 3), "max_rss_gb": rss_gb(), "device": str(jax.devices()[0])}
        print(json.dumps(row), flush=True)


def fit(h, f, order=2):
    """Least squares f = f0 + b h (+ c h^2); returns (coefficients, largest residual)."""
    A = np.vander(np.asarray(h, float), order + 1, increasing=True)
    c, *_ = np.linalg.lstsq(A, np.asarray(f, float), rcond=None)
    return c, float(np.abs(A @ c - f).max()) if len(f) > order + 1 else 0.0


def cmd_summary(a):
    rows = [json.loads(s) for s in Path(a.rows).read_text().splitlines() if s.strip().startswith("{")]
    bands = [r for r in rows if r["case"] == "hole-slab-bands"]
    fdtd = [r for r in rows if r["case"] == "hole-slab-fdtd"]
    ref = {}
    for crystal in ("7/4", "triangular"):
        sel = sorted((r for r in bands if r["crystal"] == crystal), key=lambda r: (r["sub"], r["height"], r["n"]))
        for r in sel:
            print(f"bands_gpu {crystal:10s} n {r['n']:3d} sub {r['sub']:2d} height {r['height']:.4f}: band1_K "
                  f"{r['band1_K']:.6f} band2_M {r['band2_M']:.6f} converged {r['converged']}")
        main = [r for r in sel if r["sub"] == 8 and r["shape"][2] == 4 * r["n"] + 1
                and abs(r["height"] - (4 * r["n"] + 1) / r["n"]) < 1e-12]
        if main:
            ref[crystal] = max(main, key=lambda r: r["n"])
    if "7/4" in ref and "triangular" in ref:
        t, s = ref["triangular"], ref["7/4"]
        if t["n"] == s["n"]:
            print(f"triangular - 7/4 (n {t['n']}): band1_K {t['band1_K'] - s['band1_K']:+.6f} "
                  f"band2_M {t['band2_M'] - s['band2_M']:+.6f}")
    for key in sorted({(r["diagonal"], r["air"], r["t_after"], r["point"]) for r in fdtd}):
        sel = sorted((r for r in fdtd if (r["diagonal"], r["air"], r["t_after"], r["point"]) == key),
                     key=lambda r: (r["res"], r["shift"]))
        name = "band1_K" if key[3] == "K" else "band2_M"
        tag = f"fdtd {key[0]:7s} air {key[1]} t_after {key[2]} {key[3]}"
        for r in sel:
            ident = ""
            if "7/4" in ref and key[3] in ref["7/4"]:
                fb = ref["7/4"][key[3]]
                ident = " by index against bands_gpu " + ", ".join(
                    f"{x:.5f}->{m:.5f} ({m / x - 1:+.2%})" for x, m in zip(fb, r["strong"]))
            print(f"{tag} res {r['res']:3d} shift {r['shift']}: {name} {r['band']} "
                  f"(strong {['%.5f' % f for f in r['strong']]}){ident}")
        series = {"no shift": {}, "mean of 4 shifts": {}}
        for res in sorted({r["res"] for r in sel}):
            vals = {tuple(r["shift"]): r["band"] for r in sel if r["res"] == res and r["band"] is not None}
            if (0.0, 0.0) in vals:
                series["no shift"][res] = vals[(0.0, 0.0)]
            if len(vals) == len(SHIFTS):
                v = np.array(list(vals.values()))
                series["mean of 4 shifts"][res] = float(v.mean())
                print(f"{tag} res {res:3d}: mean of 4 shifts {v.mean():.6f}, spread (max - min) {np.ptp(v):.1e}")
        for lab, ser in series.items():
            rs = sorted(ser)
            for order, sub in ((2, rs), (2, rs[1:]), (1, rs[-3:]), (1, rs[-4:])):       # rs: ascending resolution
                if len(sub) < order + 1 or (order == 1 and len(rs) < 4):
                    continue
                c, res_max = fit([1 / q for q in sub], [ser[q] for q in sub], order)
                terms = f"b {c[1]:+.4f}" + (f" c {c[2]:+.4f}" if order == 2 else "")
                msg = (f"{tag} {lab}, fit f0 + b h{' + c h^2' if order == 2 else ''} (res {sub}): f0 {c[0]:.6f} "
                       f"{terms} max residual {res_max:.1e}")
                if "7/4" in ref:
                    msg += f"; f0 - bands_gpu (7/4, n {ref['7/4']['n']}) {c[0] - ref['7/4'][name]:+.6f}"
                print(msg)


ap = argparse.ArgumentParser()
sp = ap.add_subparsers(dest="cmd", required=True)
p = sp.add_parser("fdtd")
p.add_argument("--res", default="16")
p.add_argument("--point", default="K,M")
p.add_argument("--air", type=float, default=1.5)
p.add_argument("--pml", type=float, default=1.0)
p.add_argument("--t-after", type=float, default=150.0)
p.add_argument("--diagonal", choices=["inverse", "forward"], default="inverse")
p.add_argument("--shifts", type=int, default=1, help="the first 1 to 4 crystal shifts of SHIFTS (in voxels)")
p = sp.add_parser("bands")
p.add_argument("--n", default="16")
p.add_argument("--triangular", action="store_true")
p.add_argument("--height", type=float, default=None, help="the cell height (a); default (layers n + 1) / n")
p.add_argument("--layers", type=int, default=4, help="layers n + 1 points along z")
p.add_argument("--sub", type=int, default=8)
p.add_argument("--tol", type=float, default=1e-8)
p = sp.add_parser("summary")
p.add_argument("rows")
a = ap.parse_args()
{"fdtd": cmd_fdtd, "bands": cmd_bands, "summary": cmd_summary}[a.cmd](a)
