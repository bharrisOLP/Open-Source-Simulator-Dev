"""EME against 3-D FDTD with subpixel averaging on one synthetic SOI taper, over a small band.

The device (photonsuite segments; textbook indices, constant over the band: Si 3.476, SiO2 1.444): a Si strip 220 nm
thick in SiO2, 1.5 um straight at the input width w0, a linear taper to w1 over `length` um in `segments` steps (each
segment has the width at its center), 1 um straight at w1. Window x -2 to 2 um, y -0.8 to 1.02 um (the core is y 0
to 0.22). TE (E mostly along x).

    python bench/eme_fdtd_taper.py eme  [--step 0.02] [--modes 10] [--passivity invert] [--rcond 1e-3]
    python bench/eme_fdtd_taper.py fdtd [--resolution 40] [--staircase] [--no-calibrate] [--walls pml]
    common: [--wavelengths 1.50,1.525,1.55,1.575,1.60] [--length 2] [--segments 20] [--w0 0.5] [--w1 2.0]

eme: photonsuite.eme (meow + meow-gpu), one EME cell per segment, so the EME samples the device exactly along z; it
solves each wavelength separately. fdtd: photonsuite.fdtd_avg (fdtd_gpu, subpixel averaging unless --staircase), one
run for the whole band, plus the two straight-guide runs of the mode factor (unless --no-calibrate). --walls pec: metal
side walls at the window edges (the closed window of the EME) in place of the PML outside the window; the row gives
the wall positions ("walls_um") and whether each run decayed before max_time ("decayed"). --pec-grid (with the PML):
the side PML is 0.5 / resolution thicker, so that the cell has the same parity of cells across as with --walls pec
on each transverse axis, and the shape edges have the same place on the grid (a cell face or a cell center) in the
two runs ("pec_grid_aligned" in the row). Prints one JSON line (with the peak memory of the process, "max_rss_gb").
"""
import argparse
import json
import resource
import time

import numpy as np

from photonsuite import CrossSection, Material, Rect
from photonsuite.fdtd import Segment

SI, SIO2 = Material("Si", n=3.476), Material("SiO2", n=1.444)
WINDOW = ((-2.0, 2.0), (-0.8, 1.02))


def strip(w):
    return CrossSection(SIO2, (Rect(-w / 2, w / 2, 0.0, 0.22, SI),))


def taper(w0, w1, length, n_seg):
    segs = [Segment(0.0, 1.5, strip(w0))]
    for k in range(n_seg):
        segs.append(Segment(1.5 + length * k / n_seg, 1.5 + length * (k + 1) / n_seg,
                            strip(w0 + (w1 - w0) * (k + 0.5) / n_seg)))
    return segs + [Segment(1.5 + length, 2.5 + length, strip(w1))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("method", choices=["eme", "fdtd"])
    ap.add_argument("--wavelengths", default="1.50,1.525,1.55,1.575,1.60")
    ap.add_argument("--length", type=float, default=2.0)
    ap.add_argument("--segments", type=int, default=20)
    ap.add_argument("--w0", type=float, default=0.5)
    ap.add_argument("--w1", type=float, default=2.0)
    ap.add_argument("--step", type=float, default=0.02, help="EME transverse grid (um)")
    ap.add_argument("--modes", type=int, default=10)
    ap.add_argument("--passivity", default="invert", help="meow passivity_method: invert (default), clip, none")
    ap.add_argument("--rcond", type=float, default=1e-3, help="meow tsvd_rcond")
    ap.add_argument("--resolution", type=float, default=40.0, help="FDTD cells per um")
    ap.add_argument("--staircase", action="store_true")
    ap.add_argument("--no-calibrate", action="store_true")
    ap.add_argument("--walls", choices=["pml", "pec"], default="pml",
                    help="FDTD side walls: pml (absorbing, outside the window) or pec (metal, at the window edges)")
    ap.add_argument("--pec-grid", action="store_true",
                    help="with --walls pml: the side PML 0.5 / resolution thicker (the transverse grid of --walls pec)")
    ap.add_argument("--max-time", type=float, default=None,
                    help="FDTD run time after the source (um/c; default: photonsuite.fdtd_avg)")
    a = ap.parse_args()
    wl = [float(v) for v in a.wavelengths.split(",")]
    segs = taper(a.w0, a.w1, a.length, a.segments)
    row = {"case": "SOI taper", "w0": a.w0, "w1": a.w1, "length_um": a.length, "segments": a.segments,
           "method": a.method, "wavelengths": wl}
    t0 = time.perf_counter()
    if a.method == "eme":
        from photonsuite.eme import transmission
        ikw = None
        if a.passivity != "invert" or a.rcond != 1e-3:
            ikw = {"passivity_method": a.passivity, "tsvd_rcond": a.rcond}
        rs = [transmission(segs, *WINDOW, w, step=a.step, cell_length=None, num_modes=a.modes, interface_kwargs=ikw)
              for w in wl]
        row.update({"step_um": a.step, "modes": a.modes, "passivity": a.passivity, "rcond": a.rcond,
                    "cells": rs[0].n_cells, "min_modes_kept": min(min(r.modes_kept) for r in rs),
                    "T00": [r.transmission for r in rs], "P_right": [r.total_right for r in rs],
                    "P_left": [r.total_left for r in rs]})
    else:
        from photonsuite.fdtd_avg import _cells_between_walls, mode_transmission, wall_positions
        if a.pec_grid and a.walls != "pml":
            ap.error("--pec-grid is for --walls pml")
        side = 0.4 + 0.5 / a.resolution if a.pec_grid else None
        r = mode_transmission(segs, *WINDOW, wl, a.resolution, eps_averaging=not a.staircase,
                              calibrate=not a.no_calibrate, walls=a.walls, max_time=a.max_time, side_pml=side)
        pec_shape = [_cells_between_walls(hi - lo, a.resolution) + 1 for lo, hi in WINDOW]
        row["side_pml"] = side
        row["pec_grid_aligned"] = all((n - m) % 2 == 0 for n, m in zip(r.run.shape[1:], pec_shape))
        runs = (r.run,) + tuple(r.calibration)
        row.update({"walls": a.walls, "window_um": [list(v) for v in WINDOW], "max_time": a.max_time,
                    "steps_all": [q.steps for q in runs],
                    "walls_um": ([list(v) for v in wall_positions(*WINDOW, a.resolution)] if a.walls == "pec"
                                 else None),
                    "shape": list(r.run.shape), "decayed": [q.decayed for q in runs]})
        row.update({"resolution": a.resolution, "eps_averaging": not a.staircase, "calibrated": not a.no_calibrate,
                    "cells": r.run.cells, "steps": r.run.steps, "T00": r.T00.tolist(), "T00_raw": r.T00_raw.tolist(),
                    "T_flux": r.T_flux.tolist(), "R00": r.R00.tolist(), "b_in": r.b_in.tolist(),
                    "b_out": r.b_out.tolist(), "input_mode_share": r.input_mode_share.tolist(),
                    "src_rel": r.src_rel.tolist(), "neff_in": r.run.neff_in.tolist(),
                    "neff_out": r.run.neff_out.tolist(),
                    "cal_T_flux": [float(np.mean(c.p_out / c.p_in)) for c in r.calibration],
                    "cal_neff_phase": [c.phase_index(wl, c.neff_out).tolist() for c in r.calibration],
                    "cal_neff_solver": [c.neff_out.tolist() for c in r.calibration],
                    "fdtd_seconds": round(sum(q.seconds for q in runs), 1),
                    "setup_seconds": round(sum(q.setup_seconds for q in runs), 1), "device": r.run.device})
    row["seconds"] = round(time.perf_counter() - t0, 1)
    row["max_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)   # Linux: kB
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
