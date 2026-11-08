"""EME against 3-D FDTD with subpixel averaging on one synthetic SOI taper, over a small band.

The device (photonsuite segments; textbook indices, constant over the band: Si 3.476, SiO2 1.444): a Si strip 220 nm
thick in SiO2, 1.5 um straight at the input width w0, a linear taper to w1 over `length` um in `segments` steps (each
segment has the width at its center), 1 um straight at w1. Window x -2 to 2 um, y -0.8 to 1.02 um (the core is y 0
to 0.22). TE (E mostly along x).

    python bench/eme_fdtd_taper.py eme  [--step 0.02] [--modes 10] [--passivity invert] [--rcond 1e-3]
    python bench/eme_fdtd_taper.py fdtd [--resolution 40] [--staircase] [--no-calibrate]
    common: [--wavelengths 1.50,1.525,1.55,1.575,1.60] [--length 2] [--segments 20] [--w0 0.5] [--w1 2.0]

eme: photonsuite.eme (meow + meow-gpu), one EME cell per segment, so the EME samples the device exactly along z; it
solves each wavelength separately. fdtd: photonsuite.fdtd_avg (fdtd_gpu, subpixel averaging unless --staircase), one
run for the whole band, plus the two straight-guide runs of the mode factor (unless --no-calibrate). Prints one JSON
line.
"""
import argparse
import json
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
        from photonsuite.fdtd_avg import mode_transmission
        r = mode_transmission(segs, *WINDOW, wl, a.resolution, eps_averaging=not a.staircase,
                              calibrate=not a.no_calibrate)
        runs = (r.run,) + tuple(r.calibration)
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
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
