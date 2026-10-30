"""Cross-method check with Polygon shapes: a linear Si taper with sloped sidewalls (default 80 degrees; bottom width
0.5 -> 2.0 um over 10 um, 220 nm thick, SiO2, 1 um straight input and output), by EME (photonsuite.eme, meow +
meow-gpu) and by calibrated 3-D FDTD (photonsuite.fdtd, as fdtd_taper_calibrated.py) on the same segment list.
Prints one JSON line.

    python bench/sloped_taper.py [fdtd step um] [segments] [EME cell length um] [sidewall angle deg]
"""
import json
import math
import sys
import time

from photonsuite import CrossSection, Material, Polygon
from photonsuite.eme import transmission as eme_transmission
from photonsuite.fdtd import Segment
from photonsuite.fdtd import transmission as fdtd_transmission

step = float(sys.argv[1]) if len(sys.argv) > 1 else 0.025
n_seg = int(sys.argv[2]) if len(sys.argv) > 2 else 100
cell_length = float(sys.argv[3]) if len(sys.argv) > 3 else 0.2
SI, SIO2 = Material("Si", n=3.476), Material("SiO2", n=1.444)
H, ANGLE = 0.22, float(sys.argv[4]) if len(sys.argv) > 4 else 80.0
INSET = H / math.tan(math.radians(ANGLE))          # 0.039 um: the top is narrower by this on each side


def rib(w):
    return CrossSection(SIO2, (Polygon(((-w / 2, 0.0), (w / 2, 0.0), (w / 2 - INSET, H), (-w / 2 + INSET, H)), SI),))


segs = [Segment(0.0, 1.0, rib(0.5))]
for k in range(n_seg):
    segs.append(Segment(1.0 + 10.0 * k / n_seg, 1.0 + 10.0 * (k + 1) / n_seg, rib(0.5 + 1.5 * (k + 0.5) / n_seg)))
segs.append(Segment(11.0, 12.0, rib(2.0)))
win = ((-2.0, 2.0), (-0.8, 1.02))
t0 = time.perf_counter()
eme = eme_transmission(segs, *win, 1.55, step=0.02, cell_length=cell_length, num_modes=10)
t_eme = time.perf_counter() - t0
cal = fdtd_transmission([Segment(0.0, 12.0, rib(2.0))], *win, 1.55, step=step)
tap = fdtd_transmission(segs, *win, 1.55, step=step)
print(json.dumps({"case": "sloped taper", "sidewall_deg": ANGLE, "segments": n_seg, "eme_cells": eme.n_cells,
                  "eme_T00": eme.transmission, "eme_P_right": eme.total_right, "eme_s": round(t_eme, 1),
                  "fdtd_step_um": step, "fdtd_T_flux": tap.flux_transmission,
                  "fdtd_T00": tap.flux_transmission * tap.out_mode_fraction / cal.out_mode_fraction,
                  "fdtd_cal_T_flux": cal.flux_transmission, "fdtd_s": round(tap.seconds + cal.seconds, 1)}), flush=True)
