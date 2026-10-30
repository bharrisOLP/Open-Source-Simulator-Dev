"""EME of the FDTD taper (suite/bench/fdtd_taper_calibrated.py) through the suite adapter photonsuite.eme: the same
segment list (1 um input, 0.5 -> 2.0 um over 10 um in n_seg segments, 1 um output), meow + meow-gpu. Prints one JSON
line: T00 = |S21|^2 of the fundamental mode, the power in all kept right-side modes, the cells."""
import json
import sys
import time

from photonsuite import CrossSection, Material, Rect
from photonsuite.eme import transmission
from photonsuite.fdtd import Segment

cell_length = float(sys.argv[1]) if len(sys.argv) > 1 else 0.2
n_seg = int(sys.argv[2]) if len(sys.argv) > 2 else 100
SI, SIO2 = Material("Si", n=3.476), Material("SiO2", n=1.444)


def strip(w):
    return CrossSection(SIO2, (Rect(-w / 2, w / 2, 0.0, 0.22, SI),))


segs = [Segment(0.0, 1.0, strip(0.5))]
for k in range(n_seg):
    segs.append(Segment(1.0 + 10.0 * k / n_seg, 1.0 + 10.0 * (k + 1) / n_seg, strip(0.5 + 1.5 * (k + 0.5) / n_seg)))
segs.append(Segment(11.0, 12.0, strip(2.0)))
t0 = time.perf_counter()
r = transmission(segs, (-2.0, 2.0), (-0.8, 1.02), 1.55, step=0.02, cell_length=cell_length, num_modes=10)
print(json.dumps({"cell_length_um": cell_length, "segments": n_seg, "cells": r.n_cells, "T00": r.transmission,
                  "P_right": r.total_right, "min_modes_kept": min(r.modes_kept),
                  "seconds": round(time.perf_counter() - t0, 1)}), flush=True)
