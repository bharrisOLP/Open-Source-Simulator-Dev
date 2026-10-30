"""Cross-method check: a linear Si taper 0.5 -> 2.0 um over 10 um (220 nm thick, SiO2), 1 um straight input and
output, at 1.55 um. 3-D FDTD (fdtdx, staircase of n_seg segments) gives the fundamental-mode transmission as the
ratio of two mode overlaps (biased when the input and output modes differ: use fdtd_taper_calibrated.py). The EME of
the same taper is meow_gpu/bench/bench_eme_taper.py. Prints one JSON line."""
import json
import sys

from photonsuite import CrossSection, Material, Rect
from photonsuite.fdtd import Segment, transmission

step = float(sys.argv[1]) if len(sys.argv) > 1 else 0.025
n_seg = int(sys.argv[2]) if len(sys.argv) > 2 else 50
SI, SIO2 = Material("Si", n=3.476), Material("SiO2", n=1.444)


def strip(w):
    return CrossSection(SIO2, (Rect(-w / 2, w / 2, 0.0, 0.22, SI),))


segs = [Segment(0.0, 1.0, strip(0.5))]
for k in range(n_seg):
    z0, z1 = 1.0 + 10.0 * k / n_seg, 1.0 + 10.0 * (k + 1) / n_seg
    segs.append(Segment(z0, z1, strip(0.5 + 1.5 * (k + 0.5) / n_seg)))
segs.append(Segment(11.0, 12.0, strip(2.0)))
res = transmission(segs, (-2.0, 2.0), (-0.8, 1.02), 1.55, step=step)
print(json.dumps({"step_um": step, "segments": n_seg, "T_fdtd": res.transmission, "cells": res.cells,
                  "seconds": round(res.seconds, 2), "backend": res.device}))
