"""Where the time of an FDTD run goes, with a Rect, a rectangle Polygon and a sloped Polygon: a straight Si strip,
3 um, on a step grid. Prints one JSON line per case: setup (build, place), run (time loop), overlaps (mode overlaps
and fluxes of the detectors).

    python bench/fdtd_polygon_speed.py [step um]
"""
import json
import sys
import time

import fdtdx
import jax

from photonsuite import CrossSection, Material, Polygon, Rect
from photonsuite.fdtd import Segment, _setup

step = float(sys.argv[1]) if len(sys.argv) > 1 else 0.04
SI, SIO2 = Material("Si", n=3.476), Material("SiO2", n=1.444)
shapes = {"rect": Rect(-0.25, 0.25, 0.0, 0.22, SI),
          "rect polygon": Polygon(((-0.25, 0.0), (0.25, 0.0), (0.25, 0.22), (-0.25, 0.22)), SI),
          "sloped polygon": Polygon(((-0.25, 0.0), (0.25, 0.0), (0.211, 0.22), (-0.211, 0.22)), SI)}
for name, shape in shapes.items():
    with jax.enable_x64(False):
        t0 = time.perf_counter()
        objs, arrays, config, key, backend, cells = _setup([Segment(0.0, 3.0, CrossSection(SIO2, (shape,)))],
                                                           (-1.2, 1.2), (-0.8, 1.0), 1.55, step)
        jax.block_until_ready(arrays.inv_permittivities)
        t1 = time.perf_counter()
        _, arrays = fdtdx.run_fdtd(arrays, objs, config, key)
        jax.block_until_ready(arrays)
        t2 = time.perf_counter()
        a = complex(objs["out"].compute_overlap(arrays.detector_states["out"])[0])
        t3 = time.perf_counter()
    print(json.dumps({"shape": name, "step_um": step, "cells": cells, "setup_s": round(t1 - t0, 2),
                      "run_s": round(t2 - t1, 2), "overlap_s": round(t3 - t2, 2), "abs_a_out": abs(a)}), flush=True)
