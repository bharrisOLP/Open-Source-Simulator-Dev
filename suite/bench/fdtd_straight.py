"""FDTD check of the suite adapter: a straight Si strip (0.5 x 0.22 um) in SiO2 transmits about 1 into its
fundamental mode (PML on all sides). Prints one JSON line."""
import json
import sys

import jax

from photonsuite import CrossSection, Material, Rect
from photonsuite.fdtd import Segment, transmission

step = float(sys.argv[1]) if len(sys.argv) > 1 else 0.025
SI, SIO2 = Material("Si", n=3.476), Material("SiO2", n=1.444)
xs = CrossSection(SIO2, (Rect(-0.25, 0.25, 0.0, 0.22, SI),))
res = transmission([Segment(0.0, 5.0, xs)], (-1.5, 1.5), (-1.0, 1.2), 1.55, step=step)
print(json.dumps({"step_um": step, "T": res.transmission, "cells": res.cells, "seconds": round(res.seconds, 2),
                  "backend": res.device, "jax_device": str(jax.devices()[0])}))
