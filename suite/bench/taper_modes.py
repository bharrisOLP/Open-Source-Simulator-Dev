"""The TE0 effective index of the input and output cross-sections of bench/eme_fdtd_taper.py by the EME mode solver
(meow + meow-gpu, the meow subpixel smoothing) on transverse grids `steps`. The FDTD rows of eme_fdtd_taper.py give
the same index from the fdtdx mode solver (neff_in, neff_out). Prints one JSON line per grid.

    python bench/taper_modes.py [steps, default 0.04,0.02,0.01] [wavelength, default 1.55]
"""
import json
import sys

import numpy as np

from photonsuite import CrossSection, Material, Rect

SI, SIO2 = Material("Si", n=3.476), Material("SiO2", n=1.444)
WINDOW = ((-2.0, 2.0), (-0.8, 1.02))
steps = [float(v) for v in (sys.argv[1] if len(sys.argv) > 1 else "0.04,0.02,0.01").split(",")]
wl = float(sys.argv[2]) if len(sys.argv) > 2 else 1.55


def te0(w, step):
    import meow as mw

    from meow_gpu.meow_backend import GPUModeSolver, compute_modes_gpu
    from photonsuite.eme import structures
    from photonsuite.fdtd import Segment

    st = structures([Segment(0.0, 1.0, CrossSection(SIO2, (Rect(-w / 2, w / 2, 0.0, 0.22, SI),)))], *WINDOW)
    mesh = mw.Mesh2D(x=np.arange(WINDOW[0][0], WINDOW[0][1] + step / 2, step),
                     y=np.arange(WINDOW[1][0], WINDOW[1][1] + step / 2, step))
    cs = mw.CrossSection.from_cell(cell=mw.create_cells(st, mesh, np.array([1.0]))[0],
                                   env=mw.Environment(wl=wl, T=25.0))
    modes = compute_modes_gpu(cs, num_modes=4, solver=GPUModeSolver(precision="double"))
    return [float(np.real(m.neff)) for m in modes]


for step in steps:
    print(json.dumps({"wavelength": wl, "step_um": step, "neff_w0.5": te0(0.5, step), "neff_w2.0": te0(2.0, step)}),
          flush=True)
