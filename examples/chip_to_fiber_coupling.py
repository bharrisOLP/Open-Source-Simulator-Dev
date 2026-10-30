"""Chip-to-fiber coupling through a lens pair: waveguide mode (meow-gpu) -> lenses (Optiland rays + optics-gpu
physical optics) -> SMF-28 overlap, with a focus scan.

Synthetic case (textbook values, no process data): a SiN tip 0.3 x 0.3 um (n 1.996) in SiO2 (n 1.444) at 1.55 um,
whose large mode is imaged with magnification 2.5 by two ideal lenses (f 2 mm and 5 mm, 4f) onto SMF-28
(MFD 10.4 um, w = 5.2 um). The focus scan moves the fiber along z by angular-spectrum propagation of the image field.

    uv run --project optics_gpu --with-editable ./meow_gpu python examples/chip_to_fiber_coupling.py
"""
import json
import time
import warnings

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np  # noqa: E402
from optiland.optic import Optic  # noqa: E402
from scipy.interpolate import RegularGridInterpolator  # noqa: E402

from meow_gpu.fde.operator import make_problem  # noqa: E402
from meow_gpu.meow_backend import GPUModeSolver  # noqa: E402
from optics_gpu.pop import gaussian, overlap_efficiency, propagate_through  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
WL, N_CORE, N_CLAD, W_FIBER = 1.55, 1.996, 1.444, 5.2

# 1. the waveguide mode (um), on a 50 nm grid in a 14 x 14 um window (the mode is weakly guided: an 8 um window
#    cut its tails and changed eta by about 4 %; the one review found it)
g = np.arange(-7.0, 7.0 + 1e-9, 0.05)
xc, xe = 0.5 * (g[1:] + g[:-1]), g[:-1]


def eps(xx, yy):
    X, Y = np.meshgrid(xx, yy, indexing="ij")
    core = (np.abs(X) < 0.15) & (np.abs(Y) < 0.15)
    return np.where(core, N_CORE**2, N_CLAD**2).astype(complex)


t0 = time.perf_counter()
# the weakly guided mode sits just above a dense band of window (box) modes: a larger block is needed
mode_solver = GPUModeSolver(precision="mixed", extra=8)
neff, E, _ = mode_solver.solve(make_problem(eps(xc, xe), eps(xe, xc), eps(xe, xe), g, g, WL), 4)
if not mode_solver.last_info.converged or np.real(neff[0]) <= N_CLAD:
    raise SystemExit(f"the mode solve did not find the guided mode: {mode_solver.last_info}, neff {neff[0]}")
ex = np.abs(E[0, 0]) if np.abs(E[0, 0]).max() > np.abs(E[0, 1]).max() else np.abs(E[0, 1])   # the major component
t_mode = time.perf_counter() - t0

# 2. the source field on a 0.25 um grid
dx = 0.25
xs = np.arange(-48, 48) * dx
interp = RegularGridInterpolator((xc, xc), ex, bounds_error=False, fill_value=0.0)
Xs, Ys = np.meshgrid(xs, xs, indexing="ij")
src = interp(np.stack([Xs.ravel(), Ys.ravel()], -1)).reshape(Xs.shape).astype(complex)
I = np.abs(src) ** 2
w_src = 2 * np.sqrt((I * Xs**2).sum() / I.sum())

# 3. the lens pair, and the image field
o = Optic()
o.add_surface(index=0, thickness=2.0)
o.add_surface(index=1, surface_type="paraxial", f=2.0, thickness=7.0, is_stop=True)
o.add_surface(index=2, surface_type="paraxial", f=5.0, thickness=5.0)
o.add_surface(index=3)
o.set_aperture(aperture_type="EPD", value=3.0)
o.add_wavelength(value=WL, is_primary=True)
di = 0.5
xi = np.arange(-64, 64) * di
t0 = time.perf_counter()
img = propagate_through(o, src, dx, WL, xi, xi, n_pad=4)
t_pop = time.perf_counter() - t0

# 4. the focus scan: angular-spectrum propagation of the image field by dz (um)
fx = np.fft.fftfreq(xi.size, d=di)
FX, FY = np.meshgrid(fx, fx, indexing="ij")
kz = 2 * np.pi * np.sqrt(np.clip(1 / WL**2 - FX**2 - FY**2, 0, None))
spec = np.fft.fft2(img.E)
fiber = gaussian(xi, xi, W_FIBER)
scan = {}
for dz in np.arange(-60, 61, 10.0):
    scan[float(dz)] = round(overlap_efficiency(np.fft.ifft2(spec * np.exp(1j * kz * dz)), fiber), 4)
best = max(scan, key=scan.get)
print(json.dumps({"neff": round(float(np.real(neff[0])), 5), "mode_w_um": round(float(w_src), 3),
                  "image_w_expected_um": round(float(w_src * 2.5), 3), "rays": img.n_rays, "pop_device": img.device,
                  "mode_s": round(t_mode, 2), "pop_s": round(t_pop, 3), "eta_at_focus": scan[0.0],
                  "best_dz_um": best, "eta_best": scan[best], "focus_scan": scan}))
