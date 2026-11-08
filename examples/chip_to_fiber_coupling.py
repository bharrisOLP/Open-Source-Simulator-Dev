"""Chip-to-fiber coupling through a lens pair: waveguide mode (meow-gpu) -> lenses (Optiland rays + optics-gpu
physical optics) -> SMF-28 overlap, with a focus scan.

Synthetic case (textbook values, no process data): a SiN tip 0.3 x 0.3 um (n 1.996) in SiO2 (n 1.444) at 1.55 um,
whose large mode is imaged with magnification -2.5 (inverted) by two ideal lenses (f 2 mm and 5 mm, 4f) onto
SMF-28 (MFD 10.4 um, w = 5.2 um). The focus scan moves the fiber along z by angular-spectrum propagation of the
image field.

The mode solver (meow-gpu, JAX) and Optiland with PyTorch are in two environments of the GPU computer, so the
example has two runs. From the repository folder on that computer:

    bash tools/gpu_run.sh optics_gpu ../.venv-rocm/bin/python ../examples/chip_to_fiber_coupling.py --save-mode
    bash tools/gpu_run.sh optics_gpu python ../examples/chip_to_fiber_coupling.py

The first run writes the mode to results/chip_to_fiber/sin_tip_mode.npz (results/ is not in git); --mode-file
gives another path for both runs.

The source mode (the same rules as optics_gpu/examples/coupling_scans_on_device.py):
- The core is built from integer cell indices (0.05 um cells; the core is the 6 x 6 cells of |x|, |y| <= 0.15
  um), so it is symmetric. Each permittivity sample gets the mean over the cell around it: a sample on a face of
  the core gets half core and half cladding, one on a corner of the core a quarter. Before 2026-11-07 the core
  test used the coordinates of np.arange with |x| < 0.15: the rounding put the samples on one face into the core
  and those on the other face into the cladding, so the core lost its mirror symmetries.
- The square core has two degenerate fundamental modes, and the solver can give any basis of that pair, so
  "mode 0" is not a defined source. --pol x (default, the usual TE source) or y: optics_gpu.sources.polarized_mode
  takes the combination of modes 0 and 1 with the most power in that component, and the source must have at least
  MIN_FRACTION of its power there. Before 2026-11-07 the example took the component with the larger maximum of
  mode 0.
"""
import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
WL, N_CORE, N_CLAD, W_FIBER = 1.55, 1.996, 1.444, 5.2
H, HALF, CORE = 0.05, 140, 3      # cell size (um), the window +-HALF cells (+-7 um), the core +-CORE cells (+-0.15 um)
MIN_FRACTION = 0.9                # the least power fraction of the source in the chosen component
MODE_FILE = Path(__file__).resolve().parents[1] / "results" / "chip_to_fiber" / "sin_tip_mode.npz"

ap = argparse.ArgumentParser()
ap.add_argument("--save-mode", action="store_true", help="solve the waveguide mode (meow-gpu) and save it")
ap.add_argument("--mode-file", type=Path, default=MODE_FILE)
ap.add_argument("--pol", choices=("x", "y"), default="x", help="the polarization of the source mode (--save-mode)")
args = ap.parse_args()


def solve_mode() -> None:
    """1. The waveguide mode (um), on a 50 nm grid in a 14 x 14 um window (the mode is weakly guided: an 8 um
    window cut its tails and changed eta by about 4 %; the one review found it). The output gives the neff and the
    x power fraction |Ex|^2 / (|Ex|^2 + |Ey|^2) of the 4 modes, the degeneracy and the overlap of modes 0 and 1,
    and the fraction of the source. The mode file keeps the major component of the source (Ex for x, Ey for y) on
    its Yee positions (the positions of its permittivity)."""
    import sys

    import jax

    jax.config.update("jax_enable_x64", True)
    from meow_gpu.fde.operator import make_problem
    from meow_gpu.meow_backend import GPUModeSolver

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "optics_gpu" / "src"))   # .venv-rocm: no optics_gpu
    from optics_gpu.sources import DEGENERATE_TOL, polarized_mode

    i = np.arange(-HALF, HALF + 1)
    g = i * H                                                                   # the cell corners (um)
    xc, xe = 0.5 * (g[1:] + g[:-1]), g[:-1]
    ic, ie = i[:-1] + 0.5, i[:-1]                  # the same positions in cells: exact halves and integers

    def eps(a, b):
        """The permittivity at the positions a (x) and b (y), in cells: the mean over the cell around each sample
        (the core is |x|, |y| <= CORE cells)."""
        frac = lambda v: np.clip(CORE + 0.5 - np.abs(v), 0.0, 1.0)  # noqa: E731   (the part of the cell in the core)
        F = frac(a)[:, None] * frac(b)[None, :]
        return (N_CLAD**2 + (N_CORE**2 - N_CLAD**2) * F).astype(complex)

    t0 = time.perf_counter()
    # the weakly guided mode sits just above a dense band of window (box) modes: a larger block is needed
    solver = GPUModeSolver(precision="mixed", extra=8)
    neff, E, _ = solver.solve(make_problem(eps(ic, ie), eps(ie, ic), eps(ie, ie), g, g, WL), 4)
    t_mode = time.perf_counter() - t0
    info = solver.last_info
    if not info.converged:
        raise SystemExit(f"the mode solve did not converge: {info}, neff {neff}")
    E = np.asarray(E)
    field, sel = polarized_mode(neff[:2], E[:2], args.pol, min_fraction=MIN_FRACTION)
    n_src = float(np.real(sel["neff"]))
    if n_src <= N_CLAD:
        raise SystemExit(f"the source is not a guided mode: neff {neff}, source {sel}")
    c = 0 if args.pol == "x" else 1
    px, py = (xc, xe) if c == 0 else (xe, xc)                                   # the Yee positions of the component
    args.mode_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.mode_file, x=px, y=py, ex=np.abs(field[c]), neff=n_src, pol=args.pol, fraction=sel["fraction"])
    x_fraction = [float(np.vdot(e[0], e[0]).real / (np.vdot(e[0], e[0]).real + np.vdot(e[1], e[1]).real))
                  for e in E]
    print(json.dumps({"mode_file": args.mode_file.name, "neff": round(n_src, 5), "neff_full": n_src,
                      "pol": args.pol, "neff_modes": [float(np.real(n)) for n in neff],
                      "x_fraction_modes": x_fraction, "dneff_01": sel["dneff"], "degenerate_tol": DEGENERATE_TOL,
                      "degenerate": sel["degenerate"], "source": sel["source"], "overlap_01": sel.get("overlap"),
                      "part_powers": sel.get("part_powers"), "fraction": sel["fraction"],
                      "min_fraction": MIN_FRACTION, "polish_used": info.polish_used,
                      "residual": float(info.residual), "iterations": int(info.iterations),
                      "mode_s": round(t_mode, 2)}))


if args.save_mode:
    solve_mode()
    raise SystemExit(0)

from optiland.optic import Optic  # noqa: E402
from scipy.interpolate import RegularGridInterpolator  # noqa: E402

from optics_gpu.pop import gaussian, overlap_efficiency, propagate_through  # noqa: E402

if not args.mode_file.exists():
    raise SystemExit(f"no mode file {args.mode_file}: run this script first with --save-mode in the environment "
                     f"of meow-gpu (see the docstring)")
m = np.load(args.mode_file)

# 2. the source field on a 0.25 um grid (the major component, interpolated from its Yee positions)
dx = 0.25
xs = np.arange(-48, 48) * dx
interp = RegularGridInterpolator((m["x"], m["y"]), m["ex"], bounds_error=False, fill_value=0.0)
Xs, Ys = np.meshgrid(xs, xs, indexing="ij")
src = interp(np.stack([Xs.ravel(), Ys.ravel()], -1)).reshape(Xs.shape).astype(complex)
I = np.abs(src) ** 2
w_src = [2 * np.sqrt((I * C**2).sum() / I.sum()) for C in (Xs, Ys)]   # 2 sqrt(<x^2>), 2 sqrt(<y^2>)

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
print(json.dumps({"mode_file": args.mode_file.name, "neff": round(float(m["neff"]), 5), "pol": str(m["pol"]),
                  "fraction": float(m["fraction"]), "mode_w_xy_um": [round(float(w), 3) for w in w_src],
                  "image_w_expected_xy_um": [round(float(w * 2.5), 3) for w in w_src], "rays": img.n_rays,
                  "pop_device": img.device,
                  "pop_s": round(t_pop, 3), "eta_at_focus": scan[0.0], "best_dz_um": best, "eta_best": scan[best],
                  "focus_scan": scan}))
