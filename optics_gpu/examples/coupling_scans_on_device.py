"""Chip-to-fiber coupling through a lens pair, with a focus scan and a lateral scan fully on the GPU (the device
pipeline of optics_gpu.pop: trace_bundle, image_sum, lateral_scan).

The synthetic system of examples/chip_to_fiber_coupling.py (textbook values, no process data): a SiN tip
0.3 x 0.3 um (n 1.996) in SiO2 (n 1.444) at 1.55 um, its mode imaged with magnification -2.5 (inverted) by two
ideal lenses (f 2 mm and 5 mm, 4f) onto SMF-28 (MFD 10.4 um, w = 5.2 um).

- The rays are traced one time (trace_bundle). Each focus point is one image sum with the plane-wave phase
  k N' dz per ray (image_sum); each lateral point is one overlap with the moved fiber mode (lateral_scan: all
  points of the 2-D scan in two matrix products). Only the efficiencies come back to the host. The efficiencies
  use p0, the power on the image grid at the image surface, for all planes (see the comment at p0); the output
  also gives the power on the grid of each plane and the Parseval power of the kept rays, relative to p0.
- Reference (host): the NumPy pipeline (propagate_through, trace_device None) and the focus scan of
  examples/chip_to_fiber_coupling.py (FFT angular-spectrum propagation of the image field, NumPy), and the
  lateral overlaps with NumPy (overlap_efficiency with the moved mode).

Two runs, because the mode solver (meow-gpu, JAX) and Optiland with PyTorch are in two environments of the GPU
computer. From the repository folder on that computer:

    bash tools/gpu_run.sh optics_gpu ../.venv-rocm/bin/python examples/coupling_scans_on_device.py --save-mode
    bash tools/gpu_run.sh optics_gpu python examples/coupling_scans_on_device.py

The first run writes the mode to results/optics_gpu/sin_tip_mode.npz (results/ is not in git); --mode-file
gives another path.
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
MODE_FILE = Path(__file__).resolve().parents[2] / "results" / "optics_gpu" / "sin_tip_mode.npz"

ap = argparse.ArgumentParser()
ap.add_argument("--save-mode", action="store_true", help="solve the waveguide mode (meow-gpu) and save it")
ap.add_argument("--mode-file", type=Path, default=MODE_FILE)
ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
ap.add_argument("--repeat", type=int, default=3, help="runs of each scan for the time (the best is given)")
args = ap.parse_args()


def solve_mode() -> None:
    """The mode solve of examples/chip_to_fiber_coupling.py (the same grid, window and solver settings), with the
    complex128 Davidson polish of meow-gpu: on the GPU computer (2026-10-30) the default polish ("refine") stopped
    at a residual of 3.2e-10 after 255 iterations without the converged flag, for this weakly guided mode."""
    import jax

    jax.config.update("jax_enable_x64", True)
    from meow_gpu.fde.operator import make_problem
    from meow_gpu.meow_backend import GPUModeSolver

    g = np.arange(-7.0, 7.0 + 1e-9, 0.05)
    xc, xe = 0.5 * (g[1:] + g[:-1]), g[:-1]

    def eps(xx, yy):
        X, Y = np.meshgrid(xx, yy, indexing="ij")
        core = (np.abs(X) < 0.15) & (np.abs(Y) < 0.15)
        return np.where(core, N_CORE**2, N_CLAD**2).astype(complex)

    t0 = time.perf_counter()
    solver = GPUModeSolver(precision="mixed", extra=8, polish="davidson")
    neff, E, _ = solver.solve(make_problem(eps(xc, xe), eps(xe, xc), eps(xe, xe), g, g, WL), 4)
    if not solver.last_info.converged or np.real(neff[0]) <= N_CLAD:
        raise SystemExit(f"the mode solve did not find the guided mode: {solver.last_info}, neff {neff[0]}")
    ex = np.abs(E[0, 0]) if np.abs(E[0, 0]).max() > np.abs(E[0, 1]).max() else np.abs(E[0, 1])
    args.mode_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.mode_file, x=xc, ex=ex, neff=np.real(neff[0]))
    print(json.dumps({"mode_file": args.mode_file.name, "neff": round(float(np.real(neff[0])), 5),
                      "mode_s": round(time.perf_counter() - t0, 2)}))


if args.save_mode:
    solve_mode()
    raise SystemExit(0)

import torch  # noqa: E402
from scipy.interpolate import RegularGridInterpolator  # noqa: E402

from optics_gpu.pop import (gaussian, image_sum, lateral_scan, overlap_efficiency,  # noqa: E402
                            overlap_efficiency_device, propagate_through, trace_bundle)
from optics_gpu.systems import four_f  # noqa: E402

if not args.mode_file.exists():
    raise SystemExit(f"no mode file {args.mode_file}: run this script first with --save-mode in the environment "
                     f"of meow-gpu (see the docstring)")
m = np.load(args.mode_file)
xc, ex = m["x"], m["ex"]

# the source field on a 0.25 um grid, the image grid, the fiber mode (as in examples/chip_to_fiber_coupling.py)
dx = 0.25
xs = np.arange(-48, 48) * dx
Xs, Ys = np.meshgrid(xs, xs, indexing="ij")
interp = RegularGridInterpolator((xc, xc), ex, bounds_error=False, fill_value=0.0)
src = interp(np.stack([Xs.ravel(), Ys.ravel()], -1)).reshape(Xs.shape).astype(complex)
I = np.abs(src) ** 2
w_src = 2 * np.sqrt((I * Xs**2).sum() / I.sum())
di = 0.5
xi = np.arange(-64, 64) * di
optic = four_f()                       # the lens pair of examples/chip_to_fiber_coupling.py
dzs = np.arange(-60, 61, 5.0)
shifts = np.arange(-3.0, 3.0 + 1e-9, 0.25)
dev = torch.device(args.device)


def sync():
    if dev.type == "cuda":
        torch.cuda.synchronize()


def best_of(fn):
    times, out = [], None
    for _ in range(args.repeat):
        sync()
        t0 = time.perf_counter()
        out = fn()
        sync()
        times.append(time.perf_counter() - t0)
    return out, min(times)


fiber = torch.as_tensor(gaussian(xi, xi, W_FIBER), device=dev)
trace_bundle(optic, src, dx, WL, n_pad=4, device=args.device)                 # warm-up (kernels, FFT plans)
bundle, t_bundle = best_of(lambda: trace_bundle(optic, src, dx, WL, n_pad=4, device=args.device))


# The power for all planes, p0: the power on the 64 um grid at the image surface. The power on the grid changes
# with dz (wide-angle power leaves the grid, and power that is off the grid at dz = 0 comes onto it); normalized by
# the power on the grid at each plane, eta at dz = +-60 um was 7.8e-4 higher than the FFT reference, which keeps
# the power of its periodic grid. p0 is not the total power: the output gives the power on the grid of each plane
# and the Parseval power of the kept rays (RayBundle.power), both relative to p0.
E0 = image_sum(bundle, xi, xi)
p0 = torch.vdot(E0.ravel(), E0.ravel()).real


def focus_scan():
    return torch.stack([overlap_efficiency_device(image_sum(bundle, xi, xi, dz=dz), fiber, power=p0)
                        for dz in dzs])


eta_z, t_focus = best_of(focus_scan)
eta_z = eta_z.cpu().numpy()
i_best = int(np.argmax(eta_z))
E_best = image_sum(bundle, xi, xi, dz=float(dzs[i_best]))
lat, t_lat = best_of(lambda: lateral_scan(E_best, xi, xi, W_FIBER, shifts, shifts, power=p0))
lat = lat.cpu().numpy()

# the host reference: NumPy pipeline, FFT focus scan, NumPy overlaps with the moved mode
t0 = time.perf_counter()
ref = propagate_through(optic, src, dx, WL, xi, xi, n_pad=4)
t_ref = time.perf_counter() - t0
fx = np.fft.fftfreq(xi.size, d=di)
FX, FY = np.meshgrid(fx, fx, indexing="ij")
kz = 2 * np.pi * np.sqrt(np.clip(1 / WL**2 - FX**2 - FY**2, 0, None))
spec = np.fft.fft2(ref.E)
fiber_np = gaussian(xi, xi, W_FIBER)
t0 = time.perf_counter()
eta_ref = np.array([overlap_efficiency(np.fft.ifft2(spec * np.exp(1j * kz * dz)), fiber_np) for dz in dzs])
t_ref_focus = time.perf_counter() - t0
E_best_np = E_best.cpu().numpy()
scale = np.vdot(E_best_np, E_best_np).real / float(p0)            # the NumPy overlap uses the power at its plane
X, Y = np.meshgrid(xi, xi, indexing="ij")
picks = [(0, 0), (len(shifts) - 1, len(shifts) // 2), (len(shifts) // 3, 2 * len(shifts) // 3)]
d_lat = max(abs(lat[i, j] - scale * overlap_efficiency(E_best_np, np.exp(-((X - shifts[i]) ** 2
                                                                           + (Y - shifts[j]) ** 2) / W_FIBER**2)))
            for i, j in picks)
i_lat = np.unravel_index(int(np.argmax(lat)), lat.shape)
c = len(shifts) // 2
lat_x = lat[:, c] / lat[c, c]
within_1db = shifts[lat_x >= 10 ** -0.1]


def width_1db(s, v):
    """The full width at 1 dB below the maximum of v(s), with linear interpolation between the scan points."""
    thr, i = 10 ** -0.1 * v.max(), int(np.argmax(v))
    lo, hi = i, i
    while lo > 0 and v[lo - 1] >= thr:
        lo -= 1
    while hi < v.size - 1 and v[hi + 1] >= thr:
        hi += 1
    if lo == 0 or hi == v.size - 1:
        return float("nan")                                            # the scan does not reach 1 dB
    left = s[lo - 1] + (s[lo] - s[lo - 1]) * (thr - v[lo - 1]) / (v[lo] - v[lo - 1])
    right = s[hi] + (s[hi + 1] - s[hi]) * (v[hi] - thr) / (v[hi] - v[hi + 1])
    return float(right - left)


# power on the grid at each plane and the Parseval power of the kept rays, relative to p0 (outside the timed scans)
grid_power = np.array([float(torch.vdot(e.ravel(), e.ravel()).real / p0)
                       for e in (image_sum(bundle, xi, xi, dz=dz) for dz in dzs)])
parseval = float(bundle.power / (di * di) / p0)
Iimg = np.abs(E0.cpu().numpy()) ** 2
Xi, Yi = np.meshgrid(xi, xi, indexing="ij")
centroid = lambda w, x, y: [round(float((w * x).sum() / w.sum()), 4), round(float((w * y).sum() / w.sum()), 4)]  # noqa

print(json.dumps({
    "neff": round(float(m["neff"]), 5), "mode_w_um": round(float(w_src), 3),
    "image_w_expected_um": round(float(w_src * 2.5), 3), "device": str(bundle.device),
    "rays_traced": bundle.n_traced, "rays_kept": bundle.n_rays,
    "trace_and_weights_s": round(t_bundle, 4),
    "focus_points": int(dzs.size), "focus_s_per_point": round(t_focus / dzs.size, 5),
    "lateral_points": int(lat.size), "lateral_s_per_point": round(t_lat / lat.size, 8),
    "eta_at_focus": round(float(eta_z[dzs.size // 2]), 4), "best_dz_um": float(dzs[i_best]),
    "eta_best": round(float(eta_z[i_best]), 4),
    "focus_scan": {float(z): round(float(e), 4) for z, e in zip(dzs, eta_z)},
    "eta_at_focus_parseval_power": round(float(eta_z[dzs.size // 2]) / parseval, 4),
    "parseval_power_over_p0": round(parseval, 6),
    "grid_power_over_p0_min_max": [round(float(grid_power.min()), 6), round(float(grid_power.max()), 6)],
    "lateral_1dB_full_width_um": round(float(within_1db.max() - within_1db.min()), 2),
    "lateral_1dB_full_width_interp_um": round(width_1db(shifts, lat[:, i_lat[1]]), 3),
    "lateral_best_um": [float(shifts[i_lat[0]]), float(shifts[i_lat[1]])],
    "source_centroid_um": centroid(I, Xs, Ys), "image_centroid_um": centroid(Iimg, Xi, Yi),
    "lateral_scan_x_at_best_dz": {float(s): round(float(e), 4) for s, e in zip(shifts, lat[:, c])},
    "max_d_eta_focus_vs_numpy_fft": float(np.max(np.abs(eta_z - eta_ref))),
    "max_d_eta_lateral_vs_numpy": float(d_lat),
    "numpy_pipeline_s": round(t_ref, 4), "numpy_fft_focus_s_per_point": round(t_ref_focus / dzs.size, 5),
}))
