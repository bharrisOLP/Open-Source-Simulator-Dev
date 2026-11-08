"""Chip-to-fiber coupling through a lens pair, with a focus scan and a lateral scan fully on the GPU (the device
pipeline of optics_gpu.pop: trace_bundle, image_planes, lateral_scan).

The synthetic system of examples/chip_to_fiber_coupling.py (textbook values, no process data): a SiN tip
0.3 x 0.3 um (n 1.996) in SiO2 (n 1.444) at 1.55 um, its mode imaged with magnification -2.5 (inverted) by two
ideal lenses (f 2 mm and 5 mm, 4f) onto SMF-28 (MFD 10.4 um, w = 5.2 um).

- The rays are traced one time (trace_bundle). The focus scan computes all planes in one pass over the rays, with
  the plane-wave phase k N' dz per ray and plane (image_planes); the output also gives the time and the result of
  one image sum per plane (the method before image_planes), of the complex64 option and of the non-uniform FFT
  (image_planes with method "nufft", the default tol 1e-12; the grid is uniform). Each lateral point is one
  overlap with the moved fiber mode (lateral_scan: all points of the 2-D scan in two matrix products). Only the
  efficiencies come back to the host. The efficiencies use p0, the power on the image grid at the image surface,
  for all planes (see the comment at p0); the output also gives the power on the grid of each plane and the
  Parseval power of the kept rays, relative to p0.
- Reference (host): the NumPy pipeline (propagate_through, trace_device None) and the focus scan of
  examples/chip_to_fiber_coupling.py (FFT angular-spectrum propagation of the image field, NumPy), and the
  lateral overlaps with NumPy (overlap_efficiency with the moved mode).

Two runs, because the mode solver (meow-gpu, JAX) and Optiland with PyTorch are in two environments of the GPU
computer. From the repository folder on that computer:

    bash tools/gpu_run.sh optics_gpu ../.venv-rocm/bin/python examples/coupling_scans_on_device.py --save-mode
    bash tools/gpu_run.sh optics_gpu python examples/coupling_scans_on_device.py

The first run writes the mode to results/optics_gpu/sin_tip_mode.npz (results/ is not in git); --mode-file
gives another path. --pol x (default, the usual TE source) or y sets the polarization of the source: the square
tip has two degenerate fundamental modes, and the solver can give any basis of that pair (either mode first, or
two rotated modes that are not orthogonal), so the first mode is not a defined source
(optics_gpu.sources.polarized_mode selects the field, and the source must have at least MIN_FRACTION of its power
in the chosen component). The mode file keeps the major component of the source (Ex for x, Ey for y) as "ex", on
its own Yee positions ("x", "y"), with its polarization. The x and y sources are transposes of each other, and the
system is rotationally symmetric: x and y give the same efficiencies, and the lateral scan is transposed.

The core is built from integer cell indices (the grid has 0.05 um cells; the core is the 6 x 6 cells of
|x|, |y| <= 0.15 um), so it is symmetric. Each sample gets the mean permittivity of the cell around it: a sample on
a face of the core (the Yee positions on cell corners) gets half core and half cladding, one on a corner of the
core a quarter. The rule matters, because 6 cells resolve the 0.3 um core poorly: with the samples on the faces in
the cladding, neff was 1.44507; in the core, 1.45079 (review of 2026-11-07); with the mean, 1.44719 (development
runs on the GPU computer). Before 2026-11-07 the core test used the coordinates of np.arange: the rounding put the
samples on one face into the core and those on the other face into the cladding (g[143] = 0.1499999999999746 <
0.15, but g[137] = -0.15000000000002434), so the core lost its two mirror symmetries. That split the pair by
2.06e-4 into two diagonal polarizations and moved the mode off the axis (review of 2026-11-07).
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
MODE_FILE = Path(__file__).resolve().parents[2] / "results" / "optics_gpu" / "sin_tip_mode.npz"

ap = argparse.ArgumentParser()
ap.add_argument("--save-mode", action="store_true", help="solve the waveguide mode (meow-gpu) and save it")
ap.add_argument("--mode-file", type=Path, default=MODE_FILE)
ap.add_argument("--polish", choices=("refine", "davidson"), default=None,
                help="the polish of the mode solve (default: the default of meow-gpu, 'refine')")
ap.add_argument("--pol", choices=("x", "y"), default="x", help="the polarization of the source mode (--save-mode)")
ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
ap.add_argument("--repeat", type=int, default=3, help="runs of each scan for the time (the best is given)")
args = ap.parse_args()


def solve_mode() -> None:
    """The mode solve of examples/chip_to_fiber_coupling.py (the same grid, window and solver settings), with the
    default polish of meow-gpu (--polish davidson: the earlier complex128 Davidson polish, for a comparison). The
    output gives the polish that ran (SolveInfo.polish_used: "refine+davidson" when the fallback of meow-gpu
    finished the solve), the residual and the time of the solve (with the JAX compilation of a new process).

    The source is the mode of polarization --pol (polarized_mode, at least MIN_FRACTION of the power in that
    component): if modes 0 and 1 are near-degenerate (|dneff| < DEGENERATE_TOL), the combination of the two with the
    largest power fraction in that component; else the one of the two with the larger fraction. The output gives
    the neff and the x power fraction |Ex|^2 / (|Ex|^2 + |Ey|^2) of the 4 modes, the degeneracy, the overlap of
    modes 0 and 1, the fraction of the source, the centroid of its major component and its largest mirror
    asymmetry (max |a - a mirrored| / max a, about x = 0 and y = 0). The major component is on its Yee positions
    (the positions of its permittivity: Ex at x centers and y corners, Ey at x corners and y centers); the mode
    file keeps them. Before 2026-11-07 the mode file placed it on the cell centers in x and y (a shift of half a
    cell, 0.025 um, along one axis)."""
    import sys

    import jax

    jax.config.update("jax_enable_x64", True)
    from meow_gpu.fde.operator import make_problem
    from meow_gpu.meow_backend import GPUModeSolver

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))     # .venv-rocm has no optics_gpu
    from optics_gpu.sources import DEGENERATE_TOL, polarized_mode

    i = np.arange(-HALF, HALF + 1)
    g = i * H                                                                   # the cell corners (um)
    xc, xe = 0.5 * (g[1:] + g[:-1]), g[:-1]
    ic, ie = i[:-1] + 0.5, i[:-1]                  # the same positions in cells: exact halves and integers

    def eps(a, b):
        """The permittivity at the positions a (x) and b (y), in cells: the mean over the cell around each sample
        (the core is |x|, |y| <= CORE cells), so a sample on a face of the core has the mean of core and cladding,
        and one on a corner of the core a quarter of the core."""
        frac = lambda v: np.clip(CORE + 0.5 - np.abs(v), 0.0, 1.0)  # noqa: E731   (the part of the cell in the core)
        F = frac(a)[:, None] * frac(b)[None, :]
        return (N_CLAD**2 + (N_CORE**2 - N_CLAD**2) * F).astype(complex)

    t0 = time.perf_counter()
    solver = GPUModeSolver(precision="mixed", extra=8, **({} if args.polish is None else {"polish": args.polish}))
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
    ex = np.abs(field[c])
    args.mode_file.parent.mkdir(parents=True, exist_ok=True)
    px, py = (xc, xe) if c == 0 else (xe, xc)                                   # the Yee positions of the component
    np.savez(args.mode_file, x=px, y=py, ex=ex, neff=n_src, pol=args.pol, fraction=sel["fraction"])
    w = ex**2 / (ex**2).sum()
    ax = ex if c == 0 else ex[1:, :]       # mirror in x: Ex on the centers; Ey on the corners without -7.0 um
    ay = ex[:, 1:] if c == 0 else ex       # mirror in y: Ex on the corners without -7.0 um; Ey on the centers
    mirror = max(float(np.abs(ax - ax[::-1, :]).max()), float(np.abs(ay - ay[:, ::-1]).max())) / float(ex.max())
    x_fraction = [float(np.vdot(e[0], e[0]).real / (np.vdot(e[0], e[0]).real + np.vdot(e[1], e[1]).real))
                  for e in E]
    print(json.dumps({"mode_file": args.mode_file.name, "neff": round(n_src, 5), "neff_full": n_src,
                      "pol": args.pol, "neff_modes": [float(np.real(n)) for n in neff],
                      "x_fraction_modes": x_fraction, "dneff_01": sel["dneff"], "degenerate_tol": DEGENERATE_TOL,
                      "degenerate": sel["degenerate"], "source": sel["source"], "overlap_01": sel.get("overlap"),
                      "part_powers": sel.get("part_powers"), "fraction": sel["fraction"],
                      "min_fraction": MIN_FRACTION,
                      "centroid_yee_um": [float((w.sum(1) * px).sum()), float((w.sum(0) * py).sum())],
                      "mirror_asymmetry": mirror,
                      "polish": solver.polish,
                      "polish_used": info.polish_used, "converged": bool(info.converged),
                      "residual": float(info.residual), "iterations": int(info.iterations),
                      "mode_s": round(t_mode, 2)}))


if args.save_mode:
    solve_mode()
    raise SystemExit(0)

import torch  # noqa: E402
from scipy.interpolate import RegularGridInterpolator  # noqa: E402

from optics_gpu.pop import (_image_sum_direct, gaussian, image_planes, image_sum, lateral_scan,  # noqa: E402
                            overlap_efficiency, overlap_efficiency_device, propagate_through, trace_bundle)
from optics_gpu.systems import four_f  # noqa: E402

if not args.mode_file.exists():
    raise SystemExit(f"no mode file {args.mode_file}: run this script first with --save-mode in the environment "
                     f"of meow-gpu (see the docstring)")
m = np.load(args.mode_file)
xc, ex = m["x"], m["ex"]
yc = m["y"] if "y" in m else xc                    # a mode file before 2026-11-07: the cell centers on both axes

# the source field on a 0.25 um grid, the image grid, the fiber mode (as in examples/chip_to_fiber_coupling.py)
dx = 0.25
xs = np.arange(-48, 48) * dx
Xs, Ys = np.meshgrid(xs, xs, indexing="ij")
interp = RegularGridInterpolator((xc, yc), ex, bounds_error=False, fill_value=0.0)
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


def focus_scan(dtype=torch.complex128, method="direct"):
    """All planes in one pass over the rays (image_planes), then one eta per plane."""
    return overlap_efficiency_device(image_planes(bundle, xi, xi, dzs, dtype=dtype, method=method), fiber, power=p0)


def focus_scan_per_plane():
    """One image sum per plane: the method before image_planes (one matrix product per chunk of rays)."""
    return torch.stack([overlap_efficiency_device(_image_sum_direct(bundle, xi, xi, dz=dz), fiber, power=p0)
                        for dz in dzs])


focus_scan(), focus_scan(torch.complex64), focus_scan_per_plane(), focus_scan(method="nufft")   # warm-up
eta_z, t_focus = best_of(focus_scan)
eta_pp, t_pp = best_of(focus_scan_per_plane)
eta_32, t_32 = best_of(lambda: focus_scan(torch.complex64))
eta_nu, t_nu = best_of(lambda: focus_scan(method="nufft"))
eta_z, eta_pp, eta_32, eta_nu = (v.cpu().numpy() for v in (eta_z, eta_pp, eta_32, eta_nu))
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
stack = image_planes(bundle, xi, xi, dzs)
grid_power = ((stack.real**2 + stack.imag**2).sum((1, 2)) / p0).cpu().numpy()
parseval = float(bundle.power / (di * di) / p0)
Iimg = np.abs(E0.cpu().numpy()) ** 2
Xi, Yi = np.meshgrid(xi, xi, indexing="ij")
centroid = lambda w, x, y: [round(float((w * x).sum() / w.sum()), 4), round(float((w * y).sum() / w.sum()), 4)]  # noqa

print(json.dumps({
    "mode_file": args.mode_file.name, "neff": round(float(m["neff"]), 5),
    "pol": str(m["pol"]) if "pol" in m else None,
    "fraction": float(m["fraction"]) if "fraction" in m else None, "mode_w_um": round(float(w_src), 3),
    "image_w_expected_um": round(float(w_src * 2.5), 3), "device": str(bundle.device),
    "rays_traced": bundle.n_traced, "rays_kept": bundle.n_rays,
    "trace_and_weights_s": round(t_bundle, 4),
    "focus_points": int(dzs.size), "focus_s_per_point": round(t_focus / dzs.size, 5),
    "focus_per_plane_s_per_point": round(t_pp / dzs.size, 5), "focus_c64_s_per_point": round(t_32 / dzs.size, 6),
    "max_d_eta_focus_batched_vs_per_plane": float(np.max(np.abs(eta_z - eta_pp))),
    "max_d_eta_focus_c64_vs_c128": float(np.max(np.abs(eta_32 - eta_z))),
    "focus_nufft_s_per_point": round(t_nu / dzs.size, 6),
    "max_d_eta_focus_nufft_vs_c128": float(np.max(np.abs(eta_nu - eta_z))),
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
