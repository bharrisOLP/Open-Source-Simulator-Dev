"""Time of the image sum: the direct sum (image_planes, complex128 and complex64) against the non-uniform FFT
(image_planes with method "nufft") for --rays rays, --image x --image grids (pitch --pitch um) and --planes planes.
One JSON line per case (ray count, grid and plane count).

The rays: a point source through the aberrated singlet (default) or the 2.5x lens pair (--system 4f), traced one
time on the GPU (trace_bundle), as in bench/bench_image_sum.py (|L|, |M| <= --na, about --rays rays in the box of
the kept directions). The planes: --planes values of dz, evenly from -span/2 to +span/2 (--span, um).

- direct_c128_s, direct_c64_s: the time per plane of image_planes (best of --repeat runs after one untimed run).
  The direct sum costs rays x Nx x Ny x planes; above --direct-max of that product it is timed on every s-th ray
  of the bundle (direct_rays_timed rays, the smallest s that keeps the product at most --direct-max) and scaled to
  all rays (direct_scaled true: the direct sum is linear in the rays, chunk by chunk). complex64 is null if its
  grid-phase limit (PHASE32_MAX) refuses the grid.
- nufft_s_<tol>: the time per plane of the non-uniform FFT on all rays, for each --tol.
- speedup_c128_<tol>, speedup_c64_<tol>: direct time over the time of the non-uniform FFT.
- err_<tol>: max |E_nufft - E_direct| / sum_r |a_r|, and d_eta_<tol>: the largest difference of the coupling into
  a Gaussian mode of radius --mode-w um (normalized by the power on the grid of each plane), both on the rays of the
  direct timing (the same rays for the two methods).
- nufft32_s_<tol>, err32_<tol>, d_eta32_<tol> (for each --tol32): the same for the non-uniform FFT with
  dtype=torch.complex64 (the float32 spread), err32 against the direct complex128 sum; bound32_<tol>: the largest
  value of nufft32_bound + tol on the grid (the stated bound of err32); spread32_gain_<tol>: the complex128
  non-uniform FFT at the same tol (if it is in --tol) over the float32 one; speedup32_c64_<tol>: direct complex64
  over the float32 non-uniform FFT. nufft32_s_<tol>_terms<n>, err32_<tol>_terms<n> (for each --terms32 n): the
  float32 spread with NUFFT_TERMS32 = n (patched; a larger n: fewer flushes, a larger bound).
- auto_c128, auto_c64: the method that image_planes(method="auto") selects for the case (dtype complex128 with
  tol 1e-12, complex64 with tol 1e-6: the defaults; "auto" uses the complex128 spread for both); auto_c128_loss,
  auto_c64_loss: the time of that method (direct_c128_s or nufft_s_1e-12; direct_c64_s or nufft_s_1e-06) over the
  time of the faster of the two (1 if "auto" selects the faster one; null if a time is missing).

On the GPU computer, from the repository folder (the full run for the log):

    bash tools/gpu_run.sh optics_gpu python bench/bench_nufft.py --rays 1e5 1e6 1e7 \
        --image 128 256 512 1024 --planes 1 21

The rows of the rule of method "auto" (benchmarks/2026-11-08/optics-nufft-auto.jsonl):

    bash tools/gpu_run.sh optics_gpu python bench/bench_nufft.py --rays 1e3 1e4 1e5 1e6 \
        --image 16 32 64 128 256 512 1024 --planes 1 5 21 --tol 1e-12 1e-6 --terms32 16384
    bash tools/gpu_run.sh optics_gpu python bench/bench_nufft.py --system 4f --rays 1e3 1e4 1e5 1e6 \
        --image 32 64 128 256 512 --planes 1 5 21 --tol 1e-12 1e-6
"""
import argparse
import contextlib
import json
import time
import warnings

import numpy as np
import torch

from optics_gpu import pop
from optics_gpu.pop import (NUFFT_TOL, NUFFT_TOL32, RayBundle, _auto_nufft, gaussian, image_planes, nufft32_bound,
                            nufft_width, overlap_efficiency_device, trace_bundle)
from optics_gpu.systems import aberrated_singlet, four_f

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
WL, N_PAD = 1.55, 4

ap = argparse.ArgumentParser()
ap.add_argument("--system", choices=("singlet", "4f"), default="singlet")
ap.add_argument("--rays", type=float, nargs="+", default=[1e5, 1e6])
ap.add_argument("--image", type=int, nargs="+", default=[128, 512], help="image grid points per axis")
ap.add_argument("--pitch", type=float, default=1.0, help="image grid pitch (um)")
ap.add_argument("--planes", type=int, nargs="+", default=[1, 21])
ap.add_argument("--span", type=float, default=200.0, help="dz range of the planes (um)")
ap.add_argument("--tol", type=float, nargs="+", default=[1e-12, 1e-9, 1e-6])
ap.add_argument("--tol32", type=float, nargs="+", default=[1e-6], help="tol of the complex64 non-uniform FFT")
ap.add_argument("--terms32", type=int, nargs="*", default=[],
                help="other values of NUFFT_TERMS32 (the float32 terms per value between flushes) to time")
ap.add_argument("--direct-max", type=float, default=2e11, help="rays x Nx x Ny x planes timed with the direct sum")
ap.add_argument("--repeat", type=int, default=2)
ap.add_argument("--na", type=float, default=0.09)
ap.add_argument("--mode-w", type=float, default=8.0, help="radius of the Gaussian mode of the efficiency (um)")
a = ap.parse_args()
if not torch.cuda.is_available():
    raise SystemExit("no GPU: this bench times the GPU image sum")
optic = (aberrated_singlet if a.system == "singlet" else four_f)()
dx = WL / (2.1 * a.na)


def point(n):
    src = np.zeros((n, n), complex)
    src[n // 2, n // 2] = 1.0
    return src


def timed(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - t0


def best(fn):
    out, _ = timed(fn)                                                  # untimed run
    times = []
    for _ in range(a.repeat):
        out = None
        out, t = timed(fn)
        times.append(t)
    return out, min(times)


@contextlib.contextmanager
def patched(terms):
    """pop.NUFFT_TERMS32 = terms inside the block."""
    old, pop.NUFFT_TERMS32 = pop.NUFFT_TERMS32, terms
    try:
        yield
    finally:
        pop.NUFFT_TERMS32 = old


def every(bundle, s):
    """Every s-th ray of each block of the bundle."""
    blocks = [tuple(v[::s] for v in b) for b in bundle.blocks]
    n = sum(b[3].numel() for b in blocks)
    return RayBundle(blocks, bundle.wavelength, n, n, bundle.device, bundle.power)


env = {"torch": torch.__version__, "hip": torch.version.hip, "gpu": torch.cuda.get_device_name(0),
       "system": a.system, "pitch_um": a.pitch, "na": a.na, "mode_w_um": a.mode_w}
kw = {"na_cut": a.na, "amp_cut": 0.0, "n_pad": N_PAD, "device": "cuda"}
warm = trace_bundle(optic, point(16), dx, WL, **kw)                     # warm-up (contexts, kernels)
image_planes(warm, np.arange(8.0), np.arange(8.0), [0.0, 1.0], method="nufft")
image_planes(warm, np.arange(8.0), np.arange(8.0), [0.0, 1.0], method="nufft", dtype=torch.complex64)
del warm


def case(bundle, nimg, nz):
    xi = (np.arange(nimg) - nimg // 2) * a.pitch
    dz = np.linspace(-a.span / 2, a.span / 2, nz) if nz > 1 else np.zeros(1)
    mode = torch.as_tensor(gaussian(xi, xi, a.mode_w), device="cuda")
    s = max(1, int(np.ceil(bundle.n_rays * nimg * nimg * nz / a.direct_max)))
    sub = every(bundle, s) if s > 1 else bundle
    scale = bundle.n_rays / sub.n_rays
    e128, t128 = best(lambda: image_planes(sub, xi, xi, dz))
    try:
        _, t64 = best(lambda: image_planes(sub, xi, xi, dz, dtype=torch.complex64))
    except ValueError:                                                  # the grid-phase limit of complex64
        t64 = None
    row = {"rays_traced": bundle.n_traced, "rays_kept": bundle.n_rays, "image": nimg, "planes": nz,
           "span_um": a.span if nz > 1 else 0.0, "direct_rays_timed": sub.n_rays, "direct_scaled": s > 1,
           "direct_c128_s": t128 * scale / nz, "direct_c64_s": None if t64 is None else t64 * scale / nz}
    s1 = float(sum(b[3].abs().sum() for b in sub.blocks))
    eta_ref = overlap_efficiency_device(e128, mode)
    for tol in a.tol:
        _, t = best(lambda: image_planes(bundle, xi, xi, dz, method="nufft", tol=tol))  # noqa: B023
        e = image_planes(sub, xi, xi, dz, method="nufft", tol=tol)
        row |= {f"nufft_s_{tol:g}": t / nz, f"w_{tol:g}": nufft_width(tol)[0],
                f"speedup_c128_{tol:g}": round(row["direct_c128_s"] / (t / nz), 2),
                f"speedup_c64_{tol:g}": None if t64 is None else round(row["direct_c64_s"] / (t / nz), 2),
                f"err_{tol:g}": float((e - e128).abs().max()) / s1,
                f"d_eta_{tol:g}": float((overlap_efficiency_device(e, mode) - eta_ref).abs().max())}
    c64, t64_direct = {"method": "nufft", "dtype": torch.complex64}, row["direct_c64_s"]
    for tol in a.tol32:
        _, t = best(lambda: image_planes(bundle, xi, xi, dz, tol=tol, **c64))  # noqa: B023
        e = image_planes(sub, xi, xi, dz, tol=tol, **c64)
        t64, t32 = row.get(f"nufft_s_{tol:g}"), t / nz
        row |= {f"nufft32_s_{tol:g}": t32, f"err32_{tol:g}": float((e - e128).abs().max()) / s1,
                f"bound32_{tol:g}": float(nufft32_bound(nimg, nimg, tol).max()) + tol,
                f"d_eta32_{tol:g}": float((overlap_efficiency_device(e, mode) - eta_ref).abs().max()),
                f"spread32_gain_{tol:g}": None if t64 is None else round(t64 / t32, 2),
                f"speedup32_c64_{tol:g}": None if t64_direct is None else round(t64_direct / t32, 2)}
        for terms in a.terms32:                                         # the float32 spread with another limit
            with patched(terms):
                _, t = best(lambda: image_planes(bundle, xi, xi, dz, tol=tol, **c64))  # noqa: B023
                e = image_planes(sub, xi, xi, dz, tol=tol, **c64)
            row |= {f"nufft32_s_{tol:g}_terms{terms}": t / nz,
                    f"err32_{tol:g}_terms{terms}": float((e - e128).abs().max()) / s1}
    for name, dt, tol in (("c128", torch.complex128, NUFFT_TOL), ("c64", torch.complex64, NUFFT_TOL32)):
        nufft = _auto_nufft(bundle.n_rays, nz, nimg, nimg, dt)
        t_dir = row[f"direct_{name}_s"]
        t_nu = row.get(f"nufft_s_{tol:g}")                             # "auto": the complex128 spread
        loss = None
        if t_dir is not None and t_nu is not None:
            loss = round((t_nu if nufft else t_dir) / min(t_nu, t_dir), 2)
        row |= {f"auto_{name}": "nufft" if nufft else "direct", f"auto_{name}_loss": loss}
    return row


for n in a.rays:
    src = point(max(4, int(round(1.05 * np.sqrt(n) / N_PAD))))
    trace_bundle(optic, src, dx, WL, **kw)                              # the first FFT of a new size (rocFFT)
    bundle, t_bundle = timed(lambda: trace_bundle(optic, src, dx, WL, **kw))  # noqa: B023
    for nimg in a.image:
        for nz in a.planes:
            print(json.dumps(case(bundle, nimg, nz) | {"trace_bundle_s": round(t_bundle, 4)} | env), flush=True)
    bundle = None
