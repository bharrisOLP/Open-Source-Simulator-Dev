"""Time of a focus scan with the image sum of the device pipeline: one image sum per plane (the method before
image_planes: one matrix product per chunk of rays, complex128) against image_planes (all planes in one pass over
the rays, groups of GROUP_RAYS rays in one batched matrix product) in complex128 and in complex64. One JSON line per
case (ray count and plane count).

The rays: a point source through the aberrated singlet (default) or the 2.5x lens pair (--system 4f), traced one
time on the GPU (trace_bundle), as in bench/bench_pipeline.py (|L|, |M| <= --na, about --rays rays in the box of the
kept directions). The planes: --planes values of dz, evenly from -span/2 to +span/2 (--span, um); the image grid
--image x --image points, pitch 1 um.

- per_plane_s: the time per plane of the per-plane method (pop._image_sum_direct), over the first --ref-planes
  planes (one run after one untimed plane).
- single_s: the time per plane of image_sum (image_planes with one plane, complex128) over the same planes.
- batched_c128_s, batched_c64_s: the time of image_planes for all planes (best of --repeat runs after one untimed
  run), divided by the plane count.
- max_rel_dE_*: the largest field difference from the per-plane method (relative to the largest |E|) over the
  reference planes; max_d_eta_*: the largest difference of the coupling efficiency into a Gaussian mode of radius
  --mode-w um (normalized by the power on the grid of each plane) over the reference planes;
  max_d_eta_c64_vs_c128_all: the same between complex64 and complex128 over all planes; *_p0: normalized by one
  power for all planes (the power on the grid of the middle plane of complex128), as in a focus scan.

On the GPU computer, from the repository folder:

    bash tools/gpu_run.sh optics_gpu python bench/bench_image_sum.py --rays 1e6 1e7 --planes 1 21 41
"""
import argparse
import json
import time
import warnings

import numpy as np
import torch

from optics_gpu.pop import (GROUP_RAYS, _image_sum_direct, gaussian, image_planes, image_sum,
                            overlap_efficiency_device, trace_bundle)
from optics_gpu.systems import aberrated_singlet, four_f

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
WL, N_PAD = 1.55, 4

ap = argparse.ArgumentParser()
ap.add_argument("--system", choices=("singlet", "4f"), default="singlet")
ap.add_argument("--rays", type=float, nargs="+", default=[1e6, 1e7])
ap.add_argument("--planes", type=int, nargs="+", default=[1, 21, 41])
ap.add_argument("--span", type=float, default=200.0, help="dz range of the planes (um)")
ap.add_argument("--image", type=int, default=128, help="image grid points per axis (pitch 1 um)")
ap.add_argument("--ref-planes", type=int, default=3, help="planes timed with the per-plane methods")
ap.add_argument("--repeat", type=int, default=2)
ap.add_argument("--na", type=float, default=0.09)
ap.add_argument("--mode-w", type=float, default=8.0, help="radius of the Gaussian mode of the efficiency (um)")
a = ap.parse_args()
if not torch.cuda.is_available():
    raise SystemExit("no GPU: this bench times the GPU image sum")
optic = (aberrated_singlet if a.system == "singlet" else four_f)()
xi = (np.arange(a.image) - a.image // 2) * 1.0
dx = WL / (2.1 * a.na)
mode = torch.as_tensor(gaussian(xi, xi, a.mode_w), device="cuda")


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


def rel(e, ref):
    return float((e - ref).abs().max() / ref.abs().max())


def d_eta(e, ref, power=None):
    return float((overlap_efficiency_device(e, mode, power) - overlap_efficiency_device(ref, mode, power)).abs().max())


env = {"torch": torch.__version__, "hip": torch.version.hip, "gpu": torch.cuda.get_device_name(0),
       "image": a.image, "group_rays": GROUP_RAYS, "mode_w_um": a.mode_w}
kw = {"na_cut": a.na, "amp_cut": 0.0, "n_pad": N_PAD, "device": "cuda"}
warm = trace_bundle(optic, point(16), dx, WL, **kw)                     # warm-up (contexts, kernels, FFT plans)
for dt in (torch.complex128, torch.complex64):
    image_planes(warm, xi, xi, [0.0, 1.0], dtype=dt)
_image_sum_direct(warm, xi, xi, 1.0)
del warm


def case(bundle, t_bundle, nz):
    """One JSON row: the focus scan of nz planes with the three methods."""
    dz = np.linspace(-a.span / 2, a.span / 2, nz) if nz > 1 else np.zeros(1)
    nref = min(nz, a.ref_planes)
    _image_sum_direct(bundle, xi, xi, float(dz[0]))                     # untimed plane
    ref, t_ref = timed(lambda: torch.stack([_image_sum_direct(bundle, xi, xi, float(d)) for d in dz[:nref]]))
    single, t_single = timed(lambda: torch.stack([image_sum(bundle, xi, xi, float(d)) for d in dz[:nref]]))
    e128, t128 = best(lambda: image_planes(bundle, xi, xi, dz))
    e64, t64 = best(lambda: image_planes(bundle, xi, xi, dz, dtype=torch.complex64))
    return {"system": a.system, "rays_traced": bundle.n_traced, "rays_kept": bundle.n_rays, "planes": nz,
            "span_um": a.span if nz > 1 else 0.0, "ref_planes": nref, "trace_bundle_s": round(t_bundle, 4),
            "per_plane_s": round(t_ref / nref, 5), "single_s": round(t_single / nref, 5),
            "batched_c128_s": round(t128 / nz, 5), "batched_c64_s": round(t64 / nz, 5),
            "speedup_c128": round(t_ref / nref / (t128 / nz), 2),
            "speedup_c64": round(t_ref / nref / (t64 / nz), 2),
            "max_rel_dE_single": rel(single, ref), "max_rel_dE_c128": rel(e128[:nref], ref),
            "max_rel_dE_c64": rel(e64[:nref], ref),
            "max_d_eta_c128": d_eta(e128[:nref], ref), "max_d_eta_c64": d_eta(e64[:nref], ref),
            "max_d_eta_c64_vs_c128_all": d_eta(e64, e128),
            "max_d_eta_c64_vs_c128_all_p0": d_eta(e64, e128, (e128[nz // 2].abs() ** 2).sum()),
            "eta_at_dz0": round(float(overlap_efficiency_device(e128[nz // 2], mode)), 6)}


for n in a.rays:
    src = point(max(4, int(round(1.05 * np.sqrt(n) / N_PAD))))
    trace_bundle(optic, src, dx, WL, **kw)                              # the first FFT of a new size (rocFFT)
    bundle, t_bundle = timed(lambda: trace_bundle(optic, src, dx, WL, **kw))  # noqa: B023
    for nz in a.planes:
        print(json.dumps(case(bundle, t_bundle, nz) | env), flush=True)
    bundle = None
