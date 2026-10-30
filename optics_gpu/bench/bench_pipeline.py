"""End-to-end time per image field (angular spectrum + trace + Jacobian and weights + image sum + the field to the
host) of optics_gpu.pop.propagate_through: the NumPy reference pipeline (trace_device None: NumPy trace, NumPy
Jacobian and weights, image sum on the GPU) against the device pipeline (trace_device "cuda" or "cpu": all steps
on that device). One JSON line per case (ray count and pipeline).

The source is a point (one pixel, a flat angular spectrum) through the aberrated singlet (default; its 6 mm stop
clips a part of the rays) or the 2.5x lens pair (--system 4f). The direction grid has |L|, |M| <= --na (0.09): the
pixel step dx = lambda / (2.1 na) puts the bounding box of the kept directions on about 95 % of the spectrum grid,
and the source size n_src = 1.05 sqrt(rays) / n_pad gives about --rays rays in the box. total_s is the best of
--repeat runs after one untimed run of the same case; for the device pipeline, bundle_s (trace_bundle: spectrum,
trace, Jacobian, weights) and sum_s (image_sum) are the parts, each the best of --repeat separate runs. Each device
row gives its largest field difference from the NumPy field (relative to the largest |E|) and the ray counts.

On the GPU computer, from the repository folder:

    bash tools/gpu_run.sh optics_gpu python bench/bench_pipeline.py --rays 1e5 1e6 1e7
"""
import argparse
import json
import time
import warnings

import numpy as np
import torch

from optics_gpu.pop import image_sum, propagate_through, trace_bundle
from optics_gpu.systems import aberrated_singlet, four_f

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
WL, N_PAD = 1.55, 4

ap = argparse.ArgumentParser()
ap.add_argument("--system", choices=("singlet", "4f"), default="singlet")
ap.add_argument("--rays", type=float, nargs="+", default=[1e5, 1e6, 1e7])
ap.add_argument("--pipelines", nargs="+", choices=("numpy", "cpu", "cuda"),
                default=["numpy"] + (["cuda"] if torch.cuda.is_available() else ["cpu"]))
ap.add_argument("--image", type=int, default=128, help="image grid points per axis (pitch 1 um)")
ap.add_argument("--repeat", type=int, default=2)
ap.add_argument("--na", type=float, default=0.09)
a = ap.parse_args()
optic = (aberrated_singlet if a.system == "singlet" else four_f)()
xi = (np.arange(a.image) - a.image // 2) * 1.0
dx = WL / (2.1 * a.na)


def point(n):
    src = np.zeros((n, n), complex)
    src[n // 2, n // 2] = 1.0
    return src


def sync(p):
    if p == "cuda":
        torch.cuda.synchronize()


def best(fn, p):
    """One untimed run (the first FFT of a new grid size compiles its rocFFT kernels: about 2 s), then --repeat
    timed runs."""
    times, out = [], fn()
    for _ in range(a.repeat):
        out = None
        sync(p)
        t0 = time.perf_counter()
        out = fn()
        sync(p)
        times.append(time.perf_counter() - t0)
    return out, times


env = {"torch": torch.__version__, "hip": torch.version.hip, "threads": torch.get_num_threads(),
       "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "image": a.image}
kw = {"na_cut": a.na, "amp_cut": 0.0, "n_pad": N_PAD}
for p in a.pipelines:                                                     # warm-up (contexts, kernels, FFT plans)
    propagate_through(optic, point(16), dx, WL, xi, xi, trace_device=None if p == "numpy" else p, **kw)
    sync(p)
for n in a.rays:
    n_src = max(4, int(round(1.05 * np.sqrt(n) / N_PAD)))
    src = point(n_src)
    ref = None
    for p in a.pipelines:
        dev = None if p == "numpy" else p
        img, runs = best(lambda: propagate_through(optic, src, dx, WL, xi, xi, trace_device=dev, **kw), p)
        row = {"system": a.system, "pipeline": "numpy" if p == "numpy" else "device", "device": img.device,
               "trace_device": img.trace_device, "rays_kept": img.n_rays, "spectrum_grid": n_src * N_PAD,
               "total_s": round(min(runs), 4), "total_s_runs": [round(t, 4) for t in runs]}
        if p == "numpy":
            ref = img
        else:
            bundle, t_b = best(lambda: trace_bundle(optic, src, dx, WL, device=p, **kw), p)
            _, t_s = best(lambda: image_sum(bundle, xi, xi), p)
            row |= {"rays_traced": bundle.n_traced, "bundle_s": round(min(t_b), 4), "sum_s": round(min(t_s), 4)}
            del bundle
            if ref is not None:
                row["rays_equal"] = ref.n_rays == img.n_rays
                row["max_rel_dE"] = float(np.max(np.abs(img.E - ref.E)) / np.max(np.abs(ref.E)))
        row["mrays_per_s"] = round(img.n_rays / min(runs) / 1e6, 3)
        print(json.dumps(row | env), flush=True)
        del img
