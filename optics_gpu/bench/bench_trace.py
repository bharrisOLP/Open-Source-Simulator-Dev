"""Ray trace time of Optiland on its NumPy backend (CPU) against its PyTorch backend in float64 on the CPU and on
the GPU (CUDA or ROCm), for 1e4 to 1e7 rays through the aberrated singlet (or the 2.5x lens pair, --system 4f).
One JSON line per case (ray count and backend).

The rays start at the object point on a square grid of directions |L|, |M| <= --half (0.09 overfills the 6 mm stop
of the singlet, so its physical aperture clips a part of the rays). trace_s is the time of one
optics_gpu.pop.trace_rays call (with the PyTorch copy of the optic, its check rays, and the copy of the ray
inputs to the device), the best of --repeat runs after a warm-up trace; to_host_s is the copy of x, y, L, M, OPL
and intensity back to NumPy. Each PyTorch row gives its largest difference from the NumPy trace of the same rays.

    cd optics_gpu && PYTHONDONTWRITEBYTECODE=1 uv run python bench/bench_trace.py --rays 1e4 1e5 1e6
"""
import argparse
import json
import time
import warnings

import numpy as np
import torch

from optics_gpu.pop import trace_rays
from optics_gpu.systems import aberrated_singlet, four_f

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
WL = 1.55
FIELDS = ("x", "y", "L", "M", "opd", "i")

ap = argparse.ArgumentParser()
ap.add_argument("--system", choices=("singlet", "4f"), default="singlet")
ap.add_argument("--rays", type=float, nargs="+", default=[1e4, 1e5, 1e6, 1e7])
ap.add_argument("--backends", nargs="+", choices=("numpy", "cpu", "cuda"),
                default=["numpy", "cpu"] + (["cuda"] if torch.cuda.is_available() else []))
ap.add_argument("--repeat", type=int, default=2)
ap.add_argument("--half", type=float, default=0.09)
a = ap.parse_args()
make = aberrated_singlet if a.system == "singlet" else four_f
optic = make()


def directions(n):
    s = np.linspace(-a.half, a.half, max(2, int(np.sqrt(n) + 0.5)))
    L, M = np.meshgrid(s, s, indexing="ij")
    return L.ravel(), M.ravel()


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def to_np(v):
    return v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)


env = {"torch": torch.__version__, "hip": torch.version.hip, "threads": torch.get_num_threads(),
       "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
for b in a.backends:                                                     # warm-up (CUDA context, kernels)
    trace_rays(optic, *directions(1e3), WL, None if b == "numpy" else b)
    sync(b)
for n in a.rays:
    L, M = directions(n)
    ref = None
    for b in a.backends:
        device = None if b == "numpy" else b
        runs = []
        for _ in range(a.repeat):
            rays = None
            sync(b)
            t0 = time.perf_counter()
            rays = trace_rays(optic, L, M, WL, device)
            sync(b)
            runs.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        out = {k: to_np(getattr(rays, k)) for k in FIELDS}
        to_host = time.perf_counter() - t0
        del rays
        row = {"system": a.system, "rays": int(L.size), "backend": "numpy" if b == "numpy" else "torch",
               "device": "cpu" if b == "numpy" else b, "trace_s": round(min(runs), 4),
               "trace_s_runs": [round(t, 4) for t in runs], "mrays_per_s": round(L.size / min(runs) / 1e6, 3),
               "clipped_fraction": round(float(np.mean(out["i"] == 0)), 4)}
        if b == "numpy":
            ref = out
        else:
            row["to_host_s"] = round(to_host, 4)
        if ref is not None and b != "numpy":
            fin = np.isfinite(ref["opd"])
            row["nan_pattern_equal"] = all(np.array_equal(np.isnan(ref[k]), np.isnan(out[k])) for k in FIELDS)
            row["max_dpos_mm"] = float(max(np.abs(out[k][fin] - ref[k][fin]).max() for k in ("x", "y")))
            row["max_ddir"] = float(max(np.abs(out[k][fin] - ref[k][fin]).max() for k in ("L", "M")))
            row["max_dopl_mm"] = float(np.abs(out["opd"][fin] - ref["opd"][fin]).max())
            row["intensity_equal"] = bool(np.array_equal(out["i"], ref["i"]))
        print(json.dumps(row | env), flush=True)
