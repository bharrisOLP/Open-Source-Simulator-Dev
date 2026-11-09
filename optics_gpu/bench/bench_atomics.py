"""Time of the atomic adds of the spread of the non-uniform FFT: torch index_add_ (dim 0) of --terms rows of
--cols values into --targets rows, with random target rows (uniform; fewer targets: more adds into one value), in
float32, float64, complex64 and complex128. One JSON line per case: the time (best of --repeat runs after one
untimed run) and the adds per second (rows x cols x components (2 for complex) / time).

The spread of image_planes(method="nufft") adds w^2 values per ray and plane in this way (index_add_ into the
copies of the fine grid; the planes of a batch are the columns). The NUFFT bench (bench/bench_nufft.py) times the
whole method in complex128 and complex64.

On the GPU computer, from the repository folder (the full run for the log):

    bash tools/gpu_run.sh optics_gpu python bench/bench_atomics.py --targets 256 4096 65536 1048576 \
        --cols 1 21
"""
import argparse
import json
import time

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--terms", type=float, default=2**24, help="rows added (the terms of the spread)")
ap.add_argument("--targets", type=int, nargs="+", default=[256, 4096, 65536, 1 << 20], help="rows of the target")
ap.add_argument("--cols", type=int, nargs="+", default=[1, 21], help="values per row (planes of a batch)")
ap.add_argument("--repeat", type=int, default=3)
a = ap.parse_args()
if not torch.cuda.is_available():
    raise SystemExit("no GPU: this bench times the GPU atomic adds")
dev = torch.device("cuda")
env = {"torch": torch.__version__, "hip": torch.version.hip, "gpu": torch.cuda.get_device_name(0)}
DTYPES = {"float32": torch.float32, "float64": torch.float64, "complex64": torch.complex64,
          "complex128": torch.complex128}


def best(fn):
    fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(a.repeat):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return min(times)


g = torch.Generator(device=dev).manual_seed(1)
for cols in a.cols:
    rows = max(1, int(a.terms) // cols)
    for m in a.targets:
        idx = torch.randint(0, m, (rows,), device=dev, generator=g)
        row = {"terms": rows, "cols": cols, "targets": m, "adds_per_target": rows * cols / (m * cols)}
        for name, dt in DTYPES.items():
            vals = torch.randn((rows, cols), dtype=dt, device=dev, generator=g)
            out = torch.zeros((m, cols), dtype=dt, device=dev)
            t = best(lambda: out.index_add_(0, idx, vals))  # noqa: B023
            parts = 2 if dt.is_complex else 1
            row |= {f"{name}_s": t, f"{name}_adds_per_s": rows * cols * parts / t}
            del vals, out
        row["c128_over_c64"] = round(row["complex128_s"] / row["complex64_s"], 2)
        row["f64_over_f32"] = round(row["float64_s"] / row["float32_s"], 2)
        print(json.dumps(row | env), flush=True)
