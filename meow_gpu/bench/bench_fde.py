"""Benchmark: GPU Davidson FDE solver against the tidy3d CPU solver on a synthetic Si strip (same grid).

    python bench/bench_fde.py --dx 0.02 --modes 6 [--repeat 3] [--warm]
Prints one JSON line per case. Times are wall-clock; the first GPU run includes the JIT compile and is reported apart.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from scipy.constants import c  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from structures import strip  # noqa: E402

from meow_gpu.fde.eigensolver import davidson, make_preconditioner, precondition  # noqa: E402
from meow_gpu.fde.operator import make_problem  # noqa: E402


EXTRA = None
TOL = None


def tidy3d_modes(exx, eyy, ezz, x, y, wl, k):
    from tidy3d.components.mode.solver import compute_modes
    ms = SimpleNamespace(num_modes=k, target_neff=None, num_pml=(0, 0), filter_pol=None, angle_theta=0.0,
                         angle_phi=0.0, bend_radius=None, precision="double", bend_axis=None, track_freq="central",
                         group_index_step=False)
    z = 0 * exx
    out = compute_modes(eps_cross=[exx, z, z, z, eyy, z, z, z, ezz], coords=[x, y], freq=c / (wl * 1e-6),
                        mode_spec=ms, precision="double")
    return np.sort(np.real(out[1].squeeze()))[::-1]


def gpu_modes(exx, eyy, ezz, x, y, wl, k, x0=None, extra=4, precision="mixed"):
    extra = EXTRA if EXTRA is not None else extra
    p = make_problem(exx, eyy, ezz, x, y, wl)
    op = (p.device(jnp.complex64), p.device(jnp.complex128)) if precision == "mixed" else \
        p.device(jnp.complex64 if precision == "single" else jnp.complex128)
    pc = make_preconditioner(p)
    target = -(np.sqrt(np.max(np.real(exx))) * (1 + 10 * np.finfo(float).eps)) ** 2
    b = k + extra
    if x0 is None:
        rng = np.random.default_rng(0)
        x0 = jnp.asarray(rng.normal(size=(b, 2) + exx.shape) + 0j)
        x0 = precondition(pc, x0, jnp.full(b, target, complex))
    res = davidson(op, pc, k, target, x0, tol=TOL or (1e-10 if precision != "single" else 1e-5), max_iter=500,
                   precision=precision)
    jax.block_until_ready(res.vectors)
    neff = np.sqrt(-res.values)
    return np.sort(neff.real)[::-1], res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", type=float, default=0.02)
    ap.add_argument("--width", type=float, default=0.5)
    ap.add_argument("--w-dom", type=float, default=3.0)
    ap.add_argument("--h-dom", type=float, default=2.0)
    ap.add_argument("--tol", type=float, default=None)
    ap.add_argument("--modes", type=int, default=6)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--skip-ref", action="store_true")
    ap.add_argument("--precision", default="mixed", choices=["mixed", "double", "single"])
    ap.add_argument("--extra", type=int, default=4, help="block vectors beyond the wanted modes")
    a = ap.parse_args()
    global EXTRA, TOL
    EXTRA = a.extra
    TOL = a.tol
    exx, eyy, ezz, x, y = strip(width=a.width, w_dom=a.w_dom, h_dom=a.h_dom, dx=a.dx, dy=a.dx)
    row = {"width": a.width, "tol": a.tol, "extra": a.extra, "precision": a.precision, "dx": a.dx, "grid": list(exx.shape), "unknowns": 2 * exx.size, "modes": a.modes,
           "device": str(jax.devices()[0])}
    if not a.skip_ref:
        t0 = time.perf_counter()
        n_ref = tidy3d_modes(exx, eyy, ezz, x, y, 1.55, a.modes)
        row["tidy3d_s"] = time.perf_counter() - t0
        row["n_ref"] = n_ref.tolist()
    t0 = time.perf_counter()
    n_gpu, res = gpu_modes(exx, eyy, ezz, x, y, 1.55, a.modes, precision=a.precision)
    row["gpu_first_s"] = time.perf_counter() - t0
    times = []
    for _ in range(a.repeat):
        t0 = time.perf_counter()
        n_gpu, res = gpu_modes(exx, eyy, ezz, x, y, 1.55, a.modes, precision=a.precision)
        times.append(time.perf_counter() - t0)
    row.update(gpu_s=float(np.median(times)), iterations=res.iterations, matvecs=res.matvecs,
               converged=res.converged, n_gpu=n_gpu.tolist())
    # warm start: the same structure 5 nm wider, started from the modes above
    exx2, eyy2, ezz2, _, _ = strip(width=a.width + 0.005, w_dom=a.w_dom, h_dom=a.h_dom, dx=a.dx, dy=a.dx)
    t0 = time.perf_counter()
    n_w, res_w = gpu_modes(exx2, eyy2, ezz2, x, y, 1.55, a.modes, x0=res.vectors, precision=a.precision)
    row.update(warm_s=time.perf_counter() - t0, warm_iterations=res_w.iterations, warm_converged=res_w.converged)
    if "n_ref" in row:
        row["max_dn"] = float(np.max(np.abs(np.array(row["n_ref"]) - n_gpu)))
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
