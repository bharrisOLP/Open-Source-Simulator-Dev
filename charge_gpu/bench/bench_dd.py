"""Benchmark of the drift-diffusion solver (charge_gpu.drift_diffusion). One JSON line per case.

Cases:
- diode: the 1-D long-base diode of the Shockley check (1e18 cm^-3, tau 100 us), Newton sweep 0 to 0.75 V.
- pin-newton: the PIN rib of examples/pin_phase_shifter.py (tau 1 ns), Newton sweep 0 to 1.0 V (CPU direct LU).
  Skipped with --no-sweep.
- At each bias of --biases, from the Newton solution 50 mV lower (the same start for all rows):
  - pin-newton-step: one Newton solve (SuperLU on the CPU), the reference of the other rows;
  - pin-newton-device: one Newton solve with the device linear solver (coupled.py: FGMRES, block multigrid; the
    Jacobian assembled on the host), with a one-step warm-up (the JIT compile) timed apart, and the times of the
    assembly, the setup (hierarchies, transfers) and the FGMRES loops;
  - pin-gummel-superlu: Gummel, the Poisson step on the JAX device, the continuity steps with SuperLU on the CPU
    (skipped with --no-gummel-superlu);
  - pin-gummel-device: Gummel, the Poisson step and the continuity steps (BiCGSTAB, Slotboom multigrid) on the
    JAX device (the GPU when there is one);
  - pin-gummel-device-cpu: the same code with JAX on the CPU (CHARGE_GPU_DEVICE=cpu).
  The Gummel rows give the wall time of a one-iteration warm-up (the JIT compile for this mesh) apart from the
  timed solve, the times of the Poisson and continuity steps, and the differences to the Newton solution.

    bash tools/gpu_run.sh charge_gpu python bench/bench_dd.py                                   (diode and 10 nm)
    bash tools/gpu_run.sh charge_gpu python bench/bench_dd.py --dx 10e-9 5e-9 --no-diode --no-sweep --no-cpu
"""
import argparse
import json
import os
import time

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np  # noqa: E402

from charge_gpu import drift_diffusion as dd  # noqa: E402
from charge_gpu.injection import diode_1d, rib_index_change, rib_pin, shockley_j0  # noqa: E402
from charge_gpu.poisson import solver_device, thermal_voltage  # noqa: E402
from charge_gpu.rc import masetti_mobility  # noqa: E402

TAU = 1e-9


def diode_case():
    n, tau, vt = 1e24, 1e-4, thermal_voltage()
    mu_n, mu_p = masetti_mobility(n, "n"), masetti_mobility(n, "p")
    l_n, l_p = np.sqrt(mu_n * vt * tau), np.sqrt(mu_p * vt * tau)
    dev = diode_1d(n, n, 6 * l_n, 6 * l_p)
    j0 = shockley_j0(n, n, mu_n, mu_p, tau, tau, length_p=6 * l_n, length_n=6 * l_p)
    biases = [0.55, 0.65, 0.75]
    t0 = time.perf_counter()
    sols = dd.sweep(dev, biases, tau_n=tau, tau_p=tau)
    return {"case": "diode", "cells": int(dev.n_net.size), "method": "newton", "device": sols[0].device,
            "time_s": round(time.perf_counter() - t0, 3), "converged": all(s.converged for s in sols),
            "J_over_shockley": [round(s.current / dev.y[-1] / (j0 * np.expm1(v / vt)), 6)
                                for v, s in zip(biases, sols)]}


def _diff(sol, ref):
    m = ref.mesh
    n, p = sol.n.ravel()[m.sflat], sol.p.ravel()[m.sflat]
    nr, pr = ref.n.ravel()[m.sflat], ref.p.ravel()[m.sflat]
    kn, kp = nr > dd.CARRIER_FLOOR, pr > dd.CARRIER_FLOOR
    return {"I_rel_vs_newton": float(sol.current / ref.current - 1),
            "dpsi_V": float(np.max(np.abs(sol.psi - ref.psi))),
            "dn_rel": float(np.max(np.abs(n[kn] / nr[kn] - 1))), "dp_rel": float(np.max(np.abs(p[kp] / pr[kp] - 1)))}


def _gummel_row(dev, case, bias, start, ref, linear, cpu=False):
    old = os.environ.get("CHARGE_GPU_DEVICE")
    if cpu:
        os.environ["CHARGE_GPU_DEVICE"] = "cpu"
    try:
        kw = dict(guess=start, method="gummel", linear=linear, tau_n=TAU, tau_p=TAU)
        t0 = time.perf_counter()
        dd.solve(dev, 0.0, bias, max_iter=1, **kw)                 # warm-up: the JIT compile of this mesh
        t_warm = time.perf_counter() - t0
        t0 = time.perf_counter()
        g = dd.solve(dev, 0.0, bias, max_iter=4000, **kw)
        t = time.perf_counter() - t0
        where = str(solver_device())
    finally:
        if cpu:
            if old is None:
                os.environ.pop("CHARGE_GPU_DEVICE", None)
            else:
                os.environ["CHARGE_GPU_DEVICE"] = old
    return {"case": case, "bias_V": bias, "method": "gummel", "linear": linear, "jax_device": where,
            "time_s": round(t, 3), "warmup_1_iteration_s": round(t_warm, 3),
            "poisson_s": round(g.timings["poisson_s"], 3), "continuity_s": round(g.timings["continuity_s"], 3),
            "iterations": g.iterations, "krylov_iterations": g.krylov_iterations, "converged": g.converged,
            "linear_converged": g.linear_converged, "linear_backward_error": g.linear_error,
            "linear_fallbacks": g.linear_fallbacks, **_diff(g, ref)}


def _newton_device_row(dev, bias, start, ref):
    """One Newton solve with the device linear solver (coupled.py), from the same start as the reference; a solve
    with one Newton step first (the JIT compile of this mesh), timed apart."""
    kw = dict(guess=start, method="newton", linear="device", tau_n=TAU, tau_p=TAU)
    t0 = time.perf_counter()
    dd.solve(dev, 0.0, bias, max_iter=1, **kw)
    t_warm = time.perf_counter() - t0
    t0 = time.perf_counter()
    s = dd.solve(dev, 0.0, bias, **kw)
    t = time.perf_counter() - t0
    return {"case": "pin-newton-device", "bias_V": bias, "method": "newton", "linear": "device",
            "jax_device": str(solver_device()), "time_s": round(t, 3), "warmup_1_iteration_s": round(t_warm, 3),
            **{k: round(v, 3) for k, v in s.timings.items()}, "iterations": s.iterations,
            "krylov_iterations": s.krylov_iterations, "converged": s.converged, "linear_residual": s.linear_error,
            "linear_fallbacks": s.linear_fallbacks, **_diff(s, ref)}


def pin_cases(dx, biases, sweep=True, cpu=True, gummel_superlu=True):
    dev = rib_pin(dx=dx)
    common = {"dx_nm": dx * 1e9, "cells": int(dev.n_net.size), "semiconductor_cells": int(dev.semiconductor.sum())}
    if sweep:
        vs = [0.8, 0.9, 1.0]
        t0 = time.perf_counter()
        sols = dd.sweep(dev, [0.0] + vs, tau_n=TAU, tau_p=TAU)
        t = time.perf_counter() - t0
        r = [rib_index_change(dev, s, sols[0]) for s in sols[1:]]
        yield {"case": "pin-newton", **common, "unknowns": int(dev.n_net.size + 2 * dev.semiconductor.sum()),
               "method": "newton", "device": sols[-1].device, "time_s": round(t, 3),
               "newton_last": sols[-1].iterations, "converged": all(s.converged for s in sols),
               "I_mA_per_mm": [round(s.current, 6) for s in sols[1:]],
               "n_rib_cm3": [float(f"{x['n_mean'] * 1e-6:.5g}") for x in r]}
    for bias in biases:
        start = dd.sweep(dev, [round(bias - 0.05, 6)], tau_n=TAU, tau_p=TAU)[0]
        t0 = time.perf_counter()
        ref = dd.solve(dev, 0.0, bias, guess=start, tau_n=TAU, tau_p=TAU)
        yield {"case": "pin-newton-step", **common, "bias_V": bias, "method": "newton", "linear": "superlu",
               "time_s": round(time.perf_counter() - t0, 3), **{k: round(v, 3) for k, v in ref.timings.items()},
               "iterations": ref.iterations, "converged": ref.converged, "I_mA_per_mm": round(ref.current, 6)}
        yield {**_newton_device_row(dev, bias, start, ref), **common}
        rows = [("pin-gummel-superlu", "superlu", False)] if gummel_superlu else []
        rows.append(("pin-gummel-device", "device", False))
        if cpu:
            rows.append(("pin-gummel-device-cpu", "device", True))
        for case, linear, on_cpu in rows:
            row = _gummel_row(dev, case, bias, start, ref, linear, on_cpu)
            row.update(common)
            yield row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", type=float, nargs="+", default=[10e-9])
    ap.add_argument("--biases", type=float, nargs="+", default=[0.6, 0.9])
    ap.add_argument("--no-diode", action="store_true")
    ap.add_argument("--no-sweep", action="store_true", help="skip the Newton sweep 0 to 1.0 V")
    ap.add_argument("--no-cpu", action="store_true", help="skip the device path with JAX on the CPU")
    ap.add_argument("--no-gummel-superlu", action="store_true", help="skip Gummel with the SuperLU continuity steps")
    a = ap.parse_args()
    print(json.dumps({"case": "env", "jax_default_device": str(solver_device()), "cpu_count": os.cpu_count(),
                      "jax": jax.__version__}), flush=True)
    if not a.no_diode:
        print(json.dumps(diode_case()), flush=True)
    for dx in a.dx:
        for row in pin_cases(dx, a.biases, sweep=not a.no_sweep, cpu=not a.no_cpu,
                             gummel_superlu=not a.no_gummel_superlu):
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
