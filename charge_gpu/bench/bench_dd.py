"""Benchmark of the drift-diffusion solver (charge_gpu.drift_diffusion). One JSON line per case.

Cases:
- diode: the 1-D long-base diode of the Shockley check (1e18 cm^-3, tau 100 us), Newton sweep 0 to 0.75 V.
- pin-newton: the PIN rib of examples/pin_phase_shifter.py (tau 1 ns), Newton sweep 0 to 1.0 V (CPU direct LU).
- pin-gummel: the same rib, Gummel at --gummel-bias from the Newton solution at the bias below (the Poisson step
  runs on the JAX device: multigrid CG; the continuity steps on the CPU). This case shows the device part.

    bash tools/gpu_run.sh charge_gpu python bench/bench_dd.py                             (diode and 10 nm)
    bash tools/gpu_run.sh charge_gpu python bench/bench_dd.py --dx 10e-9 5e-9 --no-diode  (CHARGE_GPU_DEVICE=cpu: CPU)
"""
import argparse
import json
import time

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np  # noqa: E402

from charge_gpu import drift_diffusion as dd  # noqa: E402
from charge_gpu.injection import diode_1d, rib_index_change, rib_pin, shockley_j0  # noqa: E402
from charge_gpu.poisson import solver_device, thermal_voltage  # noqa: E402
from charge_gpu.rc import masetti_mobility  # noqa: E402


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


def pin_cases(dx, gummel_bias):
    dev = rib_pin(dx=dx)
    biases = [0.8, 0.9, 1.0]
    t0 = time.perf_counter()
    sols = dd.sweep(dev, [0.0] + biases, tau_n=1e-9, tau_p=1e-9)
    t = time.perf_counter() - t0
    r = [rib_index_change(dev, s, sols[0]) for s in sols[1:]]
    yield {"case": "pin-newton", "dx_nm": dx * 1e9, "cells": int(dev.n_net.size),
           "unknowns": int(dev.n_net.size + 2 * dev.semiconductor.sum()), "method": "newton",
           "device": sols[-1].device, "time_s": round(t, 3),
           "newton_last": sols[-1].iterations, "converged": all(s.converged for s in sols),
           "I_mA_per_mm": [round(s.current, 6) for s in sols[1:]],
           "n_rib_cm3": [float(f"{x['n_mean'] * 1e-6:.5g}") for x in r]}
    start = dd.sweep(dev, [gummel_bias - 0.05], tau_n=1e-9, tau_p=1e-9)[0]
    ref = dd.solve(dev, 0.0, gummel_bias, guess=start, tau_n=1e-9, tau_p=1e-9)
    t0 = time.perf_counter()
    g = dd.solve(dev, 0.0, gummel_bias, guess=start, method="gummel", tau_n=1e-9, tau_p=1e-9, tol=1e-8,
                 max_iter=3000)
    yield {"case": "pin-gummel", "dx_nm": dx * 1e9, "cells": int(dev.n_net.size), "method": "gummel",
           "device": str(solver_device()), "bias_V": gummel_bias, "time_s": round(time.perf_counter() - t0, 3),
           "iterations": g.iterations, "converged": g.converged,
           "I_rel_vs_newton": float(g.current / ref.current - 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", type=float, nargs="+", default=[10e-9])
    ap.add_argument("--gummel-bias", type=float, default=0.6)
    ap.add_argument("--no-diode", action="store_true")
    a = ap.parse_args()
    if not a.no_diode:
        print(json.dumps(diode_case()), flush=True)
    for dx in a.dx:
        for row in pin_cases(dx, a.gummel_bias):
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
