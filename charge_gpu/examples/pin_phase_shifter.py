"""Forward-biased PIN phase shifter (carrier injection): drift-diffusion of a lateral PIN in an SOI rib.

The synthetic rib of ``injection.rib_pin`` (500 x 220 nm rib on a 90 nm slab, intrinsic zone with p 1e15 cm^-3 up to
0.3 um from the rib sides, p+ and n+ 1e18 cm^-3 beyond, contacts at the slab ends 1.5 um from the center; 10 nm
cells; textbook values, no process data). SRH lifetimes: 1 ns (default), about the effective lifetime of SOI rib
waveguides (Dimitropoulos et al., Appl. Phys. Lett. 86, 071115 (2005)); --tau gives others.

Per bias: the current and the power per length, the mean carrier densities in the rib, the area mean of the index
change and of the change of the loss in the rib (Soref and Bennett at 1.55 um, modulator.plasma_dispersion, from the
carriers at 0 V). With --modes, meow-gpu gives the change of the complex mode index (the mode grid and the cell means
of examples/pn_modulator_rc.py), L_pi = lambda / (2 |dneff|) and the free-carrier loss of the mode.

    bash tools/gpu_run.sh charge_gpu python examples/pin_phase_shifter.py           (charge only)
    bash tools/gpu_run.sh charge_gpu python examples/pin_phase_shifter.py --modes   (GPU computer: .venv-rocm has
    meow-gpu; on another computer use an environment with meow-gpu and charge-gpu)
"""
import argparse
import json
import time

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np  # noqa: E402

from charge_gpu import drift_diffusion as dd  # noqa: E402
from charge_gpu.injection import rib_index_change, rib_pin  # noqa: E402

WL = 1.55e-6
BIASES = [0.0, 0.8, 0.85, 0.9, 0.95, 1.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, nargs="+", default=[1e-9], help="SRH lifetimes tau_n = tau_p (s)")
    ap.add_argument("--dx", type=float, default=10e-9)
    ap.add_argument("--modes", action="store_true", help="mode overlap with meow-gpu")
    args = ap.parse_args()
    rows = []
    for tau in args.tau:
        dev = rib_pin(dx=args.dx)
        t0 = time.perf_counter()
        sols = dd.sweep(dev, BIASES, tau_n=tau, tau_p=tau)
        t_charge = time.perf_counter() - t0
        neff = None
        if args.modes:
            from pn_modulator_rc import mode_indices          # the same folder (the script folder is on sys.path)
            t0 = time.perf_counter()
            neff = mode_indices(dev, sols)
            t_modes = time.perf_counter() - t0
        for i, (v, s) in enumerate(zip(BIASES, sols)):
            r = rib_index_change(dev, s, sols[0])
            row = {"tau_s": tau, "dx_nm": args.dx * 1e9, "cells": int(dev.n_net.size), "bias_V": v,
                   "converged": s.converged, "newton": s.iterations,
                   "I_mA_per_mm": s.current, "P_mW_per_mm": s.current * v,
                   "n_rib_cm3": r["n_mean"] * 1e-6, "p_rib_cm3": r["p_mean"] * 1e-6,
                   "dn_rib": r["dn_mean"], "dloss_rib_dB_per_cm": 10 * np.log10(np.e) * r["dalpha_mean"] * 1e-2,
                   "continuity_rel": float(np.max(np.abs(s.x_currents() - s.current)) / max(abs(s.current), 1e-30)),
                   "charge_s": round(t_charge, 2)}
            if neff is not None:
                loss = 10 * np.log10(np.e) * 4 * np.pi * neff[i].imag / (WL * 1e2)            # dB/cm
                row["loss_mode_dB_per_cm"] = loss
                row["modes_s"] = round(t_modes, 2)
                if v > 0:
                    d = neff[i].real - neff[0].real
                    row["dneff"] = d
                    row["L_pi_mm"] = WL / (2 * abs(d)) * 1e3
                    row["loss_in_L_pi_dB"] = loss * row["L_pi_mm"] * 0.1
                    row["P_pi_mW"] = s.current * v * row["L_pi_mm"]
            rows.append(row)
            print(json.dumps(row), flush=True)
    cols = ["tau_s", "bias_V", "I_mA_per_mm", "n_rib_cm3", "p_rib_cm3", "dn_rib", "dloss_rib_dB_per_cm", "dneff",
            "L_pi_mm", "loss_mode_dB_per_cm", "loss_in_L_pi_dB", "P_pi_mW"]
    cols = [c for c in cols if any(c in r for r in rows)]
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join(f"{r[c]:.3g}" if c in r else "-" for c in cols) + " |")


if __name__ == "__main__":
    main()
