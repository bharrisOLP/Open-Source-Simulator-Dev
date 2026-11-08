"""The two synthetic phase shifters at 300, 330 and 360 K (Device.T; models in charge_gpu.temperature).

- Depletion modulator: the rib of ``modulator.rib_pn`` (3e17 cm^-3 on both sides, 10 nm cells, contacts 3 um from
  the rib center), reverse bias 0 to 4 V: C', R', f_3dB (charge_gpu.rc) and, with --modes, V_pi L and the
  free-carrier loss of the mode (meow-gpu, the mode grid and cell means of examples/pn_modulator_rc.py).
- Injection phase shifter: the lateral PIN of ``injection.rib_pin`` (10 nm cells, contacts 1.5 um from the center),
  forward bias 0 to 0.9 V, SRH lifetimes 1 ns at 300 K (scaled with (T / 300)^-1/2): the current, the mean carriers
  and the index change in the rib and, with --modes, L_pi and P_pi = I V L_pi (examples/pin_phase_shifter.py).

What changes with T: V_T, n_i(T) (Misiakos and Tsamakis 1993), the mobility (Masetti at 300 K times the ratio of
Arora et al. 1982) and the lifetimes. What does not: eps_r, the plasma dispersion coefficients (Soref and Bennett,
300 K) and the base index of Si and SiO2 (no thermo-optic effect: V_pi L and L_pi come from index differences at one
temperature, so the thermo-optic shift of the base index is left out). Textbook values, no process data.

    bash tools/gpu_run.sh charge_gpu python examples/temperature_dependence.py              (charge only)
    bash tools/gpu_run.sh charge_gpu python examples/temperature_dependence.py --modes      (with meow-gpu)
"""
import argparse
import json
import time

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np  # noqa: E402

from charge_gpu import drift_diffusion as dd  # noqa: E402
from charge_gpu.injection import rib_index_change, rib_pin  # noqa: E402
from charge_gpu.modulator import rib_pn  # noqa: E402
from charge_gpu.rc import small_signal  # noqa: E402

WL = 1.55e-6
TEMPS = [300.0, 330.0, 360.0]
REVERSE = [0.0, 1.0, 2.0, 4.0]
FORWARD = [0.0, 0.8, 0.85, 0.9]
TAU_300 = 1e-9
LENGTH, R_SOURCE = 0.5e-3, 50.0
DB = 10 * np.log10(np.e)


def depletion(t, modes):
    dev = rib_pn(doping=3e23, dx=10e-9)
    dev.T = t
    t0 = time.perf_counter()
    res = small_signal(dev, REVERSE, dv=0.02)
    t_charge = time.perf_counter() - t0
    neff = None
    if modes:
        from pn_modulator_rc import mode_indices          # the same folder (the script folder is on sys.path)
        neff = mode_indices(dev, [r.solution for r in res])
    rows = []
    for i, r in enumerate(res):
        row = {"device": "depletion", "T_K": t, "reverse_V": r.bias, "converged": r.converged,
               "C_fF_per_um": r.c * 1e9, "R_ohm_mm": r.r * 1e3, "f3dB_intrinsic_GHz": r.f3db() * 1e-9,
               "f3dB_0.5mm_50ohm_GHz": r.f3db(LENGTH, R_SOURCE) * 1e-9, "charge_s": round(t_charge, 2)}
        if neff is not None:
            row["loss_mode_dB_per_cm"] = DB * 4 * np.pi * neff[i].imag / (WL * 1e2)
            if r.bias > 0:
                d = neff[i].real - neff[0].real
                row["dneff"] = d
                row["VpiL_Vcm"] = r.bias * WL * 1e2 / (2 * abs(d))
        rows.append(row)
    return rows


def injection(t, modes, linear):
    dev = rib_pin(dx=10e-9)
    dev.T = t
    t0 = time.perf_counter()
    sols = dd.sweep(dev, FORWARD, tau_n=TAU_300, tau_p=TAU_300, linear=linear)
    t_charge = time.perf_counter() - t0
    neff = None
    if modes:
        from pn_modulator_rc import mode_indices
        neff = mode_indices(dev, sols)
    rows = []
    for i, (v, s) in enumerate(zip(FORWARD, sols)):
        r = rib_index_change(dev, s, sols[0])
        row = {"device": "pin", "T_K": t, "bias_V": v, "converged": s.converged, "tau_s": s.tau_n,
               "I_mA_per_mm": s.current, "n_rib_cm3": r["n_mean"] * 1e-6, "p_rib_cm3": r["p_mean"] * 1e-6,
               "dn_rib": r["dn_mean"], "dloss_rib_dB_per_cm": DB * r["dalpha_mean"] * 1e-2,
               "continuity_rel": float(np.max(np.abs(s.x_currents() - s.current)) / max(abs(s.current), 1e-30)),
               "charge_s": round(t_charge, 2)}
        if neff is not None:
            row["loss_mode_dB_per_cm"] = DB * 4 * np.pi * neff[i].imag / (WL * 1e2)
            if v > 0:
                d = neff[i].real - neff[0].real
                row["dneff"] = d
                row["L_pi_mm"] = WL / (2 * abs(d)) * 1e3
                row["P_pi_mW"] = s.current * v * row["L_pi_mm"]
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--temps", type=float, nargs="+", default=TEMPS, help="temperatures (K)")
    ap.add_argument("--part", choices=["both", "depletion", "pin"], default="both")
    ap.add_argument("--modes", action="store_true", help="mode indices with meow-gpu")
    ap.add_argument("--linear", choices=["superlu", "device"], default="superlu", help="linear solver of the PIN")
    args = ap.parse_args()
    rows = []
    for t in args.temps:
        new = []
        if args.part in ("both", "depletion"):
            new += depletion(t, args.modes)
        if args.part in ("both", "pin"):
            new += injection(t, args.modes, args.linear)
        for row in new:
            print(json.dumps(row, default=float), flush=True)
        rows += new
    for kind, cols in (("depletion", ["T_K", "reverse_V", "C_fF_per_um", "R_ohm_mm", "f3dB_intrinsic_GHz",
                                      "f3dB_0.5mm_50ohm_GHz", "VpiL_Vcm", "loss_mode_dB_per_cm"]),
                       ("pin", ["T_K", "bias_V", "I_mA_per_mm", "n_rib_cm3", "dn_rib", "dloss_rib_dB_per_cm",
                                "dneff", "L_pi_mm", "P_pi_mW", "loss_mode_dB_per_cm"])):
        part = [r for r in rows if r["device"] == kind]
        cols = [c for c in cols if any(c in r for r in part)]
        if not part:
            continue
        print("| " + " | ".join(cols) + " |")
        print("|" + "---|" * len(cols))
        for r in part:
            print("| " + " | ".join(f"{r[c]:.4g}" if c in r else "-" for c in cols) + " |")


if __name__ == "__main__":
    main()
