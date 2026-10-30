"""Depletion modulator trade-off: V_pi L, loss, capacitance, series resistance and the RC bandwidth.

The synthetic SOI rib of examples/pn_modulator.py (``modulator.rib_pn``: 500 x 220 nm rib on a 90 nm slab, lateral
PN junction at the rib center, contacts at the slab ends 3 um away; textbook values, no process data), at three
dopings (both sides the same) and reverse biases 0 to 4 V. Cells: 10 nm at 1e17 and 3e17 cm^-3, 5 nm at
1e18 cm^-3. A cell larger than about two Debye lengths (4 nm at 1e18 cm^-3) gives an error of C' that oscillates with
the bias (up to 1.9 % at 1e18 cm^-3 with 10 nm cells), because the depletion edges cross the cells.

- charge-gpu: C' = dQ_n/dV (central difference, 20 mV), R' (1-D path model, Masetti mobility), see charge_gpu.rc.
- meow-gpu: the complex mode index from the absolute carrier distribution (Soref and Bennett at 1.55 um, from
  undoped Si). V_pi L = V lambda / (2 |Re(neff(V) - neff(0))|); loss = the free-carrier loss of the mode at V.
- f_3dB: intrinsic (R_s = 0, the same for all lengths) and for a lumped 0.5 mm device with a 50 Ohm source.

    uv run --frozen --project meow_gpu --with-editable ./charge_gpu python charge_gpu/examples/pn_modulator_rc.py
    cd charge_gpu && uv run python examples/pn_modulator_rc.py --no-modes      (charge only: C, R, f_3dB)
"""
import argparse
import json
import time

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np  # noqa: E402

from charge_gpu.modulator import plasma_dispersion, rib_pn  # noqa: E402
from charge_gpu.rc import small_signal  # noqa: E402

N_SI, N_SIO2, WL = 3.476, 1.444, 1.55
DOPINGS_CM3 = [1e17, 3e17, 1e18]
CELL = {1e17: 10e-9, 3e17: 10e-9, 1e18: 5e-9}          # m; Debye length 13, 7.5 and 4.1 nm
BIASES = [0.0, 1.0, 2.0, 4.0]
LENGTH, R_SOURCE = 0.5e-3, 50.0


def mode_indices(dev, sols, sub=8):
    """Complex neff (meow-gpu) for each carrier solution; the mode grid of examples/pn_modulator.py (20 nm).

    The index change of the carriers is the mean over sub x sub points in the 20 nm cell of each sample, not the value
    at the sample point: at 1e18 cm^-3 each depletion edge moves by 6 to 11 nm per volt, less than one mode cell, and
    point samples see that movement only when an edge crosses a sample point (V_pi L at 1 V: 0.93, 0.82, 0.78 V cm
    with 10, 5, 2.5 nm charge cells; with cell means 0.923, 0.930, 0.932 V cm)."""
    from scipy.interpolate import RegularGridInterpolator

    from meow_gpu.fde.operator import make_problem
    from meow_gpu.meow_backend import GPUModeSolver

    xc, yc = 0.5 * (dev.x[1:] + dev.x[:-1]) * 1e6, 0.5 * (dev.y[1:] + dev.y[:-1]) * 1e6    # um
    mx = np.arange(-1.5, 1.5 + 1e-9, 0.02)
    my = np.arange(-0.5, 0.9 + 1e-9, 0.02)
    xce, yce = 0.5 * (mx[1:] + mx[:-1]), 0.5 * (my[1:] + my[:-1])
    off = ((np.arange(sub) + 0.5) / sub - 0.5) * 0.02
    OX, OY = (a.ravel() for a in np.meshgrid(off, off, indexing="ij"))
    solver = GPUModeSolver(precision="mixed", extra=4)
    out = []
    for s in sols:
        z = np.zeros_like(s.n)
        dn_idx, dalpha = plasma_dispersion(z, z, s.n, s.p)                # from undoped Si: the absolute loss
        n_complex = dn_idx + 1j * dalpha * (WL * 1e-6) / (4 * np.pi)        # alpha = 4 pi k / lambda
        f = RegularGridInterpolator((xc, yc), n_complex, bounds_error=False, fill_value=0.0)
        eps = []
        for xx, yy in ((xce, my[:-1]), (mx[:-1], yce), (mx[:-1], my[:-1])):
            X, Y = np.meshgrid(xx, yy, indexing="ij")
            core = ((Y > 0) & (Y < 0.09) | (np.abs(X) < 0.25) & (Y > 0) & (Y < 0.22))
            pts = np.stack([(X.ravel()[:, None] + OX).ravel(), (Y.ravel()[:, None] + OY).ravel()], -1)
            n = np.where(core, N_SI, N_SIO2) + f(pts).reshape(X.size, OX.size).mean(axis=1).reshape(X.shape)
            eps.append(n**2)
        nv, _, _ = solver.solve(make_problem(*eps, mx, my, WL), 1)
        out.append(complex(nv[0]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-modes", action="store_true", help="charge only (no meow-gpu): C, R and f_3dB")
    args = ap.parse_args()
    rows, timing = [], {}
    for dop in DOPINGS_CM3:
        dev = rib_pn(doping=dop * 1e6, dx=CELL[dop])
        t0 = time.perf_counter()
        res = small_signal(dev, BIASES, dv=0.02)
        timing[f"charge_{dop:.0e}"] = round(time.perf_counter() - t0, 2)
        neff = None
        if not args.no_modes:
            t0 = time.perf_counter()
            neff = mode_indices(dev, [r.solution for r in res])
            timing[f"modes_{dop:.0e}"] = round(time.perf_counter() - t0, 2)
        for i, r in enumerate(res):
            row = {"doping_cm3": dop, "cell_nm": CELL[dop] * 1e9, "cells": int(dev.n_net.size), "bias_V": r.bias,
                   "C_fF_per_um": r.c * 1e15 * 1e-6,
                   "R_ohm_mm": r.r * 1e3, "R_n_ohm_mm": r.r_n * 1e3, "R_p_ohm_mm": r.r_p * 1e3,
                   "f3dB_intrinsic_GHz": r.f3db() * 1e-9,
                   "f3dB_0.5mm_50ohm_GHz": r.f3db(LENGTH, R_SOURCE) * 1e-9,
                   "converged": r.converged, "newton": r.solution.newton_iterations,
                   "cg": r.solution.cg_iterations}
            if neff is not None:
                loss = 10 * np.log10(np.e) * 4 * np.pi * neff[i].imag / (WL * 1e-4)          # dB/cm
                row["loss_dB_per_cm"] = loss
                if r.bias > 0:
                    d = neff[i].real - neff[0].real
                    row["dneff"] = d
                    row["VpiL_Vcm"] = r.bias * WL * 1e-4 / (2 * abs(d))
                    row["VpiL_x_loss_VdB"] = row["VpiL_Vcm"] * loss
            rows.append(row)
    print(json.dumps({"device": res[0].solution.device, "timing_s": timing, "rows": rows}, default=float))
    cols = ["doping_cm3", "bias_V", "VpiL_Vcm", "loss_dB_per_cm", "VpiL_x_loss_VdB", "C_fF_per_um", "R_ohm_mm",
            "f3dB_intrinsic_GHz", "f3dB_0.5mm_50ohm_GHz"]
    cols = [c for c in cols if any(c in r for r in rows)]
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join(f"{r[c]:.3g}" if c in r else "-" for c in cols) + " |")


if __name__ == "__main__":
    main()
