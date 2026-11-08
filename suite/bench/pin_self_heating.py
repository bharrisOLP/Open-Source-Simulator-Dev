"""A forward-biased PIN phase shifter with self-heating (photonsuite.pin): the carrier and the thermal parts of the
mode index change, the net phase, P_pi, I_pi, the loss and the cross-over, on the PIN rib of charge-gpu
(pin.pin_rib: textbook values, no process data).

One JSON row per bias and lifetime, then one row per lifetime with the cross-over (or none in the range).

    bash tools/gpu_run.sh suite python bench/pin_self_heating.py --tau 1e-9 1e-8 --device-temperature
    bash tools/gpu_run.sh suite python bench/pin_self_heating.py --thermal-step 0.02
    bash tools/gpu_run.sh suite python bench/pin_self_heating.py --heat slab,core

--heat slab,core: I' V spread uniformly over the intrinsic zone (the tags of pin_rib) in place of the heat density
of the drift-diffusion solution. --device-temperature: also solve the drift-diffusion again at each bias with the
device at 300 K plus the mean rise of its Si (Device.T of charge-gpu, one pass from the 300 K solution) and give the
change of the current (the size of the electro-thermal feedback that the workflow leaves out).

The thermal environment (the default: the window and the substrate +-5 um wide, a substrate 3 um deep with its bottom
at 300 K, adiabatic window sides, the slab ending at the contacts +-1.5 um from the center, in oxide):
--thermal-half-width W: the thermal window and the substrate +-W um wide (the adiabatic sides farther away).
--substrate-depth D: the substrate D um deep (the thermal window bottom at y = -2 - D um).
--heat-sink: a crude heat path to the contacts: the doped slab continues (p+ and n+) to the window sides, and both
window sides are at 300 K over their whole height. The charge window and the carriers do not change.

    bash tools/gpu_run.sh suite python bench/pin_self_heating.py --thermal-half-width 20
    bash tools/gpu_run.sh suite python bench/pin_self_heating.py --heat-sink
"""
import argparse
import dataclasses
import json
import resource
import time

import photonsuite  # noqa: F401  (float64 in JAX)
from photonsuite.physics import Window
from photonsuite.pin import DRIFT_DIFFUSION, pin_electrical, pin_optical, pin_rib

WL = 1.55
BIASES = [0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, nargs="+", default=[1e-9], help="SRH lifetimes tau_n = tau_p (s, 300 K)")
    ap.add_argument("--biases", type=float, nargs="+", default=BIASES, help="forward biases (V), increasing")
    ap.add_argument("--thermal-step", type=float, default=0.05, help="thermal grid step (um)")
    ap.add_argument("--optics-step", type=float, default=0.02, help="mode grid step (um)")
    ap.add_argument("--sub", type=int, default=8, help="carrier index: sub x sub points per optics cell")
    ap.add_argument("--heat", default=DRIFT_DIFFUSION, help="'drift-diffusion' or tags (comma-separated): I' V uniform")
    ap.add_argument("--device-temperature", action="store_true", help="the current at 300 K + the mean Si rise")
    ap.add_argument("--thermal-half-width", type=float, default=5.0, help="thermal window and substrate +-W (um)")
    ap.add_argument("--substrate-depth", type=float, default=3.0, help="substrate thickness (um)")
    ap.add_argument("--heat-sink", action="store_true", help="the slab to the window sides, the sides at 300 K")
    args = ap.parse_args()
    w = args.thermal_half_width
    xs = pin_rib(substrate_half_width=w, substrate=args.substrate_depth,
                 slab_half_width=w if args.heat_sink else None)
    optics = Window(-1.5, 1.5, -0.5, 0.9, args.optics_step)
    charge = Window(-1.5, 1.5, -0.5, 0.72, 0.01)
    thermal = Window(-w, w, -2.0 - args.substrate_depth, 2.3, args.thermal_step)
    sides = 300.0 if args.heat_sink else None
    heat = args.heat if args.heat == DRIFT_DIFFUSION else tuple(args.heat.split(","))
    for tau in args.tau:
        t0 = time.perf_counter()
        el = pin_electrical(xs, charge, args.biases, tau)
        r = pin_optical(xs, optics, thermal, WL, el, heat=heat, sub=args.sub, core_tag="core", side_temperature=sides)
        seconds = time.perf_counter() - t0
        base = {"tau_s": tau, "thermal_step_um": args.thermal_step, "optics_step_um": args.optics_step,
                "thermal_half_width_um": w, "substrate_depth_um": args.substrate_depth, "heat_sink": args.heat_sink,
                "sub": args.sub, "heat": r.heat_model, "neff0": [r.neff0.real, r.neff0.imag]}
        for k, v in enumerate(r.biases):
            sol = el.solutions[k]
            row = dict(base, bias_V=float(v), I_A_per_m=float(r.current[k]), P_W_per_m=float(r.power[k]),
                       heat_W_per_m=float(r.heat[k]), heat_rel=float(r.heat[k] / r.power[k] - 1),
                       joule_W_per_m=float(r.heat_joule[k]), recombination_W_per_m=float(r.heat_recombination[k]),
                       dd_heat_rel=float((r.heat_joule[k] + r.heat_recombination[k]) / r.power[k] - 1),
                       je_W_per_m=float(r.je[k]), newton=sol.iterations,
                       thermal_iterations=r.temperatures[k].iterations,
                       rise_si_K=float(r.rise_semiconductor[k]), rise_core_K=float(r.rise_core[k]),
                       dneff_carrier=[r.dneff_carrier[k].real, r.dneff_carrier[k].imag],
                       dneff_thermal=float(r.dneff_thermal[k].real), dneff=[r.dneff[k].real, r.dneff[k].imag],
                       first_carrier=[r.first_carrier[k].real, r.first_carrier[k].imag],
                       first_thermal=float(r.first_thermal[k].real),
                       additivity=float((r.dneff[k] - r.dneff_carrier[k] - r.dneff_thermal[k]).real),
                       phase_rad_per_mm=float(r.phase_per_length[k] * 1e-3), L_pi_mm=float(r.l_pi[k] * 1e3),
                       P_pi_mW=float(r.p_pi[k] * 1e3), I_pi_mA=float(r.i_pi[k] * 1e3),
                       loss_dB_per_cm=float(r.loss_db_per_cm[k]), loss_in_L_pi_dB=float(r.loss_in_l_pi_db[k]))
            if args.device_temperature:
                from charge_gpu import drift_diffusion as dd
                hot = dataclasses.replace(el.device, T=300.0 + float(r.rise_semiconductor[k]))
                s_hot = dd.solve(hot, 0.0, float(v), guess=sol, tau_n=tau, tau_p=tau)
                row.update(device_T_K=hot.T, hot_converged=s_hot.converged,
                           I_hot_rel=float(s_hot.current / sol.current - 1))
            print(json.dumps(row), flush=True)
        print(json.dumps(dict(base, crossover=r.crossover(), charge_s=round(el.seconds, 2),
                              optics_thermal_s=round(r.seconds, 2), seconds=round(seconds, 2),
                              max_rss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 3))),
              flush=True)


if __name__ == "__main__":
    main()
