"""The switching time of the thermo-optic phase shifter of the suite (photonsuite.switching): the phase of the mode
against time after the heater switches on or off, with a constant kappa and with kappa(T). One JSON line per model
and switch.

The cross-section of tests/test_workflows.py (0.5 x 0.22 um Si core on 2 um oxide, 2 um heater 2.2 um above the
oxide-core interface, 3 um Si substrate, 10 um wide), with the heat capacities of Incropera et al., "Fundamentals of
Heat and Mass Transfer", Tables A.1 (Si, tungsten for the heater metal) and A.2 (fused silica). Models:
    constant        constant kappa and rho c_p
    kappa_T         Si kappa(T) (thermal_gpu.nonlinear.SI_POWER_LAW) and SiO2 kappa(T) (SIO2_LINEAR)
Columns: the steady dneff of phase_shifter_ppi and of the full solve at the final time; at each full-solve time the
first-order and the full dneff; the 10-90 % times of the phase and of the mean core temperature; the spread of
dneff / core rise; the heat-diffusion estimates (photonsuite.switching.slab_switch_time); run times.
--scale-check: full solves with the steady field times 0.1 and 1 (the first-order error must fall by 10 when the
formula is the exact derivative of the discrete mode index)."""
import argparse
import json
import time

import numpy as np

from photonsuite import CrossSection, Material, Rect
from photonsuite.physics import Window, mode_index, phase_shifter_ppi
from photonsuite.switching import phase_transient, slab_switch_time

SI = Material("Si", n=3.476, dn_dT=1.86e-4, kappa=148.0, rho_cp=2330.0 * 712.0, eps_r=11.7, semiconductor=True)
SIO2 = Material("SiO2", n=1.444, dn_dT=0.95e-5, kappa=1.4, rho_cp=2220.0 * 745.0, eps_r=3.9)
METAL = Material("heater", n=1.444, kappa=100.0, rho_cp=19300.0 * 132.0)

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", choices=("constant", "kappa_T"), default=["constant", "kappa_T"])
ap.add_argument("--switches", nargs="+", choices=("on", "off"), default=["on", "off"])
ap.add_argument("--thermal-step", type=float, default=0.05)
ap.add_argument("--optics-step", type=float, default=0.02)
ap.add_argument("--power", type=float, default=100.0)
ap.add_argument("--per-stage", type=int, default=20)
ap.add_argument("--dt0", type=float, default=1e-8)
ap.add_argument("--full-times", type=float, nargs="*", default=[1.5e-6, 5e-6, 20e-6])
ap.add_argument("--scale-check", action="store_true")
a = ap.parse_args()

xs = CrossSection(SIO2, (Rect(-5, 5, -5, -2, SI, tag="substrate"), Rect(-0.25, 0.25, 0, 0.22, SI, tag="core"),
                         Rect(-1, 1, 2.2, 2.3, METAL, tag="heater")))
optics = Window(-1.5, 1.5, -0.9, 1.1, a.optics_step)
thermal = Window(-5, 5, -5, 2.3, a.thermal_step)
t0 = time.perf_counter()
ppi, dneff_ppi, _ = phase_shifter_ppi(xs, optics, thermal, "heater", 1.55, a.power)
t_ppi = time.perf_counter() - t0
alpha = SIO2.kappa / SIO2.rho_cp
est = {"tau_us": 4 * 4.2e-6**2 / (np.pi**2 * alpha) * 1e6,                 # 4.2 um oxide from substrate to heater
       "single_exp_t10_t90_rise_us": [t * 1e6 for t in slab_switch_time(4.2e-6, alpha)],
       "slab_core_center_t10_t90_rise_us": [t * 1e6 for t in slab_switch_time(4.2e-6, alpha, 2.11e-6)]}
print(json.dumps({"case": "steady", "P_pi_mW": ppi * 1e3, "dneff": dneff_ppi, "seconds": round(t_ppi, 2),
                  "diffusion_estimate": est}), flush=True)
laws = None
for model in a.models:
    if model == "kappa_T":
        from thermal_gpu.nonlinear import SI_POWER_LAW, SIO2_LINEAR
        laws = {"Si": SI_POWER_LAW, "SiO2": SIO2_LINEAR}
    for switch in a.switches:
        t0 = time.perf_counter()
        r = phase_transient(xs, optics, thermal, "heater", 1.55, a.power, switch=switch, core_tag="core",
                            kappa_laws=laws if model == "kappa_T" else None, dt0=a.dt0, per_stage=a.per_stage,
                            full_times=a.full_times)
        sec = time.perf_counter() - t0
        k_end = len(r.times) - 1
        full = []
        for t, d in r.full.items():
            k = int(np.argmin(np.abs(r.times - t)))
            full.append({"t_us": t * 1e6, "first_order": r.dneff[k], "full": d, "rel": (r.dneff[k] - d) / d})
        target = r.dneff_steady if switch == "on" else 0.0
        ok = r.core_rise > 0.01 * abs(r.core_steady)
        ratio = r.dneff[ok] / r.core_rise[ok]
        row = {"case": f"{model}/{switch}", "thermal_step_um": a.thermal_step, "optics_step_um": a.optics_step,
               "steps": len(r.times) - 1, "neff0": r.neff0.real, "dneff_steady_first_order": r.dneff_steady,
               "dneff_end": r.dneff[k_end], "end_vs_final": (r.dneff[k_end] - target) / r.dneff_steady,
               "full_end_vs_ppi": (r.full[float(r.times[-1])] - (dneff_ppi if switch == "on" else 0.0)) / dneff_ppi,
               "full": full, "t_end_us": r.times[-1] * 1e6, "t10_us": r.t10 * 1e6, "t90_us": r.t90 * 1e6,
               "switch_time_us": r.switch_time * 1e6,
               "core_switch_time_us": r.core_switch_time * 1e6, "core_steady_K": r.core_steady,
               "ratio_dneff_per_K": [float(ratio.min()), float(ratio.max())],
               "steady_ratio": r.dneff_steady / r.core_steady, "uniform_slope": float(r.sensitivity.sum()),
               "picard": getattr(r.thermal, "picard_iterations", None), "seconds": round(sec, 2)}
        if model == "constant" and switch == "on":          # a uniform rise of 1 K: full solve against sum S
            d1 = float(np.real(mode_index(xs, optics, 1.55, dn=lambda X, Y: xs.sample(X, Y)[2]["dn_dT"])[0]
                               - r.neff0))
            row["one_kelvin_full"] = d1
            row["uniform_slope_vs_full"] = row["uniform_slope"] / d1 - 1
        print(json.dumps(row), flush=True)
        if a.scale_check and switch == "on" and model == "constant":
            from scipy.interpolate import RegularGridInterpolator
            box = r.thermal.monitors[-1, :r.x.size * r.y.size].reshape(r.x.size, r.y.size) - 300.0
            n0 = mode_index(xs, optics, 1.55)[0]
            for s in (0.1, 1.0):
                f = RegularGridInterpolator((r.x, r.y), s * box, bounds_error=False, fill_value=None)

                def dn(X, Y, f=f):
                    return xs.sample(X, Y)[2]["dn_dT"] * f(np.stack([X.ravel(), Y.ravel()], -1)).reshape(X.shape)

                d_full = float(np.real(mode_index(xs, optics, 1.55, dn=dn)[0] - n0))
                d_lin = s * float(np.sum(r.sensitivity * box))
                print(json.dumps({"case": "scale-check", "scale": s, "full": d_full, "first_order": d_lin,
                                  "rel": (d_lin - d_full) / d_full}), flush=True)
