"""Power-correct FDTD transmission of the taper into its fundamental mode.

T00 = T_flux * f_out / f_cal: T_flux = P_out / P_in (Poynting flux), f_out = |a_out|^2 / P_out (taper run), f_cal = the
same ratio for a straight 2.0 um guide (a pure output mode). The overlap normalization bias of the output mode cancels.
Prints one JSON line."""
import json
import sys

from photonsuite import CrossSection, Material, Rect
from photonsuite.fdtd import Segment, transmission

step = float(sys.argv[1]) if len(sys.argv) > 1 else 0.025
n_seg = int(sys.argv[2]) if len(sys.argv) > 2 else 100
SI, SIO2 = Material("Si", n=3.476), Material("SiO2", n=1.444)


def strip(w):
    return CrossSection(SIO2, (Rect(-w / 2, w / 2, 0.0, 0.22, SI),))


win = ((-2.0, 2.0), (-0.8, 1.02))
cal = transmission([Segment(0.0, 12.0, strip(2.0))], *win, 1.55, step=step)
segs = [Segment(0.0, 1.0, strip(0.5))]
for k in range(n_seg):
    segs.append(Segment(1.0 + 10.0 * k / n_seg, 1.0 + 10.0 * (k + 1) / n_seg, strip(0.5 + 1.5 * (k + 0.5) / n_seg)))
segs.append(Segment(11.0, 12.0, strip(2.0)))
tap = transmission(segs, *win, 1.55, step=step)
print(json.dumps({"step_um": step, "segments": n_seg, "T_overlap_ratio": tap.transmission,
                  "T_flux": tap.flux_transmission, "f_out": tap.out_mode_fraction, "f_cal": cal.out_mode_fraction,
                  "cal_T_flux": cal.flux_transmission,
                  "T00": tap.flux_transmission * tap.out_mode_fraction / cal.out_mode_fraction,
                  "seconds": round(tap.seconds + cal.seconds, 1)}))
