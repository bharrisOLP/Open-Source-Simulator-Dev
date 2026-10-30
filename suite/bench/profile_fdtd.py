"""Profile (cProfile) of the FDTD run of a taper with sloped Polygon or Rect shapes. Prints the 25 functions with the
largest cumulative time.

    python bench/profile_fdtd.py [sloped|rect] [step um] [n_seg]
"""
import cProfile
import math
import pstats
import sys

from photonsuite import CrossSection, Material, Polygon, Rect
from photonsuite.fdtd import Segment, transmission

kind = sys.argv[1] if len(sys.argv) > 1 else "sloped"
step = float(sys.argv[2]) if len(sys.argv) > 2 else 0.04
n_seg = int(sys.argv[3]) if len(sys.argv) > 3 else 100
SI, SIO2 = Material("Si", n=3.476), Material("SiO2", n=1.444)
inset = 0.22 / math.tan(math.radians(80.0))


def xs(w):
    if kind == "rect":
        return CrossSection(SIO2, (Rect(-w / 2, w / 2, 0.0, 0.22, SI),))
    verts = ((-w / 2, 0.0), (w / 2, 0.0), (w / 2 - inset, 0.22), (-w / 2 + inset, 0.22))
    return CrossSection(SIO2, (Polygon(verts, SI),))


segs = [Segment(0.0, 1.0, xs(0.5))]
for k in range(n_seg):
    segs.append(Segment(1.0 + 10.0 * k / n_seg, 1.0 + 10.0 * (k + 1) / n_seg, xs(0.5 + 1.5 * (k + 0.5) / n_seg)))
segs.append(Segment(11.0, 12.0, xs(2.0)))
prof = cProfile.Profile()
prof.enable()
r = transmission(segs, (-2.0, 2.0), (-0.8, 1.02), 1.55, step=step)
prof.disable()
print(f"{kind}: T {r.transmission:.4f}, run {r.seconds:.1f} s", flush=True)
pstats.Stats(prof).sort_stats("cumulative").print_stats(25)
