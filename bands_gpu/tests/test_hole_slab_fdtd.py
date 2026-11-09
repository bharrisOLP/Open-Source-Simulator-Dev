"""The hole slab against 3-D FDTD (fdtd_gpu, an independent method; the setup: tests/hole_slab_fdtd.py): the z-even
(TE-like) bands of the 7/4 crystal at K and M, FDTD at resolutions 16 and 32 against bands_gpu (32 x 32 x 129).

- Selection by index: the modes that the FDTD finds (|Q| >= 1e3, amplitude >= 1e-2 of the largest), in order, are the
  even bands of bands_gpu at that k, in order, each 0 to 2.5 % above its band (no mode of another k point that folds
  onto Gamma).
- z-odd control: the same runs with E_z dipoles and probes (parity="odd") find the z-odd modes of the same grid. No
  mode of the z-even run is a mode of the z-odd run (the same grid gives the same frequency to Harminv precision), so
  the z-even runs do not pick up a z-odd mode. This matters at M: there the z-odd band 1 is near the even band 2, and
  at 16 and 32 voxels per a the FDTD error is larger than their distance.
- Convergence: the FDTD error falls with the resolution, and the first-order extrapolation 2 f(32) - f(16) (the
  diagonal averaging of fdtd_gpu is first order for E in the plane of the curved hole wall) agrees with bands_gpu and
  is much nearer to the even band than to any z-odd band of bands_gpu.

Measured (GPU computer; test prints, not logged rows): band 1 at K, FDTD 0.266168 (16) and 0.264919 (32),
extrapolation 0.263670, bands_gpu 0.263947; band 2 at M, 0.347643 and 0.344073, extrapolation 0.340503, bands_gpu
0.340313.
"""
import sys
from pathlib import Path

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, str(Path(__file__).resolve().parent))
pytest.importorskip("fdtd_gpu")

import hole_slab_fdtd as hs  # noqa: E402


@pytest.mark.slow
def test_hole_slab_fdtd_against_bands_gpu():
    cell = hs.bands_cell(32)
    ref = hs.even_bands(cell, ("K", "M"), num=4)
    odd = hs.even_bands(cell, ("K", "M"), num=4, parity="odd")
    assert ref["K"][1] and ref["M"][1] and odd["K"][1] and odd["M"][1]
    pick = {"K": 0, "M": 1}                                  # band 1 at K, band 2 at M: the gap edges
    for point in ("K", "M"):
        bands, bands_odd = ref[point][0], odd[point][0]
        f = {}
        for res in (16, 32):
            strong = hs.strong_modes(hs.run_fdtd(point, res)["modes"])
            control = hs.strong_modes(hs.run_fdtd(point, res, parity="odd")["modes"])
            # by index: each FDTD mode is the even band of the same index, 0 to 2.5 % above it (the FDTD error)
            rel = np.array(strong) / bands[: len(strong)] - 1
            gap = min(abs(a - b) for a in strong for b in control) if control else np.inf
            print(point, res, "FDTD even", strong, "bands_gpu", bands, "relative", rel)
            print(point, res, "FDTD odd control", control, "bands_gpu odd", bands_odd, "nearest even-odd", gap)
            assert 2 <= len(strong) <= 3 and np.all(rel > 0) and np.all(rel < 0.025)
            assert control and gap > 1e-4                    # no z-odd mode of the same grid in the z-even run
            f[res] = strong[pick[point]]
        b = bands[pick[point]]
        err16, err32 = f[16] - b, f[32] - b
        extrapolated = 2 * f[32] - f[16]
        print(point, "errors", err16, err32, "extrapolated", extrapolated, "bands_gpu", b,
              "nearest odd band", bands_odd[np.argmin(abs(bands_odd - extrapolated))])
        assert 0 < err32 < 0.6 * err16                       # first order: about 1/2
        assert abs(extrapolated / b - 1) < 1.5e-3
        assert abs(extrapolated - b) < 0.1 * np.min(np.abs(bands_odd - extrapolated))
