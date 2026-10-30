"""The FDTD adapter (fdtdx) on a short straight guide: the transmission into the fundamental mode and the flux
transmission are about 1, also with float64 on (the adapter runs fdtdx in float32). Needs fdtdx: it runs in the GPU
environment (.venv-rocm) and is skipped elsewhere."""
import jax
import pytest

pytest.importorskip("fdtdx")

from photonsuite import CrossSection, Material, Rect  # noqa: E402
from photonsuite.fdtd import Segment, transmission  # noqa: E402

SI, SIO2 = Material("Si", n=3.476), Material("SiO2", n=1.444)


def test_straight_guide_with_float64_on():
    assert jax.config.jax_enable_x64          # import photonsuite enables it
    xs = CrossSection(SIO2, (Rect(-0.25, 0.25, 0.0, 0.22, SI),))
    r = transmission([Segment(0.0, 3.0, xs)], (-1.2, 1.2), (-0.8, 1.0), 1.55, step=0.04)
    assert abs(r.transmission - 1.0) < 1e-3
    assert abs(r.flux_transmission - 1.0) < 1e-2
    assert jax.config.jax_enable_x64          # restored after the run
