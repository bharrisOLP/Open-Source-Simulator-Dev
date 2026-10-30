"""The EME adapter (meow + meow-gpu) on the suite's device description: a straight guide transmits 1, also when
half of it is a Polygon (meow 0.15 drops z prisms; the adapter corrects that)."""
import warnings

import jax

jax.config.update("jax_enable_x64", True)
warnings.filterwarnings("ignore")

from photonsuite import CrossSection, Material, Polygon, Rect  # noqa: E402
from photonsuite.eme import transmission  # noqa: E402
from photonsuite.fdtd import Segment  # noqa: E402

SI, SIO2 = Material("Si", n=3.476), Material("SiO2", n=1.444)


def test_straight_guide_transmits_one():
    xs = CrossSection(SIO2, (Rect(-0.25, 0.25, 0.0, 0.22, SI),))
    r = transmission([Segment(0.0, 2.0, xs)], (-1.5, 1.5), (-0.8, 1.02), 1.55, step=0.04, cell_length=1.0,
                     num_modes=6)
    assert abs(r.transmission - 1.0) < 1e-4 and r.total_right <= 1.0 + 1e-6


def test_a_polygon_segment_continues_the_guide():
    rect = CrossSection(SIO2, (Rect(-0.25, 0.25, 0.0, 0.22, SI),))
    poly = CrossSection(SIO2, (Polygon(((-0.25, 0.0), (0.25, 0.0), (0.25, 0.22), (-0.25, 0.22)), SI),))
    r = transmission([Segment(0.0, 1.0, rect), Segment(1.0, 2.0, poly)], (-1.5, 1.5), (-0.8, 1.02), 1.55,
                     step=0.04, cell_length=1.0, num_modes=6)
    assert abs(r.transmission - 1.0) < 1e-4          # without the prism fix: the second half is oxide, T << 1


def test_structures_keep_the_fill_order():
    """A later shape replaces an earlier one, also between materials (review finding: one mesh_order for all
    shapes lost the SiN core inside an SiO2 trench of an SiN slab)."""
    import meow as mw
    import numpy as np

    from photonsuite.eme import structures

    sin = Material("SiN", n=2.0)
    xs = CrossSection(SIO2, (Rect(-1.0, 1.0, 0.0, 0.4, sin), Rect(-0.5, 0.5, 0.0, 0.4, SIO2),
                             Rect(-0.2, 0.2, 0.0, 0.4, sin)))
    st = structures([Segment(0.0, 1.0, xs)], (-1.5, 1.5), (-0.5, 0.9))
    mesh = mw.Mesh2D(x=np.arange(-1.5, 1.5001, 0.05), y=np.arange(-0.5, 0.9001, 0.05))
    cs = mw.CrossSection.from_cell(cell=mw.create_cells(st, mesh, np.array([1.0]))[0],
                                   env=mw.Environment(wl=1.55, T=25.0))
    n = np.real(np.asarray(cs.nx))
    X, Y = np.asarray(cs.mesh.Xx), np.asarray(cs.mesh.Yx)

    def n_at(x, y):
        return n[np.argmin(np.abs(X[:, 0] - x)), np.argmin(np.abs(Y[0] - y))]

    assert n_at(0.0, 0.2) == 2.0 and n_at(0.35, 0.2) == 1.444 and n_at(0.8, 0.2) == 2.0 and n_at(1.3, 0.2) == 1.444


def test_segments_are_checked():
    import pytest

    from photonsuite.fdtd import check_segments

    a = CrossSection(SIO2, (Rect(-0.25, 0.25, 0.0, 0.22, SI),))
    with pytest.raises(ValueError, match="contiguous"):
        check_segments([Segment(0.0, 1.0, a), Segment(1.5, 2.0, a)])
    with pytest.raises(ValueError, match="background"):
        check_segments([Segment(0.0, 1.0, a), Segment(1.0, 2.0, CrossSection(SI, a.shapes))])
    with pytest.raises(ValueError, match="input port"):
        check_segments([Segment(0.0, 0.8, a), Segment(0.8, 2.0, a)], first_min=1.0)
    check_segments([Segment(0.0, 1.0, a), Segment(1.0, 2.0, a)], first_min=1.0, last_min=0.5)
    from photonsuite.fdtd import transmission
    with pytest.raises(ValueError, match="shorter than the grid step"):          # raised before fdtdx is imported
        transmission([Segment(0.0, 1.0, a), Segment(1.0, 1.01, a), Segment(1.01, 2.0, a)], (-1, 1), (-1, 1), 1.55,
                     step=0.04)
