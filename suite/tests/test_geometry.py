"""Polygon shapes: a rib with sloped sidewalls (top 0.45 um, bottom 0.55 um) has a mode index between those of the
0.45 um and the 0.55 um rectangles."""
import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from photonsuite import CrossSection, Material, Polygon, Rect  # noqa: E402
from photonsuite.physics import Window, mode_index  # noqa: E402

SI, SIO2 = Material("Si", n=3.476), Material("SiO2", n=1.444)
WIN = Window(-1.5, 1.5, -0.9, 1.1, 0.02)


def test_polygon_contains():
    trap = Polygon(((-0.275, 0.0), (0.275, 0.0), (0.225, 0.22), (-0.225, 0.22)), SI)
    X, Y = np.array([0.0, 0.26, 0.26, 0.0]), np.array([0.1, 0.01, 0.21, 0.3])
    assert list(trap.contains(X, Y)) == [True, True, False, False]


def test_sloped_rib_index_between_its_rectangles():
    def n_of(shape):
        return float(np.real(mode_index(CrossSection(SIO2, (shape,)), WIN, 1.55)[0]))
    narrow = n_of(Rect(-0.225, 0.225, 0.0, 0.22, SI))
    wide = n_of(Rect(-0.275, 0.275, 0.0, 0.22, SI))
    trap = n_of(Polygon(((-0.275, 0.0), (0.275, 0.0), (0.225, 0.22), (-0.225, 0.22)), SI))
    assert narrow < trap < wide


def test_polygon_vertices_and_validity():
    import pytest
    p = Polygon([[-0.2, 0.0], [0.2, 0.0], [0.2, 0.2], [-0.2, 0.2]], SI)       # a list: stored as a tuple, hashable
    assert isinstance(p.vertices, tuple) and hash(p) == hash(Polygon(p.vertices, SI))
    with pytest.raises(ValueError):
        Polygon(((0, 0), (1, 1), (1, 0), (0, 1)), SI)                        # a bow tie (self-intersecting)
    with pytest.raises(ValueError):
        Polygon(((0, 0), (1, 1)), SI)
