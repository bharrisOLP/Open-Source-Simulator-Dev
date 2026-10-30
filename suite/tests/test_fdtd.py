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


def _inv_eps(segments, step=0.04):
    import numpy as np

    from photonsuite.fdtd import _setup

    with jax.enable_x64(False):
        _, arrays, *_ = _setup(segments, (-1.2, 1.2), (-0.8, 1.0), 1.55, step)
        return np.asarray(arrays.inv_permittivities)


def test_a_rectangle_polygon_places_like_the_rect():
    import numpy as np

    from photonsuite import Polygon

    rect = CrossSection(SIO2, (Rect(-0.24, 0.24, 0.0, 0.2, SI),))
    poly = CrossSection(SIO2, (Polygon(((-0.24, 0.0), (0.24, 0.0), (0.24, 0.2), (-0.24, 0.2)), SI),))
    a, b = _inv_eps([Segment(0.0, 2.0, rect)]), _inv_eps([Segment(0.0, 2.0, poly)])
    assert np.array_equal(a, b)
    si_cells = np.sum(np.isclose(a, 1 / SI.n**2))
    assert si_cells == pytest.approx(12 * 5 * a.shape[-3], rel=0.2)   # 0.48 x 0.2 um at 40 nm, along z


def test_a_sloped_polygon_and_the_fill_order():
    import numpy as np

    from photonsuite import Polygon

    def si_cells(xs):
        return int(np.sum(np.isclose(_inv_eps([Segment(0.0, 2.0, xs)]), 1 / SI.n**2)))

    narrow = si_cells(CrossSection(SIO2, (Rect(-0.2, 0.2, 0.0, 0.2, SI),)))
    wide = si_cells(CrossSection(SIO2, (Rect(-0.32, 0.32, 0.0, 0.2, SI),)))
    trap = si_cells(CrossSection(SIO2, (Polygon(((-0.32, 0.0), (0.32, 0.0), (0.2, 0.2), (-0.2, 0.2)), SI),)))
    assert narrow < trap < wide
    # a later shape replaces an earlier one: an oxide trench in a Si slab, then a Si core in the trench
    filled = CrossSection(SIO2, (Rect(-1.0, 1.0, 0.0, 0.2, SI), Rect(-0.5, 0.5, 0.0, 0.2, SIO2),
                                 Rect(-0.2, 0.2, 0.0, 0.2, SI)))
    inv = _inv_eps([Segment(0.0, 2.0, filled)])[0]             # first component: (z, x, y)
    iz, iy = inv.shape[0] // 2, int((0.1 + 1.2) / 0.04)           # with the PML (0.4 um), x starts at -1.6, y at -1.2

    def si_at(x):
        return bool(np.isclose(inv[iz, int((x + 1.6) / 0.04), iy], 1 / SI.n**2))

    assert [si_at(x) for x in (-0.8, -0.35, 0.0, 0.35, 0.8, 1.1)] == [True, False, True, False, True, False]


def test_segments_that_are_not_whole_cells_leave_no_gaps():
    """Review of the 40 nm taper (T 0.17): 100 nm segments are 2.5 cells, and fdtdx rounds the position and the size
    of each box separately, which left oxide gaps between segments. The adapter rounds the shared edges once."""
    import numpy as np

    xs = CrossSection(SIO2, (Rect(-0.25, 0.25, 0.0, 0.22, SI),))
    cut = [Segment(0.0, 1.0, xs)] + [Segment(1.0 + 0.1 * k, 1.1 + 0.1 * k, xs) for k in range(10)]
    cut += [Segment(2.0, 3.0, xs)]
    assert np.array_equal(_inv_eps(cut), _inv_eps([Segment(0.0, 3.0, xs)]))


def test_runs_on_the_gpu_after_an_eme_run():
    """The adapter places the run on the first GPU, also after an EME run in the same process (that made the FDTD
    9 times slower before), and reports the device of the field arrays."""
    from photonsuite.eme import transmission as eme_transmission

    gpus = [d for d in jax.devices() if d.platform != "cpu"]
    if not gpus:
        pytest.skip("no GPU")
    xs = CrossSection(SIO2, (Rect(-0.25, 0.25, 0.0, 0.22, SI),))
    eme_transmission([Segment(0.0, 2.0, xs)], (-1.5, 1.5), (-0.8, 1.02), 1.55, step=0.04, cell_length=1.0,
                     num_modes=4)
    r = transmission([Segment(0.0, 3.0, xs)], (-1.2, 1.2), (-0.8, 1.0), 1.55, step=0.04)
    assert r.device == str(gpus[0])



def _si_mask(segments):
    """The Si cells of the middle z plane of the placed simulation, as a bool (x, y) array."""
    import numpy as np

    inv = _inv_eps(segments)[0]
    return np.isclose(inv[inv.shape[0] // 2], 1 / SI.n**2)


def _centers_inside(shape):
    """The cells whose centers are inside the shape, on the grid of _inv_eps (40 nm, x from -1.6, y from -1.2)."""
    import numpy as np

    xc, yc = -1.6 + (np.arange(80) + 0.5) * 0.04, -1.2 + (np.arange(65) + 0.5) * 0.04
    X, Y = np.meshgrid(xc, yc, indexing="ij")
    return np.asarray(shape.contains(X, Y), bool)


def test_shapes_with_corners_off_the_grid_place_the_cells_of_their_centers():
    """Review finding: fdtdx placed a Polygon by its bounding box, rounded in position and size separately (up to
    half a cell off). Now each shape fills exactly the cells whose centers are inside it."""
    import numpy as np

    from photonsuite import Polygon

    rect = Rect(-0.264, 0.232, 0.013, 0.207, SI)
    poly = Polygon(((-0.264, 0.013), (0.232, 0.013), (0.232, 0.207), (-0.264, 0.207)), SI)
    sloped = Polygon(((-0.31, 0.0), (0.29, 0.0), (0.21, 0.22), (-0.23, 0.22)), SI)
    for shape in (rect, poly, sloped):
        got = _si_mask([Segment(0.0, 2.0, CrossSection(SIO2, (shape,)))])
        assert np.array_equal(got, _centers_inside(shape)), shape
    assert np.array_equal(_si_mask([Segment(0.0, 2.0, CrossSection(SIO2, (rect,)))]),
                          _si_mask([Segment(0.0, 2.0, CrossSection(SIO2, (poly,)))]))


def test_a_polygon_next_to_a_rect_and_at_the_volume_edge():
    import numpy as np

    from photonsuite import Polygon

    left = Polygon(((-0.5, 0.0), (0.013, 0.0), (0.013, 0.2), (-0.5, 0.15)), SI)
    right = Rect(0.013, 0.5, 0.0, 0.2, SI)
    both = _si_mask([Segment(0.0, 2.0, CrossSection(SIO2, (left, right)))])
    assert np.array_equal(both, _centers_inside(left) | _centers_inside(right))      # no gap column
    tall = Polygon(((-0.3, 0.22), (0.3, 0.22), (0.3, 5.0), (-0.3, 5.0)), SI)          # beyond the volume
    assert np.array_equal(_si_mask([Segment(0.0, 2.0, CrossSection(SIO2, (tall,)))]), _centers_inside(tall))
    sliver = Polygon(((0.0, 0.0), (0.01, 0.0), (0.01, 0.2), (0.0, 0.2)), SI)           # no cell center inside
    assert not _si_mask([Segment(0.0, 2.0, CrossSection(SIO2, (sliver,)))]).any()
