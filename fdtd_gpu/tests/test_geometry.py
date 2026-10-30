"""Geometry, rasterization and Harminv checks (no FDTD run)."""

import numpy as np
import pytest

import fdtd_gpu as fg
from fdtd_gpu.geometry import rasterize


def test_vector3_and_medium():
    v = fg.Vector3(1, 2) + fg.Vector3(0, 0, 3) * 2
    assert tuple(v) == (1, 2, 6) and abs(fg.Vector3(3, 4).norm() - 5) < 1e-15
    assert fg.Medium(index=2.0).epsilon == 4.0 and fg.Medium(epsilon=2.25).index == 1.5
    with pytest.raises(NotImplementedError):
        fg.Medium(epsilon=2.0, E_susceptibilities=[])


def test_rasterize_areas_and_order():
    """Cell-center sampling: the cell areas of a disk and of a triangle converge to the exact areas; a later
    object replaces an earlier one."""
    dx, n = 0.01, 400                                           # cell [-2, 2] um
    disk = fg.Cylinder(radius=1.0, material=fg.Medium(index=3))
    tri = fg.Prism([fg.Vector3(-1, -1), fg.Vector3(1, -1), fg.Vector3(-1, 1)], height=fg.inf,
                   material=fg.Medium(index=2))
    idx, mats = rasterize([disk], fg.Medium(), (n, n, 3), dx, (False, False, True))
    assert abs((idx[:, :, 0] == 1).sum() * dx**2 - np.pi) < 2e-3
    idx, mats = rasterize([disk, tri], fg.Medium(), (n, n, 3), dx, (False, False, True))
    assert abs((idx[:, :, 0] == 2).sum() * dx**2 - 2.0) < 2e-2           # triangle area 2 (diagonal: staircase)
    assert mats[2].epsilon == 4.0 and (idx[:, :, 0] == 1).sum() * dx**2 < np.pi - 1.0


def test_block_and_prism_center():
    idx, _ = rasterize([fg.Block(size=fg.Vector3(1.02, 0.52, fg.inf), center=fg.Vector3(0.25, 0),
                                 material=fg.Medium(index=2))], fg.Medium(), (40, 40, 3), 0.1, (False, False, True))
    ii, jj = np.nonzero(idx[:, :, 0])
    assert ii.min() == 17 and ii.max() == 27 and jj.min() == 17 and jj.max() == 22    # centers -1.95 + 0.1 i
    p = fg.Prism([fg.Vector3(0, 0), fg.Vector3(1, 0), fg.Vector3(1, 1), fg.Vector3(0, 1)], height=0.2,
                 center=fg.Vector3(2, 3, 4))
    lo, hi = p.bounds()
    assert np.allclose(lo, (1.5, 2.5, 3.9)) and np.allclose(hi, (2.5, 3.5, 4.1))


def test_harminv_two_modes():
    """Two damped cosines with known complex frequencies, plus noise at 1e-6."""
    dt = 0.02
    t = np.arange(20000) * dt
    rng = np.random.default_rng(0)
    s = (np.exp(-0.002 * t) * np.cos(2 * np.pi * 0.61 * t + 0.3) + 0.5 * np.exp(-0.01 * t)
         * np.cos(2 * np.pi * 0.67 * t) + 1e-6 * rng.standard_normal(len(t)))
    modes = fg.harminv(s, dt, 0.55, 0.75)
    strong = sorted(modes, key=lambda m: -abs(m.amp))[:2]
    strong = sorted(strong, key=lambda m: m.freq)
    assert abs(strong[0].freq - 0.61) < 1e-6 and abs(strong[1].freq - 0.67) < 1e-6
    assert abs(strong[0].decay + 0.002 / (2 * np.pi)) < 1e-6 and abs(strong[1].decay + 0.01 / (2 * np.pi)) < 1e-6


def test_unsupported_options_raise():
    with pytest.raises(NotImplementedError):
        fg.Simulation(cell_size=fg.Vector3(1, 1, 0), resolution=10, eps_averaging=True)
    with pytest.raises(NotImplementedError):
        fg.Block(size=fg.Vector3(1, 1, 1), e1=fg.Vector3(1, 1, 0))
    with pytest.raises(NotImplementedError):
        fg.Simulation(cell_size=fg.Vector3(1, 1, 0), resolution=10, k_point=fg.Vector3(0.1, 0, 0))


def test_prism_center_is_vertex_centroid():
    """With a center, the centroid of all vertices (top and bottom) goes to the center (the MIT package rule): for
    this triangle the centroid (1, 1) differs from the center of the bounding box (1.5, 1.5)."""
    tri = [fg.Vector3(0, 0), fg.Vector3(3, 0), fg.Vector3(0, 3)]
    p = fg.Prism(tri, height=1.0, center=fg.Vector3(2, 2, 4))
    lo, hi = p.bounds()
    assert np.allclose(lo, (1, 1, 3.5)) and np.allclose(hi, (4, 4, 4.5))
    assert np.allclose(tuple(fg.Prism(tri, height=1.0).center), (1, 1, 0.5))
    assert np.allclose(tuple(fg.Prism(tri, height=1.0, axis=fg.Vector3(0, 0, -1)).center), (1, 1, -0.5))


def test_2d_in_plane_mode_raises():
    """2-D cell: an in-plane (Hz, EVEN_Z) mode source or mode coefficient raises (fdtdx gives NaN or the Ez mode);
    1-D mode monitors raise."""
    src = fg.GaussianSource(0.65, fwidth=0.1)
    cell = fg.Vector3(6, 4, 0)
    with pytest.raises(NotImplementedError):
        fg.Simulation(cell_size=cell, resolution=10, sources=[fg.EigenModeSource(src, size=fg.Vector3(0, 2),
                                                                                  eig_parity=fg.EVEN_Z)])
    sim = fg.Simulation(cell_size=cell, resolution=10, sources=[fg.EigenModeSource(src, size=fg.Vector3(0, 2),
                                                                                    eig_parity=fg.ODD_Z)])
    mon = sim.add_mode_monitor(0.65, 0.1, 3, fg.ModeRegion(center=fg.Vector3(1), size=fg.Vector3(0, 2)))
    with pytest.raises(NotImplementedError):
        sim.get_eigenmode_coefficients(mon, [1], eig_parity=fg.EVEN_Z)
    assert sim._filter_pol(fg.ODD_Z, 0) == "tm" and sim._filter_pol(fg.NO_PARITY, 0) == "tm"   # E along z
    sim1 = fg.Simulation(cell_size=fg.Vector3(0, 0, 6), resolution=10)
    with pytest.raises(NotImplementedError):
        sim1.add_mode_monitor(0.65, 0.1, 3, fg.ModeRegion(center=fg.Vector3(0, 0, 1)))
