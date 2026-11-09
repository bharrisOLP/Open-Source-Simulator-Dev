"""photonsuite.fdtd_avg with metal side walls (walls="pec"): the closed window of the EME in the FDTD.

The position of the walls: the TE01 mode of a metal box filled with SiO2 (E along x, cos(pi y / b) between the walls
y = +-b/2; the box is too narrow in x for other modes) has an exact discrete dispersion on the Yee grid, so the FDTD
phase index of a straight box must equal it for the wall distance b of the module text, and differ clearly from the
value for walls on the cell faces (b + dx). In the closed box the modes near cutoff ring for a long time, so the runs
stop at max_time (decayed False; the module text of photonsuite.fdtd_avg). Needs fdtd_gpu and fdtdx: it runs in the
GPU environment and is skipped elsewhere."""
import math

import numpy as np
import pytest

pytest.importorskip("fdtdx")
pytest.importorskip("fdtd_gpu")

from photonsuite import CrossSection, Material, Rect  # noqa: E402
from photonsuite.fdtd import Segment  # noqa: E402
from photonsuite.fdtd_avg import geometry, mode_transmission, simulation, wall_positions  # noqa: E402

SI, SIO2 = Material("Si", n=3.476), Material("SiO2", n=1.444)


def yee_box_index(wavelength, n, b, dx):
    """The phase index beta / k0 of the TE01 mode of a metal box (wall distance b along y, uniform index n) on the
    3-D Yee grid of fdtd_gpu (dt = 0.99 dx / sqrt(3), c = 1): (2/dx)^2 sin^2(beta dx/2) = n^2 (2/dt)^2
    sin^2(omega dt/2) - (2/dx)^2 sin^2(k_y dx/2), with k_y = pi / b (the discrete sine on the nodes between the
    walls)."""
    dt = 0.99 * dx / math.sqrt(3)
    omega = 2 * math.pi / wavelength
    rhs = (n * 2 / dt * math.sin(omega * dt / 2)) ** 2 - (2 / dx * math.sin(math.pi / b * dx / 2)) ** 2
    beta = 2 / dx * math.asin(dx / 2 * math.sqrt(rhs))
    return beta / omega


def test_pec_cell_has_the_walls_on_the_window_edges():
    win = ((-2.0, 2.0), (-0.8, 1.02))
    seg = [Segment(0.0, 3.0, CrossSection(SIO2, (Rect(-0.25, 0.25, 0.0, 0.22, SI),)))]
    _, size, _ = geometry(seg, *win, 0.4, walls="pec", resolution=50.0)
    assert size[1] == pytest.approx(4.02) and size[2] == pytest.approx(1.84)        # 201 and 92 cells
    np.testing.assert_allclose(np.ravel(wall_positions(*win, 50.0)), [-2.0, 2.0, -0.8, 1.02], rtol=0, atol=1e-12)
    (x0, x1), (y0, y1) = wall_positions(*win, 30.0)                                  # 1.82 um = 54.6 cells
    assert (x0, x1) == pytest.approx((-2.0, 2.0)) and (y1 - y0) == pytest.approx(55 / 30)
    assert 0.5 * (y0 + y1) == pytest.approx(0.11)
    sim, _ = simulation(seg, *win, [1 / 1.55], 30.0, walls="pec")
    assert sim.shape == (round(3.8 * 30), 121, 56)
    with pytest.raises(ValueError, match="walls"):
        geometry(seg, *win, 0.4, walls="pmc", resolution=30.0)
    with pytest.raises(ValueError, match="resolution"):
        geometry(seg, *win, 0.4, walls="pec")


def test_pec_walls_give_the_exact_yee_box_mode():
    """A box 0.4 x 1.0 um filled with SiO2, 3 um long, 20 cells per um: only TE01 propagates (TE10 is cut off below
    2 x 0.4 x 1.444 = 1.155 um; TE02 below 1.444 um, and it rings near that cutoff, but it is odd about the center
    and the TE01 projection does not see it). The FDTD phase index equals the exact Yee value for b = 1.0 within
    2e-4, and the value for walls on the cell faces (b = 1.05) is more than 1e-2 away."""
    wl = np.array([1.5, 1.55, 1.6])
    res = 20.0
    box = CrossSection(SIO2, ())
    r = mode_transmission([Segment(0.0, 3.0, box)], (-0.2, 0.2), (-0.5, 0.5), wl, res, calibrate=False,
                          walls="pec", max_time=60.0)
    assert r.run.shape[1:] == (9, 21)
    assert np.all(np.abs(r.T00_raw - 1) < 1e-2)
    assert np.all(np.abs(r.T_flux - 1) < 1e-2)
    exact = np.array([yee_box_index(w, SIO2.n, 1.0, 1 / res) for w in wl])
    faces = np.array([yee_box_index(w, SIO2.n, 1.05, 1 / res) for w in wl])
    got = r.run.phase_index(wl, exact)
    assert np.all(np.abs(got - exact) < 2e-4), (got, exact)
    assert np.all(np.abs(faces - exact) > 1e-2)
    analytic = np.sqrt(SIO2.n ** 2 - (wl / 2.0) ** 2)            # the continuous box: the grid moves it by < 1e-2
    assert np.all(np.abs(exact - analytic) < 1e-2)


def test_straight_guide_with_pec_walls_transmits_one():
    """The 0.5 um strip in a window with the walls 0.6 um from the core: T00 within 3e-3 of 1 and T_flux within 1e-2.
    The modes of the window near cutoff ring, so the run does not decay by 1e-7 (decayed False). T00 at 60 and 120
    um/c after the source: within 1e-4 at 1.55 and 1.60 um; at 1.50 um, near a cutoff frequency of the window, the
    DFT converges more slowly (a development run: 1.3e-3), and only the 3e-3 bound is checked."""
    xs = CrossSection(SIO2, (Rect(-0.25, 0.25, 0.0, 0.22, SI),))
    rs = [mode_transmission([Segment(0.0, 3.0, xs)], (-0.85, 0.85), (-0.6, 0.82), [1.5, 1.55, 1.6], 20.0,
                            calibrate=False, walls="pec", max_time=t) for t in (60.0, 120.0)]
    assert not rs[0].run.decayed
    for r in rs:
        assert np.all(np.abs(r.T00 - 1) < 3e-3)
        assert np.all(np.abs(r.T_flux - 1) < 1e-2)
    assert np.max(np.abs(rs[0].T00[1:] - rs[1].T00[1:])) < 1e-4


def test_side_pml_and_its_errors():
    """side_pml changes only the PML of the side faces (the cell across); not with walls='pec'."""
    win = ((-2.0, 2.0), (-0.8, 1.02))
    seg = [Segment(0.0, 3.0, CrossSection(SIO2, (Rect(-0.25, 0.25, 0.0, 0.22, SI),)))]
    sim, _ = simulation(seg, *win, [1 / 1.55], 30.0, side_pml=0.4 + 0.5 / 30)
    ref, _ = simulation(seg, *win, [1 / 1.55], 30.0)
    assert sim.shape == (ref.shape[0], ref.shape[1] + 1, round(2.62 * 30 + 1))
    with pytest.raises(ValueError, match="side_pml"):
        simulation(seg, *win, [1 / 1.55], 30.0, walls="pec", side_pml=0.5)
