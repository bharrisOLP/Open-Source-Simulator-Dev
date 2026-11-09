"""The fdtd_gpu adapter (photonsuite.fdtd_avg): the axes and the shapes map to the suite's device (staircase and
subpixel average), the planes stay off the segment ends, and a straight guide transmits 1: without the calibration
of the mode factor b, with it (calibrate=True, the default of mode_transmission), and with the parity "tm". The taper
of bench/eme_fdtd_taper.py against the EME is not a test (its rows are logged: README). Needs fdtd_gpu and fdtdx: it
runs in the GPU environment and is skipped elsewhere."""
import numpy as np
import pytest

pytest.importorskip("fdtdx")
pytest.importorskip("fdtd_gpu")

from photonsuite import CrossSection, Material, Polygon, Rect  # noqa: E402
from photonsuite.fdtd import Segment  # noqa: E402
from photonsuite.fdtd_avg import mode_transmission, simulation  # noqa: E402

SI, SIO2 = Material("Si", n=3.476), Material("SiO2", n=1.444)
WIN = ((-1.2, 1.2), (-0.8, 1.0))
RES = 25.0                                    # 0.04 um: no shape edge below is on a cell center


def _device():
    """Three segments: a Rect core, a sloped Polygon core, and a core in an oxide trench of a Si slab (fill order).
    Edges and segment ends are off the cell centers of the RES grid."""
    rect = CrossSection(SIO2, (Rect(-0.27, 0.23, 0.01, 0.2, SI),))
    poly = CrossSection(SIO2, (Polygon(((-0.31, 0.01), (0.29, 0.01), (0.21, 0.2), (-0.23, 0.2)), SI),))
    fill = CrossSection(SIO2, (Rect(-0.93, 0.93, 0.01, 0.2, SI), Rect(-0.53, 0.53, 0.01, 0.2, SIO2),
                               Rect(-0.21, 0.21, 0.01, 0.2, SI)))
    return [Segment(0.0, 1.0, rect), Segment(1.0, 2.03, poly), Segment(2.03, 3.0, fill)]


def _suite_eps(segments, sim):
    """n^2 of the suite's own sampling (CrossSection.sample) at the cell centers of sim, in the fdtd_gpu axes."""
    zc, xc, yc = 1.5, 0.0, 0.1
    c = [(np.arange(n) + 0.5) * sim.dx - n * sim.dx / 2 for n in sim.shape]
    out = np.zeros(sim.shape)
    for i, zf in enumerate(c[0]):
        z = zf + zc
        seg = next((s for s in segments if s.z0 <= z < s.z1), segments[0] if z < segments[0].z0 else segments[-1])
        X, Y = np.meshgrid(c[1] + xc, c[2] + yc, indexing="ij")
        out[i] = seg.xs.sample(X, Y)[2]["n"] ** 2
    return out


def test_staircase_maps_the_suite_axes_and_shapes():
    segs = _device()
    sim, _ = simulation(segs, *WIN, [1 / 1.55], RES, pml=0.4, eps_averaging=False)
    assert sim.shape == (95, 80, 65)          # (z, x, y) of the suite: 3.8, 3.2 and 2.6 um at 0.04 um
    eps = sim.get_epsilon()
    np.testing.assert_array_equal(eps, _suite_eps(segs, sim))


def test_the_average_keeps_the_area_of_each_cross_section():
    """E along the propagation axis is parallel to every face of a z-invariant segment, so the rule of Kottke et al.
    gives the arithmetic mean <eps> there: the plane sum of eps_xx (fdtd_gpu x = the suite's z) is the exact integral
    of eps over the cross-section, also with the edges between grid lines."""
    segs = _device()
    sim, _ = simulation(segs, *WIN, [1 / 1.55], RES, pml=0.4)
    _, _, eps3 = sim._permittivity()
    area = (sim.shape[1] * sim.dx) * (sim.shape[2] * sim.dx)
    d = SI.n ** 2 - SIO2.n ** 2
    for i, core in ((12, 0.5 * 0.19), (40, 0.5 * (0.60 + 0.44) * 0.19), (80, (1.86 - 1.06 + 0.42) * 0.19)):
        got = (eps3[0, i].sum() * sim.dx ** 2 - SIO2.n ** 2 * area) / d
        assert got == pytest.approx(core, rel=2e-3), i


def test_the_planes_stay_off_the_segment_ends():
    """Review of the first taper runs: with a 1.0 um input segment the input plane was on the start of the taper."""
    xs = CrossSection(SIO2, (Rect(-0.25, 0.25, 0.0, 0.22, SI),))
    with pytest.raises(ValueError, match="first segment"):
        mode_transmission([Segment(0.0, 1.0, xs), Segment(1.0, 3.0, xs)], *WIN, [1.55], 20.0)
    with pytest.raises(ValueError, match="last segment"):
        mode_transmission([Segment(0.0, 2.0, xs), Segment(2.0, 2.8, xs)], *WIN, [1.55], 20.0)
    with pytest.raises(ValueError, match="apart"):                 # one segment: the two planes would coincide
        mode_transmission([Segment(0.0, 1.6, xs)], *WIN, [1.55], 20.0)


def test_straight_guide_transmits_one():
    xs = CrossSection(SIO2, (Rect(-0.25, 0.25, 0.0, 0.22, SI),))
    r = mode_transmission([Segment(0.0, 3.0, xs)], *WIN, [1.5, 1.55, 1.6], 20.0, calibrate=False)
    assert np.all(np.abs(r.T00 - 1) < 3e-3)
    assert np.all(np.abs(r.T_flux - 1) < 1e-2)
    share = np.abs(r.run.a_out) ** 2 / r.run.p_out             # the mode factor b of a pure mode
    assert np.all((share > 0.9) & (share < 1.01))
    # the phase of a_out / a_in gives the index of the FDTD propagation: near the mode solver (the Yee dispersion)
    assert r.run.distance == pytest.approx(1.5, abs=0.05)
    assert np.all(np.abs(r.run.phase_index(r.wavelengths, r.run.neff_out) - r.run.neff_out) < 0.05)


@pytest.fixture(scope="module")
def calibrated():
    """A straight guide with calibrate=True: the device run and the two calibration runs are the same simulation."""
    xs = CrossSection(SIO2, (Rect(-0.25, 0.25, 0.0, 0.22, SI),))
    return mode_transmission([Segment(0.0, 3.0, xs)], *WIN, [1.5, 1.55, 1.6], 20.0)


def test_the_calibrated_path_of_a_straight_guide(calibrated):
    """Review cfae980e777a: only the bench rows ran calibrate=True. b_in and b_out are |alpha|^2 / P at the output
    plane of the two calibration runs (straight guides of the input and the output cross-section, the same length,
    ports and grid). For a straight guide the three runs are the same simulation: b_in = b_out = the b of the device
    run, and T00 = T00_raw is near 1."""
    r = calibrated
    assert len(r.calibration) == 2
    for b, c in ((r.b_in, r.calibration[0]), (r.b_out, r.calibration[1])):
        np.testing.assert_array_equal(b, np.abs(c.a_out) ** 2 / c.p_out)
    b_run = np.abs(r.run.a_out) ** 2 / r.run.p_out
    print("calibrated straight guide: b_in", r.b_in, "b_out / b of the device run - 1", r.b_out / b_run - 1,
          "T00 - 1", r.T00 - 1, "input_mode_share - 1", r.input_mode_share - 1)
    assert np.all((r.b_in > 0.9) & (r.b_in < 1.01))
    np.testing.assert_allclose(r.b_out, r.b_in, rtol=1e-5, atol=0)
    np.testing.assert_allclose(r.b_out, b_run, rtol=1e-5, atol=0)
    np.testing.assert_allclose(r.T00, r.T00_raw * r.b_in / r.b_out, rtol=1e-12, atol=0)
    assert np.all(np.abs(r.T00 - 1) < 3e-3)
    assert np.all(np.abs(r.input_mode_share - 1) < 5e-3)        # |a_in|^2 / b_in / P_in: the input is the mode


def test_the_tm_parity():
    """parity "tm" (ODD_Z of fdtd_gpu: E mostly vertical; the run stops on the decay of Ez): the source and the
    monitors take the TM0 mode of the 0.5 um strip (index about 1.77 at 1.55 um, below the TE0 index 2.45), and a
    straight guide transmits it. The TM0 field reaches farther into the cladding: in the window of the TE tests
    (y -0.8 to 1.0 um) T00 was 0.99725 at 1.6 um (test print; 0.99857 in the taller window here)."""
    xs = CrossSection(SIO2, (Rect(-0.25, 0.25, 0.0, 0.22, SI),))
    r = mode_transmission([Segment(0.0, 3.0, xs)], WIN[0], (-1.2, 1.4), [1.5, 1.55, 1.6], 20.0, parity="tm",
                          calibrate=False)
    b = np.abs(r.run.a_out) ** 2 / r.run.p_out
    print(f"tm: neff_in {r.run.neff_in}, neff_out {r.run.neff_out}, T00 - 1 {r.T00 - 1}, T_flux - 1 {r.T_flux - 1}, "
          f"b {b}, phase index {r.run.phase_index(r.wavelengths, r.run.neff_out)}")
    assert np.all((r.run.neff_out > 1.6) & (r.run.neff_out < 1.9))           # TM0, not TE0
    assert np.all(np.abs(r.run.neff_in - r.run.neff_out) < 1e-3)
    assert np.all(np.abs(r.T00 - 1) < 3e-3)
    assert np.all(np.abs(r.T_flux - 1) < 1e-2)
    assert np.all((b > 0.9) & (b < 1.01))
    assert np.all(np.abs(r.run.phase_index(r.wavelengths, r.run.neff_out) - r.run.neff_out) < 0.05)
