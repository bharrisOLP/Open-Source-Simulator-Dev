"""The switching time of the thermo-optic phase shifter (photonsuite.switching), on the cross-section of
test_workflows.py with the heat capacities of Incropera et al., "Fundamentals of Heat and Mass Transfer", Tables A.1
(Si; tungsten for the heater metal) and A.2 (fused silica). The measured values are in the comments; the test bounds
are larger.

Checks: the end of the transient equals the steady phase_shifter_ppi result; the first-order index change against
full mode solves at 3 times (a second-order error); the phase against the mean core temperature (the expected
relation: dneff = (dneff/dT for a uniform rise) x the core rise); the switch-off is the mirror image of the
switch-on for a constant kappa; the rise time against the thermal-gpu rise time of the waveguide (its own structure)
and against the heat-diffusion estimate of the stack; kappa(T) of Si and SiO2."""
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from scipy.interpolate import RegularGridInterpolator  # noqa: E402

from photonsuite import CrossSection, Material, Rect  # noqa: E402
from photonsuite.physics import ConvergenceError, Window, mode_index, phase_shifter_ppi  # noqa: E402
from photonsuite.switching import phase_transient, slab_rise, slab_switch_time, to_cells  # noqa: E402

SI = Material("Si", n=3.476, dn_dT=1.86e-4, kappa=148.0, rho_cp=2330.0 * 712.0, eps_r=11.7, semiconductor=True)
SIO2 = Material("SiO2", n=1.444, dn_dT=0.95e-5, kappa=1.4, rho_cp=2220.0 * 745.0, eps_r=3.9)
METAL = Material("heater", n=1.444, kappa=100.0, rho_cp=19300.0 * 132.0)
XS = CrossSection(SIO2, (Rect(-5, 5, -5, -2, SI, tag="substrate"), Rect(-0.25, 0.25, 0, 0.22, SI, tag="core"),
                         Rect(-1, 1, 2.2, 2.3, METAL, tag="heater")))
OPTICS, THERMAL = Window(-1.5, 1.5, -0.9, 1.1, 0.02), Window(-5, 5, -5, 2.3, 0.05)
FULL_TIMES = (1.5e-6, 5e-6, 20e-6)
ALPHA_SIO2 = SIO2.kappa / SIO2.rho_cp          # 8.47e-7 m^2/s


def kappa_t_laws():
    from thermal_gpu.nonlinear import SI_POWER_LAW, SIO2_LINEAR
    return {"Si": SI_POWER_LAW, "SiO2": SIO2_LINEAR}


@pytest.fixture(scope="module")
def ppi():
    return phase_shifter_ppi(XS, OPTICS, THERMAL, "heater", 1.55)


@pytest.fixture(scope="module")
def on():
    return phase_transient(XS, OPTICS, THERMAL, "heater", 1.55, switch="on", core_tag="core", full_times=FULL_TIMES)


@pytest.fixture(scope="module")
def off():
    return phase_transient(XS, OPTICS, THERMAL, "heater", 1.55, switch="off", core_tag="core")


def test_the_end_of_the_transient_equals_the_steady_phase_shifter(on, ppi):
    p_pi, dneff_ppi, _ = ppi
    end = on.full[float(on.times[-1])]
    assert on.times[-1] >= 200e-6
    assert abs(end / dneff_ppi - 1) < 1e-8                               # 2.8e-10
    assert 1.55e-6 * 100.0 / (2 * end) * 1e3 == pytest.approx(18.997, rel=1e-3)
    assert abs(on.dneff[-1] / on.dneff_steady - 1) < 1e-8                 # first order, end against steady: 2.4e-10
    assert p_pi * 1e3 == pytest.approx(18.997, rel=1e-3)


def test_first_order_against_full_mode_solves(on):
    rel = []
    for t, d_full in on.full.items():
        k = int(np.argmin(np.abs(on.times - t)))
        rel.append((on.dneff[k] - d_full) / d_full)
        # the error is second order: rel = -c dneff, c = 0.0584 to 0.0586 at 1.48, 4.92, 20.3 and 205 us
        assert -0.07 < rel[-1] / d_full < -0.05
    assert len(rel) == 4 and max(abs(r) for r in rel) < 5e-4              # 2.6e-5, 1.1e-4, 2.2e-4, 2.4e-4


def test_the_phase_is_linear_in_the_core_temperature(on, off):
    # expected: dneff(t) = s dT_core(t), s = dneff / dT of a uniform rise (the thermo-optic coefficient of the mode)
    s = on.sensitivity.sum()
    one_kelvin = float(np.real(mode_index(XS, OPTICS, 1.55, dn=lambda X, Y: XS.sample(X, Y)[2]["dn_dT"])[0]
                               - on.neff0))
    assert abs(s / one_kelvin - 1) < 1e-4                                 # first order against a full solve: 1.2e-5
    assert s == pytest.approx(2.0124e-4, rel=1e-3)
    for r in (on, off):
        ok = r.core_rise > 0.01 * r.core_steady
        ratio = r.dneff[ok] / r.core_rise[ok] / s
        assert np.max(np.abs(ratio - 1)) < 3e-3                           # on: 2.05e-3, off: 3.9e-5
    assert abs(on.dneff_steady / on.core_steady / s - 1) < 1e-4           # 3.9e-5


def test_the_switch_off_is_the_mirror_image_of_the_switch_on(on, off):
    assert np.allclose(on.times, off.times)
    assert np.max(np.abs(on.dneff + off.dneff - on.dneff_steady)) < 1e-8 * on.dneff_steady
    assert abs(off.switch_time / on.switch_time - 1) < 1e-8               # 1e-9
    assert abs(off.full[float(off.times[-1])]) < 1e-9


def test_the_rise_time_against_thermal_gpu_and_the_diffusion_estimate(on):
    from thermal_gpu.structures import heat_capacity, heater_stack
    from thermal_gpu.transient3d import geometric_stages, rise_time, solve_transient2d

    # thermal-gpu with its own structure (the same stack): the 10-90 % time of the waveguide temperature
    x, y, kappa, q, bc, masks = heater_stack(dx=0.05e-6, power_per_length=100.0)
    tr = solve_transient2d(x, y, kappa, q, bc, heat_capacity(kappa), geometric_stages(1e-8, 200e-6), scheme="bdf2",
                           monitors=[masks["waveguide"]])
    t_wg = rise_time(tr.times, tr.monitors[:, 0], 300.0 + on.core_steady)[2]
    assert abs(on.core_switch_time / t_wg - 1) < 1e-8
    assert t_wg * 1e6 == pytest.approx(17.939, rel=1e-4)                 # the logged value of thermal_gpu
    assert on.switch_time * 1e6 == pytest.approx(17.9385, rel=1e-4)
    assert abs(on.switch_time / t_wg - 1) < 1e-4                         # 1.2e-5: the cladding adds little
    # heat diffusion through the 4.2 um of oxide between the substrate and the heater (BOX 2 um + 2.2 um)
    t_single = slab_switch_time(4.2e-6, ALPHA_SIO2)[2]                   # tau ln 9, tau = 4 L^2 / (pi^2 alpha)
    t_slab = slab_switch_time(4.2e-6, ALPHA_SIO2, 2.11e-6)[2]            # 1-D series at the core center
    assert t_single * 1e6 == pytest.approx(18.557, rel=1e-4)
    assert t_slab * 1e6 == pytest.approx(18.717, rel=1e-4)
    assert 0.94 < on.switch_time / t_slab < 0.98                          # 0.958 (single exponential: 0.967)
    assert 0.94 < on.switch_time / t_single < 0.98


def test_the_slab_estimate_against_a_1d_transient():
    from thermal_gpu.transient3d import geometric_stages, rise_time, solve_transient2d

    # an oxide column on a heat sink (300 K at y = 0), adiabatic top, 10 nm cells, 1e7 W/m^2 into the top cell (the
    # modes are those of the thickness L; the source in the top cell instead of at y = L changes their amplitudes by
    # cos(l_n dy / 2): 1 - 2e-6 and 1 - 2e-5 for the two slowest modes)
    L, n = 4.2e-6, 420
    y = np.linspace(0.0, L, n + 1)
    dy, yc = L / n, 0.5 * (y[1:] + y[:-1])
    q = np.zeros((1, n))
    q[0, -1] = 1e7 / dy
    tr = solve_transient2d(np.array([0.0, 1e-6]), y, np.full((1, n), SIO2.kappa), q, {"ymin": ("dirichlet", 300.0)},
                           SIO2.rho_cp, geometric_stages(2e-9, 120e-6, per_stage=40), scheme="bdf2",
                           monitors=[(0, 210)])
    h = yc[210]                                                           # 2.105 um, near the core center
    t10, t90, rise = rise_time(tr.times, tr.monitors[:, 0], 300.0 + 1e7 * h / SIO2.kappa)
    e10, e90, e_rise = slab_switch_time(L, ALPHA_SIO2, h)
    for got, want in ((t10, e10), (t90, e90), (rise, e_rise)):          # 3.4e-5, 2.9e-6, 6.6e-6
        assert abs(got / want - 1) < 2e-4, (got, want)
    assert abs(slab_rise(L, ALPHA_SIO2, h, 0.0)) < 1e-2 and slab_rise(L, ALPHA_SIO2, h, 1.0) == pytest.approx(1.0)


def test_kappa_of_t(on, ppi):
    laws = kappa_t_laws()
    up = phase_transient(XS, OPTICS, THERMAL, "heater", 1.55, switch="on", core_tag="core", kappa_laws=laws)
    down = phase_transient(XS, OPTICS, THERMAL, "heater", 1.55, switch="off", core_tag="core", kappa_laws=laws)
    # the steady rise falls by 0.93 % (thermal_gpu: 20.0780 K against 20.2664 K), and the phase with it
    assert up.core_steady == pytest.approx(20.0780, rel=1e-4)
    end = up.full[float(up.times[-1])]
    assert abs(end / ppi[1] - up.core_steady / on.core_steady) < 1e-4     # -0.9301 % and -0.9299 %
    assert abs(up.dneff[-1] / up.dneff_steady - 1) < 1e-8                 # 2e-11
    # rise 17.606 us (thermal_gpu, waveguide: 17.606, ratio 0.98145); fall 17.870 us: kappa is lower when cold
    assert up.core_switch_time / on.core_switch_time == pytest.approx(0.98145, abs=1e-4)
    assert abs(up.switch_time / up.core_switch_time - 1) < 1e-4
    assert up.switch_time * 1e6 == pytest.approx(17.6057, rel=1e-4)
    assert down.switch_time * 1e6 == pytest.approx(17.8705, rel=1e-4)
    assert up.switch_time < down.switch_time < on.switch_time
    s = on.sensitivity.sum()
    for r in (up, down):
        ok = r.core_rise > 0.01 * r.core_steady
        assert np.max(np.abs(r.dneff[ok] / r.core_rise[ok] / s - 1)) < 3e-3


def test_to_cells_is_the_transpose_of_the_interpolation():
    rng = np.random.default_rng(1)
    xg, yg = np.cumsum(rng.uniform(0.5, 1.5, 7)), np.cumsum(rng.uniform(0.5, 1.5, 5))
    pts = [(rng.uniform(xg[0] - 1, xg[-1] + 1, (4, 3)), rng.uniform(yg[0] - 1, yg[-1] + 1, (4, 3)))
           for _ in range(3)]                                                 # also points outside: extrapolation
    w = [rng.normal(size=(4, 3)) for _ in range(3)]
    f = rng.normal(size=(7, 5))
    interp = RegularGridInterpolator((xg, yg), f, bounds_error=False, fill_value=None)
    direct = sum(np.sum(wc * interp(np.stack([X.ravel(), Y.ravel()], -1)).reshape(X.shape))
                 for wc, (X, Y) in zip(w, pts))
    assert np.sum(to_cells(w, pts, xg, yg) * f) == pytest.approx(direct, rel=1e-12)


def test_errors(monkeypatch):
    with pytest.raises(ValueError, match="switch"):
        phase_transient(XS, OPTICS, THERMAL, "heater", 1.55, switch="up")
    with pytest.raises(ValueError, match="no material"):
        phase_transient(XS, OPTICS, THERMAL, "heater", 1.55, kappa_laws={"Ge": None})
    import thermal_gpu.transient3d as t3
    real = t3.solve_transient2d
    monkeypatch.setattr(t3, "solve_transient2d", lambda *a, **k: real(*a, **k, max_iter=1))
    with pytest.raises(ConvergenceError, match="transient"):
        phase_transient(XS, OPTICS, THERMAL, "heater", 1.55, t_end=1e-7)
