"""The suite reproduces the standalone examples (same structures, same grids): P_pi of the heater
(examples/thermo_optic_phase_shifter.py, 50 nm thermal grid: 18.997 mW) and V_pi L of the PN rib
(examples/pn_modulator.py: 1.386, 1.600, 1.955 V cm at 1, 2, 4 V), and the loss decreases. Also: the rectangle edge
rule, the float64 requirement, and the convergence errors."""
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from photonsuite import CrossSection, Material, Rect  # noqa: E402
from photonsuite.physics import ConvergenceError, Window, modulator_vpil, phase_shifter_ppi, temperature  # noqa: E402

SI = Material("Si", n=3.476, dn_dT=1.86e-4, kappa=148.0, eps_r=11.7, semiconductor=True)
SIO2 = Material("SiO2", n=1.444, dn_dT=0.95e-5, kappa=1.4, eps_r=3.9)
METAL = Material("heater", n=1.444, kappa=100.0, eps_r=1.0)        # outside the optics window


def test_phase_shifter_matches_the_example():
    xs = CrossSection(SIO2, (Rect(-5, 5, -5, -2, SI, tag="substrate"), Rect(-0.25, 0.25, 0, 0.22, SI, tag="core"),
                             Rect(-1, 1, 2.2, 2.3, METAL, tag="heater")))
    ppi, dneff, _ = phase_shifter_ppi(xs, Window(-1.5, 1.5, -0.9, 1.1, 0.02), Window(-5, 5, -5, 2.3, 0.05),
                                      "heater", 1.55)
    # the example prints 19.0; its dneff 0.004079614 gives 1.55e-6 * 100 / (2 * dneff) = 18.997 mW
    assert ppi * 1e3 == pytest.approx(18.997, rel=1e-3)


@pytest.mark.slow
def test_modulator_matches_the_example():
    d = 3e17
    xs = CrossSection(SIO2, (Rect(-3, 0, 0, 0.09, SI, doping=-d), Rect(0, 3, 0, 0.09, SI, doping=d),
                             Rect(-0.25, 0, 0, 0.22, SI, doping=-d), Rect(0, 0.25, 0, 0.22, SI, doping=d)))
    out = modulator_vpil(xs, Window(-1.5, 1.5, -0.5, 0.9, 0.02), Window(-3, 3, -0.5, 0.72, 0.01), 1.55)
    for v, want in ((1.0, 1.3858), (2.0, 1.5996), (4.0, 1.9553)):
        assert out[v]["VpiL_Vcm"] == pytest.approx(want, rel=1e-3)
        assert out[v]["dloss_dB_per_cm"] < 0


def test_rect_edges_do_not_depend_on_the_rounding_of_the_grid():
    r = Rect(-0.25, 0.25, 0.0, 0.22, SI)
    eps = 1e-15
    y = np.array([-eps, 0.0, eps, 0.22 - eps, 0.22, 0.22 + eps])
    assert list(r.contains(np.zeros_like(y), y)) == [True] * 3 + [False] * 3


def test_refuses_float32():
    xs = CrossSection(SIO2, (Rect(-1, 1, 0, 0.1, METAL, tag="heater"),))
    jax.config.update("jax_enable_x64", False)
    try:
        with pytest.raises(RuntimeError, match="float64"):
            temperature(xs, Window(-2, 2, -2, 1, 0.05), "heater", 100.0)
    finally:
        jax.config.update("jax_enable_x64", True)


def test_a_thermal_solve_that_does_not_converge_raises(monkeypatch):
    import thermal_gpu.solver as ts
    real = ts.solve_steady
    monkeypatch.setattr(ts, "solve_steady", lambda *a, **k: real(*a, **k, max_iter=2))
    xs = CrossSection(SIO2, (Rect(-1, 1, 0, 0.1, METAL, tag="heater"),))
    with pytest.raises(ConvergenceError):
        temperature(xs, Window(-2, 2, -2, 1, 0.05), "heater", 100.0)
