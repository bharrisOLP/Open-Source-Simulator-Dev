"""The PIN phase shifter with self-heating (photonsuite.pin), on the PIN rib of charge-gpu (pin.pin_rib: textbook
values, lifetimes 1 ns). The measured values (GPU computer) are in the comments; the test bounds are larger.

Checks: the cross-section gives the Device of charge_gpu.injection.rib_pin, and the same currents as the logged
charge-gpu example; the sum of the heat source equals I' V (also on a 1-D diode, and after the move to the thermal
grid); the sum of J . E equals I' (V - V_c) of the contact potentials (analytic); the thermal part against
physics.phase_shifter_ppi with the same heat in the same cells; both parts against the first-order perturbation
(a second-order difference); the limits dn/dT = 0 and no plasma dispersion; the mode is the fundamental mode
(the largest index of a 4-mode solve); the signs and the cross-over."""
import dataclasses

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from photonsuite.physics import (ConvergenceError, Window, charge_device, phase_shifter_ppi,  # noqa: E402
                                 thermal_problem)
from photonsuite.pin import (SI, SIO2, PinElectrical, cell_overlap, heat_source, pin_electrical,  # noqa: E402
                             pin_optical, pin_rib, to_thermal, uniform_heat)

XS = pin_rib()
OPTICS = Window(-1.5, 1.5, -0.5, 0.9, 0.02)
CHARGE = Window(-1.5, 1.5, -0.5, 0.72, 0.01)
THERMAL = Window(-5.0, 5.0, -5.0, 2.3, 0.05)
BIASES = (0.9, 1.0, 1.1, 1.2)
# benchmarks/2026-10-30/charge-dd.jsonl, examples/pin_phase_shifter.py of charge-gpu (tau 1 ns): I' at 0.9 and 1.0 V
LOGGED_CURRENT = {0.9: 3.8282742963924745, 1.0: 15.920944272975659}
LOGGED_DNEFF_09 = -0.0007230039242824837          # the same file, the example with --modes: dneff at 0.9 V


@pytest.fixture(scope="module")
def electrical():
    return pin_electrical(XS, CHARGE, BIASES, tau=1e-9)


@pytest.fixture(scope="module")
def result(electrical):
    return pin_optical(XS, OPTICS, THERMAL, 1.55, electrical, core_tag="core")


def one_bias(el, k):
    return PinElectrical(el.device, el.window, el.biases[k:k + 1], el.solutions[k:k + 1], el.reference, 0.0)


def test_the_cross_section_gives_the_rib_of_charge_gpu():
    from charge_gpu.injection import rib_pin

    b = rib_pin()
    # the thermal options of pin_rib (a longer slab, a wider and deeper substrate) do not change the charge device
    for xs in (XS, pin_rib(slab_half_width=5.0, substrate_half_width=20.0, substrate=18.0)):
        a = charge_device(xs, CHARGE)
        assert np.allclose(a.x, b.x, rtol=0, atol=1e-15) and np.allclose(a.y, b.y, rtol=0, atol=1e-15)
        for name in ("eps_r", "n_net", "semiconductor"):
            assert np.array_equal(getattr(a, name), getattr(b, name)), name
        assert a.n_i == b.n_i and a.T == b.T
        assert [(c.side, list(c.mask)) for c in a.contacts] == [(c.side, list(c.mask)) for c in b.contacts]
    with pytest.raises(ValueError, match="slab_half_width"):
        pin_rib(slab_half_width=1.0)


def test_the_currents_equal_the_charge_gpu_example(electrical):
    for v, sol in zip(electrical.biases, electrical.solutions):
        assert sol.converged
        if v in LOGGED_CURRENT:
            print(f"I' at {v} V: {sol.current!r} A/m, against the logged row {sol.current / LOGGED_CURRENT[v] - 1:.1e}")
            assert abs(sol.current / LOGGED_CURRENT[v] - 1) < 1e-9          # 3.7e-13 (0.9 V), 3.5e-14 (1.0 V)


def test_the_carrier_part_against_the_charge_gpu_example(result):
    # the same rib, carriers and plasma dispersion, with the mode grid and the cell means of the example; the example
    # samples the Si with strict inequalities (the Yee points on y = 0 are oxide there, Si here)
    rel = result.dneff_carrier[0].real / LOGGED_DNEFF_09 - 1
    print(f"dneff_carrier at 0.9 V: {result.dneff_carrier[0].real:.6e}, the example {LOGGED_DNEFF_09:.6e}, "
          f"relative {rel:.1e}")
    assert abs(rel) < 1e-3                                                # 2.3e-5


def test_the_heat_is_the_electrical_power(electrical, result):
    from charge_gpu.poisson import Q, thermal_voltage

    v_c = 2 * thermal_voltage(300.0) * np.arcsinh(1e24 / (2 * 1e16))         # N_A = N_D = 1e18 cm^-3: 0.9524 V
    for k, (v, sol) in enumerate(zip(electrical.biases, electrical.solutions)):
        h = heat_source(sol)
        p = sol.current * v
        # the sum is an identity of the discrete scheme (summation by parts; it holds for any cell potentials): it
        # checks the continuity residual and the flux code, not where the heat goes (the local checks follow)
        assert abs(h.total.sum() / p - 1) < 1e-9                              # 2.1e-13 or less
        assert abs(result.heat[k] / p - 1) < 1e-9                             # after the move to the thermal cells
        assert h.joule.sum() > 0 and h.recombination.sum() > 0
        assert h.joule.min() > -1e-12 * p and h.recombination.min() > -1e-12 * p   # each cell: no heat sink
        # the recombination heat of each cell from the public methods of the solution: q R A (phi_p - phi_n)
        phi_n, phi_p = sol.quasi_fermi()
        area = np.outer(np.diff(electrical.device.x), np.diff(electrical.device.y))
        semi = electrical.device.semiconductor
        rec = Q * sol.recombination()[semi] * area[semi] * (phi_p - phi_n)[semi]
        rel = np.max(np.abs(h.recombination[semi] - rec)) / np.max(np.abs(rec))
        print(f"{v} V: recombination heat per cell against the public methods: {rel:.1e}")
        assert rel < 1e-10                                                    # 0.0
        # the electrostatic J . E: I' (V - V_c) (-0.20 W/m at 0.9 V, 20.1 W/m at 1.2 V)
        print(f"{v} V: heat / I' V - 1 = {h.total.sum() / p - 1:.1e}, thermal grid {result.heat[k] / p - 1:.1e}, "
              f"J.E = {h.je:.4f} W/m, (J.E - I' (V - V_c)) / I' V = {(h.je - sol.current * (v - v_c)) / p:.1e}")
        assert abs(h.je - sol.current * (v - v_c)) < 1e-9 * p
    # the recombination heat dominates at low bias, the Joule heat at high bias (0.778 at 0.9 V, 0.731 at 1.2 V)
    assert result.heat_recombination[0] > 0.7 * result.power[0]
    assert result.heat_joule[-1] > 0.7 * result.power[-1]


def test_the_heat_of_a_1d_diode():
    from charge_gpu import drift_diffusion as dd
    from charge_gpu.injection import diode_1d

    dev = diode_1d(1e23, 1e23, 2e-6, 2e-6, fine=2e-9, ratio=1.05)              # 1e17 cm^-3 on both sides
    for sol in dd.sweep(dev, [0.5, 0.7], tau_n=1e-7, tau_p=1e-7):
        h = heat_source(sol)
        rel = h.total.sum() / (sol.current * sol.bias) - 1
        print(f"1-D diode at {sol.bias} V: I' = {sol.current:.4e} A/m, heat / I' V - 1 = {rel:.1e}")
        assert abs(rel) < 1e-6                     # 3.7e-8 at 0.5 V (I' = 5.8e-5 A/m), -8.2e-12 at 0.7 V


def test_the_move_to_the_thermal_grid_keeps_the_sum():
    rng = np.random.default_rng(1)
    x, y = np.linspace(-1.0, 1.3, 47), np.linspace(-0.4, 0.5, 23)            # not aligned with the thermal cells
    power = rng.random((46, 22))
    win = Window(-2.0, 2.0, -1.0, 1.0, 0.07)
    xt, yt = win.edges()
    q = to_thermal(power, x, y, win)
    assert abs(np.sum(q * np.outer(np.diff(xt), np.diff(yt))) * 1e-12 / power.sum() - 1) < 1e-12
    one = np.zeros((46, 22))
    one[10, 5] = 1.0                                       # one source cell over 2 x 2 thermal cells
    q1 = to_thermal(one, x, y, win) * np.outer(np.diff(xt), np.diff(yt)) * 1e-12
    i, j = np.nonzero(q1)
    # x: -0.5 to -0.45 um over the cells 21 (-0.53 to -0.46) and 22; y: -0.195 to -0.155 um over 11 and 12
    assert list(np.unique(i)) == [21, 22] and list(np.unique(j)) == [11, 12]
    fx = (xt[22] - x[10]) / (x[11] - x[10])               # 0.8
    fy = (yt[12] - y[5]) / (y[6] - y[5])
    assert np.allclose(q1[21:23, 11:13], [[fx * fy, fx * (1 - fy)], [(1 - fx) * fy, (1 - fx) * (1 - fy)]],
                       rtol=0, atol=1e-14)
    assert np.allclose(cell_overlap(x, xt)[10, 21:23], [fx, 1 - fx], rtol=0, atol=1e-14)
    with pytest.raises(ValueError, match="not inside"):
        to_thermal(power, x, y, Window(-0.5, 2.0, -1.0, 1.0, 0.07))


def test_the_thermal_part_against_phase_shifter_ppi(electrical):
    k = 1
    p = electrical.solutions[k].current * electrical.biases[k]                # 15.9 W/m
    assert np.array_equal(uniform_heat(XS, THERMAL, "core", p), thermal_problem(XS, THERMAL, "core", p)[4])
    r = pin_optical(XS, OPTICS, THERMAL, 1.55, one_bias(electrical, k), heat="core")
    _, dneff_ppi, _ = phase_shifter_ppi(XS, OPTICS, THERMAL, "core", 1.55, power_per_length=p)
    print(f"thermal part, heat in the core: {r.dneff_thermal[0].real:.6e}, phase_shifter_ppi {dneff_ppi:.6e}, "
          f"relative {r.dneff_thermal[0].real / dneff_ppi - 1:.1e}")
    assert abs(r.dneff_thermal[0].real / dneff_ppi - 1) < 1e-6           # 1.9e-11, 6.7e-12, 9.1e-12
    assert abs(r.heat[0] / p - 1) < 1e-12


def test_first_order_against_full_mode_solves(result):
    for full, first in ((result.dneff_carrier, result.first_carrier), (result.dneff_thermal, result.first_thermal)):
        err = np.abs(first - full) / np.abs(full)
        print("first order, relative difference:", np.array2string(err, precision=2))
        assert np.all(err < 1e-3)                    # carrier 7.1e-5 to 3.2e-4, thermal 1.0e-5 to 2.8e-4
        # a second-order difference: the relative error grows as |dneff| (the ratio within a factor 2)
        c = err / np.abs(full)
        assert c.max() / c.min() < 2.0


def test_the_limits_without_heat_and_without_carriers(electrical, result):
    el = one_bias(electrical, 2)
    cold = dataclasses.replace(XS, background=dataclasses.replace(SIO2, dn_dT=0.0),
                               shapes=tuple(dataclasses.replace(s, material=dataclasses.replace(SI, dn_dT=0.0))
                                            for s in XS.shapes))
    r = pin_optical(cold, OPTICS, THERMAL, 1.55, el)
    print(f"dn/dT = 0: thermal {abs(r.dneff_thermal[0]):.1e}, "
          f"both/carrier - 1 {abs(r.dneff[0] / r.dneff_carrier[0] - 1):.1e}, "
          f"carrier/carrier of the full run - 1 {abs(r.dneff_carrier[0] / result.dneff_carrier[2] - 1):.1e}")
    assert abs(r.dneff_thermal[0]) < 1e-10                                # 6.0e-14 or less
    assert abs(r.dneff[0] / r.dneff_carrier[0] - 1) < 1e-7                # 4.6e-12 or less
    assert abs(r.dneff_carrier[0] / result.dneff_carrier[2] - 1) < 1e-7   # 7.8e-12 or less
    r = pin_optical(XS, OPTICS, THERMAL, 1.55, el, plasma=lambda n0, p0, n1, p1: (0 * n1, 0 * n1))
    print(f"no plasma: carrier {abs(r.dneff_carrier[0]):.1e}, "
          f"both/thermal - 1 {abs(r.dneff[0] / r.dneff_thermal[0] - 1):.1e}, "
          f"thermal/thermal of the full run - 1 {abs(r.dneff_thermal[0] / result.dneff_thermal[2] - 1):.1e}")
    assert abs(r.dneff_carrier[0]) < 1e-10                                # 3.0e-13 or less
    assert abs(r.dneff[0] / r.dneff_thermal[0] - 1) < 1e-7                # 3.3e-11, 4.7e-11
    assert abs(r.dneff_thermal[0] / result.dneff_thermal[2] - 1) < 1e-7   # 1.6e-11 or less


def test_the_mode_is_the_fundamental_mode(result):
    # the workflow solves one mode at the default target (just above the index of Si): it must be the mode with the
    # largest index (meow_gpu/README.md: the modes that the solver misses have neff <= 1.444)
    from photonsuite.physics import mode_index

    four = mode_index(XS, OPTICS, 1.55, num_modes=4)
    print("neff of a 4-mode solve:", np.array2string(four, precision=6), "neff0:", result.neff0)
    assert abs(np.max(four.real) - result.neff0.real) < 1e-9
    assert result.neff0.real > 2.5


def test_the_thermal_environment_changes_the_thermal_part(electrical, result):
    # the same carriers (1.0 V) in two other thermal models: the window and the substrate +-20 um wide (the adiabatic
    # sides farther away), and a heat path to the contacts (the doped slab to the window sides, both sides at 300 K)
    el = one_bias(electrical, 1)
    wide = pin_optical(pin_rib(substrate_half_width=20.0), OPTICS, Window(-20.0, 20.0, -5.0, 2.3, 0.05), 1.55, el)
    sink = pin_optical(pin_rib(slab_half_width=5.0), OPTICS, THERMAL, 1.55, el, side_temperature=300.0)
    base = result.dneff_thermal[1].real
    print(f"dneff_thermal at 1.0 V: {base:.4e}; window +-20 um {wide.dneff_thermal[0].real:.4e}; heat sink "
          f"{sink.dneff_thermal[0].real:.4e}")
    # 6.9200e-4; +-20 um: 6.5187e-4 (-5.8 %); heat sink: 2.5135e-4 (0.363)
    assert 0 < wide.dneff_thermal[0].real < base and 0 < sink.dneff_thermal[0].real < 0.6 * base
    for r in (wide, sink):
        assert abs(r.dneff_carrier[0] / result.dneff_carrier[1] - 1) < 1e-7       # the same carriers
        assert abs(r.heat[0] / r.power[0] - 1) < 1e-9


def test_the_crossover_at_a_sample_and_without_a_zero(result):
    def with_dneff(d):
        return dataclasses.replace(result, dneff=np.asarray(d, complex))

    x = with_dneff([-2e-4, -1e-4, 0.0, 1e-4]).crossover()                        # exactly 0 at the third bias
    assert x["bias"] == result.biases[2] and x["interval"] == (result.biases[2],) * 2
    assert x["current"] == result.current[2]
    x = with_dneff([-2e-4, -1e-4, 1e-4, 2e-4]).crossover()                       # half way in I' from 1.0 to 1.1 V
    assert x["interval"] == (result.biases[1], result.biases[2])
    assert x["current"] == pytest.approx(0.5 * (result.current[1] + result.current[2]), rel=1e-12)
    assert with_dneff([0.0, -1e-4, -2e-4, -3e-4]).crossover() is None            # 0 only at the first bias
    assert with_dneff([-1e-4, -2e-4, -3e-4, -4e-4]).crossover() is None


def test_the_signs_and_the_crossover(result):
    dc, dt, d = result.dneff_carrier, result.dneff_thermal, result.dneff
    assert np.all(dc.real < 0) and np.all(dc.imag > 0)              # fewer index, more loss
    assert np.all(dt.real > 0) and np.all(np.abs(dt.imag) < 1e-9 * dt.real + 1e-12)
    ratio = -dt.real / dc.real                                       # 0.21, 0.43, 0.76, 1.19
    assert np.all(np.diff(ratio) > 0)
    assert np.all(np.abs(d - dc - dt) < 2e-3 * np.abs(dc))          # the cross term: 2.4e-5 to 6.3e-4 of |dc|
    x = result.crossover()
    print("ratio", ratio, "cross term / |dc|", np.abs(d - dc - dt) / np.abs(dc), "crossover", x)
    print("| V | I' (A/m) | I' V (W/m) | Joule share | core rise (K) | dneff_carrier | dneff_thermal | dneff | "
          "L_pi (mm) | P_pi (mW) | I_pi (mA) | loss (dB/cm) | loss in L_pi (dB) |")
    for k, v in enumerate(result.biases):
        print(f"| {v:.2f} | {result.current[k]:.4g} | {result.power[k]:.4g} | "
              f"{result.heat_joule[k] / result.power[k]:.3f} | {result.rise_core[k]:.4g} | {dc[k].real:.4e} | "
              f"{dt[k].real:.4e} | {d[k].real:.4e} | {result.l_pi[k] * 1e3:.4g} | {result.p_pi[k] * 1e3:.4g} | "
              f"{result.i_pi[k] * 1e3:.4g} | {result.loss_db_per_cm[k]:.4g} | {result.loss_in_l_pi_db[k]:.4g} |")
    # 1.148 V, 60.5 A/m: linear in I' between 1.1 and 1.2 V (1.157 V with the 50 mV steps of the bench)
    assert x is not None and x["interval"] == (1.1, 1.2)
    assert 1.1 < x["bias"] < 1.2 and result.current[2] < x["current"] < result.current[3]
    assert d.real[2] < 0 < d.real[3]
    assert np.all(result.loss_db_per_cm > 0)
    assert np.allclose(result.p_pi, result.power * result.l_pi) and np.allclose(result.i_pi * result.biases,
                                                                               result.p_pi)
    assert np.all(np.diff(result.rise_core) > 0) and result.rise_core[0] > 0


def test_input_checks(electrical):
    with pytest.raises(ValueError, match="increasing"):
        pin_electrical(XS, CHARGE, [1.0, 0.9])
    with pytest.raises(ValueError, match="increasing"):
        pin_electrical(XS, CHARGE, [0.0, 0.9])
    with pytest.raises(ValueError, match="p contact and an n contact"):
        pin_electrical(XS, Window(-1.5, 0.0, -0.5, 0.72, 0.01), [0.5])      # the right side: intrinsic (p)
    with pytest.raises(ValueError, match="1.55 um"):
        pin_optical(XS, OPTICS, THERMAL, 1.31, one_bias(electrical, 0))
    with pytest.raises(ValueError, match="not inside"):
        pin_optical(XS, OPTICS, Window(-1.0, 1.0, -5.0, 2.3, 0.05), 1.55, one_bias(electrical, 0))
    with pytest.raises(ValueError, match="no shape with the tags"):
        uniform_heat(XS, THERMAL, ("core", "heater"), 1.0)


def test_a_thermal_solve_that_does_not_converge_raises(monkeypatch, electrical):
    import thermal_gpu.solver as ts

    real = ts.solve_steady
    monkeypatch.setattr(ts, "solve_steady", lambda *a, **k: real(*a, **k, max_iter=2))
    with pytest.raises(ConvergenceError, match="thermal"):
        pin_optical(XS, OPTICS, THERMAL, 1.55, one_bias(electrical, 0))
