"""The PIN phase shifter with self-heating (photonsuite.pin), on the PIN rib of charge-gpu (pin.pin_rib: textbook
values, lifetimes 1 ns). The measured values (GPU computer) are in the comments; the test bounds are larger.

Checks: the cross-section gives the Device of charge_gpu.injection.rib_pin, and the same currents as the logged
charge-gpu example; the sum of the heat source equals I' V (also on a 1-D diode, and after the move to the thermal
grid); the sum of J . E equals I' (V - V_c) of the contact potentials (analytic); the thermal part against
physics.phase_shifter_ppi with the same heat in the same cells; both parts against the first-order perturbation
(a second-order difference); the limits dn/dT = 0 and no plasma dispersion; the mode is the fundamental mode
(the largest index of a 4-mode solve); the signs and the cross-over. The heat through the public face currents of
charge-gpu is the heat of the private fields (0753ed0) bit for bit. The two-way electro-thermal loop: without a
temperature rise it gives the 300 K result; it converges (the residual per step); its first step is the logged
one-pass feedback; its fixed point is that of a root finder (brentq) and does not depend on the damping; the heat
equals I' V at the converged state."""
import dataclasses

import jax
import numpy as np
import pytest
from scipy.interpolate import RegularGridInterpolator

jax.config.update("jax_enable_x64", True)

from photonsuite.physics import (ConvergenceError, Window, charge_device, phase_shifter_ppi,  # noqa: E402
                                 thermal_problem)
from photonsuite.pin import (SI, SIO2, PinElectrical, cell_overlap, electrothermal, heat_source,  # noqa: E402
                             pin_electrical, pin_optical, pin_phase_shifter, pin_rib, steady_temperature,
                             to_thermal, uniform_heat)

XS = pin_rib()
OPTICS = Window(-1.5, 1.5, -0.5, 0.9, 0.02)
CHARGE = Window(-1.5, 1.5, -0.5, 0.72, 0.01)
THERMAL = Window(-5.0, 5.0, -5.0, 2.3, 0.05)
BIASES = (0.9, 1.0, 1.1, 1.2)
# benchmarks/2026-10-30/charge-dd.jsonl, examples/pin_phase_shifter.py of charge-gpu (tau 1 ns): I' at 0.9 and 1.0 V
LOGGED_CURRENT = {0.9: 3.8282742963924745, 1.0: 15.920944272975659}
LOGGED_DNEFF_09 = -0.0007230039242824837          # the same file, the example with --modes: dneff at 0.9 V
# benchmarks/2026-11-08/suite-pin.jsonl (the first run, --device-temperature): the one-pass feedback, I_hot_rel
LOGGED_ONE_PASS = {0.9: 0.018622148456324483, 1.0: 0.04516366047616849, 1.1: 0.0641118753021257,
                   1.2: 0.06381327412045867}


@pytest.fixture(scope="module")
def electrical():
    return pin_electrical(XS, CHARGE, BIASES, tau=1e-9)


@pytest.fixture(scope="module")
def result(electrical):
    return pin_optical(XS, OPTICS, THERMAL, 1.55, electrical, core_tag="core")


def one_bias(el, k):
    return dataclasses.replace(el, biases=el.biases[k:k + 1], solutions=el.solutions[k:k + 1], seconds=0.0)


@pytest.fixture(scope="module")
def two_way(electrical):
    el = electrothermal(XS, THERMAL, electrical)
    return el, pin_optical(XS, OPTICS, THERMAL, 1.55, el, core_tag="core")


def heat_0753ed0(sol):
    """heat_source of 0753ed0 (it read the private _fluxes and _Mesh fields of charge-gpu): the reference of the
    change to the public face currents. Returns (joule, recombination, je)."""
    from charge_gpu.drift_diffusion import _fluxes, srh
    from charge_gpu.poisson import Q

    m = sol.mesh
    psi = np.asarray(sol.psi, float).ravel()
    n, p = sol.n.ravel()[m.sflat], sol.p.ravel()[m.sflat]
    (_, _, _, jn, jp), (_, _, _, jnc, jpc) = _fluxes(m, psi, n, p)
    ps = psi[m.sflat]
    phi_n = ps - m.vt * np.log(n / m.n_i)
    phi_p = ps + m.vt * np.log(p / m.n_i)
    v_c = np.where(np.asarray(sol.contact_kinds)[m.cid] == "n", sol.v_n, sol.v_p)
    face = -Q * (jn * (phi_n[m.sb] - phi_n[m.sa]) + jp * (phi_p[m.sb] - phi_p[m.sa]))
    cface = -Q * (jnc * (v_c - phi_n[m.cs]) + jpc * (v_c - phi_p[m.cs]))
    ns = m.n_semi
    joule = 0.5 * (np.bincount(m.sa, face, ns) + np.bincount(m.sb, face, ns)) + np.bincount(m.cs, cface, ns)
    rec = Q * srh(n, p, m.n_i, sol.tau_n, sol.tau_p)[0] * m.svol * (phi_p - phi_n)
    je = -Q * (np.sum((jn + jp) * (psi[m.fb] - psi[m.fa])) + np.sum((jnc + jpc) * (m.cpsi - psi[m.cflat])))

    def grid(v):
        out = np.zeros(m.n_cells)
        out[m.sflat] = v
        return out.reshape(m.shape)

    return grid(joule), grid(rec), float(je)


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
    # largest index (meow_gpu/README.md, "The modes of the PEC walls": the solver excludes the spurious wall modes)
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


def test_the_heat_of_the_public_face_currents_is_that_of_0753ed0(electrical, two_way):
    # the same numbers bit for bit, at 300 K (the one-way path) and at the loop temperatures
    for sol in [*electrical.solutions, *two_way[0].solutions]:
        h = heat_source(sol)
        joule, rec, je = heat_0753ed0(sol)
        assert np.array_equal(h.joule, joule) and np.array_equal(h.recombination, rec) and h.je == je


def test_the_loop_without_a_temperature_rise_is_the_300_K_result(monkeypatch, electrical, result):
    # dT forced to 0 (the steady temperature of the module returns t_ambient everywhere): one solve per bias at
    # Device.T = 300 K, from the 300 K solution; the same currents, carriers and carrier part
    import photonsuite.pin as pin

    real = pin.steady_temperature

    def no_rise(xs, win, q, t_ambient=300.0, *a, **k):
        th = real(xs, win, q, t_ambient, *a, **k)
        zero = RegularGridInterpolator((th.x, th.y), np.zeros_like(th.T), bounds_error=False, fill_value=None)
        return dataclasses.replace(th, rise=zero, T=np.full_like(th.T, t_ambient))

    monkeypatch.setattr(pin, "steady_temperature", no_rise)
    el = electrothermal(XS, THERMAL, electrical)
    assert np.all(el.loop.iterations == 1) and np.all(el.loop.temperature == 300.0)
    assert np.all(el.loop.residual == 0.0) and np.all(el.loop.spread == 0.0)
    worst = {"I": 0.0, "n": 0.0, "psi": 0.0}
    for s, s0 in zip(el.solutions, electrical.solutions):
        assert s.converged and s.mesh.vt == s0.mesh.vt and s.mesh.n_i == s0.mesh.n_i and s.tau_n == s0.tau_n
        semi = electrical.device.semiconductor
        worst["I"] = max(worst["I"], abs(s.current / s0.current - 1))
        worst["n"] = max(worst["n"], np.max(np.abs(s.n[semi] / s0.n[semi] - 1)), np.max(np.abs(s.p[semi] / s0.p[semi]
                                                                                               - 1)))
        worst["psi"] = max(worst["psi"], np.max(np.abs(s.psi - s0.psi)))
    r = pin_optical(XS, OPTICS, THERMAL, 1.55, one_bias(el, 2))
    rel = abs(r.dneff_carrier[0] / result.dneff_carrier[2] - 1)
    print(f"dT = 0: I' {worst['I']:.1e}, n and p {worst['n']:.1e}, psi {worst['psi']:.1e} V; dneff_carrier at 1.1 V "
          f"{rel:.1e}, dneff_thermal {abs(r.dneff_thermal[0]):.1e}")
    assert worst["I"] < 1e-10 and worst["n"] < 1e-9 and worst["psi"] < 1e-12
    assert rel < 1e-8 and abs(r.dneff_thermal[0]) < 1e-10


def test_the_loop_converges_to_a_consistent_state(electrical, result, two_way):
    from charge_gpu.poisson import thermal_voltage

    el, r = two_way
    loop = el.loop
    print("| V | T (K) | solves | residual per step (K) | I' one-way | I' two-way | one pass | two-way |")
    for k, v in enumerate(el.biases):
        h = np.array(loop.history[k])                     # (steps, 4): T, I', T from the heat, residual
        res = np.abs(h[:, 3])
        ratio = res[1:] / res[:-1]
        print(f"| {v} | {loop.temperature[k]:.6f} | {loop.iterations[k]} | "
              f"{' '.join(f'{x:.1e}' for x in h[:, 3])} | {h[0, 1]:.6g} | {el.solutions[k].current:.6g} | "
              f"{h[1, 1] / h[0, 1] - 1:.5f} | {el.solutions[k].current / h[0, 1] - 1:.5f} |")
        assert loop.iterations[k] == len(h) - 1 and res[-1] <= loop.tol and loop.residual[k] == h[-1, 3]
        assert np.all(ratio < 0.2)                         # a contraction: the gain dG/dT (0.018 to 0.066)
        # step 0: the 300 K solution of the sweep; step 1: the one-pass feedback of the logged bench
        assert h[0, 0] == 300.0 and h[0, 1] == electrical.solutions[k].current and h[1, 0] == h[0, 2]
        assert abs(h[1, 1] / h[0, 1] - 1 - LOGGED_ONE_PASS[v]) < 1e-8
        # the converged state: the device temperature of the solution; the optical part finds the same mean rise
        assert el.solutions[k].mesh.vt == thermal_voltage(loop.temperature[k]) == thermal_voltage(h[-1, 0])
        assert abs(300.0 + r.rise_semiconductor[k] - loop.temperature[k]) <= loop.tol
        assert 0 < loop.spread[k] < r.rise_semiconductor[k]
        # energy: the heat integral is I' V at the converged state (on the charge cells and on the thermal grid)
        assert abs(heat_source(el.solutions[k]).total.sum() / r.power[k] - 1) < 1e-9
        assert abs(r.heat[k] / r.power[k] - 1) < 1e-9
    # the feedback raises the current, the carrier change and the heat; the 0 V reference stays at 300 K
    assert np.all(r.current > result.current) and np.all(r.rise_core > result.rise_core)
    assert np.all(r.dneff_carrier.real < result.dneff_carrier.real)
    assert np.all(r.dneff_thermal.real > result.dneff_thermal.real)
    assert el.reference is electrical.reference and np.array_equal(r.biases, result.biases)
    x, x1 = r.crossover(), result.crossover()
    print("| V | dneff_carrier one-way | two-way | dneff_thermal one-way | two-way | dneff one-way | two-way |")
    for k, v in enumerate(r.biases):
        print(f"| {v} | {result.dneff_carrier[k].real:.4e} | {r.dneff_carrier[k].real:.4e} | "
              f"{result.dneff_thermal[k].real:.4e} | {r.dneff_thermal[k].real:.4e} | {result.dneff[k].real:.4e} | "
              f"{r.dneff[k].real:.4e} |")
    print("crossover one-way", x1, "two-way", x)
    assert x is not None and x["interval"] == (1.1, 1.2)


def test_the_fixed_point_of_a_root_finder(electrical, two_way):
    # 1.2 V (the largest feedback): G(T) - T = 0 by brentq, with G from a fresh solve at T (from the 300 K solution)
    # and the mean of charge_gpu.temperature.mean_temperature; the loop has |G(T) - T| <= tol = 1e-4 K and a gain
    # below 0.1, so it is within about 1.1e-4 K of the root
    from charge_gpu import drift_diffusion as dd
    from charge_gpu.temperature import mean_temperature
    from scipy.optimize import brentq

    el, _ = two_way
    k, v = 3, 1.2
    dev, sol0 = electrical.device, electrical.solutions[k]
    xe, ye = np.asarray(dev.x) * 1e6, np.asarray(dev.y) * 1e6
    xc, yc = 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1])
    pts = np.stack([a.ravel() for a in np.meshgrid(xc, yc, indexing="ij")], -1)
    calls = []

    def f(t):
        s = dd.solve(dataclasses.replace(dev, T=t), 0.0, v, guess=sol0, tau_n=1e-9, tau_p=1e-9)
        assert s.converged
        th = steady_temperature(XS, THERMAL, to_thermal(heat_source(s).total, xe, ye, THERMAL))
        calls.append(t)
        return mean_temperature(dev, 300.0 + th.rise(pts).reshape(xc.size, yc.size))[0] - t

    root = brentq(f, 300.0, el.loop.history[k][1][0] + 10.0, xtol=1e-7)
    print(f"1.2 V: brentq {root:.7f} K ({len(calls)} solves), the loop {el.loop.temperature[k]:.7f} K "
          f"({el.loop.iterations[k]} solves), difference {el.loop.temperature[k] - root:.1e} K")
    assert abs(el.loop.temperature[k] - root) < 1.2e-4


def test_the_damping_does_not_change_the_fixed_point(electrical, two_way):
    el, _ = two_way
    d = electrothermal(XS, THERMAL, one_bias(electrical, 0), damping=0.5).loop
    h = np.array(d.history[0])
    print(f"0.9 V: damping 0.5: {d.temperature[0]:.7f} K in {d.iterations[0]} solves; damping 1: "
          f"{el.loop.temperature[0]:.7f} K in {el.loop.iterations[0]} solves; residuals "
          f"{' '.join(f'{x:.1e}' for x in h[:, 3])}")
    assert d.iterations[0] > el.loop.iterations[0] and abs(d.residual[0]) <= d.tol
    assert abs(d.temperature[0] - el.loop.temperature[0]) < 2.5e-4
    assert np.all(np.abs(h[1:, 3]) < np.abs(h[:-1, 3]))


def test_input_checks_of_the_loop(electrical):
    el = one_bias(electrical, 3)
    with pytest.raises(ConvergenceError, match="electro-thermal loop at 1.2 V"):
        electrothermal(XS, THERMAL, el, max_iter=1)                  # one solve leaves a residual of about 1.5 K
    for kw, match in ((dict(tol=0.0), "tol"), (dict(damping=0.0), "damping"), (dict(damping=1.5), "damping"),
                      (dict(max_iter=0), "max_iter")):
        with pytest.raises(ValueError, match=match):
            electrothermal(XS, THERMAL, el, **kw)
    with pytest.raises(ValueError, match="tau"):
        electrothermal(XS, THERMAL, PinElectrical(el.device, el.window, el.biases, el.solutions, el.reference, 0.0))
    with pytest.raises(ValueError, match="mobility"):
        electrothermal(XS, THERMAL, dataclasses.replace(el, options={"mobility": (0.1, 0.04)}))
    with pytest.raises(ValueError, match="coupling"):
        pin_phase_shifter(XS, OPTICS, CHARGE, THERMAL, 1.55, [0.9], coupling="both")
