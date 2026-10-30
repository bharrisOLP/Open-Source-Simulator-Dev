"""Checks of the small-signal capacitance, the series resistance and the mobility model (charge_gpu.rc), on CPU-size
cases: the 1-D abrupt junction of test_junction.py (N_A = N_D = 1e18 cm^-3) and uniform slabs."""
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from charge_gpu.poisson import EPS0, Q, Contact, Device, solve, thermal_voltage  # noqa: E402
from charge_gpu.rc import (masetti_mobility, path_resistance, rc_bandwidth, series_resistance,  # noqa: E402
                           small_signal)
from test_junction import EPS_SI, NA, ND, NI, junction  # noqa: E402

BIASES = [0.0, 1.0, 3.0]
HEIGHT = 1e-6                    # y extent of the 1-D junction: per length = per area x HEIGHT


@pytest.fixture(scope="module")
def sweep():
    dev, _ = junction()
    return dev, small_signal(dev, BIASES, dv=0.02)


def w_sze(v_r):
    """Depletion width with the Debye term (Sze), as in test_junction.py."""
    vt = thermal_voltage()
    vbi = vt * np.log(NA * ND / NI**2)
    return np.sqrt(2 * EPS_SI * EPS0 * (vbi + v_r - 2 * vt) * (NA + ND) / (Q * NA * ND))


def test_capacitance_against_the_depletion_approximation(sweep):
    """C = dQ_n/dV per area against eps / W(V) (Sze: C_D = eps_s / W, with the same W as the depletion-width check).
    Tolerance 3e-4: the central difference (h = 20 mV) has a truncation error of h^2 / (8 (V_bi + V - 2 V_T)^2),
    6e-5 at 0 V; the rest is the Debye-tail approximation of the W formula."""
    _, res = sweep
    for r in res:
        assert r.converged
        assert r.c / HEIGHT == pytest.approx(EPS_SI * EPS0 / w_sze(r.bias), rel=3e-4)


def test_capacitance_decreases_with_reverse_bias(sweep):
    """Strictly decreasing at 0, 1 and 3 V. (The ratio C(0) / C(3 V) = W(3 V) / W(0) follows from the first test,
    so it is not a separate check.)"""
    _, res = sweep
    assert np.all(np.diff([r.c for r in res]) < 0)


def test_charge_neutral_total(sweep):
    """The n side and the p side hold opposite charges (the contact regions are neutral), so -dQ_p/dV = dQ_n/dV."""
    _, res = sweep
    for r in res:
        assert r.q_n > 0 > r.q_p
        assert abs(r.q_n + r.q_p) < 1e-9 * r.q_n
        assert r.c_p == pytest.approx(r.c, rel=1e-8)


def test_series_resistance_of_the_junction_against_the_neutral_regions(sweep):
    """R' = sum of the neutral lengths L/2 - W/2 (N_A = N_D: W_n = W_p = W/2) over q mu N t, with W from Sze.
    R' falls with reverse bias (the depletion region takes more of the path)."""
    dev, res = sweep
    mu_n, mu_p = masetti_mobility(ND, "n"), masetti_mobility(NA, "p")
    for r in res:
        neutral = 0.5e-6 - w_sze(r.bias) / 2
        assert r.r_n == pytest.approx(neutral / (Q * mu_n * ND * HEIGHT), rel=1e-4)
        assert r.r_p == pytest.approx(neutral / (Q * mu_p * NA * HEIGHT), rel=1e-4)
    assert np.all(np.diff([r.r for r in res]) < 0)
    s = res[0].solution
    with pytest.raises(ValueError, match="small-signal charge"):  # the same bias three times: no charge change
        series_resistance(dev, s, s, s)


def test_series_resistance_rejects_a_column_with_n_and_p_cells():
    """An L-shaped junction (rib junction at x = 0, slab junction at x = +0.15 um): the columns between them have n
    cells (rib) and p cells (slab), so the 1-D path model does not apply. The lateral rib passes (a control)."""
    from charge_gpu.modulator import rib_pn
    lateral = rib_pn(half_span=1.0e-6, dx=20e-9)
    r = small_signal(lateral, [1.0])[0]
    assert r.converged and r.r > 0
    dev = rib_pn(half_span=1.0e-6, dx=20e-9)
    xc = 0.5 * (dev.x[1:] + dev.x[:-1])
    yc = 0.5 * (dev.y[1:] + dev.y[:-1])
    X, Y = np.meshgrid(xc, yc, indexing="ij")
    dev.n_net = np.where((X > 0) & (X < 0.15e-6) & (Y > 0) & (Y < 0.09e-6), -3e23, dev.n_net)
    with pytest.raises(ValueError, match="n and p cells"):
        small_signal(dev, [1.0])


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_uniform_slab_against_the_sheet_resistance(sign):
    """A 2 um long, 100 nm thick slab (1e17 cm^-3, n or p) in oxide, contacts at both ends at 0 V:
    R' = L R_sheet, R_sheet = 1 / (q mu N t). The oxide rows carry no current."""
    dop, length, t, d = 1e23, 2e-6, 100e-9, 10e-9
    x, y = np.arange(0, length + d / 2, d), np.arange(0, 3 * t + d / 2, d)
    yc = 0.5 * (y[1:] + y[:-1])
    slab = (yc > t) & (yc < 2 * t)
    semi = np.broadcast_to(slab, (x.size - 1, yc.size)).copy()
    dev = Device(x=x, y=y, eps_r=np.where(semi, 11.7, 3.9), n_net=np.where(semi, sign * dop, 0.0),
                 semiconductor=semi, contacts=[Contact("xmin", slab, 0.0), Contact("xmax", slab, 0.0)], n_i=NI)
    sol = solve(dev, 0.0, 0.0)
    assert sol.converged
    mu = masetti_mobility(dop, "n" if sign > 0 else "p")
    r_sheet = 1 / (Q * mu * dop * t)
    assert path_resistance(dev, sol) == pytest.approx(length * r_sheet, rel=1e-9)
    # the dissipation weight: a conduction current that grows linearly along the path (a uniform RC line) gives R/3
    f = (np.arange(x.size - 1) + 0.5) / (x.size - 1)
    assert path_resistance(dev, sol, fraction=f) == pytest.approx(length * r_sheet / 3, rel=1e-4)
    with pytest.raises(ValueError):                              # no junction: no series resistance of a capacitor
        series_resistance(dev, sol, sol, sol)


def test_mobility_against_the_published_model():
    """Masetti, Severi, Solmi (1983), Table I (phosphorus: electrons; boron: holes), 300 K. Reference values in
    cm^2/(V s), evaluated by hand from the published parameters (a separate script, not this code), at 1e16 and
    1e18 cm^-3; and the limits: mu_max at low doping (1414, 470.5)."""
    to_cm2 = 1e4
    assert masetti_mobility(1e22, "n") * to_cm2 == pytest.approx(1183.79, abs=0.01)
    assert masetti_mobility(1e24, "n") * to_cm2 == pytest.approx(276.96, abs=0.01)
    assert masetti_mobility(1e22, "p") * to_cm2 == pytest.approx(424.91, abs=0.01)
    assert masetti_mobility(1e24, "p") * to_cm2 == pytest.approx(160.31, abs=0.01)
    assert masetti_mobility(1e14, "n") * to_cm2 == pytest.approx(1414.0, rel=1e-4)
    assert masetti_mobility(1e14, "p") * to_cm2 == pytest.approx(470.5, rel=1e-4)
    mu = masetti_mobility(np.logspace(20, 26, 13), "n")
    assert np.all(np.diff(mu) < 0)


def test_rc_bandwidth(sweep):
    """f = 1 / (2 pi (R'/L + R_s) C' L): no source resistance gives 1 / (2 pi R' C') for all L. SmallSignal.f3db
    needs the length when it has a source resistance."""
    r, c = 0.02, 2.5e-10                                         # 20 Ohm mm, 0.25 pF/mm
    assert rc_bandwidth(r, c, 1e-3) == pytest.approx(1 / (2 * np.pi * r * c), rel=1e-12)
    assert rc_bandwidth(r, c, 0.2e-3) == pytest.approx(rc_bandwidth(r, c, 2e-3), rel=1e-12)
    assert rc_bandwidth(r, c, 1e-3, r_source=50.0) == pytest.approx(1 / (2 * np.pi * 70.0 * 2.5e-13), rel=1e-12)
    s = sweep[1][0]
    assert s.f3db() == pytest.approx(1 / (2 * np.pi * s.r * s.c), rel=1e-12)
    assert s.f3db(0.5e-3, 50.0) == pytest.approx(rc_bandwidth(s.r, s.c, 0.5e-3, 50.0), rel=1e-12)
    with pytest.raises(ValueError):
        s.f3db(r_source=50.0)
