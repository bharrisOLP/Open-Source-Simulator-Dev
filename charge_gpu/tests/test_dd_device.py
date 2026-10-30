"""Checks of the device continuity solver (charge_gpu.continuity) and of the Gummel iteration with linear="device":
the Slotboom operator against the SG matrix of the SuperLU path, the linear solves against SuperLU, and the whole
solution against the coupled Newton method with SuperLU (the reference) at 0 V, reverse bias and forward bias up to
high injection; the Shockley diode and the current continuity with the device path."""
import jax
import numpy as np
import pytest
import scipy.sparse.linalg as spl

jax.config.update("jax_enable_x64", True)

from charge_gpu import continuity  # noqa: E402
from charge_gpu import drift_diffusion as dd  # noqa: E402
from charge_gpu.injection import rib_index_change, rib_pin  # noqa: E402
from charge_gpu.poisson import solve as poisson_solve, thermal_voltage  # noqa: E402
from test_drift_diffusion import TAU_LB, _Guess  # noqa: E402
from test_junction import junction  # noqa: E402

VT = thermal_voltage()
TAU_PIN = 1e-9


def _dense_rel(a, b, floor=dd.CARRIER_FLOOR):
    """Max relative difference of two carrier arrays where the reference exceeds the floor of the metric."""
    k = b > floor
    return float(np.max(np.abs(a[k] / b[k] - 1)))


def _compare(sol, ref):
    """psi (V), n and p (relative), current (relative to the larger contact current)."""
    m = ref.mesh
    n, p = sol.n.ravel()[m.sflat], sol.p.ravel()[m.sflat]
    nr, pr = ref.n.ravel()[m.sflat], ref.p.ravel()[m.sflat]
    di = np.max(np.abs(sol.contact_currents - ref.contact_currents)) / max(np.max(np.abs(ref.contact_currents)),
                                                                           1e-30)
    return float(np.max(np.abs(sol.psi - ref.psi))), _dense_rel(n, nr), _dense_rel(p, pr), float(di)


@pytest.fixture(scope="module")
def pin():
    """The coarse PIN rib of test_drift_diffusion (10 nm cells, contacts 1 um from the center): Newton (SuperLU) at
    0.85, 0.9, 0.95 and 1.0 V."""
    dev = rib_pin(dx=10e-9, half_span=1.0e-6, clad=0.1e-6)
    biases = [0.85, 0.9, 0.95, 1.0]
    return dev, dict(zip(biases, dd.sweep(dev, biases, tau_n=TAU_PIN, tau_p=TAU_PIN)))


def test_slotboom_operator_equals_the_sg_matrix(pin):
    """S u = -M (E u) for electrons and S_p v = M_p (F v) for holes (M: the SG matrix of the SuperLU path), and
    b = -rhs (electrons), b = rhs (holes); S is symmetric with positive weights. On the PIN rib at 1.0 V."""
    sols = pin[1]
    s = sols[1.0]
    m = s.mesh
    psi, n, p = s.psi.ravel(), s.n.ravel()[m.sflat], s.p.ravel()[m.sflat]
    den = TAU_PIN * (n + m.n_i) + TAU_PIN * (p + m.n_i)
    solver = continuity.Solver(m, jax.devices("cpu")[0])
    rng = np.random.default_rng(3)
    for carrier, other, sign in (("n", p, -1.0), ("p", n, 1.0)):
        gx, gy, extra, b, scale = solver.operator(psi, carrier, other, den)
        assert np.all(gx >= 0) and np.all(gy >= 0) and np.all(extra >= 0)
        mat, rhs = dd._continuity_matrix(m, psi, carrier, other, den, m.n_semi)
        u = rng.uniform(0.5, 2.0, m.n_semi) * (n if carrier == "n" else p) / scale
        lv = continuity.hierarchy(gx, gy, extra, jax.devices("cpu")[0])[0][0]
        su = np.asarray(continuity._apply(lv, solver.to_grid(u))).ravel()[solver.cell]
        mu = sign * (mat @ (scale * u))
        assert np.max(np.abs(su - mu) / (np.abs(mat) @ (scale * u))) < 1e-13
        assert np.max(np.abs(b - sign * rhs) / np.abs(rhs)) < 1e-14


def test_linear_solve_equals_superlu(pin):
    """The device solve of one Gummel continuity step (electrons, then holes) at the 1.0 V state against SuperLU,
    from a start 30 % off: the same carriers to 1e-11, the componentwise backward error of the true residual (here
    recomputed on the host with the SG matrix) below 1e-11, converged = True. With 1 iteration and no restart the
    flag is False."""
    sols = pin[1]
    s = sols[1.0]
    m = s.mesh
    psi, n, p = s.psi.ravel(), s.n.ravel()[m.sflat], s.p.ravel()[m.sflat]
    solver = continuity.Solver(m)
    rng = np.random.default_rng(4)
    den = TAU_PIN * (n + m.n_i) + TAU_PIN * (p + m.n_i)
    for carrier, other, c in (("n", p, n), ("p", n, p)):
        mat, rhs = dd._continuity_matrix(m, psi, carrier, other, den, m.n_semi)
        ref = spl.spsolve(mat, rhs)
        start = c * np.exp(0.3 * rng.standard_normal(c.size))
        res = solver.solve(psi, carrier, other, den, start)
        assert res.converged and res.backward_error < 1e-11
        assert np.max(np.abs(res.c / ref - 1)) < 1e-11
        omega = np.max(np.abs(rhs - mat @ res.c) / (np.abs(mat) @ np.abs(res.c) + np.abs(rhs)))
        assert omega < 1e-11
        bad = solver.solve(psi, carrier, other, den, start, max_iter=1, restarts=0)
        assert not bad.converged and bad.backward_error > 1e-11


@pytest.mark.parametrize("bias", [0.9, 1.0])
def test_pin_high_injection_equals_newton(pin, bias):
    """Gummel with the device continuity solves at 0.9 and 1.0 V (high injection: n and p in the rib about 1e17 to
    5e17 cm^-3), started from the Newton solution 50 mV lower: the solution of the Newton method with SuperLU
    (psi 1e-11 V, n and p 1e-8, contact currents 1e-8), the same total current through every column (1e-9 of I),
    and the same rib index change."""
    dev, sols = pin
    start = sols[round(bias - 0.05, 2)]
    ref = sols[bias]
    g = dd.solve(dev, 0.0, bias, guess=start, method="gummel", linear="device", tau_n=TAU_PIN, tau_p=TAU_PIN,
                 max_iter=4000)
    assert g.converged and g.linear_converged and g.linear_error < 1e-11 and g.linear_fallbacks == 0
    d_psi, d_n, d_p, d_i = _compare(g, ref)
    assert d_psi < 1e-11 and d_n < 1e-8 and d_p < 1e-8 and d_i < 1e-8, (d_psi, d_n, d_p, d_i)
    assert np.max(np.abs(g.x_currents() - g.current)) < 1e-9 * g.current
    r_g = rib_index_change(dev, g, start)
    r_ref = rib_index_change(dev, ref, start)
    assert r_g["dn_mean"] == pytest.approx(r_ref["dn_mean"], rel=1e-8)


@pytest.mark.parametrize("v_n, v_p", [(0.0, 0.6), (1.0, 0.0), (3.0, 0.0), (5.0, 0.0), (10.0, 0.0), (25.0, 0.0)])
def test_junction_equals_newton(v_n, v_p):
    """The 1-D junction (1e18 cm^-3, 2.5 nm cells) at 0.6 V forward and 1, 3, 5, 10, 25 V reverse (up to 12 V_T
    per cell in the depletion region; with the pseudo-inverse on the coarsest level the device solve diverged from
    5 V): Gummel with the device continuity solves = Newton with SuperLU (psi 1e-12 V; n and p 1e-9 where they
    exceed the floor of the metric, 1 cm^-3; at 0.6 V the current to DEVICE_NOISE), with no fallback to SuperLU.
    Gummel starts from poisson.solve at the bias; the Newton reference comes from a continuation (from the
    poisson.solve guess Newton does not converge at 5 V and more)."""
    dev, _ = junction(n=400)
    bias = v_p - v_n
    a = dd.sweep(dev, [bias], step=0.5)[0]
    b = dd.solve(dev, 0.0, bias, method="gummel", linear="device")
    assert a.converged and b.converged and b.linear_converged and b.linear_fallbacks == 0
    d_psi, d_n, d_p, _ = _compare(b, a)
    assert d_psi < 1e-12 and d_n < 1e-9 and d_p < 1e-9, (d_psi, d_n, d_p)
    if v_p > 0:              # the current at 0.6 V is near the face-current floor: measured 3e-7 relative
        assert abs(b.current - a.current) < DEVICE_NOISE


def test_a_device_solve_that_fails_is_not_used(monkeypatch):
    """A device solve that is not converged (here: a result of garbage, flagged converged = False) is replaced by
    SuperLU for that step: the solution is still the Newton solution, and every step counts as a fallback. Also
    with SPAN_MAX below the psi span of the device: Solver.operator raises SpanError, and the steps use SuperLU."""
    dev, _ = junction(n=200)
    ref = dd.solve(dev, 1.0, 0.0)
    real = continuity.Solver.solve

    def bad(self, psi, carrier, other, den, c0, **kw):
        res = real(self, psi, carrier, other, den, c0, **kw)
        return continuity.LinearResult(c=res.c * 1e3, iterations=res.iterations, backward_error=0.5,
                                       converged=False)

    monkeypatch.setattr(continuity.Solver, "solve", bad)
    g = dd.solve(dev, 1.0, 0.0, method="gummel", linear="device")
    assert g.converged and g.linear_converged and g.linear_fallbacks == 2 * g.iterations
    assert np.max(np.abs(g.psi - ref.psi)) < 1e-12
    monkeypatch.setattr(continuity.Solver, "solve", real)
    monkeypatch.setattr(continuity, "SPAN_MAX", 10.0)
    m = ref.mesh
    with pytest.raises(continuity.SpanError, match="SPAN_MAX"):
        continuity.Solver(m).operator(ref.psi.ravel(), "n", ref.p.ravel()[m.sflat], np.ones(m.n_semi))
    g = dd.solve(dev, 1.0, 0.0, method="gummel", linear="device")
    assert g.converged and g.linear_fallbacks == 2 * g.iterations and g.krylov_iterations == 0
    assert np.max(np.abs(g.psi - ref.psi)) < 1e-12


def test_backward_error_of_both_linear_solvers():
    """DDSolution.linear_error is the measured componentwise backward error of the last continuity steps for both
    linear solvers of Gummel (SuperLU: round-off), and None for Newton (no continuity steps)."""
    dev, _ = junction(n=200)
    s = dd.solve(dev, 0.0, 0.6, method="gummel")
    d = dd.solve(dev, 0.0, 0.6, method="gummel", linear="device")
    n = dd.solve(dev, 0.0, 0.6)
    assert 0.0 < s.linear_error < 1e-14 and s.linear_fallbacks == 0
    assert 0.0 < d.linear_error < continuity.CHECK
    assert n.linear_error is None


# the floor of a face current on the device path (A/m): the majority-carrier drift and diffusion terms of a face of
# the 1-D junction are q D N / h x 1 um = 4.6e4 A/m (2.5 nm cells, 1e18 cm^-3, D 7.2e-4 m^2/s), and the device solve
# stops at a backward error of about 1e-15 to 1e-14 (TOL), so its carriers have relative errors of about 1e-14: a
# face current of about 5e-10 A/m (measured 2e-10 to 5e-10 A/m, it changes from run to run: the order of the sums
# on the GPU). The limit is 10x that. SuperLU leaves about 1e-16: the floor of test_drift_diffusion (NOISE, 1e-9).
DEVICE_NOISE = 5e-9


def test_zero_bias_equals_poisson():
    """0 V, from the perturbed state of test_drift_diffusion (20 mV on psi, 30 % on n and p): the device path gives
    the equilibrium of poisson.solve and no current (below DEVICE_NOISE)."""
    dev, _ = junction(n=400)
    ref = poisson_solve(dev, 0.0, 0.0)
    rng = np.random.default_rng(0)
    g = _Guess(ref.psi + 0.02 * rng.standard_normal(ref.psi.shape),
               ref.n * np.exp(0.3 * rng.standard_normal(ref.psi.shape)),
               ref.p * np.exp(0.3 * rng.standard_normal(ref.psi.shape)))
    sol = dd.solve(dev, 0.0, 0.0, guess=g, method="gummel", linear="device")
    assert sol.converged and sol.linear_converged and sol.linear_fallbacks == 0
    assert np.max(np.abs(sol.psi - ref.psi)) < 1e-11
    assert np.max(np.abs(sol.n / ref.n - 1)) < 1e-9 and np.max(np.abs(sol.p / ref.p - 1)) < 1e-9
    assert np.max(np.abs(sol.contact_currents)) < DEVICE_NOISE
    assert np.max(np.abs(sol.x_currents())) < DEVICE_NOISE


def test_short_base_diode_against_shockley_on_the_device():
    """A short-base diode (1e18 cm^-3 both sides, tau 100 us, contacts 10 um from the junction: L = 270 um, so the
    minority carriers decrease linearly to the contacts), a whole sweep 0 to 0.7 V with Gummel and the device
    continuity solves. Reference: the Shockley diode with the finite neutral widths W' = 10 um - W_dep / 2 (Sze and
    Ng, Ch. 2: J0 = q n_i^2 (D_n / (L_n N_A) coth(W'_p / L_n) + D_p / (L_p N_D) coth(W'_n / L_p)); W_dep of the
    depletion approximation, 23 to 28 nm). Newton gives J / J_Shockley = 1.00017, 1.00006, 0.99992 at 0.6, 0.65,
    0.7 V (a scratch run). The device path: within 1e-3 of the Shockley value.
    Current accuracy: the Gummel iteration has a floor here, with either linear solver. Its relative updates do not
    fall much below 1e-11 (tol 1e-12 fails at 0 V), and the contact currents (differences of majority drift and
    diffusion terms about 2e4 times larger) then differ from Newton by up to about 1e-9 A/m: scratch runs with the
    SuperLU continuity steps gave dI / I = -2.8e-5, -6.4e-6, 1.1e-6 and contact sums up to 1.5e-9 A/m; the device
    path -3.5e-5, 1.6e-6, -3.1e-7 and 5.8e-10 A/m (I = 2.2e-5, 1.5e-4, 1.0e-3 A/m). So the current, the sum of the
    contact currents and the current through each cross-section are checked to 5e-9 A/m. The long-base diode of
    test_drift_diffusion is not used here: Gummel stalls on it with both linear solvers (README)."""
    from charge_gpu.injection import diode_1d, shockley_j0
    from charge_gpu.poisson import EPS0, Q
    from charge_gpu.rc import masetti_mobility

    n_dop, width, biases = 1e24, 10e-6, [0.6, 0.65, 0.7]
    mu_n, mu_p = masetti_mobility(n_dop, "n"), masetti_mobility(n_dop, "p")
    dev = diode_1d(n_dop, n_dop, width, width)
    sols = dd.sweep(dev, biases, method="gummel", linear="device", tau_n=TAU_LB, tau_p=TAU_LB)
    refs = dd.sweep(dev, biases, tau_n=TAU_LB, tau_p=TAU_LB)
    vbi = VT * np.log(n_dop * n_dop / dev.n_i**2)
    for v, g, ref in zip(biases, sols, refs):
        assert g.converged and g.linear_converged and g.method == "gummel" and g.linear == "device"
        assert g.linear_fallbacks == 0
        w_dep = np.sqrt(2 * 11.7 * EPS0 * (vbi - v - 2 * VT) / Q * 2 / n_dop)
        j0 = shockley_j0(n_dop, n_dop, mu_n, mu_p, TAU_LB, TAU_LB, length_p=width - w_dep / 2,
                         length_n=width - w_dep / 2)
        assert g.current / dev.y[-1] / (j0 * np.expm1(v / VT)) == pytest.approx(1.0, abs=1e-3)
        assert abs(g.current - ref.current) < 5e-9
        assert abs(np.sum(g.contact_currents)) < 5e-9
        assert np.max(np.abs(g.x_currents() - g.current)) < 5e-9


def test_linear_option_is_checked():
    dev, _ = junction(n=50)
    with pytest.raises(ValueError, match="linear"):
        dd.solve(dev, 0.0, 0.1, method="gummel", linear="cg")
    with pytest.raises(ValueError, match="newton"):
        dd.solve(dev, 0.0, 0.1, linear="device")
