"""Checks of the drift-diffusion solver (charge_gpu.drift_diffusion) on CPU-size cases: the 1-D abrupt junction of
test_junction.py (N_A = N_D = 1e18 cm^-3, 1 um, 2.5 nm cells), a 1-D long-base diode against the Shockley ideal
diode, and a coarse lateral PIN rib."""
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from charge_gpu import drift_diffusion as dd  # noqa: E402
from charge_gpu.injection import diode_1d, rib_index_change, rib_pin, shockley_j0  # noqa: E402
from charge_gpu.poisson import Q, solve as poisson_solve, thermal_voltage  # noqa: E402
from charge_gpu.rc import masetti_mobility  # noqa: E402
from test_junction import junction  # noqa: E402

VT = thermal_voltage()
NOISE = 1e-9            # A/m: above the double-precision floor of the face currents in the 1-D junction (1e-10)

# the long-base diode of the Shockley check: 1e18 cm^-3 both sides, tau = 100 us, contacts 6 diffusion lengths away
N_LB, TAU_LB = 1e24, 1e-4


def long_base(fine=1e-9, ratio=1.03):
    mu_n, mu_p = masetti_mobility(N_LB, "n"), masetti_mobility(N_LB, "p")
    l_n, l_p = np.sqrt(mu_n * VT * TAU_LB), np.sqrt(mu_p * VT * TAU_LB)
    dev = diode_1d(N_LB, N_LB, 6 * l_n, 6 * l_p, fine=fine, ratio=ratio)
    return dev, shockley_j0(N_LB, N_LB, mu_n, mu_p, TAU_LB, TAU_LB, length_p=6 * l_n, length_n=6 * l_p)


class _Guess:
    def __init__(self, psi, n, p):
        self.psi, self.n, self.p = psi, n, p


def test_bernoulli_and_jacobian():
    """B(x) - B(-x) = -x, B(0) = 1, B' against a central difference; the assembled Jacobian against a central
    difference of the residual (forward bias, all three blocks)."""
    x = np.array([-800.0, -30.0, -1.0, -1e-5, 0.0, 1e-7, 1e-3, 2.0, 50.0, 800.0])
    assert np.allclose(dd.bernoulli(x) - dd.bernoulli(-x), -x, rtol=1e-13, atol=1e-13)
    assert dd.bernoulli(0.0) == 1.0
    h = 1e-6
    xd = np.array([-20.0, -0.5, -2e-4, 5e-5, 0.3, 7.0])
    fd = (dd.bernoulli(xd + h) - dd.bernoulli(xd - h)) / (2 * h)
    assert np.allclose(dd._dbernoulli(xd), fd, rtol=1e-7, atol=1e-10)

    dev, _ = junction(n=200)
    sol = dd.solve(dev, 0.0, 0.4)
    m = sol.mesh
    psi, n, p = sol.psi.ravel(), sol.n.ravel()[m.sflat], sol.p.ravel()[m.sflat]
    f, jac = dd._system(m, psi, n, p, sol.tau_n, sol.tau_p)
    rng = np.random.default_rng(1)
    d = rng.standard_normal(f.size) * np.concatenate([np.full(psi.size, 1e-6), 1e-7 * n, 1e-7 * p])
    k, s = psi.size, n.size
    fp, _ = dd._system(m, psi + d[:k], n + d[k:k + s], p + d[k + s:], sol.tau_n, sol.tau_p)
    fm, _ = dd._system(m, psi - d[:k], n - d[k:k + s], p - d[k + s:], sol.tau_n, sol.tau_p)
    jd = jac @ d
    for block in (slice(0, k), slice(k, k + s), slice(k + s, None)):
        assert np.max(np.abs((fp - fm)[block] / 2 - jd[block])) < 1e-6 * np.max(np.abs(jd[block]))


@pytest.mark.parametrize("method", ["newton", "gummel"])
def test_zero_bias_equals_poisson(method):
    """At 0 V, started from a perturbed state (20 mV noise on psi, 30 % log-normal noise on n and p): the solution
    is the equilibrium of poisson.solve, and no current flows."""
    dev, _ = junction(n=400)
    ref = poisson_solve(dev, 0.0, 0.0)
    rng = np.random.default_rng(0)
    g = _Guess(ref.psi + 0.02 * rng.standard_normal(ref.psi.shape),
               ref.n * np.exp(0.3 * rng.standard_normal(ref.psi.shape)),
               ref.p * np.exp(0.3 * rng.standard_normal(ref.psi.shape)))
    sol = dd.solve(dev, 0.0, 0.0, guess=g, method=method)
    assert sol.converged
    assert np.max(np.abs(sol.psi - ref.psi)) < 1e-11
    assert np.max(np.abs(sol.n / ref.n - 1)) < 1e-9 and np.max(np.abs(sol.p / ref.p - 1)) < 1e-9
    assert np.max(np.abs(sol.contact_currents)) < NOISE and np.max(np.abs(sol.x_currents())) < NOISE
    phi_n, phi_p = sol.quasi_fermi()
    assert np.max(np.abs(phi_n)) < 1e-11 and np.max(np.abs(phi_p)) < 1e-11


def test_reverse_bias_matches_poisson():
    """Reverse bias 1 V and 3 V by continuation from 0 V (so not started from the poisson.solve result): psi equals
    poisson.solve, and so do the majority carriers wherever they exceed n_i (the constant quasi-Fermi assumption).
    The minority carriers differ: poisson.solve puts phi_n at the n-contact voltage also in the p region (n there
    is e^(-V/V_T) of the equilibrium value); drift-diffusion gives the equilibrium value far from the junction."""
    dev, _ = junction(n=400)
    nside = dev.n_net > 0
    for v, sol in zip((1.0, 3.0), dd.sweep(dev, [-1.0, -3.0])):
        assert sol.converged
        ref = poisson_solve(dev, 0.0, -v)
        assert np.max(np.abs(sol.psi - ref.psi)) < 1e-12
        kn, kp = nside & (ref.n > dev.n_i), ~nside & (ref.p > dev.n_i)
        assert kn.sum() > 180 and kp.sum() > 180
        assert np.max(np.abs(sol.n[kn] / ref.n[kn] - 1)) < 1e-7
        assert np.max(np.abs(sol.p[kp] / ref.p[kp] - 1)) < 1e-7
        # the first cell, 1.25 nm from the p contact: n_i^2 / N_A less the linear decrease of the minority carriers
        # toward the depletion edge (about 0.45 um away: 0.3 %)
        assert sol.n[0, 0] == pytest.approx(dev.n_i**2 / 1e24, rel=5e-3)
        assert sol.n[0, 0] / ref.n[0, 0] == pytest.approx(np.exp(v / VT), rel=5e-3)
        assert abs(sol.current) < NOISE                     # generation current: far below the noise here


def test_gummel_matches_newton():
    """The two solvers (Gummel with the device CG Poisson step, coupled Newton with a direct LU) at 0.6 V forward
    and 1 V reverse: the same discrete solution."""
    dev, _ = junction(n=400)
    for v_n, v_p in ((0.0, 0.6), (1.0, 0.0)):
        a = dd.solve(dev, v_n, v_p)
        b = dd.solve(dev, v_n, v_p, method="gummel")
        assert a.converged and b.converged
        assert np.max(np.abs(a.psi - b.psi)) < 1e-12
        assert np.max(np.abs(a.n / b.n - 1)) < 1e-9 and np.max(np.abs(a.p / b.p - 1)) < 1e-9
    fwd_a, fwd_b = dd.solve(dev, 0.0, 0.6), dd.solve(dev, 0.0, 0.6, method="gummel")
    assert fwd_b.current == pytest.approx(fwd_a.current, rel=1e-6)
    with pytest.raises(ValueError):
        dd.solve(dev, 0.0, 0.6, method="lu")


@pytest.fixture(scope="module")
def long_base_sweep():
    dev, j0 = long_base()
    biases = [0.3, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    return dev, j0, biases, dd.sweep(dev, biases, tau_n=TAU_LB, tau_p=TAU_LB)


def test_long_base_diode_against_shockley(long_base_sweep):
    """J = J0 (exp(V/V_T) - 1), J0 = q n_i^2 (D_n / (L_n N_A) + D_p / (L_p N_D)) (Sze; the coth of the 6 L base is
    1 + 1.2e-5). The ideal-diode law neglects the recombination in the depletion region (Sah-Noyce-Shockley). That
    current goes as exp(V / 2 V_T), so its part of J falls by exp(-0.05 V / 2 V_T) = 0.38 per 50 mV: measured
    +2.39 %, +0.99 %, +0.41 % at 0.50, 0.55, 0.60 V (factors 0.41 and 0.41). So:
    - 0.60 to 0.75 V: within 1 % (measured 1.0041 to 0.9908; the 0.75 V end has 0.9 % of high injection);
    - 0.55 V: within 1.5 % (measured 1.0099: the expected +0.9 to 1.0 % of depletion recombination; a 1 % tolerance
      would leave a margin of 1e-4);
    - the excess J / J_Shockley - 1 from 0.50 to 0.55 and from 0.55 to 0.60 V falls by 0.3 to 0.5 per step (the
      exp(V / 2 V_T) signature, 0.38);
    - ratio 1.9 at 0.3 V (depletion recombination dominates), 0.94 at 0.8 V (high injection and ohmic drop)."""
    dev, j0, biases, sols = long_base_sweep
    ratio = {v: s.current / dev.y[-1] / (j0 * np.expm1(v / VT)) for v, s in zip(biases, sols)}
    for v in (0.6, 0.65, 0.7, 0.75):
        assert ratio[v] == pytest.approx(1.0, abs=0.01), (v, ratio[v])
    assert ratio[0.55] == pytest.approx(1.0, abs=0.015)
    excess = [ratio[v] - 1 for v in (0.5, 0.55, 0.6)]
    assert all(e > 0 for e in excess)
    assert 0.3 < excess[1] / excess[0] < 0.5 and 0.3 < excess[2] / excess[1] < 0.5
    assert ratio[0.3] > 1.5
    assert ratio[0.8] < 0.97
    for s in sols:
        assert s.converged


def test_current_continuity(long_base_sweep):
    """The total current is the same through every cross-section and equals the contact currents (sum zero), to the
    round-off of the face currents: in the 1 nm cells at the junction the drift and diffusion terms of one face are
    about 1e8 A/m and cancel to the current, so the floor is about 2e-10 A/m (relative 1.5e-3 at 0.55 V, where
    I = 1.3e-7 A/m). The contact faces (cells of up to 50 um) are far more accurate."""
    _, _, biases, sols = long_base_sweep
    for v, s in zip(biases, sols):
        i = s.current
        assert np.max(np.abs(s.x_currents() - i)) < 1e-9
        assert abs(np.sum(s.contact_currents)) < 1e-12
        if v >= 0.65:
            assert np.max(np.abs(s.x_currents() - i)) < 1e-4 * abs(i)


def test_grid_convergence_of_the_diode_current():
    """The long-base current at 0.65 V on three grids (fine cell and growth halved each time): second order (the
    ratio of successive differences is about 4); the default grid (1 nm, 3 % growth) is within 1e-4 of the
    Richardson extrapolation."""
    j = []
    for fine, ratio in ((2e-9, 1.06), (1e-9, 1.03), (0.5e-9, 1.015)):
        dev, _ = long_base(fine, ratio)
        j.append(dd.sweep(dev, [0.65], tau_n=TAU_LB, tau_p=TAU_LB)[0].current)
    d1, d2 = j[1] - j[0], j[2] - j[1]
    assert 3.0 < d1 / d2 < 5.5
    extrapolated = j[2] + d2 / 3
    assert abs(j[1] / extrapolated - 1) < 1e-4


@pytest.fixture(scope="module")
def pin_sweep():
    dev = rib_pin(dx=10e-9, half_span=1.0e-6, clad=0.1e-6)
    return dev, dd.sweep(dev, [0.0, 0.8, 0.9], tau_n=1e-9, tau_p=1e-9)


def test_pin_rib_current_continuity_and_balance(pin_sweep):
    """2-D lateral PIN rib (10 nm cells, contacts 1 um from the center), 0.9 V: the same total current through
    every column, and the terminal current = q x (total recombination) + the minority-carrier currents of the
    contacts (the global balance)."""
    dev, sols = pin_sweep
    s = sols[-1]
    assert s.converged and s.current > 0
    assert np.max(np.abs(s.x_currents() - s.current)) < 1e-9 * s.current
    vol = np.diff(dev.x)[:, None] * np.diff(dev.y)[None, :]
    m = s.mesh
    _, (_, _, _, jnc, jpc) = dd._fluxes(m, s.psi.ravel(), s.n.ravel()[m.sflat], s.p.ravel()[m.sflat])
    # the sum of the electron equations: the electron currents out of both contacts = q x total recombination; so
    # I (out of the n+ contact, 1) = q sum R A + the hole current out of contact 1 - the electron current out of 0
    recombination = Q * np.sum(s.recombination() * vol)
    total = recombination + Q * np.sum(jpc[m.cid == 1]) - Q * np.sum(jnc[m.cid == 0])
    assert total == pytest.approx(s.current, rel=1e-8)
    assert 0.1 * s.current < recombination < s.current


def test_pin_rib_injection(pin_sweep):
    """High injection in the rib: quasi-neutral (p - n = the background acceptors, 1e15 cm^-3, within 5 %), far
    above the background doping, below the limit n_i exp(V / 2 V_T) of the full quasi-Fermi split; the index falls
    and the loss rises with the bias."""
    dev, sols = pin_sweep
    r0 = None
    prev = None
    for v, s in zip((0.8, 0.9), sols[1:]):
        r = rib_index_change(dev, s, sols[0])
        assert r["p_mean"] - r["n_mean"] == pytest.approx(1e21, rel=0.05)
        assert 1e22 < r["n_mean"] < dev.n_i * np.exp(v / (2 * VT))
        assert r["dn_mean"] < 0 and r["dalpha_mean"] > 0
        if prev is not None:
            assert r["dn_mean"] < prev["dn_mean"] and r["dalpha_mean"] > prev["dalpha_mean"]
            assert s.current > r0
        prev, r0 = r, s.current


def test_rib_pin_needs_boundaries_on_cell_edges():
    """With 20 nm cells the 90 nm slab and the 250 nm half width are not whole numbers of cells: ValueError (before
    this check, the cell centers on the boundaries made an 80 nm slab)."""
    with pytest.raises(ValueError, match="not a whole number of cells"):
        rib_pin(dx=20e-9)
    dev = rib_pin(dx=5e-9, half_span=0.6e-6, clad=0.05e-6)
    yc = 0.5 * (dev.y[1:] + dev.y[:-1])
    slab_rows = dev.semiconductor[0, :]                      # the column at xmin: slab only
    assert np.sum(np.diff(dev.y)[slab_rows]) == pytest.approx(0.09e-6, rel=1e-9)
    assert yc[slab_rows].min() > 0


def test_contact_on_a_y_side():
    """A p block with an xmin contact and an n block with a ymax contact (an L-shaped current path), 0.4 V: the
    contact currents sum to zero; x_currents (x faces only) refuses this device."""
    from charge_gpu.poisson import Contact, Device
    x = np.linspace(0, 1e-6, 21)
    y = np.linspace(0, 1e-6, 21)
    xc = 0.5 * (x[1:] + x[:-1])
    n_net = np.where(xc[:, None] < 0.5e-6, -1e23, 1e23) * np.ones((20, 20))
    dev = Device(x=x, y=y, eps_r=np.full((20, 20), 11.7), n_net=n_net, semiconductor=np.ones((20, 20), bool),
                 contacts=[Contact("xmin", np.ones(20, bool), 0.0), Contact("ymax", xc > 0.5e-6, 0.0)])
    sol = dd.solve(dev, 0.0, 0.4)
    assert sol.converged and sol.current > 0
    assert abs(np.sum(sol.contact_currents)) < 1e-7 * sol.current          # round-off: 1.7e-8 measured
    with pytest.raises(ValueError, match="xmin and xmax"):
        sol.x_currents()


def test_sweep_halves_the_step_after_a_solver_exception(monkeypatch):
    """A RuntimeError of the linear solver (for example a singular LU factor) is a failed step: the step is halved
    and the sweep goes on. sweep sets the guess itself: guess= is refused."""
    dev, _ = junction(n=200)
    real = dd.solve
    calls = []

    def flaky(dev_, v_n, v_p, **kw):
        calls.append(v_p)
        if len(calls) == 2:                                  # the first step from 0 V
            raise RuntimeError("Factor is exactly singular")
        return real(dev_, v_n, v_p, **kw)

    monkeypatch.setattr(dd, "solve", flaky)
    sol = dd.sweep(dev, [0.1], step=0.1)[0]
    assert sol.converged and sol.bias == pytest.approx(0.1)
    assert calls[1] == pytest.approx(0.1) and calls[2] == pytest.approx(0.05)
    monkeypatch.setattr(dd, "solve", real)
    with pytest.raises(ValueError, match="guess"):
        dd.sweep(dev, [0.1], guess=sol)
