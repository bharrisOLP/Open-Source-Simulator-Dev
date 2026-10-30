"""Checks of the device linear solver of the coupled Newton step (charge_gpu.coupled: method="newton",
linear="device"): the ELL product and one linear solve against SuperLU; the whole solution against the Newton
method with SuperLU (the reference) on the long-base diode (and against Shockley), the 1-D junction in reverse
bias, the PIN phase shifter rib up to high injection and in reverse bias, and the depletion-modulator rib in reverse
bias; the fallback to SuperLU. The measured values are printed (pytest -s)."""
import jax
import numpy as np
import pytest
import scipy.sparse.linalg as spl

jax.config.update("jax_enable_x64", True)

from charge_gpu import continuity, coupled  # noqa: E402
from charge_gpu import drift_diffusion as dd  # noqa: E402
from charge_gpu.injection import rib_index_change, rib_pin  # noqa: E402
from charge_gpu.modulator import rib_pn  # noqa: E402
from charge_gpu.poisson import thermal_voltage  # noqa: E402
from test_drift_diffusion import TAU_LB, _Guess, long_base  # noqa: E402
from test_junction import junction  # noqa: E402

VT = thermal_voltage()
TAU_PIN = 1e-9


def _diff(sol, ref):
    """psi (V); n and p (relative, where the reference exceeds the floor of the metric, 1 cm^-3); the contact
    currents (relative to the largest contact current of the reference)."""
    m = ref.mesh
    out = [float(np.max(np.abs(sol.psi - ref.psi)))]
    for a, b in ((sol.n, ref.n), (sol.p, ref.p)):
        a, b = a.ravel()[m.sflat], b.ravel()[m.sflat]
        k = b > dd.CARRIER_FLOOR
        out.append(float(np.max(np.abs(a[k] / b[k] - 1))))
    big = max(float(np.max(np.abs(ref.contact_currents))), 1e-30)
    out.append(float(np.max(np.abs(sol.contact_currents - ref.contact_currents)) / big))
    return tuple(out)


def _continuation(dev, biases, step=0.05, **kw):
    """Solutions at the biases (in order) by continuation from 0 V in steps of at most ``step`` (as dd.sweep, but
    without step halving: a failed step fails the test). Returns the solutions at the biases and all steps."""
    cur = dd.solve(dev, 0.0, 0.0, **kw)
    steps, out, v = [cur], [], 0.0
    for target in biases:
        while abs(target - v) > 1e-12:
            v = v + float(np.clip(target - v, -step, step))
            v = target if abs(target - v) < 1e-9 else v
            cur = dd.solve(dev, 0.0, v, guess=cur, **kw)
            assert cur.converged, (v, cur.error)
            steps.append(cur)
        out.append(cur)
    return out, steps


def _check_device_steps(steps):
    """Every Newton step of every bias was solved on the device (no fallback), to a relative residual of at most
    ACCEPT. Returns the FGMRES steps, the Newton steps and the largest relative residual."""
    for s in steps:
        assert s.linear == "device" and s.linear_fallbacks == 0, (s.bias, s.linear_fallbacks)
        assert 0.0 <= s.linear_error <= coupled.ACCEPT
    return (sum(s.krylov_iterations for s in steps), sum(s.iterations for s in steps),
            max(s.linear_error for s in steps))


@pytest.fixture(scope="module")
def pin():
    """The PIN rib of test_dd_device (10 nm cells, contacts 1 um from the center, tau 1 ns)."""
    return rib_pin(dx=10e-9, half_span=1.0e-6, clad=0.1e-6)


def test_ell_product_and_one_solve_equal_superlu(pin):
    """The first Newton step from the 0.95 V solution to 1.0 V (high injection) on the PIN rib: the ELL product of
    the device equals the scipy product of J_s (1e-14 of |J_s| |v|), and the FGMRES solution equals the SuperLU
    solution of the same scaled system; the true relative residual is below RTOL. With 2 steps the flag is
    False."""
    kw = dict(tau_n=TAU_PIN, tau_p=TAU_PIN)
    start = dd.sweep(pin, [0.95], **kw)[0]
    m = dd._mesh(pin, 0.0, 1.0)
    psi, n, p = start.psi.ravel(), start.n.ravel()[m.sflat], start.p.ravel()[m.sflat]
    js, b, rs = dd._scaled_system(m, psi, n, p, TAU_PIN, TAU_PIN)
    solver = coupled.Solver(m)
    o = solver.operators(js, rs, psi, n, p, TAU_PIN, TAU_PIN)
    v = np.random.default_rng(5).standard_normal(b.size)
    prod = np.asarray(coupled._matvec(o, jax.device_put(v, solver.device)))
    assert np.max(np.abs(prod - js @ v) / (abs(js) @ np.abs(v))) < 1e-14
    ref = spl.splu(js.tocsc()).solve(b)
    res = solver.solve(js, b, rs, psi, n, p, TAU_PIN, TAU_PIN)
    err = float(np.max(np.abs(res.y - ref)) / np.max(np.abs(ref)))
    print(f"\none solve at 1.0 V: {res.iterations} FGMRES steps, residual {res.residual:.2e}, "
          f"max |y - y_superlu| / max |y_superlu| = {err:.2e}, setup {res.setup_s:.3f} s, krylov {res.krylov_s:.3f} s")
    assert res.converged and res.residual <= coupled.RTOL
    assert err < 1e-8
    bad = solver.solve(js, b, rs, psi, n, p, TAU_PIN, TAU_PIN, max_iter=2)
    assert not bad.converged and bad.residual > coupled.ACCEPT


def test_pin_forward_to_high_injection_equals_superlu(pin):
    """The PIN rib, continuation 0 to 1.0 V in 50 mV steps with the device Newton (high injection at 0.9 and 1.0 V:
    n and p in the rib about 2e17 and 6e17 cm^-3): no fallback to SuperLU in any Newton step, and at 0.6, 0.9 and
    1.0 V the solution of the SuperLU Newton (psi 1e-11 V, n and p 1e-8, contact currents 1e-8), the same total
    current through every column (1e-9 of I) and the same rib index change (1e-8)."""
    kw = dict(tau_n=TAU_PIN, tau_p=TAU_PIN)
    biases = [0.6, 0.9, 1.0]
    dev_sols, steps = _continuation(pin, biases, linear="device", **kw)
    ref_sols = dd.sweep(pin, biases, **kw)
    zero = dd.solve(pin, 0.0, 0.0, **kw)
    krylov, newton, worst = _check_device_steps(steps)
    print(f"\nPIN 0 to 1.0 V: {len(steps)} solves, {newton} Newton, {krylov} FGMRES steps, max residual {worst:.3e}")
    for v, s, ref in zip(biases, dev_sols, ref_sols):
        d = _diff(s, ref)
        r_s, r_ref = rib_index_change(pin, s, zero), rib_index_change(pin, ref, zero)
        d_index = abs(r_s["dn_mean"] / r_ref["dn_mean"] - 1)
        print(f"  {v} V: Newton steps {s.iterations} (SuperLU {ref.iterations}), FGMRES {s.krylov_iterations}, "
              f"psi {d[0]:.1e} V, n {d[1]:.1e}, p {d[2]:.1e}, I {d[3]:.1e}, index {d_index:.1e}, "
              f"I = {s.current:.6g} A/m")
        assert d[0] < 1e-11 and d[1] < 1e-8 and d[2] < 1e-8 and d[3] < 1e-8, d
        assert np.max(np.abs(s.x_currents() - s.current)) < 1e-9 * s.current
        assert d_index < 1e-8


@pytest.mark.parametrize("device", ["pin", "pn"])
def test_ribs_in_reverse_bias_equal_superlu(pin, device):
    """Reverse bias, continuation in 0.5 V steps: the PIN rib to 2 V and the depletion-modulator rib of
    modulator.rib_pn (3e17 cm^-3, 10 nm cells, contacts 1 um from the center) to 4 V. No fallback; the solution of
    the SuperLU Newton (psi 1e-11 V, n and p 1e-8). The currents are generation currents near the round-off floor
    of the face currents: checked to 1e-9 A/m."""
    dev = pin if device == "pin" else rib_pn(half_span=1.0e-6, clad=0.1e-6)
    kw = dict(tau_n=TAU_PIN, tau_p=TAU_PIN)
    biases = [-2.0] if device == "pin" else [-1.0, -4.0]
    dev_sols, steps = _continuation(dev, biases, step=0.5, linear="device", **kw)
    ref_sols = dd.sweep(dev, biases, step=0.5, **kw)
    krylov, newton, worst = _check_device_steps(steps)
    print(f"\n{device} reverse: {len(steps)} solves, {newton} Newton, {krylov} FGMRES steps, max residual {worst:.3e}")
    for v, s, ref in zip(biases, dev_sols, ref_sols):
        d = _diff(s, ref)
        print(f"  {v} V: psi {d[0]:.1e} V, n {d[1]:.1e}, p {d[2]:.1e}, I {s.current:.3e} and {ref.current:.3e} A/m")
        assert d[0] < 1e-11 and d[1] < 1e-8 and d[2] < 1e-8, d
        assert abs(s.current - ref.current) < 1e-9


def test_junction_in_reverse_bias_equals_superlu():
    """The 1-D junction of test_junction (1e18 cm^-3, 2.5 nm cells) at 1, 5 and 25 V reverse (up to 12 V_T per cell
    in the depletion region), continuation in 0.5 V steps: no fallback; the solution of the SuperLU Newton (psi
    1e-12 V, n and p 1e-9)."""
    dev, _ = junction(n=400)
    biases = [-1.0, -5.0, -25.0]
    dev_sols, steps = _continuation(dev, biases, step=0.5, linear="device")
    ref_sols = dd.sweep(dev, biases, step=0.5)
    krylov, newton, worst = _check_device_steps(steps)
    print(f"\njunction reverse: {len(steps)} solves, {newton} Newton, {krylov} FGMRES steps, max residual {worst:.3e}")
    for v, s, ref in zip(biases, dev_sols, ref_sols):
        d = _diff(s, ref)
        print(f"  {v} V: psi {d[0]:.1e} V, n {d[1]:.1e}, p {d[2]:.1e}")
        assert d[0] < 1e-12 and d[1] < 1e-9 and d[2] < 1e-9, d


def test_long_base_diode_against_shockley_and_superlu():
    """The long-base diode of the Shockley check (1e18 cm^-3, tau 100 us, contacts 6 diffusion lengths away, cells
    from 1 nm to 50 um), continuation 0 to 0.8 V in 50 mV steps with the device Newton: no fallback; J / J_Shockley
    within 1 % from 0.6 to 0.75 V (as the SuperLU test; at 0.3 V and 0.8 V the known deviations of that test); at
    every bias the current of the SuperLU Newton (1e-8, or the face-current floor 1e-9 A/m), the contact currents
    sum to zero (1e-12 A/m) and every cross-section carries the current (1e-9 A/m)."""
    dev, j0 = long_base()
    biases = [0.3, 0.6, 0.65, 0.7, 0.75, 0.8]
    kw = dict(tau_n=TAU_LB, tau_p=TAU_LB)
    dev_sols, steps = _continuation(dev, biases, linear="device", **kw)
    ref_sols = dd.sweep(dev, biases, **kw)
    krylov, newton, worst = _check_device_steps(steps)
    print(f"\nlong-base diode 0 to 0.8 V: {len(steps)} solves, {newton} Newton, {krylov} FGMRES steps, "
          f"max residual {worst:.3e}")
    for v, s, ref in zip(biases, dev_sols, ref_sols):
        ratio = s.current / dev.y[-1] / (j0 * np.expm1(v / VT))
        d = _diff(s, ref)
        print(f"  {v} V: J / J_Shockley {ratio:.6f} (SuperLU "
              f"{ref.current / dev.y[-1] / (j0 * np.expm1(v / VT)):.6f}), psi {d[0]:.1e} V, n {d[1]:.1e}, "
              f"p {d[2]:.1e}, dI {abs(s.current - ref.current):.1e} A/m ({abs(s.current / ref.current - 1):.1e})")
        if 0.55 < v < 0.78:
            assert ratio == pytest.approx(1.0, abs=0.01)
        assert abs(s.current - ref.current) < max(1e-8 * abs(ref.current), 1e-9)
        assert d[0] < 1e-11 and d[1] < 1e-8 and d[2] < 1e-8, d
        assert abs(np.sum(s.contact_currents)) < 1e-12
        assert np.max(np.abs(s.x_currents() - s.current)) < 1e-9


def test_far_starts_without_continuation():
    """The device Newton from far starts, with no continuation: 0 V from the perturbed state of test_drift_diffusion
    (20 mV on psi, 30 % on n and p) gives poisson.solve; the 1-D junction at 0.6 V forward and 3 V reverse from the
    poisson.solve guess, and the block with an xmin and a ymax contact (an L-shaped current path) of
    test_drift_diffusion at 0.4 V: the SuperLU Newton solution (at most 2 more Newton steps), no fallback."""
    from charge_gpu.poisson import Contact, Device, solve as poisson_solve

    dev, _ = junction(n=400)
    ref = poisson_solve(dev, 0.0, 0.0)
    rng = np.random.default_rng(0)
    g = _Guess(ref.psi + 0.02 * rng.standard_normal(ref.psi.shape),
               ref.n * np.exp(0.3 * rng.standard_normal(ref.psi.shape)),
               ref.p * np.exp(0.3 * rng.standard_normal(ref.psi.shape)))
    s = dd.solve(dev, 0.0, 0.0, guess=g, linear="device")
    assert s.converged and s.linear_fallbacks == 0
    assert np.max(np.abs(s.psi - ref.psi)) < 1e-11
    assert np.max(np.abs(s.n / ref.n - 1)) < 1e-9 and np.max(np.abs(s.p / ref.p - 1)) < 1e-9
    x = np.linspace(0, 1e-6, 21)
    xc = 0.5 * (x[1:] + x[:-1])
    block = Device(x=x, y=x.copy(), eps_r=np.full((20, 20), 11.7),
                   n_net=np.where(xc[:, None] < 0.5e-6, -1e23, 1e23) * np.ones((20, 20)),
                   semiconductor=np.ones((20, 20), bool),
                   contacts=[Contact("xmin", np.ones(20, bool), 0.0), Contact("ymax", xc > 0.5e-6, 0.0)])
    for d, v_n, v_p in ((dev, 0.0, 0.6), (dev, 3.0, 0.0), (block, 0.0, 0.4)):
        s = dd.solve(d, v_n, v_p, linear="device")
        r = dd.solve(d, v_n, v_p)
        diff = _diff(s, r)
        print(f"\nfar start {v_p - v_n} V ({d.n_net.shape}): Newton steps {s.iterations} (SuperLU {r.iterations}), "
              f"FGMRES {s.krylov_iterations}, psi {diff[0]:.1e} V, n {diff[1]:.1e}, p {diff[2]:.1e}, "
              f"I {diff[3]:.1e}, max residual {s.linear_error:.2e}, linear solves {s.timings['linear_s']:.3f} s "
              f"(SuperLU {r.timings['linear_s']:.3f} s)")
        assert s.converged and s.linear_fallbacks == 0 and s.iterations <= r.iterations + 2
        assert diff[0] < 1e-12 and diff[1] < 1e-9 and diff[2] < 1e-9, diff


def test_a_device_solve_that_fails_is_not_used(monkeypatch):
    """A device solve that is not converged (here: a garbage result flagged converged = False) is replaced by
    SuperLU for that Newton step: the solution is the SuperLU solution, and every step counts as a fallback. Also
    with SPAN_MAX below the psi span: continuity.SpanError, and the steps use SuperLU."""
    dev, _ = junction(n=200)
    ref = dd.solve(dev, 0.0, 0.6)
    real = coupled.Solver.solve

    def bad(self, js, b, rs, *args, **kw):
        res = real(self, js, b, rs, *args, **kw)
        return coupled.LinearResult(y=res.y * 1e3, iterations=res.iterations, residual=0.5, converged=False,
                                    setup_s=res.setup_s, krylov_s=res.krylov_s)

    monkeypatch.setattr(coupled.Solver, "solve", bad)
    g = dd.solve(dev, 0.0, 0.6, linear="device")
    assert g.converged and g.linear_fallbacks == g.iterations and g.linear_error == 0.0
    assert np.max(np.abs(g.psi - ref.psi)) < 1e-12
    monkeypatch.setattr(coupled.Solver, "solve", real)
    monkeypatch.setattr(continuity, "SPAN_MAX", 10.0)
    g = dd.solve(dev, 0.0, 0.6, linear="device")
    assert g.converged and g.linear_fallbacks == g.iterations and g.krylov_iterations == 0
    assert np.max(np.abs(g.psi - ref.psi)) < 1e-12


def test_solution_info():
    """linear, device, linear_error and the timings of the device Newton; SuperLU keeps linear_error None."""
    dev, _ = junction(n=200)
    s = dd.solve(dev, 0.0, 0.6, linear="device")
    r = dd.solve(dev, 0.0, 0.6)
    assert s.converged and s.linear == "device" and "FGMRES" in s.device and s.krylov_iterations > 0
    assert 0.0 < s.linear_error <= coupled.ACCEPT and s.linear_fallbacks == 0
    assert set(s.timings) == {"assembly_s", "linear_s", "setup_s", "krylov_s"}
    assert r.linear_error is None and r.krylov_iterations == 0 and set(r.timings) == {"assembly_s", "linear_s"}
    with pytest.raises(ValueError, match="linear"):
        dd.solve(dev, 0.0, 0.6, linear="cg")
