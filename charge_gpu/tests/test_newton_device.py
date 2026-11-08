"""Checks of the device linear solver of the coupled Newton step (charge_gpu.coupled: method="newton",
linear="device"): the ELL product and one linear solve against SuperLU; the whole solution against the Newton
method with SuperLU (the reference) on the long-base diode (and against Shockley), the 1-D junction in reverse
bias, the PIN phase shifter rib up to high injection and in reverse bias, and the depletion-modulator rib in reverse
bias; the fallback to SuperLU. The measured values are printed (pytest -s)."""
import warnings

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
from test_drift_diffusion import NOISE, TAU_LB, _Guess, long_base  # noqa: E402
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


# the continuation tests: every device solve reaches the FGMRES target RTOL (1e-10; measured at most 9.97e-11), so
# they are checked to 1e-9 (RTOL with a margin), not only to ACCEPT (1e-6): a weaker preconditioner whose solves
# stop at 1e-7 would still give converged Newton solutions (inexact Newton steps), and would pass ACCEPT.
CONTINUATION_RESIDUAL = 1e-9


def _check_device_steps(steps):
    """Every Newton step of every bias was solved on the device (no fallback, no error), to a relative residual of
    at most CONTINUATION_RESIDUAL. Returns the FGMRES steps, the Newton steps and the largest relative residual, and
    prints the FGMRES steps per Newton step."""
    for s in steps:
        assert s.linear == "device" and s.linear_fallbacks == 0 and not s.linear_errors, (s.bias, s.linear_errors)
        assert 0.0 < s.linear_error <= CONTINUATION_RESIDUAL, (s.bias, s.linear_error)
    krylov, newton = sum(s.krylov_iterations for s in steps), sum(s.iterations for s in steps)
    print(f"\n  FGMRES steps per Newton step: {krylov / newton:.1f}", end="")
    return krylov, newton, max(s.linear_error for s in steps)


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
    test_drift_diffusion at 0.4 V: the SuperLU Newton solution (psi 1e-13 V, n and p 1e-11; Newton steps at most 2
    more than SuperLU and at most 20), no fallback; at 3 V the current to 1e-9 A/m (absolute, see below)."""
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
              f"FGMRES {s.krylov_iterations} ({s.krylov_iterations / s.iterations:.1f} per Newton step), "
              f"psi {diff[0]:.1e} V, n {diff[1]:.1e}, p {diff[2]:.1e}, I {diff[3]:.1e}, "
              f"max residual {s.linear_error:.2e}, linear solves {s.timings['linear_s']:.3f} s "
              f"(SuperLU {r.timings['linear_s']:.3f} s)")
        assert s.converged and s.linear_fallbacks == 0 and not s.linear_errors
        assert s.iterations <= min(r.iterations + 2, 20) and s.linear_error <= coupled.ACCEPT
        assert diff[0] < 1e-13 and diff[1] < 1e-11 and diff[2] < 1e-11, diff
        if v_n > 0:
            # 3 V reverse: the current is the generation current of the depletion region, at most
            # q (n_i / 2 tau) W A (SRH with n, p << n_i; W of the depletion approximation, Sze; A = 1 um x 1 m):
            # 8e-11 A/m, below the round-off floor of the contact currents (NOISE, 1e-9 A/m; the majority drift and
            # diffusion terms of a contact face are about 1e5 A/m). So a relative difference of the two currents is
            # noise (some percent in earlier runs); the check is absolute, to NOISE.
            from charge_gpu.poisson import EPS0, Q
            vbi = VT * np.log(1e24 * 1e24 / d.n_i**2)
            w_dep = np.sqrt(2 * 11.7 * EPS0 * (vbi + v_n - 2 * VT) / Q * 2 / 1e24)
            i_gen = Q * d.n_i / (2 * dd.TAU_SI) * w_dep * d.y[-1]
            print(f"  3 V: I = {s.current:.2e} A/m (SuperLU {r.current:.2e} A/m), difference "
                  f"{abs(s.current - r.current):.1e} A/m; generation current at most {i_gen:.1e} A/m")
            assert i_gen < NOISE and abs(s.current) < NOISE and abs(r.current) < NOISE
            assert abs(s.current - r.current) < NOISE


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
    assert g.converged and g.linear_fallbacks == g.iterations and g.linear_error is None and not g.linear_errors
    assert np.max(np.abs(g.psi - ref.psi)) < 1e-12
    monkeypatch.setattr(coupled.Solver, "solve", real)
    monkeypatch.setattr(continuity, "SPAN_MAX", 10.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        g = dd.solve(dev, 0.0, 0.6, linear="device")
    assert g.converged and g.linear_fallbacks == g.iterations and g.krylov_iterations == 0
    assert g.linear_errors == {"SpanError": g.iterations} and g.linear_error is None
    assert not [w for w in caught if "device linear solve failed" in str(w.message)]     # expected: no warning
    assert np.max(np.abs(g.psi - ref.psi)) < 1e-12


# the device errors that fall back to SuperLU, as they come: jax.errors.JaxRuntimeError (a device out of memory,
# RESOURCE_EXHAUSTED; a RuntimeError) and MemoryError (host memory)
DEVICE_ERRORS = [jax.errors.JaxRuntimeError, MemoryError]


def device_error_of(kind):
    def fail(*args, **kw):
        raise kind("RESOURCE_EXHAUSTED: out of memory (test)")
    return fail


@pytest.mark.parametrize("kind", DEVICE_ERRORS, ids=lambda k: k.__name__)
@pytest.mark.parametrize("where", ["solve", "setup"])
def test_a_device_error_falls_back_to_superlu(monkeypatch, where, kind):
    """A device solve that raises jax.errors.JaxRuntimeError (as at a device out of memory) or MemoryError, in the
    solve or in the setup of the solver: that Newton step uses SuperLU, the error is counted by type in
    linear_errors and given as a RuntimeWarning that points to the caller (this file), and the solution is the
    SuperLU solution. sweep() reaches the bias with no step halving (before, the error went to sweep, which halved
    the step down to min_step and then reported "did not converge")."""
    dev, _ = junction(n=200)
    ref = dd.solve(dev, 0.0, 0.6)
    monkeypatch.setattr(coupled.Solver, "solve" if where == "solve" else "__init__", device_error_of(kind))
    with pytest.warns(RuntimeWarning, match=f"device linear solve failed.*{kind.__name__}.*out of memory") as rec:
        g = dd.solve(dev, 0.0, 0.6, linear="device")
    assert all(w.filename == __file__ for w in rec if "device linear solve" in str(w.message))
    assert g.converged and g.linear_fallbacks == g.iterations
    assert g.linear_errors == {kind.__name__: g.iterations}
    assert g.krylov_iterations == 0 and g.linear_error is None
    assert np.max(np.abs(g.psi - ref.psi)) < 1e-12
    calls = []
    real_solve = dd.solve

    def counted(*args, **kw):
        calls.append(args[2])
        return real_solve(*args, **kw)

    monkeypatch.setattr(dd, "solve", counted)
    with pytest.warns(RuntimeWarning, match="device linear solve failed"):
        s = dd.sweep(dev, [0.1], step=0.05, linear="device")[0]
    assert s.converged and s.bias == pytest.approx(0.1) and calls == [0.0, 0.05, 0.1]
    assert s.linear_errors == {kind.__name__: s.iterations}


@pytest.mark.parametrize("error", [TypeError("a programming error (test)"),
                                   ValueError("operands could not be broadcast together (test)"),
                                   RuntimeError("a plain RuntimeError (test)"),
                                   NotImplementedError("not implemented (test)")])
def test_other_errors_of_the_device_solve_propagate(monkeypatch, error):
    """Only continuity.SpanError, jax.errors.JaxRuntimeError and MemoryError fall back to SuperLU. Other errors of
    the device solve (programming errors: a TypeError, a numpy broadcast ValueError, a plain RuntimeError, a
    NotImplementedError) leave dd.solve unchanged, with no fallback and no warning."""
    dev, _ = junction(n=200)

    def fail(*args, **kw):
        raise error

    monkeypatch.setattr(coupled.Solver, "solve", fail)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(type(error), match="test") as raised:
            dd.solve(dev, 0.0, 0.6, linear="device")
    assert not [w for w in caught if "device linear solve failed" in str(w.message)]
    assert raised.value.charge_gpu_device_solve and "not a device error" in " ".join(raised.value.__notes__)


@pytest.mark.parametrize("method", ["newton", "gummel"])
@pytest.mark.parametrize("kind", [RuntimeError, NotImplementedError, np.linalg.LinAlgError], ids=lambda k: k.__name__)
def test_sweep_raises_an_error_of_the_device_solve_at_once(monkeypatch, method, kind):
    """sweep() takes a RuntimeError or a LinAlgError of a solve as a failed bias step (for example a singular LU
    factor of SuperLU) and halves the step. An error of the device linear solve that is not a device error (here: a
    plain RuntimeError, a NotImplementedError, a LinAlgError, raised after the 0 V solve) is a programming error:
    sweep() raises it at once, with its type and text (before, it halved the step down to min_step and raised
    "did not converge" with no cause)."""
    dev, _ = junction(n=200)
    target = coupled.Solver if method == "newton" else continuity.Solver
    real = target.solve
    count = {"calls": 0, "solves": 0}

    def fail_later(self, *args, **kw):
        count["calls"] += 1
        if count["solves"] >= 1:
            raise kind(f"a {kind.__name__} of the device code (test)")
        return real(self, *args, **kw)

    real_solve = dd.solve

    def counted(*args, **kw):
        out = real_solve(*args, **kw)
        count["solves"] += 1
        return out

    monkeypatch.setattr(target, "solve", fail_later)
    monkeypatch.setattr(dd, "solve", counted)
    with pytest.raises(kind, match="of the device code") as raised:
        dd.sweep(dev, [0.1], step=0.05, method=method, linear="device")
    assert type(raised.value) is kind and raised.value.charge_gpu_device_solve
    assert count["solves"] == 1 and count["calls"] > 1    # the 0 V solve only: no step halving


def test_sweep_names_the_cause_of_a_failed_step(monkeypatch):
    """A RuntimeError of a solve that is not from the device solve (as a singular LU factor of SuperLU) at every
    step: sweep() halves the step down to min_step, then raises RuntimeError "did not converge" that names the last
    error and has it as its cause."""
    dev, _ = junction(n=200)
    real_solve = dd.solve
    calls = []

    def singular(dev_, v_n, v_p, **kw):
        calls.append(v_p)
        if len(calls) > 1:
            raise RuntimeError("Factor is exactly singular")
        return real_solve(dev_, v_n, v_p, **kw)

    monkeypatch.setattr(dd, "solve", singular)
    with pytest.raises(RuntimeError, match="did not converge.*last error: RuntimeError: Factor is exactly") as raised:
        dd.sweep(dev, [0.1], step=0.1, min_step=0.03)
    assert isinstance(raised.value.__cause__, RuntimeError) and "singular" in str(raised.value.__cause__)
    assert calls[1:] == pytest.approx([0.1, 0.05])


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
