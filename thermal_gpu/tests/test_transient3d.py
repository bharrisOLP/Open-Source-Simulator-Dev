"""Checks of the 3-D transient solver (transient3d): the decay of a 3-D box mode (exact discrete recurrence of each
scheme and the continuous time constant), the order of each scheme in time, an independent scipy time stepper of the
same discrete system, the steady limit, the 2-D solver, and the rise time of the 3-D heater against its 2-D
cross-section."""
import jax
import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spl

jax.config.update("jax_enable_x64", True)

from thermal_gpu.fv3d import assemble3, build3  # noqa: E402
from thermal_gpu.solver import solve_steady, solve_transient  # noqa: E402
from thermal_gpu.solver3d import solve_steady3d  # noqa: E402
from thermal_gpu.transient3d import (geometric_stages, rise_time, solve_transient2d,  # noqa: E402
                                     solve_transient3d, stages_from)

SIDES = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")


def _box(n=16, L=(2e-6, 3e-6, 4e-6), k=1.4, rho_cp=1.6e6):
    """A box with T = 0 on all sides and the first mode sin(pi x / Lx) sin(pi y / Ly) sin(pi z / Lz) (uniform grid).
    The sampled sine is an exact eigenvector of the discrete operator (the Dirichlet face at half a cell acts as the
    antisymmetric ghost cell), with lambda_h = alpha sum 4 / h^2 sin^2(pi h / (2 L)); continuous: lambda =
    alpha pi^2 sum 1 / L^2."""
    L = np.asarray(L)
    edges = [np.linspace(0, ln, n + 1) for ln in L]
    X, Y, Z = np.meshgrid(*[0.5 * (e[1:] + e[:-1]) for e in edges], indexing="ij")
    mode = np.sin(np.pi * X / L[0]) * np.sin(np.pi * Y / L[1]) * np.sin(np.pi * Z / L[2])
    alpha, h = k / rho_cp, L / n
    lam_h = alpha * np.sum(4 / h**2 * np.sin(np.pi * h / (2 * L)) ** 2)
    lam = alpha * np.pi**2 * np.sum(1 / L**2)
    return edges, np.full(mode.shape, k), rho_cp, mode, {s: ("dirichlet", 0.0) for s in SIDES}, lam_h, lam


def _recurrence(scheme, z, n):
    """Amplitude after n steps of the scalar problem y' = -lambda y, z = dt lambda (first BDF2 step: euler)."""
    if scheme == "euler":
        return (1 + z) ** -n
    if scheme == "cn":
        return ((1 - z / 2) / (1 + z / 2)) ** n
    y0, y1 = 1.0, 1 / (1 + z)
    for _ in range(n - 1):
        y0, y1 = y1, (2 * y1 - 0.5 * y0) / (1.5 + z)
    return y1


@pytest.mark.parametrize("scheme", ["euler", "bdf2", "cn"])
def test_box_mode_decay(scheme):
    """tol = 1e-12. The field stays the mode (to 1e-11), its amplitude is the exact recurrence of the scheme with
    lambda_h (to 1e-10), and the decay time constant from the amplitude at t = tau is the continuous one: measured
    1.6 % (euler, dt = tau / 40; 1.3 % from the time step, 0.3 % from the grid), 0.35 % (bdf2) and 0.32 % (cn; the
    grid: lambda_h / lambda = 0.99679 at 16 cells per side). Measured: shape 1e-12 to 9e-12, recurrence 3e-13 to
    5e-12."""
    edges, kappa, rho_cp, mode, bc, lam_h, lam = _box()
    tau, n = 1 / lam, 40
    r = solve_transient3d(*edges, kappa, np.zeros(mode.shape), bc, rho_cp, tau / n, t_end=tau, scheme=scheme,
                          monitors=[mode > 0.5, (8, 8, 8)], t0_field=mode, tol=1e-12)
    assert r.converged and r.max_residual <= 1e-12 and len(r.times) == n + 1
    ratio = r.T / mode
    assert np.std(ratio) / np.mean(ratio) < 1e-11
    assert np.mean(ratio) == pytest.approx(_recurrence(scheme, tau / n * lam_h, n), rel=1e-10)
    assert r.monitors[-1, 1] / r.monitors[0, 1] == pytest.approx(np.mean(ratio), rel=1e-10)
    tau_fit = r.times[-1] / -np.log(np.mean(ratio))
    assert tau_fit == pytest.approx(tau, rel={"euler": 0.02, "bdf2": 0.004, "cn": 0.004}[scheme])
    assert lam_h / lam == pytest.approx(0.99679, abs=1e-5)


@pytest.mark.parametrize("scheme,order", [("euler", 1), ("bdf2", 2), ("cn", 2)])
def test_order_in_time(scheme, order):
    """Error of the amplitude at t = tau against the exact semi-discrete decay exp(-lambda_h t), at dt = tau / 10,
    tau / 20, tau / 40: measured orders 0.97 and 0.99 (euler), 2.07 and 2.03 (bdf2), 2.00 and 2.00 (cn)."""
    edges, kappa, rho_cp, mode, bc, lam_h, lam = _box()
    tau, errs = 1 / lam, []
    for n in (10, 20, 40):
        r = solve_transient3d(*edges, kappa, np.zeros(mode.shape), bc, rho_cp, tau / n, t_end=tau, scheme=scheme,
                              t0_field=mode, tol=1e-12)
        assert r.converged
        errs.append(abs(np.mean(r.T / mode) - np.exp(-lam_h * tau)))
    orders = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    assert np.all(np.abs(orders - order) < 0.1), orders


def _heater_box(rng, n=(14, 12, 10)):
    """A heater block and a waveguide in oxide on Si, non-uniform grid, Robin top and mixed sides (synthetic)."""
    x, y, z = (np.cumsum(np.r_[0, rng.uniform(0.02, 0.06, m)]) * 1e-6 for m in n)
    X, Y, Z = np.meshgrid(*[0.5 * (e[1:] + e[:-1]) for e in (x, y, z)], indexing="ij")
    kappa = np.where(Y < 0.2e-6, 148.0, 1.4)
    heater = (np.abs(X - x.mean()) < 0.1e-6) & (np.abs(Y - 0.45e-6) < 0.05e-6) & (np.abs(Z - z.mean()) < 0.15e-6)
    kappa = np.where(heater, 100.0, kappa)
    rho_cp = np.where(Y < 0.2e-6, 1.66e6, np.where(heater, 2.5e6, 1.65e6))
    q = np.where(heater, 2e15, 0.0)
    bc = {"ymin": ("dirichlet", 300.0), "ymax": ("robin", 1e4, 300.0), "xmin": ("robin", 1e5, 305.0),
          "zmax": ("dirichlet", 300.0)}
    return x, y, z, kappa, q, bc, rho_cp


def _scipy_march(x, y, z, kappa, q, bc, rho_cp, stages, scheme, t0):
    """An independent time stepper: the same recurrences in T (not shifted), with scipy's sparse direct solver."""
    p = build3(x, y, z, kappa, q, bc)
    A = assemble3(p).tocsc()
    vol = np.einsum("i,j,k->ijk", *[np.diff(e) for e in (x, y, z)])
    M = sp.diags((rho_cp * vol).ravel())
    b = p.b.ravel()
    t1 = t0.ravel().copy()
    t0_ = t1.copy()
    dt_prev, out = None, []
    for dt, n in stages:
        for s in range(n):
            if scheme == "euler" or (scheme == "bdf2" and dt_prev is None):
                t2 = spl.spsolve(M / dt + A, M @ t1 / dt + b)
            elif scheme == "bdf2":
                w = dt / dt_prev
                t2 = spl.spsolve((1 + 2 * w) / ((1 + w) * dt) * M + A,
                                 M @ ((1 + w) * t1 - w * w / (1 + w) * t0_) / dt + b)
            else:
                t2 = spl.spsolve(M / dt + A / 2, M @ t1 / dt - A @ t1 / 2 + b)
            t0_, t1, dt_prev = t1, t2, dt
            out.append(t2.reshape(p.shape))
    return out


@pytest.mark.parametrize("scheme", ["euler", "bdf2", "cn"])
def test_matches_an_independent_scipy_time_stepper(scheme):
    """Non-uniform grid, contrast 148 / 1.4, three heat capacities, Robin and Dirichlet sides, three stages with
    dt ratios 2 and 0.5 (the variable-step BDF2 and the first steps of the stages), 10296 cells (3 multigrid levels,
    so V and W differ): the multigrid-CG steps equal scipy direct solves of the same recurrences to 1e-8 of the
    rise."""
    x, y, z, kappa, q, bc, rho_cp = _heater_box(np.random.default_rng(4), n=(26, 22, 18))
    stages = [(2e-9, 3), (4e-9, 4), (2e-9, 2)]
    t0 = np.full(kappa.shape, 300.0)
    ref = _scipy_march(x, y, z, kappa, q, bc, rho_cp, stages, scheme, t0)
    rise = ref[-1].max() - 300.0
    assert rise > 0.1
    zc = 0.5 * (z[1:] + z[:-1])
    mon = np.zeros(kappa.shape, bool)
    mon[10:16, :, np.argmin(np.abs(zc - z.mean()))] = True
    for cycle in ("V", "W"):
        r = solve_transient3d(x, y, z, kappa, q, bc, rho_cp, stages, scheme=scheme, monitors=[mon], cycle=cycle)
        assert r.converged and len(r.times) == 10
        assert r.times[-1] == pytest.approx(2e-9 * 3 + 4e-9 * 4 + 2e-9 * 2)
        assert np.max(np.abs(r.T - ref[-1])) < 1e-8 * rise, cycle
        vol = np.einsum("i,j,k->ijk", *[np.diff(e) for e in (x, y, z)])
        means = [np.sum(t[mon] * vol[mon]) / vol[mon].sum() for t in ref]
        assert np.max(np.abs(r.monitors[1:, 0] - means)) < 1e-8 * rise


def test_steady_limit_and_fixed_point():
    """From 300 K with growing steps up to 30 us (2000 times the slowest time constant of this box, about 15 ns):
    euler and bdf2 end at solve_steady3d (to 1e-9 of the rise; measured 3.6e-10). From the steady field, each scheme
    stays there (measured: 2.1e-10 of the rise after 5 steps)."""
    x, y, z, kappa, q, bc, rho_cp = _heater_box(np.random.default_rng(6))
    st = solve_steady3d(x, y, z, kappa, q, bc)
    rise = st.T.max() - 300.0
    stages = geometric_stages(1e-10, 30e-6, per_stage=8, factor=2.0)
    for scheme in ("euler", "bdf2"):
        r = solve_transient3d(x, y, z, kappa, q, bc, rho_cp, stages, scheme=scheme)
        assert r.converged and np.max(np.abs(r.T - st.T)) < 1e-9 * rise, scheme
    for scheme in ("euler", "bdf2", "cn"):
        r = solve_transient3d(x, y, z, kappa, q, bc, rho_cp, 1e-9, t_end=5e-9, scheme=scheme, t0_field=st.T)
        assert r.converged and np.max(np.abs(r.T - st.T)) < 1e-9 * rise, scheme


def test_2d_wrapper_equals_the_2d_solver():
    """solve_transient2d (one z cell, 3-D multigrid) = solver.solve_transient (the 2-D code, implicit Euler) on a
    heater cross-section with a non-uniform grid: 1e-8 of the rise. The 2-D code solves for T (not T - T_ref), and its
    tolerance 1e-12 is relative to (C / dt) T with T near 300 K: its error is the larger one."""
    rng = np.random.default_rng(7)
    x, y = (np.cumsum(np.r_[0, rng.uniform(0.02, 0.06, m)]) * 1e-6 for m in (40, 30))
    X, Y = np.meshgrid(0.5 * (x[1:] + x[:-1]), 0.5 * (y[1:] + y[:-1]), indexing="ij")
    kappa = np.where(Y < 0.4e-6, 148.0, 1.4)
    heater = (np.abs(X - x.mean()) < 0.2e-6) & (np.abs(Y - 1.0e-6) < 0.05e-6)
    kappa = np.where(heater, 100.0, kappa)
    q = np.where(heater, 1e15, 0.0)
    rho_cp = np.where(heater, 2.5e6, 1.65e6)
    bc = {"ymin": ("dirichlet", 300.0), "ymax": ("robin", 10.0, 300.0)}
    r2 = solve_transient(x, y, kappa, q, bc, rho_cp, t_end=200e-9, dt=20e-9, probes=[(20, 25)])
    r3 = solve_transient2d(x, y, kappa, q, bc, rho_cp, 20e-9, t_end=200e-9, monitors=[(20, 25)])
    rise = r2.T.max() - 300.0
    assert r2.converged and r3.converged and r3.T.shape == kappa.shape
    assert np.max(np.abs(r3.T - r2.T)) < 1e-8 * rise
    assert np.max(np.abs(r3.monitors[:, 0] - r2.probes[:, 0])) < 1e-8 * rise


def test_rise_time_helper_and_input_errors():
    t = np.linspace(0, 10, 100001)
    t10, t90, rt = rise_time(t, 300 + 5 * (1 - np.exp(-t)), 305.0)
    assert t10 == pytest.approx(np.log(1 / 0.9), abs=1e-7) and rt == pytest.approx(np.log(9), abs=1e-7)
    assert stages_from(1e-9, 1e-8) == [(1e-9, 10)]
    st = geometric_stages(1e-9, 1e-6, per_stage=5)
    assert sum(d * n for d, n in st) >= 1e-6 and st[1][0] == 2e-9
    x, y, z, kappa, q, bc, rho_cp = _heater_box(np.random.default_rng(1), n=(6, 5, 4))
    for kw in ({"scheme": "rk4"}, {"cycle": "F"}):
        with pytest.raises(ValueError):
            solve_transient3d(x, y, z, kappa, q, bc, rho_cp, 1e-9, t_end=1e-8, **kw)
    bad = [((1e-9, 1e-8), {}), (1e-9, {}), ([(1e-9, 2), (3e-9, 2)], {"scheme": "bdf2"}), ([(-1e-9, 2)], {})]
    for dt, kw in bad:                                   # tuple dt with t_end, no t_end, ratio 3 > 2.41, dt < 0
        with pytest.raises(ValueError):
            solve_transient3d(x, y, z, kappa, q, bc, rho_cp, dt, t_end=1e-8 if dt == (1e-9, 1e-8) else None, **kw)
    with pytest.raises(ValueError):
        solve_transient3d(x, y, z, kappa, q, bc, -rho_cp, 1e-9, t_end=1e-8)
    with pytest.raises(ValueError):
        solve_transient3d(x, y, z, kappa, q, bc, rho_cp, 1e-9, t_end=1e-8, monitors=[np.zeros((2, 2, 2), bool)])
    with pytest.raises(ValueError):
        solve_transient3d(x, y, z, kappa, q, bc, rho_cp, 1e-9, t_end=1e-8, monitors=[np.zeros(kappa.shape, bool)])
    x, y, z, kappa, q, bc, rho_cp = _heater_box(np.random.default_rng(1))     # 1680 cells: two multigrid levels
    r = solve_transient3d(x, y, z, kappa, q, bc, rho_cp, 1e-6, t_end=2e-6, max_iter=1)
    assert not r.converged and r.max_residual > 1e-10


def test_heater_3d_rise_time_at_mid_length_equals_the_2d_cross_section():
    """The 3-D heater (100 um, 10 mW; a narrow 6 um window and a coarse z grid to keep the test short) switches on
    at t = 0: at mid-length the 10 % to 90 % rise time of the waveguide is that of the 2-D cross-section (100 W/m)
    to 1e-3 (measured 9e-5), and the transient ends at the steady rise."""
    from thermal_gpu.structures import heat_capacity, heater_3d, heater_stack
    x, y, z, k, q, bc, m = heater_3d(dx=0.1e-6, width=6e-6, length=100e-6, margin=10e-6, dz_end=0.5e-6,
                                     dz_max=4e-6, power=10e-3)
    zc = 0.5 * (z[1:] + z[:-1])
    mid = m["waveguide"] & (np.abs(zc) <= np.abs(zc).min() + 1e-15)[None, None, :]
    final = solve_steady3d(x, y, z, k, q, bc).T[mid].mean()
    stages = geometric_stages(2e-8, 100e-6, per_stage=10)
    r = solve_transient3d(x, y, z, k, q, bc, heat_capacity(k), stages, scheme="bdf2", monitors=[mid])
    assert r.converged and r.monitors[-1, 0] == pytest.approx(final, abs=1e-4)
    x2, y2, k2, q2, bc2, m2 = heater_stack(dx=0.1e-6, width=6e-6, power_per_length=100.0)
    f2 = solve_steady(x2, y2, k2, q2, bc2).T[m2["waveguide"]].mean()
    r2 = solve_transient2d(x2, y2, k2, q2, bc2, heat_capacity(k2), stages, scheme="bdf2", monitors=[m2["waveguide"]])
    rt3 = rise_time(r.times, r.monitors[:, 0], final)[2]
    rt2 = rise_time(r2.times, r2.monitors[:, 0], f2)[2]
    assert r2.converged and 10e-6 < rt2 < 30e-6
    assert rt3 == pytest.approx(rt2, rel=1e-3)
