"""Checks of the transient solver with kappa(T) and rho c_p(T) (transient_kappa): the device operator against
fv3d.build3, the enthalpy against the Kirchhoff integrals of the laws, the constant case against transient3d, a 1-D
Si slab against a solve_ivp reference of the same method-of-lines system (and the order in dt of each scheme), an
exact Kirchhoff solution (constant diffusivity), an independent scipy stepper of the same nonlinear recurrences in
3-D, the long-time limit against the steady kappa(T) solver, the 2-D wrapper, the valid range and the input errors."""
import warnings

import jax
import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from scipy.integrate import solve_ivp

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from thermal_gpu.fv3d import SIDES3, assemble3, build3, full_diag  # noqa: E402
from thermal_gpu.nonlinear import (SI_POWER_LAW, SIO2_CONSTANT, SIO2_LINEAR, Constant, LinearLaw,  # noqa: E402
                                   PowerLaw, material_kappa, solve_steady_kappa2d, solve_steady_kappa3d)
from thermal_gpu.structures import K_METAL, RHO_CP_METAL, RHO_CP_SIO2, heater_stack, material_labels  # noqa: E402
from thermal_gpu.transient3d import geometric_stages, reference_temperature, solve_transient3d  # noqa: E402
from thermal_gpu.transient_kappa import (SI_RHO_CP_LINEAR, _enthalpy_density, _full_diag, _Geo,  # noqa: E402
                                         _operator, _DevLaw, law_table, solve_transient_kappa2d,
                                         solve_transient_kappa3d)

LINE = (np.array([0.0, 1e-6]), np.array([0.0, 1e-6]))    # one cell in x and z (1 um x 1 um)
RHO_CP_SI = 2330.0 * 712.0


def _geo(x, y, z, q, bc):
    """The device geometry of transient_kappa (as solve_transient_kappa3d builds it)."""
    t_ref = reference_temperature(bc)
    d = [np.diff(e) for e in (x, y, z)]
    vol = d[0][:, None, None] * d[1][None, :, None] * d[2][None, None, :]
    sides, bv = [], []
    for i, s in enumerate(SIDES3):
        spec = bc.get(s, ("neumann",))
        if spec[0] == "dirichlet":
            sides.append(i)
            bv.append((0.0, spec[1] - t_ref))
        elif spec[0] == "robin":
            sides.append(i)
            bv.append((1 / spec[1], spec[2] - t_ref))
    return _Geo(tuple(jnp.asarray(np.reshape(d[a], [-1 if b == a else 1 for b in range(3)])) for a in range(3)),
                jnp.asarray(q * vol), jnp.asarray(np.reshape(bv, (-1, 2))), tuple(sides)), t_ref, vol


def test_device_operator_and_enthalpy():
    """The conductances, the boundary terms and the diagonal of the device operator equal fv3d.build3 (random cells
    and kappa, Dirichlet and Robin sides, 1e-15 relative; b in theta: 1e-13, the round-off of the host shift
    b - T_ref G_b). The enthalpy density of each law equals k0 (kirchhoff(T) - kirchhoff(T_ref)) (to 1e-12 of
    rho c_p |T - T_ref|; measured 8.9e-15), also for a small theta (1e-6 K, against the Taylor series; 2.2e-16: no
    cancellation) and for T_ref = 0 (power law: the base t0, an offset per cell; differences 3.0e-14)."""
    rng = np.random.default_rng(3)
    n = (7, 6, 5)
    x, y, z = (np.cumsum(np.r_[0, rng.uniform(0.5, 2.0, m)]) * 1e-6 for m in n)
    kap, q = rng.uniform(1, 150, n), rng.uniform(0, 1e14, n)
    bc = {"xmin": ("dirichlet", 310.0), "xmax": ("robin", 1e5, 290.0), "ymin": ("robin", 3e4, 305.0),
          "zmax": ("dirichlet", 300.0)}
    p = build3(x, y, z, kap, q, bc)
    geo, t_ref, vol = _geo(x, y, z, q, bc)
    g, extra, bb = _operator(geo, jnp.asarray(kap))
    for ax in range(3):
        assert np.max(np.abs(np.asarray(g[ax]) / p.g[ax] - 1)) < 2e-15
    assert np.max(np.abs(np.asarray(extra) - p.diag_extra)) < 1e-15 * p.diag_extra.max()
    b_theta = p.b - t_ref * p.diag_extra
    assert np.max(np.abs(np.asarray(bb) + q * vol - b_theta)) < 1e-13 * np.abs(b_theta).max()
    fd = full_diag(p.g, p.diag_extra)
    assert np.max(np.abs(np.asarray(_full_diag(g, extra)) - fd)) < 1e-15 * fd.max()
    laws = [Constant(2.0), PowerLaw(1.6e6, 1.3), PowerLaw(1.6e6, 1.0), PowerLaw(1.6e6, -0.4, 250.0),
            LinearLaw(1.6e6, 1.1e-3), LinearLaw(1.6e6, -5e-4, 350.0)]
    labels = np.repeat(np.arange(len(laws)), 4)
    tab = law_table(material_kappa(labels, laws), labels.shape)
    dl = _DevLaw(jnp.asarray(tab.labels), *(jnp.asarray(tab.rows[:, i]) for i in range(4)))
    th = rng.uniform(-50, 200, labels.shape)
    ref = np.array([laws[k].k0 * (laws[k].kirchhoff(300.0 + th[i]) - laws[k].kirchhoff(300.0))
                    for i, k in enumerate(labels)])
    h = np.asarray(_enthalpy_density(dl, jnp.asarray(th), 300.0))
    print("H", np.max(np.abs(h - ref) / (tab(300.0 + th) * np.abs(th))))
    assert np.max(np.abs(h - ref) / (tab(300.0 + th) * np.abs(th))) < 1e-12
    small = 1e-6                                                # Taylor: c(300) th + c'(300) th^2 / 2
    c300, dc = tab(np.full(labels.shape, 300.0)), (tab(np.full(labels.shape, 300.001)) -
                                                   tab(np.full(labels.shape, 299.999))) / 0.002
    h = np.asarray(_enthalpy_density(dl, jnp.full(labels.shape, small), 300.0))
    print("H small", np.max(np.abs(h / (c300 * small + 0.5 * dc * small**2) - 1)))
    assert np.max(np.abs(h / (c300 * small + 0.5 * dc * small**2) - 1)) < 1e-12
    t1 = rng.uniform(250.0, 450.0, labels.shape)                # T_ref = 0 (theta = T): only differences count
    h1, h2 = (np.asarray(_enthalpy_density(dl, jnp.asarray(t), 0.0)) for t in (t1, t1 + 3.0))
    ref = np.array([laws[k].k0 * (laws[k].kirchhoff(t1[i] + 3.0) - laws[k].kirchhoff(t1[i]))
                    for i, k in enumerate(labels)])
    print("H diff", np.max(np.abs((h2 - h1) / ref - 1)))
    assert np.max(np.abs((h2 - h1) / ref - 1)) < 1e-10


def _heater_box(rng, n=(14, 12, 10)):
    """A heater block over Si in oxide, non-uniform grid, Robin and Dirichlet sides (synthetic): labels 0 Si,
    1 SiO2, 2 heater metal."""
    x, y, z = (np.cumsum(np.r_[0, rng.uniform(0.02, 0.06, m)]) * 1e-6 for m in n)
    X, Y, Z = np.meshgrid(*[0.5 * (e[1:] + e[:-1]) for e in (x, y, z)], indexing="ij")
    labels = np.where(Y < 0.2e-6, 0, 1)
    heater = (np.abs(X - x.mean()) < 0.1e-6) & (np.abs(Y - 0.45e-6) < 0.05e-6) & (np.abs(Z - z.mean()) < 0.15e-6)
    labels = np.where(heater, 2, labels)
    bc = {"ymin": ("dirichlet", 300.0), "ymax": ("robin", 1e4, 300.0), "xmin": ("robin", 1e5, 305.0),
          "zmax": ("dirichlet", 300.0)}
    return x, y, z, labels, np.where(heater, 2e15, 0.0), bc


K_LAWS = [SI_POWER_LAW, SIO2_LINEAR, Constant(100.0)]
C_LAWS = [SI_RHO_CP_LINEAR, Constant(1.65e6), Constant(2.5e6)]


@pytest.mark.parametrize("scheme", ["euler", "bdf2", "cn"])
def test_constant_laws_equal_the_linear_transient(scheme):
    """Constant laws (and an array rho c_p): one Picard iteration per step, and the result of
    transient3d.solve_transient3d (non-uniform 26 x 22 x 18 grid, three stages with ratios 2 and 0.5) to 1e-10 of
    the rise (measured 1.4e-14 to 3.4e-12; the conductances differ by round-off)."""
    x, y, z, labels, q, bc = _heater_box(np.random.default_rng(4), n=(26, 22, 18))
    kc, rc = np.array([148.0, 1.4, 100.0])[labels], np.array([1.66e6, 1.65e6, 2.5e6])[labels]
    stages = [(2e-9, 3), (4e-9, 4), (2e-9, 2)]
    r0 = solve_transient3d(x, y, z, kc, q, bc, rc, stages, scheme=scheme)
    r1 = solve_transient_kappa3d(x, y, z, material_kappa(labels, [Constant(148.0), Constant(1.4), Constant(100.0)]),
                                 q, bc, rc, stages, scheme=scheme)
    rise = r0.T.max() - 300.0
    assert r0.converged and r1.converged and r1.picard_iterations == 9 and r1.max_step_picard == 1
    assert np.max(np.abs(r1.T - r0.T)) < 1e-10 * rise
    assert r1.max_residual == pytest.approx(r0.max_residual, rel=1e-3)


def _mol(y, kfn, cfn, q, bc, t_init, t_end, method="BDF"):
    """Reference: solve_ivp of the method-of-lines system dT/dt = F(T) / (rho_cp(T) V) of the 1-D line (build3 with
    the numpy laws, in T; rtol 1e-12, atol 1e-10 K)."""
    n = len(y) - 1
    vol = np.diff(y) * 1e-12
    shape = (1, n, 1)

    def f(_, t):
        t3 = t.reshape(shape)
        p = build3(*LINE[:1], y, LINE[1], kfn(t3), q, bc)
        return (p.b.ravel() - assemble3(p) @ t) / (cfn(t3).ravel() * vol)

    jac = sp.diags([np.ones(n - 1), np.ones(n), np.ones(n - 1)], [-1, 0, 1])
    sol = solve_ivp(f, (0.0, t_end), t_init.ravel(), method=method, rtol=1e-12, atol=1e-10, jac_sparsity=jac,
                    t_eval=[t_end])
    assert sol.success
    return sol.y[:, -1]


def _slab(n=100, L=10e-6, amp=100.0):
    """Si slab, T = 300 K on both sides, initial 300 + amp sin(pi y / L); tau = L^2 / (pi^2 alpha(300 K))."""
    y = np.linspace(0, L, n + 1)
    yc = 0.5 * (y[1:] + y[:-1])
    bc = {"ymin": ("dirichlet", 300.0), "ymax": ("dirichlet", 300.0)}
    tau = L**2 / (np.pi**2 * SI_POWER_LAW.k0 / RHO_CP_SI)
    return y, yc, bc, tau, (300.0 + amp * np.sin(np.pi * yc / L))[None, :, None]


def test_si_slab_matches_solve_ivp_and_order_in_dt():
    """Si slab (10 um, 100 cells) with Si kappa(T) (power law) and Si rho c_p(T) (linear law), from 300 + 100
    sin(pi y / L) K, no source, to t = tau (114 ns). Reference: solve_ivp (BDF, rtol 1e-12) of the same
    method-of-lines system; Radau agrees with it to 2.0e-11 of the 100 K amplitude, and constant kappa and rho c_p
    end 7.3 K away. Errors at t = tau (of the 100 K amplitude) with dt = tau / 20, / 40, / 80, measured: euler
    7.4e-3, 3.7e-3, 1.9e-3 (orders 0.99, 0.99); bdf2 3.8e-4, 9.3e-5, 2.3e-5 (2.03, 2.01); cn 3.2e-5, 7.9e-6, 2.0e-6
    (2.00, 2.00). From dt = tau / 10, cn is not yet in its asymptotic range (order 2.73 to tau / 20)."""
    y, yc, bc, tau, t_init = _slab()
    q = np.zeros(t_init.shape)
    ref = _mol(y, SI_POWER_LAW, SI_RHO_CP_LINEAR, q, bc, t_init, tau)
    ref2 = _mol(y, SI_POWER_LAW, SI_RHO_CP_LINEAR, q, bc, t_init, tau, "Radau")
    assert np.max(np.abs(ref - ref2)) < 1e-10 * 100.0
    lin = _mol(y, Constant(148.0), Constant(RHO_CP_SI), q, bc, t_init, tau)
    print("references", np.max(np.abs(ref - ref2)) / 100.0, np.max(np.abs(lin - ref)), ref.max())
    assert np.max(np.abs(lin - ref)) > 0.05 * 100.0                     # kappa(T) matters here
    out = {}
    for scheme in ("euler", "bdf2", "cn"):
        errs = []
        for n in (20, 40, 80):
            r = solve_transient_kappa3d(*LINE[:1], y, LINE[1], SI_POWER_LAW, q, bc, SI_RHO_CP_LINEAR, tau / n,
                                        t_end=tau, scheme=scheme, t0_field=t_init)
            assert r.converged and r.out_of_range == 0 and r.max_residual <= 1e-10
            errs.append(np.max(np.abs(r.T.ravel() - ref)) / 100.0)
        out[scheme] = (errs, np.log2(np.array(errs[:-1]) / np.array(errs[1:])))
    print({k: (np.round(v[0], 8).tolist(), np.round(v[1], 3).tolist()) for k, v in out.items()})
    for scheme, order in (("euler", 1), ("bdf2", 2), ("cn", 2)):
        assert np.all(np.abs(out[scheme][1] - order) < 0.1), (scheme, out[scheme])
    assert out["bdf2"][0][-1] < 1e-3 and out["cn"][0][-1] < 1e-3


def test_si_slab_step_response_matches_solve_ivp():
    """The same slab from 300 K with a uniform source from t = 0 (steady T_max 373 K; at t = tau the rise is
    39.9 K) to t = tau, bdf2 with dt = tau / 20, / 40, / 80 against solve_ivp: errors 7.2e-4, 1.8e-4, 4.4e-5 of the
    rise, orders 2.02 and 2.01 (measured)."""
    y, yc, bc, tau, _ = _slab()
    t_init = np.full((1, len(yc), 1), 300.0)
    q = np.full(t_init.shape, 7.5e14)
    ref = _mol(y, SI_POWER_LAW, SI_RHO_CP_LINEAR, q, bc, t_init, tau)
    rise = ref.max() - 300.0
    errs = []
    for n in (20, 40, 80):
        r = solve_transient_kappa3d(*LINE[:1], y, LINE[1], SI_POWER_LAW, q, bc, SI_RHO_CP_LINEAR, tau / n, t_end=tau,
                                    scheme="bdf2")
        assert r.converged and r.out_of_range == 0
        errs.append(np.max(np.abs(r.T.ravel() - ref)) / rise)
    orders = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    print("step response", rise, errs, orders)
    assert np.all(np.abs(orders - 2) < 0.15), (errs, orders)


def test_kirchhoff_exact_solution_with_constant_diffusivity():
    """kappa = 148 (300 / T)^1.3 and rho c_p = (148 / alpha) (300 / T)^1.3 (alpha constant): the Kirchhoff variable
    u = 300 + 300 ((T / 300)^-0.3 - 1) / -0.3 solves u_t = alpha u_yy exactly, so u = 300 + U0 sin(pi y / L)
    exp(-t / tau) with U0 = 114.5 K (T_max 450 K). Error of T at t = tau (space and time, of 150 K) from (100 cells,
    tau / 40) to (200 cells, tau / 80), measured: bdf2 1.34e-4 and 3.33e-5 (ratio 4.02), cn 2.68e-5 and 6.69e-6
    (ratio 4.00)."""
    law = PowerLaw(148.0, 1.3)
    alpha = 148.0 / RHO_CP_SI
    cap = PowerLaw(148.0 / alpha, 1.3)
    L, u0 = 10e-6, law.kirchhoff(450.0) - 300.0
    tau = L**2 / (np.pi**2 * alpha)
    for scheme in ("bdf2", "cn"):
        errs = []
        for n, steps in ((100, 40), (200, 80)):
            y = np.linspace(0, L, n + 1)
            yc = 0.5 * (y[1:] + y[:-1])
            bc = {"ymin": ("dirichlet", 300.0), "ymax": ("dirichlet", 300.0)}
            exact = lambda t: law.inverse(300.0 + u0 * np.sin(np.pi * yc / L) * np.exp(-t / tau))  # noqa: E731
            r = solve_transient_kappa3d(*LINE[:1], y, LINE[1], law, np.zeros((1, n, 1)), bc, cap, tau / steps,
                                        t_end=tau, scheme=scheme, t0_field=exact(0.0)[None, :, None])
            assert r.converged
            errs.append(np.max(np.abs(r.T.ravel() - exact(tau))) / 150.0)
        print("kirchhoff", scheme, errs, errs[0] / errs[1])
        assert errs[0] < 1e-3 and errs[0] / errs[1] == pytest.approx(4.0, rel=0.05)


def _scipy_kappa_march(x, y, z, labels, q, bc, stages, scheme, t_init):
    """An independent stepper: the same nonlinear recurrences in T (not shifted), H from the Kirchhoff integrals of
    the laws, each step solved by a Picard iteration with scipy's sparse direct solver to 1e-10 K."""
    kfn, cfn = material_kappa(labels, K_LAWS), material_kappa(labels, C_LAWS)
    vol = np.einsum("i,j,k->ijk", *[np.diff(e) for e in (x, y, z)])

    def enthalpy(t):
        out = np.empty(t.shape)
        for i, law in enumerate(C_LAWS):
            m = labels == i
            out[m] = law.k0 * (law.kirchhoff(t[m]) - law.kirchhoff(300.0)) * vol[m]
        return out

    def flux(t):
        p = build3(x, y, z, kfn(t), q, bc)
        return p, p.b - (assemble3(p) @ t.ravel()).reshape(t.shape)

    t1 = t_init.copy()
    t0 = t1.copy()
    dt_prev, out = None, []
    for dt, n in stages:
        for _ in range(n):
            if scheme == "euler" or (scheme == "bdf2" and dt_prev is None):
                d, hist = 1 / dt, enthalpy(t1) / dt
            elif scheme == "bdf2":
                w = dt / dt_prev
                d = (1 + 2 * w) / ((1 + w) * dt)
                hist = ((1 + w) * enthalpy(t1) - w * w / (1 + w) * enthalpy(t0)) / dt
            else:
                d, hist = 2 / dt, 2 * enthalpy(t1) / dt + flux(t1)[1]
            t = t1.copy()
            for _ in range(100):
                p, _ = flux(t)
                c = cfn(t) * vol
                a = assemble3(p) + sp.diags(d * c.ravel())
                t_new = spl.spsolve(a.tocsc(), (hist + p.b - d * (enthalpy(t) - c * t)).ravel()).reshape(t.shape)
                change = np.max(np.abs(t_new - t))
                t = t_new
                if change < 1e-10:
                    break
            assert change < 1e-10
            t0, t1, dt_prev = t1, t, dt
            out.append(t)
    return out


@pytest.mark.parametrize("scheme", ["euler", "bdf2", "cn"])
def test_3d_heater_matches_an_independent_scipy_stepper(scheme):
    """Heater box (non-uniform 26 x 22 x 18 grid, 3 multigrid levels), Si and SiO2 kappa(T), Si rho c_p(T), the
    heater to 414 to 417 K; stages with ratios 2 and 0.5: the multigrid-CG Picard steps (V and W cycle) equal the
    scipy stepper to 1e-9 of the rise (each step's monitor and the final field; measured 9.5e-12 to 2.2e-11).
    kappa(T) changes the peak by more than 0.5 % of the rise against constant kappa. Picard: 5 to 6 iterations
    per step (48 or 49 for 9 steps; stop rule 1e-10); CG iterations 362 to 407 (V), 320 to 346 (W)."""
    x, y, z, labels, q, bc = _heater_box(np.random.default_rng(4), n=(26, 22, 18))
    q = 30 * q
    stages = [(2e-9, 3), (4e-9, 4), (2e-9, 2)]
    ref = _scipy_kappa_march(x, y, z, labels, q, bc, stages, scheme, np.full(labels.shape, 300.0))
    rise = ref[-1].max() - 300.0
    lin = solve_transient3d(x, y, z, np.array([148.0, 1.4, 100.0])[labels], q, bc, np.array(
        [RHO_CP_SI, 1.65e6, 2.5e6])[labels], stages, scheme=scheme)
    assert abs(lin.T.max() - ref[-1].max()) > 0.005 * rise
    mon = np.zeros(labels.shape, bool)
    mon[10:16, :, 9] = True
    vol = np.einsum("i,j,k->ijk", *[np.diff(e) for e in (x, y, z)])
    means = [np.sum(t[mon] * vol[mon]) / vol[mon].sum() for t in ref]
    for cycle in ("V", "W"):
        r = solve_transient_kappa3d(x, y, z, material_kappa(labels, K_LAWS), q, bc, material_kappa(labels, C_LAWS),
                                    stages, scheme=scheme, monitors=[mon], cycle=cycle)
        assert r.converged and len(r.times) == 10 and r.hierarchies == 3 and r.max_step_picard <= 8
        assert np.max(np.abs(r.T - ref[-1])) < 1e-9 * rise, cycle
        assert np.max(np.abs(r.monitors[1:, 0] - means)) < 1e-9 * rise
        print(scheme, cycle, np.max(np.abs(r.T - ref[-1])) / rise, r.picard_iterations, r.iterations)


def test_long_time_limit_and_fixed_point_equal_the_steady_kappa_solution():
    """Heater box (14 x 12 x 10), kappa(T) and rho c_p(T), the heater about 100 K above 300 K: from 300 K with steps
    growing to 30 us (2000 times the slowest time constant), euler and bdf2 end at solve_steady_kappa3d to 1e-9 of
    the rise (measured 1.4e-11 and 1.7e-11). From the steady kappa(T) field each scheme stays there (1e-9)."""
    x, y, z, labels, q, bc = _heater_box(np.random.default_rng(6))
    kf, cf, q = material_kappa(labels, K_LAWS), material_kappa(labels, C_LAWS), 20 * q
    st = solve_steady_kappa3d(x, y, z, kf, q, bc)
    rise = st.T.max() - 300.0
    assert st.converged and rise > 50.0
    stages = geometric_stages(1e-10, 30e-6, per_stage=8, factor=2.0)
    for scheme in ("euler", "bdf2"):
        r = solve_transient_kappa3d(x, y, z, kf, q, bc, cf, stages, scheme=scheme)
        assert r.converged and np.max(np.abs(r.T - st.T)) < 1e-9 * rise, scheme
        print("limit", scheme, np.max(np.abs(r.T - st.T)) / rise, r.picard_iterations, len(r.times) - 1)
    for scheme in ("euler", "bdf2", "cn"):
        r = solve_transient_kappa3d(x, y, z, kf, q, bc, cf, 1e-9, t_end=5e-9, scheme=scheme, t0_field=st.T)
        assert r.converged and np.max(np.abs(r.T - st.T)) < 1e-9 * rise, scheme


def test_2d_heater_stack_ends_at_the_steady_kappa_solution():
    """The 2-D heater stack (100 W/m, 100 nm grid) with Si and SiO2 kappa(T): the bdf2 transient (geometric stages
    to 200 us, 200 steps) ends at solve_steady_kappa2d (waveguide mean: 5.1e-10 K). Waveguide rise 20.098 K against
    20.287 K with constant kappa; rise time 10 % to 90 % 17.606 us against 17.939 us (-1.9 %). Picard: 768
    iterations for 200 steps (at most 5 per step); CG 4529 against 2072 for constant kappa. Si rho c_p(T) in
    addition changes the rise time by 8.9e-5 relative."""
    from thermal_gpu.structures import heat_capacity
    from thermal_gpu.transient3d import rise_time, solve_transient2d
    x, y, k, q, bc, masks = heater_stack(dx=0.1e-6, power_per_length=100.0)
    labels, wg = material_labels(k), masks["waveguide"]
    kf = material_kappa(labels, [SI_POWER_LAW, SIO2_LINEAR, Constant(K_METAL)])
    st = solve_steady_kappa2d(x, y, kf, q, bc)
    final = st.T[wg].mean()
    stages = geometric_stages(1e-8, 200e-6, per_stage=20)
    r = solve_transient_kappa2d(x, y, kf, q, bc, heat_capacity(k), stages, scheme="bdf2", monitors=[wg])
    assert r.converged and r.T.shape == k.shape
    assert r.monitors[-1, 0] == pytest.approx(final, abs=1e-6)
    print("2-D end", r.monitors[-1, 0] - final)
    r0 = solve_transient2d(x, y, k, q, bc, heat_capacity(k), stages, scheme="bdf2", monitors=[wg])
    from thermal_gpu.solver import solve_steady
    f0 = solve_steady(x, y, k, q, bc).T[wg].mean()
    rt, rt0 = rise_time(r.times, r.monitors[:, 0], final)[2], rise_time(r0.times, r0.monitors[:, 0], f0)[2]
    print("2-D rise", final - 300.0, f0 - 300.0, rt, rt0, r.picard_iterations, r.iterations, r0.iterations)
    assert final - 300.0 == pytest.approx(20.098, abs=1e-3) and f0 - 300.0 == pytest.approx(20.287, abs=1e-3)
    assert rt == pytest.approx(17.606e-6, rel=1e-4) and rt0 == pytest.approx(17.939e-6, rel=1e-4)
    assert r.max_step_picard <= 6 and r.picard_iterations <= 4 * (len(r.times) - 1)
    cf = material_kappa(labels, [SI_RHO_CP_LINEAR, Constant(RHO_CP_SIO2), Constant(RHO_CP_METAL)])
    r2 = solve_transient_kappa2d(x, y, kf, q, bc, cf, stages, scheme="bdf2", monitors=[wg])
    assert r2.converged and r2.out_of_range == 0
    assert r2.monitors[-1, 0] == pytest.approx(final, abs=1e-6)
    rt2 = rise_time(r2.times, r2.monitors[:, 0], final)[2]
    print("2-D rise with c_p(T)", rt2, rt2 / rt - 1)
    assert abs(rt2 / rt - 1) < 3e-4


def test_range_warning_flags_and_input_errors():
    """Si slab from 300 + 520 sin(pi y / L) K (820 K, above the 800 K of the Si law) to t = tau: a RuntimeWarning
    and out_of_range > 0 although the final field is inside the range. max_picard = 1 on a nonlinear case:
    converged False. kappa <= 0 at the start: no step, converged False. Input errors: ValueError or TypeError."""
    y, yc, bc, tau, t_init = _slab(amp=520.0)
    q = np.zeros(t_init.shape)
    with pytest.warns(RuntimeWarning, match="leave the valid temperature range"):
        r = solve_transient_kappa3d(*LINE[:1], y, LINE[1], SI_POWER_LAW, q, bc, RHO_CP_SI, tau / 10, t_end=tau,
                                    scheme="bdf2", t0_field=t_init)
    assert r.converged and r.out_of_range > 0 and r.T.max() < 800.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = solve_transient_kappa3d(*LINE[:1], y, LINE[1], SI_POWER_LAW, q, bc, RHO_CP_SI, tau / 10, t_end=tau,
                                    t0_field=_slab(amp=100.0)[4], max_picard=1)
    assert not r.converged and r.max_step_picard == 1
    r = solve_transient_kappa3d(*LINE[:1], y, LINE[1], LinearLaw(1.0, -0.01), q, bc, RHO_CP_SI, tau / 10, t_end=tau,
                                t0_field=np.full(t_init.shape, 450.0))
    assert not r.converged and len(r.times) == 1
    common = (*LINE[:1], y, LINE[1])
    bad = [({"kappa_fn": lambda t: 148.0 + 0 * t}, TypeError), ({"rho_cp": -RHO_CP_SI}, ValueError),
           ({"scheme": "rk4"}, ValueError), ({"cycle": "F"}, ValueError), ({"max_picard": 0}, ValueError),
           ({"dt": [(1e-9, 2), (3e-9, 2)], "t_end": None, "scheme": "bdf2"}, ValueError),
           ({"kappa_fn": material_kappa(np.zeros(7, int), [SI_POWER_LAW])}, ValueError)]
    for kw, err in bad:
        args = {"kappa_fn": SI_POWER_LAW, "q": q, "bc": bc, "rho_cp": RHO_CP_SI, "dt": tau / 10, "t_end": tau}
        args.update(kw)
        with pytest.raises(err):
            solve_transient_kappa3d(*common, **args)
    assert law_table(SIO2_CONSTANT, (2, 3, 1)).constant and not law_table(SI_POWER_LAW, (2, 3, 1)).constant


def test_loose_stop_rule_and_tol():
    """The Si slab (bdf2, dt = tau / 20, Si kappa(T) and rho c_p(T)). Defaults: converged, at most 8 Picard
    iterations in a step (measured 7). tol_picard = 1e-4: the stop rule is met in each step (fewer than max_picard
    iterations), but the residual (measured 1.9e-6) is above tol = 1e-10, so converged is False; with tol = 1e-5 the
    same run converges (residual 3.4e-6: the CG target tol / 2 is then looser too)."""
    y, yc, bc, tau, t_init = _slab()
    q = np.zeros(t_init.shape)
    run = lambda **kw: solve_transient_kappa3d(*LINE[:1], y, LINE[1], SI_POWER_LAW, q, bc, SI_RHO_CP_LINEAR,  # noqa
                                               tau / 20, t_end=tau, scheme="bdf2", t0_field=t_init, **kw)
    r = run()
    assert r.converged and r.max_step_picard <= 8
    loose = run(tol_picard=1e-4)
    strict_tol = run(tol_picard=1e-4, tol=1e-5)
    print("picard", r.max_step_picard, r.picard_iterations, loose.max_residual, loose.max_step_picard,
          strict_tol.max_residual)
    assert not loose.converged and loose.max_step_picard < 50 and 1e-10 < loose.max_residual < 1e-5
    assert strict_tol.converged


def test_2d_labels_must_have_the_grid_layout():
    """A 5 x 4 cross-section: material_kappa labels of shape (nx, ny) = (5, 4) are accepted; labels of shape (4, 5)
    (np.meshgrid with its default indexing "xy") and arrays of shape (4, 5) raise ValueError (no reshape)."""
    x, y = np.linspace(0, 5e-6, 6), np.linspace(0, 4e-6, 5)
    bc = {"ymin": ("dirichlet", 300.0)}
    q = np.full((5, 4), 1e14)
    yc = 0.5 * (y[1:] + y[:-1])
    good = np.broadcast_to((yc > 2e-6).astype(int)[None, :], (5, 4))
    r = solve_transient_kappa2d(x, y, material_kappa(good, [SI_POWER_LAW, SIO2_LINEAR]), q, bc, RHO_CP_SI, 1e-9,
                                t_end=3e-9)
    assert r.converged and r.T.shape == (5, 4)
    _, yy = np.meshgrid(0.5 * (x[1:] + x[:-1]), yc)                # indexing "xy": shape (ny, nx) = (4, 5)
    with pytest.raises(ValueError):
        solve_transient_kappa2d(x, y, material_kappa((yy > 2e-6).astype(int), [SI_POWER_LAW, SIO2_LINEAR]), q, bc,
                                RHO_CP_SI, 1e-9, t_end=3e-9)
    with pytest.raises(ValueError):
        law_table(material_kappa(np.zeros((4, 5), int), [SI_POWER_LAW]), (5, 4, 1))
    with pytest.raises(ValueError):
        law_table(np.ones((4, 5)), (5, 4, 1))
    with pytest.raises(ValueError):
        law_table(material_kappa(np.zeros((20, 3), int), [SI_POWER_LAW]), (4, 5, 3))
    assert law_table(material_kappa(good, [SI_POWER_LAW, SIO2_LINEAR]), (5, 4, 1)).labels[:, :, 0].tolist() == \
        good.tolist()
