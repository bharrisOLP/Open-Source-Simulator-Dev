"""Checks of the steady solve with kappa(T) (nonlinear): exact 1-D solutions from the Kirchhoff transform (one Si
layer with heating; Si and SiO2 in series), the second order of the grid error, an independent scipy Picard
iteration in 3-D, thermal runaway, the material laws against published tables, and the effect on the 2-D heater."""
import warnings

import jax
import numpy as np
import pytest
import scipy.sparse.linalg as spl
from scipy.optimize import brentq

jax.config.update("jax_enable_x64", True)

from thermal_gpu.fv3d import assemble3, build3  # noqa: E402
from thermal_gpu.nonlinear import (SI_POWER_LAW, SIO2_CONSTANT, SIO2_LINEAR, Constant, LinearLaw,  # noqa: E402
                                   PowerLaw, material_kappa, solve_steady_kappa2d, solve_steady_kappa3d)
from thermal_gpu.solver import solve_steady  # noqa: E402
from thermal_gpu.structures import K_METAL, K_SI, heater_stack, material_labels  # noqa: E402

LINE = (np.array([0.0, 1e-6]), np.array([0.0, 1e-6]))    # one cell in x and z


def test_laws_kirchhoff_and_published_values():
    """kirchhoff' = kappa / kappa(t0), inverse(kirchhoff(T)) = T; the Si power law against Incropera Table A.1
    (98.9 W/m/K at 400 K, 61.9 at 600 K: within 3 %); the SiO2 slope against Table A.2, fused silica (1.38 -> 1.51
    W/m/K from 300 to 400 K; the ratio at 600 K, 1.75 / 1.38, within 2 %); the valid ranges."""
    t = np.linspace(250.0, 700.0, 46)
    for law in (SI_POWER_LAW, PowerLaw(10.0, 1.0), SIO2_LINEAR, LinearLaw(2.0, -5e-4), SIO2_CONSTANT):
        k0 = law(np.array(getattr(law, "t0", 300.0)))
        h = 1e-4
        deriv = (law.kirchhoff(t + h) - law.kirchhoff(t - h)) / (2 * h)
        assert np.allclose(deriv, law(t) / k0, rtol=1e-8)
        assert np.allclose(law.inverse(law.kirchhoff(t)), t, rtol=1e-13)
    assert SI_POWER_LAW(np.array(400.0)) == pytest.approx(98.9, rel=0.03)
    assert SI_POWER_LAW(np.array(600.0)) == pytest.approx(61.9, rel=0.03)
    assert SIO2_LINEAR(np.array(400.0)) / SIO2_LINEAR(np.array(300.0)) == pytest.approx(1.51 / 1.38, rel=1e-12)
    assert SIO2_LINEAR(np.array(600.0)) / SIO2_LINEAR(np.array(300.0)) == pytest.approx(1.75 / 1.38, rel=0.02)
    assert np.isnan(SI_POWER_LAW.inverse(np.array(300.0 + 300.0 / 0.3 + 1.0)))      # above the runaway limit
    assert SI_POWER_LAW.out_of_range(np.array([300.0, 550.0, 800.0])) == 0
    assert SI_POWER_LAW.out_of_range(np.array([299.0, 801.0, 500.0])) == 2
    assert SIO2_LINEAR.out_of_range(np.array([601.0])) == 1 and SIO2_CONSTANT.out_of_range(np.array([5e3])) == 0
    mk = material_kappa(np.array([0, 1, 0]), [SI_POWER_LAW, SIO2_LINEAR])
    assert mk.out_of_range(np.array([900.0, 500.0, 400.0])) == 1                 # Si at 900 K
    assert mk.out_of_range(np.array([400.0, 700.0, 900.0])) == 2                 # SiO2 at 700 K, Si at 900 K


def test_si_slab_with_heating_matches_the_kirchhoff_solution():
    """-(kappa(T) T')' = q in Si (power law), T = 300 K on both sides, L = 10 um, T_max = 450 K: theta = 300 +
    q y (L - y) / (2 kappa0), T = inverse(theta). Error at the cell centers 2.86e-4 and 7.15e-5 of the rise at 100
    and 200 cells (order 2: the cell value of kappa is a midpoint rule of the half-cell resistances)."""
    L, q0, errs = 10e-6, 1.356e15, []
    for n in (100, 200):
        y = np.linspace(0, L, n + 1)
        yc = 0.5 * (y[1:] + y[:-1])
        bc = {"ymin": ("dirichlet", 300.0), "ymax": ("dirichlet", 300.0)}
        r = solve_steady_kappa3d(LINE[0], y, LINE[1], SI_POWER_LAW, np.full((1, n, 1), q0), bc)
        exact = SI_POWER_LAW.inverse(300.0 + q0 * yc * (L - yc) / (2 * SI_POWER_LAW.k0))
        assert r.converged and r.residual <= 1e-10 and r.picard_iterations <= 12
        errs.append(np.max(np.abs(r.T[0, :, 0] - exact)) / (exact.max() - 300.0))
    assert exact.max() == pytest.approx(450.0, abs=0.1)
    assert errs[0] < 3e-4 and errs[0] / errs[1] == pytest.approx(4.0, rel=0.02)


def test_si_and_oxide_in_series_match_the_kirchhoff_solution():
    """100 um Si (power law) and 1 um SiO2 (linear law), T = 300 K at the Si side and 500 K at the oxide side, no
    source. Exact: the same flux F in both layers, F = kappa0 (theta(T_i) - theta(T_a)) / t in each layer, solved for
    the interface temperature T_i (414.75 K); theta is linear in each layer. Error 4.2e-6 and 1.06e-6 of the 200 K
    step at (100 + 50) and (200 + 100) cells (order 2)."""
    tsi, tox, si, ox = 100e-6, 1e-6, SI_POWER_LAW, SIO2_LINEAR

    def flux_mismatch(ti):
        return si.k0 * (si.kirchhoff(ti) - si.kirchhoff(300.0)) / tsi - ox.k0 * (ox.kirchhoff(500.0) -
                                                                              ox.kirchhoff(ti)) / tox
    ti = brentq(flux_mismatch, 300.0, 500.0, xtol=1e-13)
    flux = si.k0 * (si.kirchhoff(ti) - si.kirchhoff(300.0)) / tsi
    errs = []
    for nsi, nox in ((100, 50), (200, 100)):
        y = np.concatenate([np.linspace(0, tsi, nsi + 1), np.linspace(tsi, tsi + tox, nox + 1)[1:]])
        yc = 0.5 * (y[1:] + y[:-1])
        labels = (yc > tsi).astype(int)[None, :, None]
        bc = {"ymin": ("dirichlet", 300.0), "ymax": ("dirichlet", 500.0)}
        r = solve_steady_kappa3d(LINE[0], y, LINE[1], material_kappa(labels, [si, ox]), np.zeros(labels.shape), bc)
        exact = np.where(yc < tsi, si.inverse(si.kirchhoff(300.0) + flux * yc / si.k0),
                         ox.inverse(ox.kirchhoff(ti) + flux * (yc - tsi) / ox.k0))
        assert r.converged
        errs.append(np.max(np.abs(r.T[0, :, 0] - exact)) / 200.0)
    assert ti == pytest.approx(414.75, abs=0.01)
    assert errs[0] < 1e-5 and errs[0] / errs[1] == pytest.approx(4.0, rel=0.05)


def _scipy_picard(x, y, z, kappa_fn, q, bc, n_iter=60):
    """An independent Picard iteration (T, not shifted) with scipy's sparse direct solver."""
    t = np.full(q.shape, 300.0)
    for _ in range(n_iter):
        p = build3(x, y, z, kappa_fn(t), q, bc)
        t_new = spl.spsolve(assemble3(p).tocsc(), p.b.ravel()).reshape(p.shape)
        if np.max(np.abs(t_new - t)) < 1e-12 * (t_new.max() - 300.0):
            return t_new
        t = t_new
    return t


def test_3d_heater_box_matches_an_independent_scipy_picard_iteration():
    """A heater block over Si in oxide, non-uniform 26 x 22 x 18 grid, Robin bottom, top and side, a heating that
    raises the Si to 409 to 410 K: the multigrid-CG Picard result (with the reused hierarchy and with a new one at
    each iteration) equals a scipy direct Picard iteration to 1e-8 of the rise (measured 1.6e-12 and 3.1e-12);
    kappa(T) changes the maximum rise by 1.6 % (124.5 K against 126.5 K with constant kappa; the Robin bottom sets
    most of the thermal resistance)."""
    rng = np.random.default_rng(2)
    x, y, z = (np.cumsum(np.r_[0, rng.uniform(0.02, 0.06, m)]) * 1e-6 for m in (26, 22, 18))
    X, Y, Z = np.meshgrid(*[0.5 * (e[1:] + e[:-1]) for e in (x, y, z)], indexing="ij")
    labels = np.where(Y < 0.3e-6, 0, 1)
    heater = (np.abs(X - x.mean()) < 0.2e-6) & (np.abs(Y - 0.6e-6) < 0.05e-6) & (np.abs(Z - z.mean()) < 0.3e-6)
    labels = np.where(heater, 2, labels)
    laws = [SI_POWER_LAW, SIO2_LINEAR, Constant(100.0)]
    kf = material_kappa(labels, laws)
    q = np.where(heater, 3.5e15, 0.0)
    bc = {"ymin": ("robin", 1e5, 300.0), "ymax": ("robin", 1e5, 300.0), "xmax": ("robin", 1e6, 300.0)}
    ref = _scipy_picard(x, y, z, kf, q, bc)
    rise = ref.max() - 300.0
    lin = _scipy_picard(x, y, z, material_kappa(labels, [Constant(148.0), Constant(1.4), Constant(100.0)]), q, bc, 1)
    assert ref[labels == 0].min() > 400.0 and abs(lin.max() - 300.0 - rise) > 0.01 * rise
    for reuse in (True, False):
        r = solve_steady_kappa3d(x, y, z, kf, q, bc, reuse_mg=reuse)
        assert r.converged and r.residual <= 1e-10
        assert np.max(np.abs(r.T - ref)) < 1e-8 * rise, reuse


def test_runaway_and_flags():
    """Si power law (n = 1.3) with a heating above the runaway limit (theta - T0 = T0 / (n - 1) = 1000 K at the
    center; here 10 times more): no steady solution, so converged is False. max_picard = 1 on a nonlinear case:
    also False."""
    y = np.linspace(0, 10e-6, 101)
    bc = {"ymin": ("dirichlet", 300.0), "ymax": ("dirichlet", 300.0)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # the runaway field is far outside 300 to 800 K
        r = solve_steady_kappa3d(LINE[0], y, LINE[1], SI_POWER_LAW, np.full((1, 100, 1), 1.356e16), bc,
                                 max_picard=60)
    assert not r.converged
    r = solve_steady_kappa3d(LINE[0], y, LINE[1], SI_POWER_LAW, np.full((1, 100, 1), 1.356e15), bc, max_picard=1)
    assert not r.converged and r.picard_iterations == 1
    with pytest.raises(ValueError):
        solve_steady_kappa3d(LINE[0], y, LINE[1], SI_POWER_LAW, np.zeros((1, 100, 1)), bc, cycle="F")
    with pytest.raises(ValueError):
        material_kappa(np.array([0, 1, 3]), [SI_POWER_LAW, SIO2_CONSTANT])


def test_range_warning_and_hierarchy_rebuild():
    """Si slab (200 cells), T = 300 K on both sides. Heated to 765 K (inside 300 to 800 K): no warning; the same
    field with a new hierarchy at each iteration, with the automatic rebuild (default) and with the first hierarchy
    only (rebuild_factor = inf), to 1e-9 of the rise. CG iterations (14 Picard iterations): 14 (a new hierarchy each
    time: here one level, an exact solve), 47 (automatic, 3 hierarchies), 143 (never). Heated to 1186 K: converged
    (the discrete problem), but a RuntimeWarning and out_of_range > 0 (16, 65 and 245 CG iterations)."""
    n, L = 200, 10e-6
    y = np.linspace(0, L, n + 1)
    bc = {"ymin": ("dirichlet", 300.0), "ymax": ("dirichlet", 300.0)}
    q = np.full((1, n, 1), 2.9e15)
    runs = {}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for name, kw in (("new", {"reuse_mg": False}), ("auto", {}), ("never", {"rebuild_factor": np.inf})):
            runs[name] = solve_steady_kappa3d(LINE[0], y, LINE[1], SI_POWER_LAW, q, bc, **kw)
    rise = runs["new"].T.max() - 300.0
    assert 700.0 < runs["new"].T.max() < 800.0
    for name, r in runs.items():
        assert r.converged and r.out_of_range == 0, name
        assert np.max(np.abs(r.T - runs["new"].T)) < 1e-9 * rise, name
    assert runs["never"].hierarchies == 1 and runs["auto"].hierarchies > 1
    assert runs["auto"].cg_iterations <= 4 * runs["new"].cg_iterations
    assert runs["never"].cg_iterations > 2.5 * runs["auto"].cg_iterations
    with pytest.warns(RuntimeWarning, match="outside the valid temperature range"):
        r = solve_steady_kappa3d(LINE[0], y, LINE[1], SI_POWER_LAW, np.full((1, n, 1), 4e15), bc)
    assert r.converged and r.T.max() > 800.0 and r.out_of_range > 0


def test_2d_heater_with_and_without_kappa_of_t():
    """The 2-D heater stack at 100 W/m, 50 nm grid: constant kappa through the Picard path = solve_steady (1e-8 of
    the rise); Si kappa(T) raises the waveguide rise by 1.3e-5 relative. The effect is small: the Si waveguide (at
    about 320 K, kappa 8 % lower) is an island in the oxide, not in series with the heat path, and the Si substrate
    stays near 300 K. Si and SiO2 kappa(T) lower the rise by 0.93 % (20.266 -> 20.078 K)."""
    x, y, kappa, q, bc, masks = heater_stack(dx=0.05e-6, power_per_length=100.0)
    labels, wg = material_labels(kappa), masks["waveguide"]
    rise0 = solve_steady(x, y, kappa, q, bc).T[wg].mean() - 300.0
    rises = {}
    for name, laws in (("constant", [Constant(K_SI), SIO2_CONSTANT, Constant(K_METAL)]),
                       ("si", [SI_POWER_LAW, SIO2_CONSTANT, Constant(K_METAL)]),
                       ("si+sio2", [SI_POWER_LAW, SIO2_LINEAR, Constant(K_METAL)])):
        r = solve_steady_kappa2d(x, y, material_kappa(labels, laws), q, bc)
        assert r.converged and r.T.shape == kappa.shape, name
        rises[name] = r.T[wg].mean() - 300.0
    assert rises["constant"] == pytest.approx(rise0, rel=1e-8)
    assert rises["si"] / rise0 - 1 == pytest.approx(1.32e-5, rel=0.02)
    assert rises["si+sio2"] / rise0 - 1 == pytest.approx(-0.0093, abs=2e-4)
    assert rise0 == pytest.approx(20.266, abs=1e-3) and rises["si+sio2"] == pytest.approx(20.078, abs=1e-3)
