"""Checks of the 3-D steady-state heat solver: exact 1-D solutions along each axis, the sparse direct solve of the
same discrete system, the 2-D solver on z-invariant problems, the Galerkin coarse operator, the multigrid
iteration counts, and the 3-D heater against the 2-D cross-section."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spl

jax.config.update("jax_enable_x64", True)

from thermal_gpu.fv3d import assemble3, assemble_op, build3, full_diag  # noqa: E402
from thermal_gpu.multigrid3d import build_hierarchy3, choose_factors, coarsen3, mg_cycle3  # noqa: E402
from thermal_gpu.solver import solve_steady  # noqa: E402
from thermal_gpu.solver3d import solve_steady3d  # noqa: E402

AXES = "xyz"


def _nonuniform(n, rng, scale=1e-6):
    return np.cumsum(np.r_[0, rng.uniform(0.02, 0.06, n)]) * scale


def _centers(e):
    return 0.5 * (e[1:] + e[:-1])


def _along(v, ax):
    s = [1, 1, 1]
    s[ax] = -1
    return np.reshape(v, s)


@pytest.mark.parametrize("ax", [0, 1, 2])
def test_parabola_along_each_axis(ax):
    """-k T'' = q on 0 < s < L along one axis, T(0) = T(L) = 0, adiabatic on the four other sides (non-uniform
    cells there). On a uniform grid the cell-centered solution is the exact parabola q s (L - s) / (2 k) plus the
    constant q h^2 / (8 k) (the half-cell Dirichlet distance), to round-off.
    The true residual cannot reach the default 1e-12 here: its round-off floor is about eps |A| |T| / |b| =
    eps n^2 / 8 (1e-12 at n = 200; measured 2e-12 to 7e-12). So the test checks the residual against 1e-11 and the
    temperature against the exact solution (measured error 1e-13 of the maximum)."""
    rng = np.random.default_rng(ax)
    L, k, q, n = 1e-6, 1.4, 1e15, 200
    edges = [_nonuniform(3, rng), _nonuniform(4, rng), _nonuniform(3, rng)]
    edges[ax] = np.linspace(0, L, n + 1)
    shape = tuple(len(e) - 1 for e in edges)
    s = _along(_centers(edges[ax]), ax)
    bc = {f"{AXES[ax]}min": ("dirichlet", 0.0), f"{AXES[ax]}max": ("dirichlet", 0.0)}
    exact = np.broadcast_to(q * s * (L - s) / (2 * k) + q * (L / n) ** 2 / (8 * k), shape)
    for precond in ("mg", "jacobi"):
        res = solve_steady3d(*edges, np.full(shape, k), np.full(shape, q), bc, precond=precond)
        assert res.residual < 1e-11, precond
        assert np.max(np.abs(res.T - exact)) / exact.max() < 1e-11, precond


@pytest.mark.parametrize("ax", [0, 1, 2])
def test_two_layers_series_resistance_along_each_axis(ax):
    """Two layers (Si 148 / SiO2 1.4 W/m/K) across one axis, T = 0 and 1 on its two sides: the flux per unit area is
    1 / (t1 / k1 + t2 / k2), exactly (the harmonic mean at the interface)."""
    rng = np.random.default_rng(10 + ax)
    k1, k2, t1, t2 = 148.0, 1.4, 0.3e-6, 2.0e-6
    edges = [_nonuniform(3, rng), _nonuniform(2, rng), _nonuniform(4, rng)]
    edges[ax] = np.concatenate([np.linspace(0, t2, 201), np.linspace(t2, t2 + t1, 31)[1:]])
    shape = tuple(len(e) - 1 for e in edges)
    kappa = np.broadcast_to(np.where(_along(_centers(edges[ax]), ax) < t2, k2, k1), shape).copy()
    bc = {f"{AXES[ax]}min": ("dirichlet", 0.0), f"{AXES[ax]}max": ("dirichlet", 1.0)}
    res = solve_steady3d(*edges, kappa, np.zeros(shape), bc)
    assert res.converged
    d0 = edges[ax][1] - edges[ax][0]
    flux = k2 * np.take(res.T, 0, axis=ax) / (d0 / 2)        # through the min side, per unit area, in each cell
    assert np.allclose(flux, 1 / (t1 / k1 + t2 / k2), rtol=1e-9, atol=0)


@pytest.mark.parametrize("ax", [0, 1, 2])
def test_two_layers_with_random_cells_and_a_robin_side(ax):
    """The same two layers with random cell sizes in each layer (d1 != d2 at the interface and at each face), a
    Robin min side (h, T_inf = 0) on the oxide and T = 1 on the Si side: the flux is 1 / (t1 / k1 + t2 / k2 + 1 / h),
    exactly (the sum of the half-cell resistances of a 1-D chain is the resistance of each layer)."""
    rng = np.random.default_rng(20 + ax)
    k1, k2, t1, t2, h = 148.0, 1.4, 0.3e-6, 2.0e-6, 1e6
    s2, s1 = np.cumsum(rng.uniform(0.2, 1.8, 150)), np.cumsum(rng.uniform(0.2, 1.8, 40))
    edges = [_nonuniform(3, rng), _nonuniform(2, rng), _nonuniform(4, rng)]
    edges[ax] = np.concatenate([[0.0], t2 * s2 / s2[-1], t2 + t1 * s1 / s1[-1]])
    shape = tuple(len(e) - 1 for e in edges)
    kappa = np.broadcast_to(np.where(_along(_centers(edges[ax]), ax) < t2, k2, k1), shape).copy()
    bc = {f"{AXES[ax]}min": ("robin", h, 0.0), f"{AXES[ax]}max": ("dirichlet", 1.0)}
    res = solve_steady3d(*edges, kappa, np.zeros(shape), bc)
    assert res.converged
    d0 = edges[ax][1] - edges[ax][0]
    flux = np.take(res.T, 0, axis=ax) / (d0 / 2 / k2 + 1 / h)  # through the Robin side, per unit area
    assert np.allclose(flux, 1 / (t1 / k1 + t2 / k2 + 1 / h), rtol=1e-9, atol=0)


def _heater_box(rng, n=(26, 22, 18)):
    """A heater block and a waveguide in oxide on a Si substrate, non-uniform grid (synthetic values)."""
    x, y, z = (_nonuniform(m, rng) for m in n)
    X, Y, Z = np.meshgrid(_centers(x), _centers(y), _centers(z), indexing="ij")
    kappa = np.where(Y < 0.3e-6, 148.0, 1.4)                                   # Si substrate, oxide above
    wg = (np.abs(X - x.mean()) < 0.1e-6) & (np.abs(Y - 0.4e-6) < 0.05e-6)      # Si waveguide along z
    heater = (np.abs(X - x.mean()) < 0.2e-6) & (np.abs(Y - 0.7e-6) < 0.05e-6) & (np.abs(Z - z.mean()) < 0.25e-6)
    kappa = np.where(wg, 148.0, np.where(heater, 100.0, kappa))
    q = np.where(heater, 1e15, 0.0)
    bc = {"ymin": ("dirichlet", 300.0), "ymax": ("robin", 10.0, 300.0)}
    return x, y, z, kappa, q, bc


@pytest.mark.parametrize("sides", ["top", "all"])
def test_matches_the_sparse_direct_solve(sides):
    """Non-uniform grid, contrast 148 / 1.4 W/m/K, Robin top: CG (multigrid V and W, Jacobi) = scipy direct.
    "all": also Dirichlet and Robin boundaries on the x and z sides."""
    x, y, z, kappa, q, bc = _heater_box(np.random.default_rng(0))
    if sides == "all":
        bc.update(xmin=("dirichlet", 305.0), xmax=("robin", 1e5, 295.0), zmin=("robin", 1e4, 300.0),
                  zmax=("dirichlet", 300.0))
    p = build3(x, y, z, kappa, q, bc)
    ref = spl.spsolve(assemble3(p).tocsc(), p.b.ravel()).reshape(p.shape)
    rise = ref.max() - ref.min()
    assert rise > 1.0
    for precond, cycle in (("mg", "V"), ("mg", "W"), ("jacobi", "V")):
        res = solve_steady3d(x, y, z, kappa, q, bc, precond=precond, cycle=cycle)
        assert res.converged and res.residual <= 1e-12
        assert np.max(np.abs(res.T - ref)) < 1e-7 * rise, (precond, cycle)


def _cross_section(rng):
    x, y = _nonuniform(40, rng), _nonuniform(30, rng)
    X, Y = np.meshgrid(_centers(x), _centers(y), indexing="ij")
    kappa = np.where(Y < 0.4e-6, 148.0, 1.4)
    heater = (np.abs(X - x.mean()) < 0.3e-6) & (np.abs(Y - 1.0e-6) < 0.05e-6)
    kappa = np.where(heater, 100.0, kappa)
    q = np.where(heater, 1e15, 0.0)
    bc = {"ymin": ("dirichlet", 300.0), "ymax": ("robin", 10.0, 300.0), "xmax": ("robin", 1e4, 310.0)}
    return x, y, kappa, q, bc, heater


def test_z_invariant_problem_equals_the_2d_solver():
    """kappa, q and the boundaries do not change along z, the z ends are adiabatic: each z cell has the 2-D solution.
    Also, with a source that changes along z (a finite heater), the z average sum_k dz_k T_k / L_z is the 2-D
    solution with the z-averaged source (the z fluxes cancel in the sum): a check of the z faces."""
    rng = np.random.default_rng(5)
    x, y, k2, q2, bc, heater = _cross_section(rng)
    z = _nonuniform(9, rng)
    dz = np.diff(z)
    r2 = solve_steady(x, y, k2, q2, bc)
    rise = r2.T.max() - 300.0
    for zz in (z, z[:2]):                                   # 9 z cells, and one z cell
        nz = zz.size - 1
        r3 = solve_steady3d(x, y, zz, np.repeat(k2[:, :, None], nz, axis=2), np.repeat(q2[:, :, None], nz, axis=2),
                            bc)
        assert r2.converged and r3.converged
        assert np.max(np.abs(r3.T - r2.T[:, :, None])) < 1e-9 * rise, nz
    k3 = np.repeat(k2[:, :, None], dz.size, axis=2)
    on = np.zeros(dz.size)
    on[2:5] = 1.0
    q3 = q2[:, :, None] * on[None, None, :]
    r3 = solve_steady3d(x, y, z, k3, q3, bc)
    r2 = solve_steady(x, y, k2, q2 * (dz * on).sum() / dz.sum(), bc)
    mean = (r3.T * dz).sum(axis=2) / dz.sum()
    assert r3.converged and r2.converged
    assert np.max(np.abs(mean - r2.T)) < 1e-9 * (r3.T.max() - 300.0)


def _cube(n):
    edges = [np.linspace(0, 1e-6 * n / 32, n + 1)] * 3
    c = _centers(edges[0])
    X, Y, Z = np.meshgrid(c, c, c, indexing="ij")
    L = edges[0][-1]
    kappa = np.where(Y < 0.3 * L, 148.0, 1.4)
    blk = (np.abs(X - L / 2) < 0.15 * L) & (np.abs(Y - 0.7 * L) < 0.05 * L) & (np.abs(Z - L / 2) < 0.2 * L)
    kappa = np.where(blk, 100.0, kappa)
    bc = {"ymin": ("dirichlet", 300.0), "ymax": ("robin", 10.0, 300.0)}
    return *edges, kappa, np.where(blk, 1e15, 0.0), bc


def test_multigrid_iterations_stay_bounded_when_the_grid_is_refined():
    """A heated block in a cube with a Si bottom: 13, 17 and 20 iterations at 16^3, 32^3 and 64^3 cells."""
    its = []
    for n in (16, 32, 64):                                  # 4k, 33k, 262k cells
        r = solve_steady3d(*_cube(n))
        assert r.converged
        its.append(r.iterations)
    assert its[2] <= its[0] + 10 and its[2] < 30, its


def test_semi_coarsening_on_long_cells():
    """Cells 25 times longer in z than in x and y: z is not aggregated on the first levels, and the iteration count
    stays near that of cubic cells (plain 2 x 2 x 2 aggregation needs about 120 here)."""
    x, y, z, kappa, q, bc = _cube(32)
    z = z * 25
    p = build3(x, y, z, kappa, q, bc)
    assert choose_factors(p.g, p.shape) == (2, 2, 1)
    r = solve_steady3d(x, y, z, kappa, q, bc)
    assert r.converged and r.iterations < 35
    p16 = build3(*_cube(16))
    assert choose_factors(p16.g, p16.shape) == (2, 2, 2)
    one = (np.array([0.0, 1.0]), np.linspace(0, 1, 5), np.linspace(0, 1, 5))
    p1 = build3(*one, np.ones((1, 4, 4)), np.zeros((1, 4, 4)), {})
    assert choose_factors(p1.g, p1.shape) == (1, 2, 2)                          # one cell: never aggregated


@pytest.mark.parametrize("shape,agg", [((5, 4, 3), (2, 2, 2)), ((1, 3, 5), (1, 2, 2)), ((7, 1, 2), (2, 1, 2)),
                                       ((6, 5, 4), (2, 2, 1)), ((3, 3, 3), (1, 1, 2))])
def test_galerkin_coarsening_equals_explicit_ptap(shape, agg):
    """The coarse operator of coarsen3 = P^T A P with the explicit piecewise-constant interpolation P."""
    rng = np.random.default_rng(sum(shape))
    g = tuple(rng.uniform(0.1, 10, tuple(n - (a == ax) for a, n in enumerate(shape))) for ax in range(3))
    extra = rng.uniform(0, 1, shape)
    A = assemble_op(g, full_diag(g, extra))
    cshape = tuple(-(-n // a) for n, a in zip(shape, agg))
    ijk = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")
    cidx = np.ravel_multi_index(tuple(i // a for i, a in zip(ijk, agg)), cshape).ravel()
    P = sp.csr_matrix((np.ones(cidx.size), (np.arange(cidx.size), cidx)), shape=(cidx.size, int(np.prod(cshape))))
    gc, ec = coarsen3(g, extra, agg)
    Ac = assemble_op(gc, full_diag(gc, ec)).toarray()
    assert np.max(np.abs(Ac - (P.T @ A @ P).toarray())) < 1e-12 * np.abs(A).max()


@pytest.mark.parametrize("aspect", [1.0, 20.0, 0.1])
def test_preconditioner_is_symmetric_positive_definite(aspect):
    """The explicit matrix B of the V- and W-cycle on a deep hierarchy (coarsest level 8 cells): B is symmetric and
    the eigenvalues of B A are in the proven bounds (docstring of multigrid3d): (0, 1.6^(L-1)] for the V-cycle
    with L levels, (0, 1.6] for the W-cycle. On the flat cells (aspect 0.1: the first levels aggregate only z) the
    V-cycle goes above 2 (5.9); the W-cycle stays below 1.6."""
    rng = np.random.default_rng(3)
    edges = [_nonuniform(12, rng), _nonuniform(10, rng), _nonuniform(9, rng) * aspect]
    Y = _centers(edges[1])[None, :, None] * np.ones((12, 1, 9))
    kappa = np.where(Y < edges[1].mean() * 0.6, 148.0, 1.4)
    kappa = np.where(rng.random(kappa.shape) < 0.05, 100.0, kappa)
    p = build3(*edges, kappa, np.zeros_like(kappa), {"ymin": ("dirichlet", 300.0), "ymax": ("robin", 10.0, 300.0)})
    A = assemble3(p).toarray()
    levels, cinv = build_hierarchy3(p.g, p.diag_extra, coarsest=8)
    assert len(levels) >= 4
    n = A.shape[0]
    eye = jnp.eye(n).reshape((n,) + p.shape)
    chol = np.linalg.cholesky(A)
    tops = {}
    for cycle, top in (("V", 1.6 ** (len(levels) - 1)), ("W", 1.6)):
        B = np.asarray(jax.vmap(lambda r: mg_cycle3(levels, cinv, r, cycle))(eye)).reshape(n, n)
        assert np.max(np.abs(B - B.T)) < 1e-12 * np.abs(B).max()
        ev = np.linalg.eigvalsh(chol.T @ B @ chol)
        assert 0 < ev.min() and ev.max() <= top * (1 + 1e-9), (cycle, ev.min(), ev.max())
        tops[cycle] = ev.max()
    if aspect == 0.1:
        assert tops["V"] > 2.0 > 1.6 >= tops["W"]


def test_flags_and_input_errors():
    x, y, z, kappa, q, bc = _heater_box(np.random.default_rng(1), n=(14, 12, 10))
    assert solve_steady3d(x, y, z, kappa, q, bc).converged
    assert not solve_steady3d(x, y, z, kappa, q, bc, max_iter=3).converged
    assert not solve_steady3d(x, y, z, kappa, q, bc, max_iter=3, precond="jacobi").converged
    r32 = solve_steady3d(x, y, z, kappa, q, bc, dtype=jnp.float32)      # float32 cannot reach 1e-12
    assert not r32.converged and r32.residual > 1e-9
    for kw in ({"precond": "amg"}, {"cycle": "F"}):
        with pytest.raises(ValueError):
            solve_steady3d(x, y, z, kappa, q, bc, **kw)
    with pytest.raises(ValueError):
        solve_steady3d(x, y, z, kappa, q, {"top": ("dirichlet", 300.0)})        # unknown side
    with pytest.raises(ValueError):
        solve_steady3d(x, y, z, kappa[:-1], q[:-1], bc)                        # wrong shape
    for spec in (("convective", 10.0), ("robin", 10.0), ("dirichlet",), ("neumann", 1.0), ("robin", 0.0, 300.0),
                 ("robin", -5.0, 300.0), ("dirichlet", np.nan), ("robin", np.inf, 300.0), ("dirichlet", "hot"),
                 "dirichlet", ()):
        with pytest.raises(ValueError):                                        # kind, count, value or h <= 0
            solve_steady3d(x, y, z, kappa, q, {"ymin": ("dirichlet", 300.0), "zmin": spec})
    bad_x, bad_k, bad_q = x.copy(), kappa.copy(), q.copy()
    bad_x[3], bad_k[1, 2, 3], bad_q[2, 3, 4] = np.nan, np.inf, np.nan
    for args in ((bad_x, y, z, kappa, q), (x, y, z, bad_k, q), (x, y, z, kappa, bad_q), (x[::-1], y, z, kappa, q)):
        with pytest.raises(ValueError):                                        # NaN edge, inf kappa, NaN q, order
            solve_steady3d(*args, bc)


def test_heater_3d_mid_length_equals_the_2d_cross_section():
    """A 100 um heater with 10 mW over the waveguide: at mid-length the waveguide rise is the 2-D rise at 100 W/m;
    at the heater ends it is about half (the heat also flows along z); 20 um past the ends it is almost 0. The z
    integral of the rise is that of the 2-D model: the heat that leaves the ends still heats the waveguide."""
    from thermal_gpu.structures import heater_3d, heater_stack
    dx = 0.1e-6
    x, y, k2, q2, bc2, m2 = heater_stack(dx=dx, power_per_length=100.0)
    r2 = solve_steady(x, y, k2, q2, bc2)
    rise2 = r2.T[m2["waveguide"]].mean() - 300.0
    x, y, z, k, q, bc, m = heater_3d(dx=dx, length=100e-6, power=10e-3)
    assert np.allclose(q[m["heater"]], q2[m2["heater"]][0])
    r3 = solve_steady3d(x, y, z, k, q, bc)
    assert r3.converged
    zc = _centers(z)
    wg = m["waveguide"][:, :, 0]
    prof = np.array([r3.T[:, :, i][wg].mean() - 300.0 for i in range(zc.size)])
    assert np.interp(0.0, zc, prof) == pytest.approx(rise2, rel=1e-4)
    assert 0.45 < np.interp(50e-6, zc, prof) / rise2 < 0.6
    assert prof[-1] < 0.01 * rise2
    assert np.sum(prof * np.diff(z)) / (rise2 * 100e-6) == pytest.approx(1.0, abs=2e-3)
    xl, yl, zl, kl, ql, bcl, ml = heater_3d(dx=dx, length=100e-6, power=10e-3, lead_w=4e-6)   # metal leads
    rl = solve_steady3d(xl, yl, zl, kl, ql, bcl)
    end_lead = np.interp(50e-6, zc, [rl.T[:, :, i][wg].mean() - 300.0 for i in range(zc.size)])
    assert rl.converged and end_lead < np.interp(50e-6, zc, prof)            # the leads cool the heater ends
    xs, ys, zs, ks, qs, bcs, _ = heater_3d(dx=dx, length=20e-6, margin=0.0, power=2e-3)   # no ends: the 2-D result
    rs = solve_steady3d(xs, ys, zs, ks, qs, bcs)
    assert rs.converged and np.max(np.abs(rs.T - r2.T[:, :, None])) < 1e-9 * (r2.T.max() - 300.0)
    with pytest.raises(ValueError):
        heater_3d(dx=dx, power_per_length=100.0)
