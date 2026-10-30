"""Checks of the steady-state heat solver: the exact 1-D slab, the sparse direct solve of the same discrete system,
and the layered (harmonic-mean) conductance."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse.linalg as spl

jax.config.update("jax_enable_x64", True)

from thermal_gpu.fv import assemble, build  # noqa: E402
from thermal_gpu.solver import solve_steady  # noqa: E402


def test_slab_with_uniform_heating_matches_the_parabola():
    """-k T'' = q on 0 < y < L, T(0) = T(L) = 0: T = q y (L - y) / (2 k). Cell-centered FV is exact for a
    parabola at the cell centers (second-order differences of a quadratic are exact)."""
    L, k, q, n = 1e-6, 1.4, 1e15, 200
    x = np.array([0.0, 1e-6])
    y = np.linspace(0, L, n + 1)
    yc = 0.5 * (y[1:] + y[:-1])
    bc = {"ymin": ("dirichlet", 0.0), "ymax": ("dirichlet", 0.0)}
    res = solve_steady(x, y, np.full((1, n), k), np.full((1, n), q), bc)
    exact = q * yc * (L - yc) / (2 * k)
    # the half-cell Dirichlet distance makes the boundary cells second-order: relative error O(1/n^2)
    assert np.max(np.abs(res.T[0] - exact)) / exact.max() < 1e-4


def test_two_layers_series_resistance():
    """Heat flux through two layers (k1, t1), (k2, t2) with T = 1 at the top and 0 at the bottom: the flux is
    1 / (t1 / k1 + t2 / k2) per unit area, exactly (the harmonic mean at the interface)."""
    k1, k2, t1, t2, n = 148.0, 1.4, 0.3e-6, 2.0e-6, 230
    y = np.concatenate([np.linspace(0, t2, 201), np.linspace(t2, t2 + t1, 31)[1:]])
    x = np.array([0.0, 1e-6])
    kappa = np.where(0.5 * (y[1:] + y[:-1]) < t2, k2, k1)[None, :]
    bc = {"ymin": ("dirichlet", 0.0), "ymax": ("dirichlet", 1.0)}
    res = solve_steady(x, y, kappa, np.zeros_like(kappa), bc)
    dy0 = y[1] - y[0]
    flux = k2 * (res.T[0, 0] - 0.0) / (dy0 / 2)              # through the bottom boundary, per unit area
    assert flux == pytest.approx(1 / (t1 / k1 + t2 / k2), rel=1e-9)


def test_matches_the_sparse_direct_solve():
    """A heater over a waveguide in oxide on a Si substrate (synthetic values): GPU CG = scipy sparse direct."""
    rng = np.random.default_rng(0)
    x = np.cumsum(np.r_[0, rng.uniform(0.02, 0.06, 80)]) * 1e-6
    y = np.cumsum(np.r_[0, rng.uniform(0.02, 0.06, 60)]) * 1e-6
    xc, yc = 0.5 * (x[1:] + x[:-1]), 0.5 * (y[1:] + y[:-1])
    X, Y = np.meshgrid(xc, yc, indexing="ij")
    kappa = np.where(Y < 1e-6, 148.0, 1.4)                       # Si substrate, oxide above
    heater = (np.abs(X - x.mean()) < 0.5e-6) & (np.abs(Y - 2.2e-6) < 0.05e-6)
    kappa = np.where(heater, 100.0, kappa)
    q = np.where(heater, 1e15, 0.0)
    bc = {"ymin": ("dirichlet", 300.0), "ymax": ("robin", 10.0, 300.0)}
    p = build(x, y, kappa, q, bc)
    ref = spl.spsolve(assemble(p).tocsc(), p.b.ravel()).reshape(p.shape)
    for precond in ("mg", "jacobi"):
        res = solve_steady(x, y, kappa, q, bc, tol=1e-12, precond=precond)
        assert res.converged
        assert np.max(np.abs(res.T - ref)) < 1e-7 * (ref.max() - 300.0), precond


def test_multigrid_iterations_do_not_grow_with_the_grid():
    from thermal_gpu.structures import heater_stack
    its = []
    for dx in (0.05e-6, 0.025e-6):
        x, y, kappa, q, bc, _ = heater_stack(dx=dx)
        r = solve_steady(x, y, kappa, q, bc)
        assert r.converged
        its.append(r.iterations)
    assert its[1] <= its[0] + 6 and its[1] < 40


def test_converged_flag_and_thin_heater_error():
    from thermal_gpu.structures import heater_stack
    x, y, kappa, q, bc, _ = heater_stack(dx=0.05e-6)
    assert solve_steady(x, y, kappa, q, bc).converged
    assert not solve_steady(x, y, kappa, q, bc, max_iter=3).converged
    assert not solve_steady(x, y, kappa, q, bc, max_iter=3, precond="jacobi").converged
    # float32 cannot reach the true residual 1e-12 (the recursive CG residual can): the flag must say so
    r32 = solve_steady(x, y, kappa, q, bc, dtype=jnp.float32)
    assert not r32.converged and r32.residual > 1e-9
    with pytest.raises(ValueError):
        solve_steady(x, y, kappa, q, bc, precond="amg")
    with pytest.raises(ValueError):
        heater_stack(dx=0.3e-6, heater_t=0.1e-6)          # cell centers at 2.05 and 2.35 um: none in the heater
