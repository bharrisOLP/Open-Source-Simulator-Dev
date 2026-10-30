"""The mixed-precision refinement (precision "mixed", polish "refine"): complex128 residuals and Rayleigh-Ritz,
complex64 GMRES correction solves.

Checks: the refined eigenpairs equal those of a double-precision solve (neff to 1e-9, fields by overlap) on a strip
and on a cluster (two far-apart identical cores: even and odd supermodes with a neff splitting below the complex64
resolution), they are eigenpairs of the assembled SciPy matrix (an independent check of the residual), the
refinement needs far fewer complex128 operator applies than the complex128 Davidson polish, and it stops with
converged=False (not silently) when it cannot reach the tolerance.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse.linalg as sla
from structures import coupler, sin_tip, strip

import meow_gpu.fde.eigensolver as eigensolver
from meow_gpu.device import put
from meow_gpu.fde.eigensolver import (
    REFINE_INNER,
    REFINE_STALL,
    _refine_with_fallback,
    _selection,
    _track,
    davidson,
    make_preconditioner,
    precondition,
    refine,
)
from meow_gpu.fde.operator import assemble, make_problem
from meow_gpu.meow_backend import GPUModeSolver

K = 4


def _case(p, b=K + 4, seed=0):
    pc = make_preconditioner(p)
    target = -(np.sqrt(np.max(np.abs(p.eps_xx))) * (1 + 1e-14)) ** 2
    x = put(np.random.default_rng(seed).normal(size=(b, 2) + p.shape) + 0j, jnp.complex64)
    x0 = precondition(pc, x, put(np.full(b, target), jnp.complex64))
    return p, pc, target, x0, (p.device(jnp.complex64), p.device(jnp.complex128))


@pytest.fixture(scope="module")
def strip_case():
    return _case(make_problem(*strip(dx=0.04, dy=0.04, w_dom=2.4, h_dom=1.6), 1.55))


@pytest.fixture(scope="module")
def cluster_case():
    # gap 56 cells (2.24 um): the TE even and odd supermodes differ by 2.7e-8 in neff (relative eigenvalue
    # splitting 2.4e-8, below the complex64 epsilon 1.2e-7); the TM pair differs by 8.8e-5
    return _case(make_problem(*coupler(gap=56), 1.55))


def _neff(values):
    return np.sqrt(-np.asarray(values, dtype=complex))


def _unit(v):
    v = np.asarray(v).reshape(len(v), -1)
    return v / np.linalg.norm(v, axis=1)[:, None]


def _cos(a, b):
    """|<a_i, b_i>| of the normalized rows."""
    return np.abs(np.sum(_unit(a).conj() * _unit(b), axis=1))


def _scipy_residuals(p, values, vectors):
    """Relative residuals |A x - theta x| / (|x| max(1, |theta|)) with the assembled matrix (CPU, SciPy)."""
    a = assemble(p)
    x = np.asarray(vectors).reshape(len(values), -1)
    return np.array([np.linalg.norm(a @ xi - th * xi) / np.linalg.norm(xi) / max(1.0, abs(th))
                     for th, xi in zip(values, x)])


def test_refine_equals_the_double_solve(strip_case):
    p, pc, target, x0, op = strip_case
    got = davidson(op, pc, K, target, x0, tol=1e-10, precision="mixed", polish="refine")
    ref = davidson(op[1], pc, K, target, x0, tol=1e-10, precision="double")
    assert got.converged and ref.converged
    assert np.all(got.residuals < 1e-10)
    assert np.max(np.abs(_neff(got.values) - _neff(ref.values))) < 1e-9
    assert np.all(_cos(got.vectors[:K], ref.vectors[:K]) > 1 - 1e-9)
    assert np.all(_scipy_residuals(p, got.values, got.vectors[:K]) < 1e-10)     # independent of the JAX operator


def test_refine_needs_few_complex128_applies(strip_case):
    """The same tolerance as the complex128 Davidson polish, with far fewer complex128 applies and iterations (on the
    RX 7900 XT: 24 against 456 complex128 applies, 49 against 106 iterations, at REFINE_INNER = 20)."""
    p, pc, target, x0, op = strip_case
    new = davidson(op, pc, K, target, x0, tol=1e-10, precision="mixed", polish="refine")
    old = davidson(op, pc, K, target, x0, tol=1e-10, precision="mixed", polish="davidson")
    assert new.converged and old.converged
    assert np.all(new.residuals < 1e-10) and np.all(old.residuals < 1e-10)
    assert np.max(np.abs(_neff(new.values) - _neff(old.values))) < 1e-9
    assert 0 < new.matvecs_hi < old.matvecs_hi / 5
    assert new.iterations < old.iterations


def test_refine_separates_a_cluster(cluster_case):
    """The reference is ARPACK shift-invert (complex128 LU) on the assembled matrix: an independent solver."""
    p, pc, target, x0, op = cluster_case
    vals, vecs = sla.eigs(assemble(p), k=K, sigma=target, tol=1e-14)
    order = np.argsort(np.abs(vals - target))
    vals, vecs = vals[order], vecs[:, order].T
    n_ref = _neff(vals)
    assert abs(n_ref[0] - n_ref[1]) < 1e-7                    # the cluster
    got = davidson(op, pc, K, target, x0, tol=1e-10, precision="mixed", polish="refine")
    dbl = davidson(op[1], pc, K, target, x0, tol=1e-10, precision="double")
    assert got.converged
    assert np.max(np.abs(_neff(got.values) - n_ref)) < 1e-9
    assert np.max(np.abs(_neff(got.values) - _neff(dbl.values))) < 1e-9
    # each vector of the pair: sin(angle) <= |r| / gap (about 3e-3 at tol 1e-10), so 1 - cos < 5e-6
    assert np.all(_cos(got.vectors[:K], vecs) > 1 - 1e-5)
    assert np.all(_cos(got.vectors[2:K], vecs[2:]) > 1 - 1e-9)       # the TM pair
    # the subspace of the pair: all principal angles about 0
    qa, qb = np.linalg.qr(_unit(got.vectors[:2]).T)[0], np.linalg.qr(_unit(vecs[:2]).T)[0]
    assert np.linalg.svd(qa.conj().T @ qb, compute_uv=False).min() > 1 - 1e-9


@pytest.fixture(scope="module")
def cluster_reference(cluster_case):
    """ARPACK shift-invert (complex128 LU) on the assembled matrix of the two-core case: the 4 eigenpairs nearest
    the target (TE even, TE odd, TM even, TM odd)."""
    p, _, target, _, _ = cluster_case
    vals, vecs = sla.eigs(assemble(p), k=K, sigma=target, tol=1e-14)
    order = np.argsort(np.abs(vals - target))
    return vals[order], vecs[:, order].T


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("k", [1, 3])
def test_mode_count_inside_a_near_degenerate_pair(cluster_case, cluster_reference, k, seed):
    """Review finding (2026-10-30): k = 1 cuts the TE pair (splitting 2.7e-8 in neff) and k = 3 cuts the TM pair
    (8.8e-5). The complex64 Ritz values (residual 1e-3) cannot order the two modes of a pair; with the wanted set
    of the first refinement step, the refinement gave the odd mode (|cos| about 1e-4 with the even mode) with
    converged=True for k = 1, seeds 0 and 1, and k = 3, seed 0. The mode identity is checked by the field overlap,
    against ARPACK and against the double solve."""
    p = cluster_case[0]
    vals, vecs = cluster_reference
    _, pc, target, x0, op = _case(p, b=k + 4, seed=seed)
    got = davidson(op, pc, k, target, x0, tol=1e-10, precision="mixed", polish="refine")
    dbl = davidson(op[1], pc, k, target, x0, tol=1e-10, precision="double")
    assert got.converged and dbl.converged
    assert np.max(np.abs(_neff(got.values) - _neff(vals[:k]))) < 1e-9
    assert np.max(np.abs(_neff(got.values) - _neff(dbl.values))) < 1e-9
    # sin(angle) <= |r| / gap for each vector of the TE pair (about 3e-3 at tol 1e-10): 1 - cos < 5e-6
    assert np.all(_cos(got.vectors[:k], vecs[:k]) > 1 - 1e-5)
    assert np.all(_cos(got.vectors[:k], dbl.vectors[:k]) > 1 - 1e-5)


def test_selection_orders_a_near_pair_by_converged_values():
    """``_selection``: the wanted pairs are the k tracked pairs nearest the target, by value (not by index), and an
    identified neighbour whose value can still be nearer the target (within its residual) must converge too."""
    target, tol = -12.0, 1e-10
    theta = np.array([-4.8370001, -4.8370002, -2.8, -1.5 + 0.2j, 0.0])
    inf = np.inf
    # pair 1 is nearer the target: its residual decides (the converged pair 0 is not wanted)
    used, worst = _selection(theta, np.array([1e-11, 1e-6, 1e-11, 4.0, inf]), 4, 1, target, tol)
    assert list(used) == [1] and worst == 1e-6
    # pair 1 converged, pair 0 not: pair 0 (error bound 4.8e-6) can still be nearer the target
    used, worst = _selection(theta, np.array([1e-6, 1e-11, 1e-11, 4.0, inf]), 4, 1, target, tol)
    assert list(used) == [1, 0] and worst == 1e-6
    # pair 0 at 1e-9 (bound 4.8e-9, less than the splitting 1e-7): the order is sure, and the pair with a residual
    # of 4 (not identified) does not block the stop
    used, worst = _selection(theta, np.array([1e-9, 1e-11, 1e-11, 4.0, inf]), 4, 1, target, tol)
    assert list(used) == [1] and worst < tol
    # after _track left a pair out, the indices are not in the order of the distance: the selection uses the values
    used, _ = _selection(np.array([-1.85, -6.9, -7.8]), np.full(3, 1e-11), 3, 2, target, tol)
    assert list(used) == [2, 1]
    # fewer tracked pairs than wanted pairs: no stop
    assert _selection(theta, np.full(5, 1e-11), 1, 2, target, tol)[1] == inf


def test_refine_stops_when_the_tolerance_is_out_of_reach(strip_case):
    """A tolerance below the complex128 floor (about 1e-14 here): the refinement stops by its stall rule, not by the
    iteration limit, with converged=False and residuals at the floor."""
    p, pc, target, x0, op = strip_case
    none = davidson(op, pc, K, target, x0, tol=1e-10, precision="mixed", polish="refine", max_iter=0)
    assert not none.converged and none.iterations == 0     # max_iter = 0: no step, not converged
    x64 = davidson(op[0], pc, K, target, x0, tol=1e-3, precision="single").vectors
    got = refine(op[0], op[1], pc, K, target, x64, tol=1e-17, max_steps=50)
    assert not got.converged
    assert REFINE_STALL <= got.iterations < 50
    assert np.all(got.residuals < 1e-12)
    # the step limit: also not converged
    short = refine(op[0], op[1], pc, K, target, x64, tol=1e-10, max_steps=1)
    assert not short.converged and short.iterations == 1 and np.max(short.residuals) > 1e-10


def test_track_keeps_each_pair():
    """The values of a GPU run (taper cell 18, 14 GMRES steps): a spurious Ritz value (-1.886 - 8.5e-4 i) and a new
    value nearer the target (-1.935) must not replace the pairs at -1.8877 and -1.8503."""
    prev = np.array([-7.809716, -6.956546, -1.887704, -1.850276])
    vals = np.array([-1.850276 + 4e-11j, -1.935, -6.956546, -1.886025 - 8.5e-4j, -7.809716, -1.887704 - 6e-11j])
    assert list(_track(vals, prev)) == [4, 2, 5, 0]
    assert list(_track(vals[:3], prev)) == [2, 1, 0]       # fewer values than pairs: pair 0 is left out
    # with the residuals of all values: -1.935 comes in only when it is identified (residual below 1e-2); then it
    # replaces the followed value farthest from the target (-1.850276). The spurious value never comes in.
    rn = np.array([1e-10, 8.6e-2, 1e-12, 4e-1, 1e-12, 1e-10])
    assert list(_track(vals, prev, rn, -12.0)) == [4, 2, 5, 0]
    rn[1] = 1e-4
    assert list(_track(vals, prev, rn, -12.0)) == [4, 2, 5, 1]


def test_solver_warns_when_the_refinement_does_not_converge():
    p = make_problem(*strip(dx=0.04, dy=0.04, w_dom=2.4, h_dom=1.6), 1.55)
    s = GPUModeSolver(precision="mixed", extra=4, tol=1e-17)
    with pytest.warns(UserWarning, match="did not converge"):
        s.solve(p, K)
    assert not s.last_info.converged and s.last_info.residual < 1e-12
    assert s.last_info.polish_used == "refine+davidson"      # the fallback ran too
    assert s.last_info.iterations < 250 < s.max_iter          # its stall rule stopped it (review, 2026-10-30)


def test_fallback_stops_by_its_stall_rule_and_keeps_the_better_result(strip_case):
    """A tolerance below the complex128 floor: the fallback (the steps of the earlier polish from the complex64 block)
    stops by its stall rule, not at the iteration limit, and the result is not worse than that of the refinement
    alone (review, 2026-10-30: without the stall rule, 500 iterations and a larger final residual)."""
    p, pc, target, x0, op = strip_case
    x64 = davidson(op[0], pc, K, target, x0, tol=1e-3, precision="single").vectors
    alone = refine(op[0], op[1], pc, K, target, x64, tol=1e-17, max_steps=400)
    both = _refine_with_fallback(op[0], op[1], pc, K, target, x64, 1e-17, 400, REFINE_INNER, False, 4 * (K + 4))
    assert not alone.converged and not both.converged and both.polish == "refine+davidson"
    assert both.iterations < 400
    assert np.max(both.residuals) <= np.max(alone.residuals) and np.max(both.residuals) < 1e-12
    assert both.matvecs_hi > alone.matvecs_hi


def test_batched_fallback():
    """The fallback in the batched path (``_refine_members``, ``BatchEigResult.polishes``): a tolerance out of reach
    (both members: "refine+davidson", stopped by the stall rule), and the weakly guided SiN tip in a batch of 1 (the
    result of ``solve``)."""
    probs = [make_problem(*strip(width=w, dx=0.04, dy=0.04, w_dom=2.4, h_dom=1.6), 1.55) for w in (0.45, 0.5)]
    s = GPUModeSolver(precision="mixed", extra=4, tol=1e-17, warm_start=False)
    with pytest.warns(UserWarning, match="did not converge"):
        s.solve_batch(probs, K)
    for info in s.last_batch_info:
        assert info.polish_used == "refine+davidson" and not info.converged
        assert info.residual < 1e-12 and info.iterations < s.max_iter
    p = make_problem(*sin_tip(), 1.55)
    one = GPUModeSolver(precision="mixed", extra=8, warm_start=False)
    n_one, e_one, _ = one.solve(p, K)
    bat = GPUModeSolver(precision="mixed", extra=8, warm_start=False)
    ((n_bat, e_bat, _),) = bat.solve_batch([p], K)
    assert bat.last_batch_info[0].converged and one.last_info.converged
    assert bat.last_batch_info[0].polish_used == one.last_info.polish_used == "refine+davidson"
    assert np.max(np.abs(n_bat - n_one)) < 1e-9
    assert np.all(_cos(e_bat[:2], e_one[:2]) > 1 - 1e-6)


def test_complex128_precisions_need_x64(strip_case):
    """Without jax_enable_x64, JAX runs complex128 arrays in complex64 without an error: "mixed" and "double" stop
    with a clear error instead (review, 2026-10-30)."""
    p, pc, target, x0, op = strip_case
    jax.config.update("jax_enable_x64", False)
    try:
        for precision in ("mixed", "double"):
            with pytest.raises(RuntimeError, match="jax_enable_x64"):
                GPUModeSolver(precision=precision).solve(p, K)
            with pytest.raises(RuntimeError, match="jax_enable_x64"):
                davidson(op if precision == "mixed" else op[1], pc, K, target, x0, precision=precision)
        with pytest.raises(RuntimeError, match="jax_enable_x64"):
            refine(op[0], op[1], pc, K, target, x0, tol=1e-10, max_steps=1)
    finally:
        jax.config.update("jax_enable_x64", True)


def test_weakly_guided_mode_converges():
    """Regression (2026-10-30, examples/chip_to_fiber_coupling.py): a weakly guided SiN tip mode just above a dense
    band of window modes. The refinement alone stalls at a residual of about 3e-10 (the halving rule stops it; before
    the fallback, "mixed" returned converged=False here, while polish="davidson" converged). The fallback runs the
    steps of the earlier polish from the complex64 block. References: the earlier polish and the double
    solve from the same start (neff and fields), ARPACK shift-invert on the assembled matrix (the neff of the two
    guided modes; ARPACK also finds two modes at exactly n = 1.444 that all precisions of this solver miss, a known
    limit), and the residuals of all 4 pairs with the assembled SciPy matrix."""
    p = make_problem(*sin_tip(), 1.55)
    s = GPUModeSolver(precision="mixed", extra=8, warm_start=False)
    n_mix, e_mix, _ = s.solve(p, K)
    assert s.last_info.converged and s.last_info.residual < 1e-10
    assert s.last_info.polish_used in ("refine", "refine+davidson")
    old = GPUModeSolver(precision="mixed", polish="davidson", extra=8, warm_start=False)
    n_old, e_old, _ = old.solve(p, K)
    assert old.last_info.converged and old.last_info.polish_used == "davidson"
    dbl = GPUModeSolver(precision="double", extra=8, warm_start=False)
    n_dbl, e_dbl, _ = dbl.solve(p, K)
    assert dbl.last_info.converged and dbl.last_info.polish_used == ""
    assert np.max(np.abs(n_mix - n_old)) < 1e-9 and np.max(np.abs(n_mix - n_dbl)) < 1e-9
    assert np.real(n_mix[0]) > 1.444                          # the guided mode
    # fields: the two guided modes (the window modes 3 and 4 are a near-degenerate pair: compare their subspace)
    assert np.all(_cos(e_mix[:2], e_dbl[:2]) > 1 - 1e-6)
    qa, qb = np.linalg.qr(_unit(e_mix[2:4]).T)[0], np.linalg.qr(_unit(e_dbl[2:4]).T)[0]
    assert np.linalg.svd(qa.conj().T @ qb, compute_uv=False).min() > 1 - 1e-6
    target = -(1.996 * (1 + 10 * np.finfo(float).eps)) ** 2
    vals = sla.eigs(assemble(p), k=K, sigma=target, tol=1e-13, return_eigenvectors=False)
    n_ref = np.sort(np.real(_neff(vals)))[::-1]
    assert np.max(np.abs(np.real(n_mix[:2]) - n_ref[:2])) < 1e-9
    u = e_mix[:, :2]                                          # (Ex, Ey): the eigenvector of the operator
    assert np.all(_scipy_residuals(p, -(n_mix**2), u) < 1e-10)


@pytest.fixture(scope="module")
def example_tip():
    """The grid of examples/chip_to_fiber_coupling.py (the SiN tip on 50 nm, 280 x 280 cells, 4 modes, extra 8) and
    the references: the earlier polish and the double solve."""
    p = make_problem(*sin_tip(dx=0.05), 1.55)
    old = GPUModeSolver(precision="mixed", polish="davidson", extra=8, warm_start=False)
    ref_old = old.solve(p, K)
    dbl = GPUModeSolver(precision="double", extra=8, warm_start=False)
    ref_dbl = dbl.solve(p, K)
    assert old.last_info.converged and dbl.last_info.converged
    return p, ref_old, ref_dbl


@pytest.mark.parametrize("admission", [True, False])
def test_example_grid_gives_the_modes_of_the_earlier_polish(example_tip, monkeypatch, admission):
    """End to end on the grid of the example (review, 2026-10-30). Before the admission rule and the fallback, the
    4th mode was 1.44122808 instead of 1.4418766 (the second mode of a near-degenerate pair of window modes was not
    in the block). admission=False turns off the admission rule of ``_track``: then only the fallback (the steps of
    the earlier polish from the complex64 block) protects the result, and it must give the same modes. All 4 neff
    against the earlier polish and the double solve; the fields of the guided modes by overlap, the near-degenerate
    pair by its subspace."""
    p, (n_old, e_old, _), (n_dbl, e_dbl, _) = example_tip
    if not admission:
        track = eigensolver._track
        monkeypatch.setattr(eigensolver, "_track", lambda vals, prev, rn_vals=None, target=0.0: track(vals, prev))
    s = GPUModeSolver(precision="mixed", extra=8, warm_start=False)
    n, e, _ = s.solve(p, K)
    assert s.last_info.converged
    assert np.max(np.abs(n - n_old)) < 1e-9 and np.max(np.abs(n - n_dbl)) < 1e-9
    assert np.real(n[3]) > 1.4418                             # not the wrong mode of the regression (1.44122808)
    for ref in (e_old, e_dbl):
        assert np.all(_cos(e[:2], ref[:2]) > 1 - 1e-6)
        qa, qb = np.linalg.qr(_unit(e[2:4]).T)[0], np.linalg.qr(_unit(ref[2:4]).T)[0]
        assert np.linalg.svd(qa.conj().T @ qb, compute_uv=False).min() > 1 - 1e-6


def test_mixed_iteration_limit_inside_the_refinement(strip_case):
    """max_iter counts the complex64 iterations and the refinement steps together."""
    p, pc, target, x0, op = strip_case
    full = davidson(op, pc, K, target, x0, tol=1e-10, precision="mixed", polish="refine")
    cut = davidson(op, pc, K, target, x0, tol=1e-10, precision="mixed", polish="refine",
                   max_iter=full.iterations - 1)
    assert full.converged and not cut.converged and cut.iterations == full.iterations - 1


@pytest.mark.parametrize("precision", ["single", "double"])
def test_complex128_apply_count(strip_case, precision):
    p, pc, target, x0, op = strip_case
    r = davidson(op[0] if precision == "single" else op[1], pc, K, target, x0, tol=1e-5, precision=precision)
    assert r.matvecs_hi == (0 if precision == "single" else r.matvecs)


def test_davidson_polish_stays_available():
    """The earlier polish (polish="davidson") through the solver and the batched solver: the same modes."""
    probs = [make_problem(*strip(width=w, dx=0.04, dy=0.04, w_dom=2.4, h_dom=1.6), 1.55) for w in (0.45, 0.5)]
    new = [GPUModeSolver(precision="mixed", extra=4, warm_start=False).solve(p, K) for p in probs]
    old_solver = GPUModeSolver(precision="mixed", polish="davidson", extra=4, warm_start=False)
    old = [old_solver.solve(p, K) for p in probs]
    batch = GPUModeSolver(precision="mixed", polish="davidson", extra=4).solve_batch(probs, K)
    for a, b, c in zip(new, old, batch):
        assert np.max(np.abs(a[0] - b[0])) < 1e-9 and np.max(np.abs(c[0] - b[0])) < 1e-9
        assert np.all(_cos(a[1], b[1]) > 1 - 1e-9)
    with pytest.raises(ValueError, match="polish"):
        GPUModeSolver(precision="mixed", polish="newton").solve(probs[0], K)
