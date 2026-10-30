"""Bend modes on the GPU: compute_modes_gpu with a bend radius gives the modes of compute_modes_tidy3d (the same radial
transform, the same discrete problem), with no tidy3d fallback. Synthetic Si strip (textbook indices)."""
import warnings
from functools import cache

import meow as mw
import numpy as np
import pytest
import scipy.sparse.linalg as spl
from tidy3d_ref import tidy3d_solver_matrix

from meow_gpu.meow_backend import (
    GPUModeSolver,
    _supported,
    compute_modes_gpu,
    compute_modes_gpu_batch,
    post_process_modes,
)

N_SI = 3.476


def bend_cross_section(radius=None, axis=1, width=0.5, wl=1.55, dx=0.04, x=(-1.2, 1.2), y=(-0.7, 0.92),
                       num_pml=(0, 0), plane_center=(0.0, 0.0)):
    """A 0.5 x 0.22 um Si strip in SiO2 (n 3.476, 1.444), as ``test_meow_backend.strip_cross_section``, with a bend
    radius (None: straight) and bend axis (1: the bend in the x-z plane; 0: in the y-z plane)."""
    oxide = mw.IndexMaterial(name="sio2", n=1.444)
    si = mw.IndexMaterial(name="si", n=N_SI)
    clad = mw.Structure(material=oxide, geometry=mw.Box(x_min=-5, x_max=5, y_min=-5, y_max=5, z_min=-1, z_max=2))
    core = mw.Structure(material=si, mesh_order=1,
                        geometry=mw.Box(x_min=-width / 2, x_max=width / 2, y_min=0, y_max=0.22, z_min=-1, z_max=2))
    bend = {} if radius is None else {"bend_radius": radius, "bend_axis": axis, "plane_center": plane_center}
    mesh = mw.Mesh2D(x=np.arange(x[0], x[1] + dx / 2, dx), y=np.arange(y[0], y[1] + dx / 2, dx), num_pml=num_pml,
                     **bend)
    cell = mw.Cell(structures=[clad, core], mesh=mesh, z_min=0.0, z_max=1.0)
    return mw.CrossSection.from_cell(cell=cell, env=mw.Environment(wl=wl, T=25.0))


def _raw(modes):
    return modes


@cache
def _tidy3d(radius, axis, k, target=None, **kw):
    """compute_modes_tidy3d with the post-processing of meow_gpu."""
    cs = bend_cross_section(radius, axis, **kw)
    return cs, mw.compute_modes_tidy3d(cs, num_modes=k, target_neff=target, post_process=post_process_modes)


def _gpu(cs, k, precision, target=None, post_process=None):
    """compute_modes_gpu; check that it did not fall back to tidy3d and that it converged."""
    s = GPUModeSolver(precision=precision, extra=4)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        modes = compute_modes_gpu(cs, num_modes=k, target_neff=target, solver=s, post_process=post_process)
    assert not [w for w in record if "not supported" in str(w.message)]
    assert s.last_info.converged
    return modes


def _assert_same(got, ref, ntol, ftol):
    assert len(got) == len(ref)
    for a, b in zip(got, ref):
        assert abs(a.neff - b.neff) < ntol
        assert abs(mw.inner_product(a, b)) == pytest.approx(1.0, abs=ftol)   # both normalized by the post-processing


def _assert_same_components(got, ref, tol):
    """All six field components, after one common phase for each mode. meow's inner product uses only Ex, Ey, Hx
    and Hy; Ez and Hz are the components that the back-transform of a bend changes (Ez = a_e Ez', Hz = a_h Hz')."""
    for a, b in zip(got, ref):
        ea = np.concatenate([np.ravel(a.Ex), np.ravel(a.Ey)])
        eb = np.concatenate([np.ravel(b.Ex), np.ravel(b.Ey)])
        c = np.vdot(eb, ea)
        c /= abs(c)
        for name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
            fa, fb = np.asarray(getattr(a, name)), np.asarray(getattr(b, name))
            assert np.linalg.norm(fa - c * fb) < tol * np.linalg.norm(fa), name


def test_bends_are_supported():
    assert _supported(bend_cross_section(5.0)) is None
    assert _supported(bend_cross_section(5.0, axis=0)) is None


@pytest.mark.parametrize("precision", ["double", "mixed"])
@pytest.mark.parametrize("axis", [1, 0])
@pytest.mark.parametrize("radius", [5.0, 10.0, 50.0])
def test_bend_modes_match_tidy3d(radius, axis, precision):
    """The two guided modes (TE0, TM0): neff and fields (all six components) of compute_modes_tidy3d."""
    cs, ref = _tidy3d(radius, axis, 2)
    straight = _tidy3d(None, 1, 2)[1]
    got = _gpu(cs, 2, precision)
    _assert_same(got, ref, 1e-8, 1e-6)
    _assert_same_components(got, ref, 1e-6)
    assert abs(got[0].neff - straight[0].neff) > 1e-4         # the bend changes the modes


def test_the_jacobian_changes_ez_and_hz():
    """Without the back-transform (Ez', Hz'), Ez and Hz differ from tidy3d by much more than the tolerance of
    test_bend_modes_match_tidy3d: so that test checks the back-transform."""
    from meow_gpu.meow_backend import _problem
    cs, ref = _tidy3d(5.0, 1, 2)
    got = _gpu(cs, 2, "double")
    a_e, a_h = _problem(cs).jac
    for a, b in zip(got, ref):
        c = np.vdot(np.ravel(b.Ex), np.ravel(a.Ex))
        c /= abs(c)
        for name, jac in (("Ez", a_e), ("Hz", a_h)):
            fa, fb = np.asarray(getattr(a, name)) / jac, np.asarray(getattr(b, name))
            assert np.linalg.norm(fa - c * fb) > 1e-3 * np.linalg.norm(fb), name


def test_bend_modes_single_precision():
    cs, ref = _tidy3d(5.0, 1, 2)
    got = _gpu(cs, 2, "single")
    _assert_same(got, ref, 2e-5, 1e-3)
    _assert_same_components(got, ref, 5e-3)


@pytest.mark.parametrize("radius,axis", [(5.0, 1), (50.0, 1), (5.0, 0)])
def test_bend_modes_are_the_eigenvalues_nearest_the_target(radius, axis):
    """4 modes: the eigenvalues of the tidy3d matrix (captured from tidy3d, ARPACK shift-invert, 8 values, tol
    1e-13) nearest the target, and residuals of the GPU eigenpairs with that matrix. Here compute_modes_tidy3d
    itself misses box modes at the outer wall (for 5 um, bend_axis 1: neff 1.6579, with its maximum at the corner of
    the PEC box), so this test does not use its mode list."""
    cs = bend_cross_section(radius, axis)
    mat = tidy3d_solver_matrix(cs.nx**2, cs.ny**2, cs.nz**2, cs.mesh.x, cs.mesh.y, cs.env.wl, bend_radius=radius,
                               bend_axis=axis, plane_center=(0.0, 0.0))
    sigma = -(N_SI * (1 + 10 * np.finfo(float).eps)) ** 2
    vals = spl.eigs(mat, k=8, sigma=sigma, tol=1e-13, return_eigenvectors=False)
    want = np.sort(np.sqrt(-vals.astype(complex))[np.argsort(np.abs(vals - sigma))[:4]].real)[::-1]
    got = _gpu(cs, 4, "double", post_process=_raw)
    assert np.max(np.abs(np.array([m.neff.real for m in got]) - want)) < 1e-9
    for m in got:
        u = np.concatenate([np.ravel(m.Ex), np.ravel(m.Ey)])
        lam = -(complex(m.neff) ** 2)
        assert np.linalg.norm(mat @ u - lam * u) < 1e-9 * abs(lam) * np.linalg.norm(u)


# A tight bend with PML in x: the fundamental TE mode radiates (Im neff > 0). The domain reaches past the radius
# where n_clad u / R = neff (the caustic), so that the radiation reaches the PML. Target 2.45 (near the TE0 mode):
# the outer cladding also gives PML modes with Re(neff) up to n_clad u_max / R, nearer the default target.
RADIATION = {"x": (-1.0, 2.4), "num_pml": (10, 0)}


@pytest.mark.parametrize("radius", [1.5, 2.0])
def test_radiation_loss_matches_tidy3d(radius):
    cs, ref = _tidy3d(radius, 1, 1, 2.45, **RADIATION)
    for precision in ("double", "mixed"):
        got = _gpu(cs, 1, precision, target=2.45)
        assert abs(got[0].neff - ref[0].neff) < 1e-9
        assert np.imag(got[0].neff) == pytest.approx(np.imag(ref[0].neff), rel=1e-4, abs=1e-11)
        assert abs(mw.inner_product(got[0], ref[0])) == pytest.approx(1.0, abs=1e-6)


def test_radiation_loss_falls_with_the_radius():
    loss = [np.imag(_gpu(bend_cross_section(r, 1, **RADIATION), 1, "double", target=2.45)[0].neff)
            for r in (1.5, 2.0, 3.0)]
    assert loss[0] > loss[1] > loss[2] > 0
    assert loss[0] > 1e-5 and loss[2] < 1e-8


def test_large_radius_tends_to_the_straight_modes():
    """The difference falls as 1/R (first order: tidy3d takes the Jacobian of all E components at the lower cell
    corner of the normal axis, half a cell from Ex)."""
    straight = _gpu(bend_cross_section(None), 2, "double")
    diffs = []
    for radius in (1e3, 1e4, 1e5):
        got = _gpu(bend_cross_section(radius), 2, "double")
        diffs.append(max(abs(a.neff - b.neff) for a, b in zip(got, straight)))
        for a, b in zip(got, straight):
            assert abs(mw.inner_product(a, b)) == pytest.approx(1.0, abs=1e-4 * 1e3 / radius)
    assert diffs[2] < 1e-6
    assert 8 < diffs[0] / diffs[1] < 12 and 8 < diffs[1] / diffs[2] < 12


def test_negative_radius_and_the_mirror_bend():
    """A negative radius (the centre of the bend on the +x side): the modes of tidy3d, with the tidy3d direction
    rule (Re(neff) > 0, Im(neff) = 0 here). The mirror bend (+5 um) has the same physics; the difference of the
    TE0 neff is the first-order discretization error of the transform (it halves with dx)."""
    cs, ref = _tidy3d(-5.0, 1, 2)
    got = _gpu(cs, 2, "double")
    _assert_same(got, ref, 1e-8, 1e-6)
    assert all(np.real(m.neff) > 0 and abs(np.imag(m.neff)) < 1e-10 for m in got)
    diff = {}
    for dx in (0.04, 0.02):
        plus, minus = (_gpu(bend_cross_section(r, 1, dx=dx), 1, "double")[0].neff for r in (5.0, -5.0))
        diff[dx] = abs(plus - minus)
    assert 5e-3 < diff[0.04] < 2e-2
    assert 1.7 < diff[0.04] / diff[0.02] < 2.4


def test_warm_start_only_from_the_same_kind():
    """The warm-start memory has one entry for straight cross-sections and one for each bend (radius, axis, and
    the shift R - plane_center of the normal axis)."""
    from meow_gpu.meow_backend import _problem
    s = GPUModeSolver(precision="double", extra=4)
    bend, straight = _problem(bend_cross_section(5.0)), _problem(bend_cross_section(None))
    s.solve(bend, 2)
    s.solve(straight, 2)
    assert not s.last_info.warm
    s.solve(_problem(bend_cross_section(5.0, axis=0)), 2)
    assert not s.last_info.warm
    s.solve(_problem(bend_cross_section(5.0, plane_center=(0.3, 0.0))), 2)      # another shift: another bend
    assert not s.last_info.warm
    s.solve(bend, 2)
    assert s.last_info.warm and s.last_info.iterations <= 2
    s.solve(_problem(bend_cross_section(5.0, plane_center=(0.0, 0.3))), 2)      # the same transform as ``bend``
    assert s.last_info.warm and s.last_info.iterations <= 2
    s.solve(straight, 2)
    assert s.last_info.warm and s.last_info.iterations <= 2


def _converged_without_warning(record, solver):
    assert not [w for w in record if "did not converge" in str(w.message)]
    assert all(i.converged for i in solver.last_batch_info)


@pytest.mark.parametrize("precision", ["double", "mixed"])
def test_batch_warm_start_keeps_each_kind(precision):
    """Review finding (2026-10-30): the batched warm start gave a member the vectors of another kind (a bend's
    modes to a straight cell); with 4 modes (box modes in the set) the straight member stopped at 500 iterations
    (residual 0.3, neff off by 0.09). Now each member starts only from vectors of its own kind, else cold."""
    from meow_gpu.meow_backend import _problem
    css = [bend_cross_section(5.0), bend_cross_section(None), bend_cross_section(None), bend_cross_section(5.0)]
    ref = {r: compute_modes_gpu(bend_cross_section(r), num_modes=4,
                                solver=GPUModeSolver(precision=precision, extra=4)) for r in (5.0, None)}
    s = GPUModeSolver(precision=precision, extra=4)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        got = compute_modes_gpu_batch(css, num_modes=4, solver=s, batch=2)      # batches (0, 2), (1, 3)
    _converged_without_warning(record, s)
    assert [i.warm for i in s.last_batch_info] == [True, True]                # each from its own kind
    for modes, r in zip(got, (5.0, None, None, 5.0)):
        _assert_same(modes, ref[r], 1e-8, 1e-6)
    # solve_batch: the per-cell memory of one kind, then batches with the kinds in both orders
    p5, ps = _problem(css[0]), _problem(css[1])
    want = {id(p): GPUModeSolver(precision=precision, extra=4).solve(p, 4)[0] for p in (p5, ps)}
    s = GPUModeSolver(precision=precision, extra=4)
    s.solve(p5, 4)
    for members, warm in (([p5, ps], [True, False]), ([ps, p5], [True, True])):
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            out = s.solve_batch(members, 4)
        _converged_without_warning(record, s)
        assert [i.warm for i in s.last_batch_info] == warm
        for p, (neff, _, _) in zip(members, out):
            assert np.max(np.abs(neff - want[id(p)])) < 1e-8


def test_a_bend_that_the_transform_cannot_represent_falls_back(monkeypatch):
    """bend_radius 0, or the centre of the bend inside the cross-section: compute_modes_gpu and
    compute_modes_gpu_batch use tidy3d for that cross-section (with the warning), as for angled propagation; the
    other cross-sections of the list stay on the GPU. tidy3d is replaced by a stub here (it does not fail on
    these inputs, but its transform is singular there)."""
    import meow
    ref = {r: _tidy3d(r, 1, 2)[1] for r in (5.0, None)}                      # before the stub
    calls = []
    monkeypatch.setattr(meow, "compute_modes_tidy3d", lambda cs, **kw: calls.append(cs) or ["tidy3d"])
    zero, inside = bend_cross_section(0.0), bend_cross_section(0.5)          # u from -0.7 to 1.7: centre inside
    assert "not 0" in _supported(zero) and "centre of the bend" in _supported(inside)
    with pytest.warns(UserWarning, match="centre of the bend"):
        assert compute_modes_gpu(inside, num_modes=2) == ["tidy3d"]
    css = [bend_cross_section(5.0), zero, bend_cross_section(None), inside]
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        got = compute_modes_gpu_batch(css, num_modes=2, solver=GPUModeSolver(precision="double", extra=4), batch=2)
    messages = [str(w.message) for w in record if "using the tidy3d CPU solver" in str(w.message)]
    assert len(messages) == 2 and "not 0" in messages[0] and "centre of the bend" in messages[1]
    assert got[1] == ["tidy3d"] and got[3] == ["tidy3d"]
    assert len(calls) == 3 and calls[1] is zero and calls[2] is inside
    for modes, r in ((got[0], 5.0), (got[2], None)):
        _assert_same(modes, ref[r], 1e-8, 1e-6)


def test_batch_with_bends_and_straight_cells():
    """Bends of three radii and a straight cell (one grid shape) in one batch: the per-cell results."""
    css = [bend_cross_section(5.0), bend_cross_section(None), bend_cross_section(10.0), bend_cross_section(50.0)]
    one = GPUModeSolver(precision="mixed", extra=4)
    ref = [compute_modes_gpu(cs, num_modes=4, solver=one) for cs in css]
    got = compute_modes_gpu_batch(css, num_modes=4, solver=GPUModeSolver(precision="mixed", extra=4), batch=4)
    for a_modes, b_modes in zip(got, ref):
        _assert_same(a_modes, b_modes, 1e-8, 1e-6)
