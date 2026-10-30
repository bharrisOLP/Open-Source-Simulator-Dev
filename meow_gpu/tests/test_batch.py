"""Batched mode solves: B cross-sections of one grid shape at the same time give the results of the per-cell solve."""
import jax.numpy as jnp
import meow as mw
import numpy as np
import pytest
from structures import strip
from test_meow_backend import _below_cutoff_eme, strip_cross_section

from meow_gpu.device import put, solver_device
from meow_gpu.fde.eigensolver import STALL_FLOOR, davidson, davidson_batch, make_preconditioner, precondition, stack
from meow_gpu.fde.operator import make_problem
from meow_gpu.meow_backend import GPUModeSolver, compute_modes_gpu, compute_modes_gpu_batch

K = 4
# (width, wavelength): different eps, targets and 1-D preconditioner bases in one batch, with the same grid shape
MEMBERS = ((0.45, 1.55), (0.5, 1.50), (0.6, 1.60))


def _problem(width, wl):
    exx, eyy, ezz, x, y = strip(width=width, dx=0.04, dy=0.04, w_dom=2.4, h_dom=1.6)
    return make_problem(exx, eyy, ezz, x, y, wl)


@pytest.fixture(scope="module")
def problems():
    return [_problem(w, wl) for w, wl in MEMBERS]


ON_CPU = solver_device().platform == "cpu"


def _same_steps(a, b) -> bool:
    """Iteration counts of the batched and the per-cell solve. On the CPU the two do the same steps (up to rounding:
    +-1); a GPU can round the batched kernels differently, and the stall rule then stops at another iteration."""
    slack = 1 if ON_CPU else max(3, 0.25 * max(a, b))
    return abs(int(a) - int(b)) <= slack


def _assert_same_modes(a, b, ntol, ftol):
    """(neff, E, H) of two solves: the same neff, and the same E and H up to one complex phase per mode."""
    (na, ea, ha), (nb, eb, hb) = a, b
    assert na.shape == nb.shape
    assert np.max(np.abs(na - nb)) < ntol
    for i in range(len(na)):
        c = np.vdot(eb[i], ea[i])
        c /= abs(c)
        for fa, fb in ((ea[i], eb[i]), (ha[i], hb[i])):     # H with the phase of E: the same direction rule
            assert np.linalg.norm(fa - c * fb) < ftol * np.linalg.norm(fa)


# single: the fields of one complex64 solve have errors up to about 7e-4 (README), so two solves (batched and
# per-cell) can differ by twice that; on the RX 7900 XT the difference was 1.4e-3
@pytest.mark.parametrize("precision,ntol,ftol", [("double", 1e-9, 1e-6), ("mixed", 1e-9, 1e-6),
                                                 ("single", 2e-5, 3e-3)])
def test_batch_equals_per_cell(problems, precision, ntol, ftol):
    one = GPUModeSolver(precision=precision, extra=4, warm_start=False)
    ref, ref_info = [], []
    for p in problems:
        ref.append(one.solve(p, K))
        ref_info.append(one.last_info)
    s = GPUModeSolver(precision=precision, extra=4)
    got = s.solve_batch(problems, K)
    assert len(got) == len(problems)
    for a, b in zip(got, ref):
        _assert_same_modes(a, b, ntol, ftol)
    assert all(i.converged and not i.warm for i in s.last_batch_info)
    # each member does the steps of the per-cell solver (the same iterations, up to rounding). Not in single
    # precision on a GPU: there the stall rule stops near the complex64 floor, and a different rounding of the
    # batched kernels moves that stop by more than the slack of _same_steps. On the CPU: also the same matvecs.
    if precision != "single" or ON_CPU:
        assert all(_same_steps(i.iterations, r.iterations) for i, r in zip(s.last_batch_info, ref_info))
    if ON_CPU:
        assert [i.matvecs for i in s.last_batch_info] == [r.matvecs for r in ref_info]


def test_batch_of_one(problems):
    p = problems[1]
    one = GPUModeSolver(precision="double", extra=4)
    ref = one.solve(p, K)
    s = GPUModeSolver(precision="double", extra=4)
    (got,) = s.solve_batch([p], K)
    _assert_same_modes(got, ref, 1e-9, 1e-6)
    assert _same_steps(s.last_info.iterations, one.last_info.iterations)
    assert s._memory[(p.shape, K + 4, 1)].devices() == {solver_device()}    # the batched arrays stay on the device


def test_batch_warm_start(problems):
    """The next batch starts from the vectors of the previous batch; the per-cell memory gets the last member."""
    s = GPUModeSolver(precision="double", extra=4)
    s.solve_batch(problems, K)
    cold = [i.iterations for i in s.last_batch_info]
    got = s.solve_batch(problems, K)
    assert all(i.warm for i in s.last_batch_info)
    assert all(w < c / 2 for w, c in zip([i.iterations for i in s.last_batch_info], cold))
    for a, p in zip(got, problems):
        _assert_same_modes(a, GPUModeSolver(precision="double", extra=4).solve(p, K), 1e-9, 1e-6)
    s.solve(problems[-1], K)
    assert s.last_info.warm and s.last_info.iterations <= 2


def test_batch_shapes_must_match(problems):
    other = make_problem(*strip(dx=0.05, dy=0.05, w_dom=2.4, h_dom=1.6), 1.55)
    with pytest.raises(ValueError, match="same grid shape"):
        GPUModeSolver().solve_batch([problems[0], other], K)


@pytest.mark.parametrize("precision", ["double", "mixed"])
def test_ragged_convergence(problems, precision):
    """Members that converge at different iterations: member 0 starts from its own eigenvectors (it stops much
    earlier: 1 iteration in double; in mixed, the complex128 polish of the rounded vectors takes about 40), the
    others from random blocks. Each member gives the result of ``davidson`` from the same initial block, so a member
    that stopped did not change after its stop."""
    b = K + 4
    pcs = [make_preconditioner(p) for p in problems]
    targets = [-(np.sqrt(np.max(np.abs(p.eps_xx))) * (1 + 1e-14)) ** 2 for p in problems]
    if precision == "mixed":
        ops = [(p.device(jnp.complex64), p.device(jnp.complex128)) for p in problems]
        op_b = (stack([o[0] for o in ops]), stack([o[1] for o in ops]))
    else:
        ops = [p.device(jnp.complex128) for p in problems]
        op_b = stack(ops)
    rng = np.random.default_rng(3)
    x0 = [precondition(pc, put(rng.normal(size=(b, 2) + problems[0].shape) + 0j, jnp.complex64),
                       put(np.full(b, t), jnp.complex64)).astype(jnp.complex128) for pc, t in zip(pcs, targets)]
    x0[0] = davidson(ops[0], pcs[0], K, targets[0], x0[0], tol=1e-10, precision=precision).vectors
    ref = [davidson(o, pc, K, t, x, tol=1e-10, precision=precision) for o, pc, t, x in zip(ops, pcs, targets, x0)]
    got = davidson_batch(op_b, stack(pcs), K, np.array(targets), jnp.stack(x0), tol=1e-10, precision=precision)
    assert got.converged.all()
    assert min(got.iterations[1:]) - got.iterations[0] >= 20              # ragged: very different stops
    for i, r in enumerate(ref):
        assert r.converged
        assert _same_steps(got.iterations[i], r.iterations)
        assert np.max(np.abs(got.values[i] - r.values)) < 1e-9
        v, w = np.asarray(got.vectors[i, :K]), np.asarray(r.vectors[:K])
        cos = np.abs(np.sum(v.conj() * w, axis=(1, 2, 3))) / np.linalg.norm(v.reshape(K, -1), axis=1) \
            / np.linalg.norm(w.reshape(K, -1), axis=1)
        assert np.all(cos > 1 - 1e-12)
    assert got.member(0).iterations == got.iterations[0]


def test_iteration_limit_of_each_member(problems):
    """Each member has its own iteration limit: a member at its limit stops (not converged) as ``davidson`` with
    that limit, a member with no iterations keeps its initial block, and the other member converges."""
    b = K + 4
    pcs = [make_preconditioner(p) for p in problems]
    ops = [p.device(jnp.complex128) for p in problems]
    targets = [-(np.sqrt(np.max(np.abs(p.eps_xx))) * (1 + 1e-14)) ** 2 for p in problems]
    x = put(np.random.default_rng(0).normal(size=(b, 2) + problems[0].shape) + 0j, jnp.complex64)
    x0 = jnp.stack([precondition(pc, x, put(np.full(b, t), jnp.complex64)) for pc, t in zip(pcs, targets)])
    limits = (3, 300, 0)
    got = davidson_batch(stack(ops), stack(pcs), K, np.array(targets), x0, tol=1e-10, max_iter=np.array(limits),
                         precision="double")
    assert list(got.converged) == [False, True, False]
    assert got.iterations[0] == 3 and got.iterations[2] == 0
    assert np.array_equal(np.asarray(got.vectors[2]), np.asarray(x0[2].astype(jnp.complex128)))
    for i, n in enumerate(limits):
        ref = davidson(ops[i], pcs[i], K, targets[i], x0[i], tol=1e-10, max_iter=n, precision="double")
        assert _same_steps(ref.iterations, got.iterations[i]) and ref.converged == got.converged[i]
        assert np.max(np.abs(got.values[i] - ref.values)) < 1e-9
        if ON_CPU or i != 1:     # at the limit, both expand the subspace once more before the stop
            assert got.matvecs[i] == ref.matvecs


def test_stall_stop_of_each_member(problems):
    """Single precision with a tolerance below the complex64 floor: two members stop by the stall rule (not
    converged, residual below STALL_FLOOR), the third member converges to its own tolerance. Each member gives the
    result of ``davidson`` with the same tolerance."""
    b = K + 4
    pcs = [make_preconditioner(p) for p in problems]
    ops = [p.device(jnp.complex64) for p in problems]
    targets = [-(np.sqrt(np.max(np.abs(p.eps_xx))) * (1 + 1e-14)) ** 2 for p in problems]
    x = put(np.random.default_rng(1).normal(size=(b, 2) + problems[0].shape) + 0j, jnp.complex64)
    x0 = [precondition(pc, x, put(np.full(b, t), jnp.complex64)) for pc, t in zip(pcs, targets)]
    tols = np.array([1e-9, 1e-9, 1e-5])
    got = davidson_batch(stack(ops), stack(pcs), K, np.array(targets), jnp.stack(x0), tol=tols, precision="single")
    if ON_CPU:
        assert list(got.converged) == [False, False, True]
        assert np.all(got.residuals[:2].max(axis=1) < STALL_FLOOR) and got.residuals[2].max() < 1e-5
    else:   # on the RX 7900 XT the complex64 floor of this case is above 1e-5: all three stop by the stall rule
        assert np.all(got.residuals.max(axis=1) < STALL_FLOOR)
    for i, tol in enumerate(tols):
        ref = davidson(ops[i], pcs[i], K, targets[i], x0[i], tol=tol, precision="single")
        if ON_CPU:
            assert ref.converged == got.converged[i]
        assert np.max(np.abs(got.values[i] - ref.values)) < (1e-12 if ON_CPU else 1e-4)
        if ON_CPU:                           # on a GPU, other rounding moves the stall stop
            assert got.iterations[i] == ref.iterations and got.matvecs[i] == ref.matvecs


def _taper(n_cells=5, dx=0.05):
    oxide = mw.IndexMaterial(name="sio2", n=1.444)
    si = mw.IndexMaterial(name="si", n=3.476)
    clad = mw.Structure(material=oxide, geometry=mw.Box(x_min=-2, x_max=2, y_min=-2, y_max=2, z_min=-1, z_max=4))
    poly = np.array([[0.0, -0.25], [3.0, -0.5], [3.0, 0.5], [0.0, 0.25]])
    core = mw.Structure(material=si, mesh_order=1, geometry=mw.Prism(poly=poly, h_min=0.0, h_max=0.22, axis="y"))
    mesh = mw.Mesh2D(x=np.arange(-1.0, 1.0 + dx / 2, dx), y=np.arange(-0.6, 0.82 + dx / 2, dx))
    cells = mw.create_cells([clad, core], mesh, np.full(n_cells, 3.0 / n_cells))
    return [mw.CrossSection.from_cell(cell=c, env=mw.Environment(wl=1.55, T=25.0)) for c in cells]


@pytest.mark.parametrize("batch", [1, 2, 8])
def test_compute_modes_gpu_batch_equals_per_cell(batch):
    """A 5-cell taper; batch 2 gives the batches (0, 3), (1, 4), (2, 4*): the last member solves cell 4 again."""
    css = _taper()
    one = GPUModeSolver(precision="mixed", extra=4)
    ref = [compute_modes_gpu(cs, num_modes=K, solver=one) for cs in css]
    s = GPUModeSolver(precision="mixed", extra=4)
    got = compute_modes_gpu_batch(css, num_modes=K, solver=s, batch=batch)
    assert len(got) == len(css)
    for a_modes, b_modes in zip(got, ref):
        assert len(a_modes) == len(b_modes)
        for a, b in zip(a_modes, b_modes):
            assert abs(a.neff - b.neff) < 1e-8
            assert abs(mw.inner_product(a, b)) == pytest.approx(1.0, abs=1e-6)
    assert all(i.warm for i in s.last_batch_info) == (batch < len(css))


def test_compute_modes_gpu_batch_with_two_grid_shapes_and_a_fallback_cell():
    """Cells with two grid shapes and one bend cell (tidy3d fallback), in mixed order: the output keeps the order
    of the input, each cell gets the per-cell result, and the fallback warning points at the caller."""
    taper = _taper(n_cells=3)
    bend = strip_cross_section(bend_radius=20.0)
    css = [taper[0], strip_cross_section(width=0.5), taper[1], bend, strip_cross_section(width=0.6), taper[2]]
    assert len({tuple(np.shape(cs.nx)) for cs in css}) == 2
    with pytest.warns(UserWarning, match="bend radius") as record:
        got = compute_modes_gpu_batch(css, num_modes=K, solver=GPUModeSolver(precision="mixed", extra=4), batch=2)
    assert {w.filename for w in record if "bend radius" in str(w.message)} == {__file__}
    one = GPUModeSolver(precision="mixed", extra=4)
    with pytest.warns(UserWarning, match="bend radius"):
        ref = [compute_modes_gpu(cs, num_modes=K, solver=one) for cs in css]
    assert len(got) == len(css)
    for a_modes, b_modes in zip(got, ref):
        assert len(a_modes) == len(b_modes)
        for a, b in zip(a_modes, b_modes):
            assert np.shape(a.Ex) == np.shape(b.Ex)
            assert abs(a.neff - b.neff) < 1e-8
            assert abs(mw.inner_product(a, b)) == pytest.approx(1.0, abs=1e-6)


def test_batch_follows_the_tidy3d_direction_rule():
    """The below-cutoff case of test_meow_backend: the batched solve gives the S21 of tidy3d (no growing modes)."""
    ref, _ = _below_cutoff_eme(None)
    got, modes = _below_cutoff_eme(GPUModeSolver(precision="double", extra=4, warm_start=False), batch=2)
    assert all(np.imag(m.neff) >= -1e-9 for ms in modes for m in ms)
    assert got == pytest.approx(ref, abs=3e-6)
