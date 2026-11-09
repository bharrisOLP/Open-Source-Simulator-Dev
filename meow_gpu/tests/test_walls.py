"""The unknowns of the PEC walls (README, "The modes of the PEC walls").

The operator of the tidy3d solver keeps Ex on the near y wall and Ey on the near x wall (tangential E on a PEC wall)
as unknowns, decoupled from all other unknowns. Their eigenpairs (1-D modes along the walls) are not modes of the
PEC waveguide; tidy3d excludes them (its start vector is zero there), and so does meow_gpu. Checks: the decoupling
in the assembled matrix and in the tidy3d matrix (straight, PML, PEC symmetry, bends), also in floating point in the
matrix-free operator; no wall unknowns with a PMC symmetry or along an axis of one cell; the wall eigenvalues equal
the 1-D Neumann formula (analytic), and they are the eigenvalues that the solver of 505c4c5 did not give (the values
and counts of benchmarks/2026-11-08/meow-solver-arpack.jsonl); the solver gives the eigenvalues nearest the target of
the other unknowns (ARPACK on the restricted matrix, an independent solver) at 40 and 60 modes on the 0.5 and 2.0 um
cells of the taper; its modes and the iterates of all paths have no wall components; windows that are uniform
except for a small change give the nearest eigenvalues (the shift of eps_ref, ``eigensolver.REF_SHIFT``)."""
import functools

import jax.numpy as jnp
import meow as mw
import numpy as np
import pytest
import scipy.sparse.linalg as spl
from structures import strip
from tidy3d_ref import tidy3d_matrix, tidy3d_solver_matrix

from meow_gpu.device import put
from meow_gpu.fde.eigensolver import davidson, davidson_batch, make_preconditioner, precondition, refine, stack
from meow_gpu.fde.operator import apply, assemble, make_problem, wall_unknowns
from meow_gpu.meow_backend import GPUModeSolver, _problem, _target, compute_modes_gpu

N_SI, N_SIO2 = 3.476, 1.444


def _random_case(seed=1, nx=23, ny=17):
    rng = np.random.default_rng(seed)
    x = np.cumsum(np.r_[0, rng.uniform(0.02, 0.05, nx)])
    y = np.cumsum(np.r_[0, rng.uniform(0.02, 0.05, ny)])
    eps = [rng.uniform(2, 12, (nx, ny)) + 1j * rng.uniform(0, 0.1, (nx, ny)) for _ in range(3)]
    return rng, eps, x, y


# (PML, symmetry, bend (radius, axis, plane centre) or None, walls (x wall: Ey[0, :], y wall: Ex[:, 0]))
CASES = [((0, 0), (0, 0), None, (True, True)),
         ((4, 3), (0, 0), None, (True, True)),
         ((0, 5), (0, -1), None, (True, True)),           # a PEC symmetry is a PEC wall
         ((4, 3), (1, 0), None, (False, True)),           # a PMC symmetry: no wall unknowns on that axis
         ((3, 3), (1, 1), None, (False, False)),
         ((0, 0), (0, 0), (5.0, 1, (0.4, 0.3)), (True, True)),
         ((4, 3), (0, 0), (1.2, 0, (0.4, 0.3)), (True, True))]


@pytest.mark.parametrize("pml,sym,bend,want", CASES)
def test_wall_unknowns_are_decoupled(pml, sym, bend, want):
    """A[W, R] = A[R, W] = 0 exactly, in our assembled matrix and in tidy3d's matrix (random media, non-uniform
    grid); the matrix-free operator keeps zero wall components at zero in complex64 and complex128 (the exact zeros
    of the stencils). With a PMC symmetry the first row is coupled, and it is not a wall row."""
    rng, eps, x, y = _random_case()
    kw = {} if bend is None else {"bend_radius": bend[0], "bend_axis": bend[1], "plane_center": bend[2]}
    p = make_problem(*eps, x, y, 1.55, pml, sym, **kw)
    walls = wall_unknowns(p)
    assert walls[1, 0, :].all() == want[0] and walls[0, :, 0].all() == want[1]
    assert walls.sum() == want[0] * p.shape[1] + want[1] * p.shape[0]
    ref = (tidy3d_matrix(*eps, x, y, 1.55, pml, sym) if bend is None else
           tidy3d_solver_matrix(*eps, x, y, 1.55, pml, sym, bend_radius=bend[0], bend_axis=bend[1],
                                plane_center=bend[2]))
    iw, ir = np.flatnonzero(walls.ravel()), np.flatnonzero(~walls.ravel())
    for a in (assemble(p).tocsr(), ref.tocsr()):
        if len(iw):
            assert abs(a[iw][:, ir]).max() == 0 and abs(a[ir][:, iw]).max() == 0
    first = np.zeros_like(walls)                        # the first rows that are not walls (PMC): coupled
    first[1, 0, :], first[0, :, 0] = not want[0], not want[1]
    if first.any():
        jf, jr = np.flatnonzero(first.ravel()), np.flatnonzero(~first.ravel())
        assert abs(ref.tocsr()[jf][:, jr]).max() > 0
    v = rng.normal(size=(3, 2) + p.shape) + 1j * rng.normal(size=(3, 2) + p.shape)
    v[:, walls] = 0
    for dtype in (jnp.complex64, jnp.complex128):
        av = np.asarray(apply(p.device(dtype), put(v, dtype)))
        assert np.all(av[:, walls] == 0) and np.all(np.isfinite(av))


def test_one_cell_axis_has_no_wall_unknowns():
    """A 1-D cross-section (one cell along y): no derivative along y, so Ex[:, 0] is all of Ex (the physical field
    of the slab), not a wall row. The x walls stay: Ey[0, 0] is decoupled (its equation is -eps_yy Ey)."""
    _, eps, x, _ = _random_case(nx=23, ny=1)
    p = make_problem(*eps, x, np.array([0.0, 0.5]), 1.55)
    walls = wall_unknowns(p)
    assert not walls[0].any() and walls[1, 0, 0] and walls.sum() == 1
    a = assemble(p).tocsr()
    iw, ir = np.flatnonzero(walls.ravel()), np.flatnonzero(~walls.ravel())
    assert abs(a[iw][:, ir]).max() == 0 and abs(a[ir][:, iw]).max() == 0
    assert abs(a[iw[0], iw[0]] + eps[1][0, 0]) < 1e-14


def taper_cell(width, step=0.02, wavelength=1.55):
    """A cell of the taper of bench/bench_interface.py (the cross-section of bench/solver_vs_arpack.py)."""
    structs = [mw.Structure(material=mw.IndexMaterial(name="SiO2", n=N_SIO2),
                            geometry=mw.Box(x_min=-3, x_max=3, y_min=-2, y_max=2, z_min=-1, z_max=2)),
               mw.Structure(material=mw.IndexMaterial(name="Si", n=N_SI), mesh_order=1,
                            geometry=mw.Box(x_min=-width / 2, x_max=width / 2, y_min=0, y_max=0.22, z_min=-1,
                                            z_max=2))]
    mesh = mw.Mesh2D(x=np.arange(-2.0, 2.0 + step / 2, step), y=np.arange(-0.8, 1.02 + step / 2, step))
    cell = mw.Cell(structures=structs, mesh=mesh, z_min=0.0, z_max=1.0)
    return mw.CrossSection.from_cell(cell=cell, env=mw.Environment(wl=wavelength, T=25.0))


def _blocks(cs):
    p = _problem(cs)
    a = assemble(p).astype(complex).tocsr()
    walls = wall_unknowns(p).ravel()
    iw, ir = np.flatnonzero(walls), np.flatnonzero(~walls)
    return p, a, walls, a[iw][:, iw].toarray(), a[ir][:, ir].tocsc()


# The eigenvalues that the solver of 505c4c5 did not give (benchmarks/2026-11-08/meow-solver-arpack.jsonl, the
# 0.5 and the 2.0 um cell, 40 to 120 modes; neff rounded to 5 decimals): above cutoff and below cutoff (Im neff).
LOGGED_ABOVE = [1.444, 1.444, 1.43094, 1.39104, 1.37979, 1.32187, 1.21849, 1.16624, 1.07105, 0.85716, 0.67428,
                0.49757]
LOGGED_BELOW = [0.56054, 0.90082, 0.97492, 1.28882, 1.56102, 1.56391]


def test_wall_modes_are_the_neumann_modes_that_505c4c5_missed():
    """The wall block of the 0.5 um cell (Ex on 200 cells along the y wall, Ey on 91 cells along the x wall, both in
    SiO2): its eigenvalues are -(n^2 - (2 sin(m pi / 2N) / (k0 dl))^2), m = 0 .. N - 1 (within 1e-11 of |target|).
    The 12 above cutoff are the 12 logged neff above cutoff, and the 6 logged neff below cutoff are wall modes too."""
    p, _, walls, a_ww, _ = _blocks(taper_cell(0.5))
    assert walls.sum() == 200 + 91
    lam = np.linalg.eigvals(a_ww)
    formula = []
    for dl in p.dl_f:
        n = len(dl)
        kt = 2 * np.sin(np.arange(n) * np.pi / (2 * n)) / (p.k0 * np.mean(dl))
        formula.append(-(N_SIO2**2 - kt**2))
    assert [int(np.sum(f < 0)) for f in formula] == [8, 4]     # above cutoff: m = 0 to 7 (y wall), 0 to 3 (x wall)
    formula = np.sort(np.concatenate(formula))
    scale = abs(_target(p, None))
    assert np.max(np.abs(np.sort(lam.real) - formula)) < 1e-11 * scale and np.max(np.abs(lam.imag)) < 1e-11 * scale
    neff = np.sqrt(-formula.astype(complex))
    above = np.sort(neff[formula < 0].real)[::-1]
    assert len(above) == 12 and np.max(np.abs(above - LOGGED_ABOVE)) < 5.1e-6
    below = np.abs(neff[formula > 0].imag)          # |Im neff| (the logged sign is that of the rounding of -lambda)
    assert all(np.min(np.abs(below - v)) < 5.1e-6 for v in LOGGED_BELOW)


@pytest.fixture(scope="module", params=[0.5, 2.0])
def taper_case(request):
    """A cell of the taper, the blocks of its matrix, and the 80 eigenvalues of the restricted matrix A[R, R] nearest
    the target (ARPACK shift-invert, complex128 LU)."""
    cs = taper_cell(request.param)
    p, a, walls, a_ww, a_rr = _blocks(cs)
    target = _target(p, None)
    lam_rest = spl.eigs(a_rr, k=80, sigma=target, which="LM", tol=1e-13, return_eigenvectors=False)
    return request.param, cs, p, a, walls, np.linalg.eigvals(a_ww), lam_rest, target


# The counts of eigenvalues that 505c4c5 did not give, of the 40 and 60 nearest the target (the logged rows).
LOGGED_MISSED = {(0.5, 40): 10, (0.5, 60): 13, (2.0, 40): 9, (2.0, 60): 12}


@pytest.mark.parametrize("modes", [40, 60])
def test_solver_gives_the_nearest_eigenvalues(taper_case, modes, monkeypatch):
    """The solver (precision "mixed", cold start) at 40 and 60 modes: each eigenvalue is an eigenvalue of A[R, R]
    (within 1e-9 of |target|), and the distances to the target are those of the ``modes`` nearest eigenvalues of
    A[R, R] (no missed eigenvalue; a cut through a complex pair or a degenerate pair has equal distances). The modes
    have exactly zero wall components and are eigenpairs of the full matrix. The eigenvalues of the full matrix are
    those of A[R, R] and of the wall block (A is block-diagonal): of its ``modes`` nearest, the number that are wall
    eigenvalues is the logged count of the eigenvalues that the solver of 505c4c5 missed. The search space of
    505c4c5 (``make_preconditioner(deflate_walls=False)``) gives the same eigenvalues: it also gave the nearest
    eigenvalues of A[R, R], and no wall mode."""
    import meow_gpu.meow_backend as backend
    width, cs, p, a, walls, lam_wall, lam_rest, target = taper_case
    scale = abs(target)
    solver = GPUModeSolver(precision="mixed")
    raw = compute_modes_gpu(cs, num_modes=modes, solver=solver, post_process=lambda ms: ms)
    assert solver.last_info.converged and len(raw) == modes
    lam = np.array([-complex(m.neff) ** 2 for m in raw])
    ref = lam_rest[np.argsort(np.abs(lam_rest - target), kind="stable")[:modes]]
    assert max(np.min(np.abs(lam_rest - v)) for v in lam) < 1e-9 * scale
    assert np.max(np.abs(np.sort(np.abs(lam - target)) - np.sort(np.abs(ref - target)))) < 1e-9 * scale
    monkeypatch.setattr(backend, "make_preconditioner", lambda q: make_preconditioner(q, deflate_walls=False))
    old_solver = GPUModeSolver(precision="mixed")
    old = compute_modes_gpu(cs, num_modes=modes, solver=old_solver, post_process=lambda ms: ms)
    lam_old = np.array([-complex(m.neff) ** 2 for m in old])
    assert old_solver.last_info.converged
    assert max(np.min(np.abs(lam_rest - v)) for v in lam_old) < 1e-9 * scale
    assert np.max(np.abs(np.sort(np.abs(lam_old - target)) - np.sort(np.abs(lam - target)))) < 1e-9 * scale
    for m, v in zip(raw, lam):
        u = np.stack([np.asarray(m.Ex), np.asarray(m.Ey)])
        assert np.all(u[walls.reshape(u.shape)] == 0)
        u = u.ravel()
        assert np.linalg.norm(a @ u - v * u) < 1e-9 * scale * np.linalg.norm(u)
    union = np.concatenate([lam_rest, lam_wall])
    is_wall = np.r_[np.zeros(len(lam_rest), bool), np.ones(len(lam_wall), bool)]
    nearest = np.argsort(np.abs(union - target), kind="stable")[:modes]
    assert np.abs(union[nearest[-1]] - target) < np.abs(lam_rest - target).max()    # lam_rest reaches far enough
    assert int(is_wall[nearest].sum()) == LOGGED_MISSED[(width, modes)]


# Windows that are uniform (n = 1.444) except for a small change (review of 2026-11-08, finding 1): a lossy cladding,
# one cell (the centre) with eps (1 + d) in all three components, eps_zz (1 + d) everywhere. Without the shift of
# eps_ref (``eigensolver.REF_SHIFT``) the solver did not converge for d = 1e-8 (lossy, eps_zz) and for one cell at
# 1e-9 to 1e-4, and it converged to a set without the physical mode at neff 1.2917 for one cell at 1e-3 and 1e-2 (and
# 1e-1 in single precision).
NEAR_UNIFORM = [("lossy", 1e-8), ("cell", 1e-9), ("cell", 1e-6), ("cell", 1e-4), ("cell", 1e-3), ("cell", 1e-2),
                ("cell", 1e-1), ("zz", 1e-8)]


@functools.cache
def _near_uniform(kind, d, n=1.444, nx=60, ny=41, dx=0.04):
    """The problem (60 x 41 cells of 0.04 um, 1.55 um) and the 16 eigenvalues of A[R, R] nearest the target
    (ARPACK shift-invert)."""
    x, y = np.arange(nx + 1) * dx, np.arange(ny + 1) * dx
    eps = [np.full((nx, ny), n**2, dtype=complex) for _ in range(3)]
    if kind == "lossy":
        eps = [e + 1j * d for e in eps]
    elif kind == "cell":
        for e in eps:
            e[nx // 2, ny // 2] *= 1 + d
    else:
        eps[2] = eps[2] * (1 + d)
    p = make_problem(*eps, x, y, 1.55)
    ir = np.flatnonzero(~wall_unknowns(p).ravel())
    target = _target(p, None)
    lam = spl.eigs(assemble(p).tocsr()[ir][:, ir].tocsc(), k=16, sigma=target, tol=1e-13, return_eigenvectors=False)
    return p, lam, target


@pytest.mark.parametrize("precision", ["single", "mixed", "double"])
@pytest.mark.parametrize("kind,d", NEAR_UNIFORM)
def test_near_uniform_window_gives_the_nearest_eigenvalues(kind, d, precision):
    """6 modes, cold start: the solver converges, and its eigenvalues are the 6 eigenvalues of A[R, R] nearest the
    target (sorted distances and values within 1e-9 of |target|; single precision 1e-4: the wrong sets of the review
    were off by 3.5e-3 to 1.1e-1)."""
    p, lam_rest, target = _near_uniform(kind, d)
    scale = abs(target)
    s = GPUModeSolver(precision=precision)
    neff, _, _ = s.solve(p, 6)
    assert s.last_info.converged, (kind, d, precision, s.last_info)
    lam = -np.asarray(neff, dtype=complex) ** 2
    ref = lam_rest[np.argsort(np.abs(lam_rest - target), kind="stable")[:6]]
    tol = (1e-4 if precision == "single" else 1e-9) * scale
    assert np.max(np.abs(np.sort(np.abs(lam - target)) - np.sort(np.abs(ref - target)))) < tol
    assert max(np.min(np.abs(lam_rest - v)) for v in lam) < tol


@pytest.fixture(scope="module")
def strip_problem():
    return make_problem(*strip(dx=0.04, dy=0.04, w_dom=2.4, h_dom=1.6), 1.55)


def test_strip_wall_modes(strip_problem):
    """The modes that ARPACK found before the 4th kept mode of the strip of tests/test_refine.py (README, "Known
    limits": neff 1.444 twice, 1.4074 and 1.3604) are the wall modes m = 0 and m = 1 of its two walls (the eigenvalues
    of the wall block, 4 decimals)."""
    p = strip_problem
    walls = wall_unknowns(p).ravel()
    iw = np.flatnonzero(walls)
    lam = np.linalg.eigvals(assemble(p).tocsr()[iw][:, iw].toarray())
    neff = np.sort(np.sqrt(-lam[lam.real < 0].astype(complex)).real)[::-1]
    assert np.max(np.abs(neff[:4] - [1.444, 1.444, 1.4074, 1.3604])) < 1e-4


def test_iterates_have_no_wall_components(strip_problem):
    """A start block with wall components (the cold start up to 505c4c5: a random block through the preconditioner
    of ``make_preconditioner(deflate_walls=False)``, which keeps them): davidson (single, mixed, double),
    davidson_batch and refine give vectors with exactly zero wall components, and the preconditioner gives zero
    there."""
    p = strip_problem
    walls = wall_unknowns(p)
    k, b = 4, 8
    target = _target(p, None)
    pc = make_preconditioner(p)
    full = make_preconditioner(p, deflate_walls=False)
    assert np.all(np.asarray(full.mask) == 1) and np.all(np.asarray(pc.mask)[walls] == 0)
    assert np.all(np.asarray(pc.mask)[~walls] == 1)
    th = put(np.full(b, target), jnp.complex64)
    rnd = put(np.random.default_rng(5).normal(size=(b, 2) + p.shape) + 0j, jnp.complex128)
    assert np.all(np.asarray(precondition(pc, rnd, th))[:, walls] == 0)
    x0 = precondition(full, rnd, th)
    assert np.all(np.asarray(x0)[:, walls] != 0)
    ops = {"single": p.device(jnp.complex64), "double": p.device(jnp.complex128),
           "mixed": (p.device(jnp.complex64), p.device(jnp.complex128))}
    for precision, op in ops.items():
        r = davidson(op, pc, k, target, x0, tol=1e-4 if precision == "single" else 1e-10, precision=precision)
        assert np.all(np.asarray(r.vectors)[:, walls] == 0), precision
        assert r.converged or precision == "single", precision
    r = refine(ops["mixed"][0], ops["mixed"][1], pc, k, target, x0, tol=1e-10, max_steps=3)
    assert np.all(np.asarray(r.vectors)[:, walls] == 0)
    rb = davidson_batch(stack([ops["double"]] * 2), stack([pc, pc]), k, target, jnp.stack([x0, x0]), tol=1e-10,
                        precision="double")
    assert rb.converged.all() and np.all(np.asarray(rb.vectors)[:, :, walls] == 0)
