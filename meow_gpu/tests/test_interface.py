"""meow_gpu.interface: the power-conserving EME interface (README, "A power-conserving interface").

Checks: a lossless step conserves power (S unitary) and is reciprocal (S = S^T), also through meow's SAX cascade;
meow's interface on the same modes does not conserve power; a step between equal guides is the identity; a step
between two uniform media gives the impedance reflection of the fundamental mode (analytic); with the complete set of
the discrete modes of a small grid the result satisfies the continuity of E_t and H_t at every sample (the exact
discrete solution) and equals meow's formula; the Yee product makes the eigenmodes orthogonal, meow's product does
not."""
import functools

import jax.numpy as jnp
import meow as mw
import numpy as np
import pytest

from meow_gpu.device import put
from meow_gpu.fde.operator import assemble, fields
from meow_gpu.interface import (compute_interface_s_matrices, compute_interface_s_matrix, interface_blocks,
                                overlap_matrix, post_process_eigenmodes, yee_inner_product)
from meow_gpu.meow_backend import GPUModeSolver, _orient, _problem, compute_modes_gpu, post_process_modes

N_SI, N_SIO2 = 3.476, 1.444


def cross_section(width, n_core=N_SI, n_clad=N_SIO2, wl=1.55, dx=0.04, x=(-1.2, 1.2), y=(-0.7, 0.92),
                  num_pml=(0, 0)):
    """A Si strip (0.22 um thick; width None: no core) in a uniform cladding, in a closed window."""
    structs = [mw.Structure(material=mw.IndexMaterial(name="clad", n=n_clad),
                            geometry=mw.Box(x_min=x[0] - 1, x_max=x[1] + 1, y_min=y[0] - 1, y_max=y[1] + 1,
                                            z_min=-1, z_max=2))]
    if width:
        structs.append(mw.Structure(material=mw.IndexMaterial(name="core", n=n_core), mesh_order=1,
                                    geometry=mw.Box(x_min=-width / 2, x_max=width / 2, y_min=0, y_max=0.22,
                                                    z_min=-1, z_max=2)))
    mesh = mw.Mesh2D(x=np.arange(x[0], x[1] + dx / 2, dx), y=np.arange(y[0], y[1] + dx / 2, dx), num_pml=num_pml)
    cell = mw.Cell(structures=structs, mesh=mesh, z_min=0.0, z_max=1.0)
    return mw.CrossSection.from_cell(cell=cell, env=mw.Environment(wl=wl, T=25.0))


@pytest.fixture(scope="module")
def step_raw():
    """The raw eigenmodes (no post-processing) of a 0.5 um and a 1.0 um strip, 8 modes each (all guided or box
    modes above cutoff in this window)."""
    solver = GPUModeSolver(precision="mixed", extra=4)
    return [compute_modes_gpu(cross_section(w), num_modes=8, solver=solver, post_process=lambda ms: ms)
            for w in (0.5, 1.0)]


def _sigma_max(s) -> float:
    return float(np.linalg.svd(np.asarray(s), compute_uv=False)[0])


def test_lossless_step_conserves_power_and_is_reciprocal(step_raw):
    left, right = (post_process_eigenmodes(m) for m in step_raw)
    assert len(left) == len(right) == 8
    assert all(abs(np.imag(m.neff)) < 1e-9 and np.real(m.neff) > 0 for ms in (left, right) for m in ms)
    s = np.asarray(compute_interface_s_matrix(left, right)[0])
    eye = np.eye(len(s))
    assert np.max(np.abs(s.conj().T @ s - eye)) < 1e-7           # unitary: power is conserved for every input
    assert np.max(np.abs(np.sum(np.abs(s) ** 2, axis=1) - 1.0)) < 1e-7
    assert np.max(np.abs(s - s.T)) < 1e-7                         # reciprocal
    r_ll, t_lr, t_rl, r_rr = interface_blocks(left, right)
    assert np.max(np.abs(t_rl - t_lr.T)) < 1e-7 and 0.5 < abs(t_lr[0, 0]) ** 2 < 1.0


def test_meow_interface_does_not_conserve_power_on_the_same_step(step_raw):
    """The cause, on the same eigenmodes: meow's interface (meow's product, the Gram-Schmidt modes of meow_gpu, no
    passivity rule) has a singular value above 1; meow's product does not make the eigenmodes orthogonal."""
    gs = [post_process_modes(m) for m in step_raw]
    s_meow = mw.compute_interface_s_matrix(gs[0], gs[1], passivity_method="none")[0]
    eig = [post_process_eigenmodes(m) for m in step_raw]
    s_ours = compute_interface_s_matrix(eig[0], eig[1])[0]
    assert _sigma_max(s_meow) > 1.0 + 1e-3
    assert abs(_sigma_max(s_ours) - 1.0) < 1e-7
    from meow.eme.interface import overlap_matrix as meow_overlap
    for modes in eig:
        g_yee = overlap_matrix(modes, modes)
        g_meow = meow_overlap(modes, modes, mw.inner_product)
        assert np.max(np.abs(g_yee - np.eye(len(modes)))) < 1e-7
        assert np.max(np.abs(g_meow - np.diag(np.diag(g_meow)))) > 1e-4


def test_cascade_through_meow_conserves_power(step_raw):
    """Narrow - wide - narrow with meow's propagation and SAX cascade: the power of the input mode goes into the kept
    modes on the two sides, without gain or loss."""
    left, right = (post_process_eigenmodes(m) for m in step_raw)
    import jax
    with jax.default_device(jax.devices("cpu")[0]):      # SAX with klujax runs on the CPU
        S, pm = mw.compute_s_matrix([left, right, left], cell_lengths=[1.0, 0.7, 1.0],
                                    interfaces_fn=compute_interface_s_matrices)
    S = np.asarray(S)
    for k in range(len(left)):
        p = np.sum(np.abs(S[pm[f"left@{k}"]]) ** 2)
        assert p == pytest.approx(1.0, abs=1e-7), k
    assert np.max(np.abs(S - S.T)) < 1e-7


def test_post_processing_removes_modes_that_break_the_interface(step_raw):
    """A copy of a mode, a backward copy (neff and H_t flipped: the branch rule flips it back to the forward mode,
    and the copy rule removes it), a mode with a NaN field and a zero mode are removed; the eight eigenmodes stay. A
    singular interface system (a mode twice) raises ValueError."""
    raw = step_raw[0]
    m = raw[3]
    copy = mw.Mode(neff=m.neff, cs=m.cs, Ex=2 * m.Ex, Ey=2 * m.Ey, Ez=2 * m.Ez, Hx=2 * m.Hx, Hy=2 * m.Hy,
                   Hz=2 * m.Hz)
    backward = mw.Mode(neff=-m.neff, cs=m.cs, Ex=m.Ex, Ey=m.Ey, Ez=-m.Ez, Hx=-m.Hx, Hy=-m.Hy, Hz=m.Hz)
    nan = mw.Mode(neff=m.neff, cs=m.cs, Ex=np.full(m.Ex.shape, np.nan), Ey=m.Ey, Ez=m.Ez, Hx=m.Hx, Hy=m.Hy, Hz=m.Hz)
    zero = mw.Mode(neff=m.neff, cs=m.cs, **{c: np.zeros(m.Ex.shape) for c in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")})
    got = post_process_eigenmodes(raw + [copy, backward, nan, zero])
    ref = post_process_eigenmodes(raw)
    assert len(ref) == 8 and len(got) == 8
    assert [m.neff for m in got] == [m.neff for m in ref]
    left = post_process_eigenmodes(raw)
    with pytest.raises(ValueError, match="singular|not finite"):
        interface_blocks(left + [left[0]], left + [left[0]])


def test_star_cascade_equals_the_sax_cascade_of_meow(step_raw):
    """compute_s_matrix (the star product; memory of the order of k^2) gives the S-matrix and the port map of
    meow's SAX cascade, with the conserving interface and with meow's interface (narrow, wide, narrow), also with
    meow's interface without its symmetrization (a non-symmetric S)."""
    import jax

    from meow_gpu.interface import compute_s_matrix
    eig = [post_process_eigenmodes(m) for m in step_raw]
    gs = [post_process_modes(m) for m in step_raw]
    lengths = [1.0, 0.7, 1.0]
    nonsym = functools.partial(mw.compute_interface_s_matrices, enforce_reciprocity=False)
    for modes, fn in (([eig[0], eig[1], eig[0]], compute_interface_s_matrices),
                      ([gs[0], gs[1], gs[0]], mw.compute_interface_s_matrices),
                      ([gs[0], gs[1], gs[0]], nonsym)):     # a non-symmetric S checks the in/out port order
        with jax.default_device(jax.devices("cpu")[0]):
            ref, pm_ref = mw.compute_s_matrix(modes, cell_lengths=lengths, interfaces_fn=fn)
        got, pm = compute_s_matrix(modes, lengths, interfaces_fn=fn)
        assert pm == pm_ref
        assert np.max(np.abs(np.asarray(ref) - got)) < 1e-10
    asym = max(np.max(np.abs(np.asarray(sm) - np.asarray(sm).T)) for sm, _ in nonsym([gs[0], gs[1]]).values())
    assert asym > 1e-6


def test_equal_guides_give_the_identity(step_raw):
    left = post_process_eigenmodes(step_raw[0])
    s = np.asarray(compute_interface_s_matrix(left, left)[0])
    n = len(left)
    expected = np.block([[np.zeros((n, n)), np.eye(n)], [np.eye(n), np.zeros((n, n))]])
    assert np.max(np.abs(s - expected)) < 1e-12


def test_uniform_media_step_gives_the_impedance_reflection():
    """A window filled with n = 1.444, then with n = 2.0 (no core): the transverse eigenvectors are the same, so a
    mode couples only to the modes of its own transverse profile. On this grid the two first modes are a degenerate
    pair with neff = n (uniform Ex and uniform Ey: the walls of the discrete operator allow them; H_t = n z x E_t /
    eta0), so r = (n1 - n2) / (n1 + n2) and |t|^2 = 1 - r^2 (the Fresnel values at normal incidence). The eigensolver
    gives any basis of the pair, so the check is on the 2 x 2 block: R = r I and T^H T = (1 - r^2) I; the fields of
    the pair are real after the post-processing."""
    solver = GPUModeSolver(precision="mixed", extra=4)
    a, b = (compute_modes_gpu(cross_section(None, n_clad=n), num_modes=4, solver=solver,
                              post_process=post_process_eigenmodes) for n in (1.444, 2.0))
    assert max(abs(a[k].neff - 1.444) for k in (0, 1)) < 1e-9 and max(abs(b[k].neff - 2.0) for k in (0, 1)) < 1e-9
    r_ll, t_lr, _, _ = interface_blocks(a, b)
    r = (1.444 - 2.0) / (1.444 + 2.0)
    assert np.max(np.abs(r_ll[:2, :2] - r * np.eye(2))) < 1e-8
    t = t_lr[:2, :2]
    assert np.max(np.abs(t.conj().T @ t - (1.0 - r**2) * np.eye(2))) < 1e-8
    assert np.max(np.abs(r_ll[2:, :2])) < 1e-7 and np.max(np.abs(t_lr[2:, :2])) < 1e-7
    for m in a[:2] + b[:2]:
        for c in ("Ex", "Ey", "Hx", "Hy"):
            f = np.asarray(getattr(m, c))
            assert np.max(np.abs(f.imag)) < 1e-7 * max(np.max(np.abs(m.Ex)), np.max(np.abs(m.Hx)), 1e-30) + 1e-30


def _raw_all_modes(cs):
    """Every eigenmode of the discrete operator (dense eig of the assembled matrix) with the direction rule of the
    solver (``_orient``), not normalized."""
    p = _problem(cs)
    lam, v = np.linalg.eig(assemble(p).toarray())
    neff = np.sqrt(-lam.astype(np.complex128))
    nx, ny = p.shape
    e, h = fields(p.device(jnp.complex128), put(v.T.reshape(-1, 2, nx, ny)), put(neff))
    neff, e, h = _orient(p, neff, np.array(e), np.array(h))
    return neff, [mw.Mode(cs=cs, neff=complex(neff[i]), Ex=e[i, 0], Ey=e[i, 1], Ez=e[i, 2], Hx=h[i, 0],
                          Hy=h[i, 1], Hz=h[i, 2]) for i in range(len(neff))]


def _all_modes(cs):
    """Every eigenmode, normalized in the Yee product, without a filter and without a change of branch."""
    from meow.mode import normalize, zero_phase
    neff, modes = _raw_all_modes(cs)
    return neff, [normalize(zero_phase(m), yee_inner_product) for m in modes]


def _stack_fields(modes, names):
    return np.stack([np.concatenate([np.asarray(getattr(m, c)).ravel() for c in names]) for m in modes], 1)


def _continuity_errors(a, b, js=(0, 1, 5)):
    """The largest E_t and H_t continuity errors (relative to the field maxima) for the left incidence of modes js."""
    r_ll, t_lr, _, _ = interface_blocks(a, b)
    ea, ha = _stack_fields(a, ("Ex", "Ey")), _stack_fields(a, ("Hx", "Hy"))
    eb, hb = _stack_fields(b, ("Ex", "Ey")), _stack_fields(b, ("Hx", "Hy"))
    err_e = err_h = 0.0
    for j in js:
        inc = np.eye(len(a))[:, j]
        err_e = max(err_e, np.max(np.abs(ea @ (inc + r_ll[:, j]) - eb @ t_lr[:, j])) /
                    (np.max(np.abs(ea)) + np.max(np.abs(eb))))
        err_h = max(err_h, np.max(np.abs(ha @ (inc - r_ll[:, j]) - hb @ t_lr[:, j])) /
                    (np.max(np.abs(ha)) + np.max(np.abs(hb))))
    return err_e, err_h


def test_complete_set_gives_the_exact_discrete_solution():
    """A 0.2 um and a 0.4 um strip on an 8 x 6 grid (96 modes each, most below cutoff): with all modes, the
    interface solution satisfies the continuity of E_t and H_t at every sample, and meow's formula (with the same
    product, no TSVD truncation and no passivity rule) gives the same S-matrix."""
    kw = dict(dx=0.1, x=(-0.4, 0.4), y=(-0.3, 0.3))
    (n_a, a), (n_b, b) = _all_modes(cross_section(0.2, **kw)), _all_modes(cross_section(0.4, **kw))
    assert len(a) == len(b) == 96 and min(np.min(np.abs(n_a)), np.min(np.abs(n_b))) > 1e-2
    r_ll, t_lr, t_rl, r_rr = interface_blocks(a, b)

    def stack(modes, names):
        return np.stack([np.concatenate([np.asarray(getattr(m, c)).ravel() for c in names]) for m in modes], 1)

    ea, ha, eb, hb = stack(a, ("Ex", "Ey")), stack(a, ("Hx", "Hy")), stack(b, ("Ex", "Ey")), stack(b, ("Hx", "Hy"))
    scale = np.max(np.abs(ea)) + np.max(np.abs(eb))
    for j in (0, 1, 5):                       # left incidence of mode j: E_t and H_t continuous at z = 0
        inc = np.eye(len(a))[:, j]
        assert np.max(np.abs(ea @ (inc + r_ll[:, j]) - eb @ t_lr[:, j])) < 1e-8 * scale
        assert np.max(np.abs(ha @ (inc - r_ll[:, j]) - hb @ t_lr[:, j])) < 1e-8 * (np.max(np.abs(ha)) +
                                                                                     np.max(np.abs(hb)))
    s_ours = np.block([[r_ll, t_rl], [t_lr, r_rr]])
    s_meow = np.asarray(mw.compute_interface_s_matrix(a, b, inner_product=yee_inner_product, conjugate=False,
                                                      tsvd_rcond=1e-15, passivity_method="none",
                                                      enforce_reciprocity=False)[0])
    assert np.max(np.abs(s_ours[:, 0] - s_meow[:, 0])) < 1e-8
    assert np.max(np.abs(s_ours - s_ours.T)) < 1e-8          # reciprocal also with the modes below cutoff


def test_unsupported_grids_raise():
    solver = GPUModeSolver(precision="mixed", extra=4)
    a = compute_modes_gpu(cross_section(0.5, dx=0.08), num_modes=2, solver=solver, post_process=lambda ms: ms)
    b = compute_modes_gpu(cross_section(0.5, dx=0.1), num_modes=2, solver=solver, post_process=lambda ms: ms)
    with pytest.raises(ValueError, match="same grid"):
        overlap_matrix(a, b)
    pml = cross_section(0.5, dx=0.08, num_pml=(4, 4))
    m = mw.Mode(cs=pml, neff=a[0].neff, Ex=np.zeros(pml.nx.shape), Ey=np.zeros(pml.nx.shape),
                Ez=np.zeros(pml.nx.shape), Hx=np.zeros(pml.nx.shape), Hy=np.zeros(pml.nx.shape),
                Hz=np.zeros(pml.nx.shape))
    with pytest.raises(ValueError, match="PML"):
        overlap_matrix([m], [m])
    with pytest.raises(ValueError, match="PML"):           # the right side is checked too (review finding)
        overlap_matrix(a[:1], [m])


@pytest.mark.parametrize("core", [(0.2, 0.4), ("uniform", "uniform")])
def test_post_processing_keeps_the_complete_set(core):
    """The documented path: post_process_eigenmodes on all 96 modes of the 8 x 6 grid keeps all of them (the members
    with Re(neff) < 0 of complex pairs, the modes flipped from Im(neff) < 0, the degenerate clusters below cutoff of
    a uniform window), all decay in +z, and the interface gives the exact discrete solution (E_t and H_t continuous
    within 1e-8). Review finding (2026-11-08): the first version removed 24 and 30 modes here (6.6 % and 17 %
    continuity errors)."""
    kw = dict(dx=0.1, x=(-0.4, 0.4), y=(-0.3, 0.3))
    if core[0] == "uniform":
        sides = [cross_section(None, n_clad=n, **kw) for n in (1.444, 2.0)]
    else:
        sides = [cross_section(w, **kw) for w in core]
    raws = [_raw_all_modes(cs)[1] for cs in sides]
    a, b = (post_process_eigenmodes(r) for r in raws)
    assert len(a) == len(b) == 96
    assert all(np.imag(m.neff) >= -1e-9 * abs(m.neff) for m in a + b)
    if core[0] != "uniform":
        assert any(np.real(m.neff) < -1e-6 and np.imag(m.neff) > 1e-6 for m in a + b), "complex pairs expected"
        assert any(np.imag(m.neff) < -1e-6 for m in raws[0] + raws[1]), "the case needs flipped modes"
    err_e, err_h = _continuity_errors(a, b)
    assert err_e < 1e-8 and err_h < 1e-8


def test_max_imag_neff_removes_the_strongly_evanescent_modes():
    kw = dict(dx=0.1, x=(-0.4, 0.4), y=(-0.3, 0.3))
    raw = _raw_all_modes(cross_section(0.2, **kw))[1]
    got = post_process_eigenmodes(raw, max_imag_neff=1.0)
    assert len(got) == sum(abs(np.imag(m.neff)) <= 1.0 for m in raw) < len(raw)
    assert all(abs(np.imag(m.neff)) <= 1.0 for m in got)


def test_real_orthonormal_keeps_a_cluster_without_a_real_basis(step_raw):
    """Two complex combinations of four different modes span no subspace with a real basis: the cluster stays as it
    is (the residual check of _real_orthonormal)."""
    from meow_gpu.interface import _combine, _real_orthonormal
    raw = step_raw[0]
    c = np.array([[1, 0], [1j, 0], [0, 1], [0, 1j]], complex)
    pair = _combine([raw[0], raw[1], raw[2], raw[3]], c)
    assert _real_orthonormal(pair) is pair


def test_step_with_different_mode_counts_conserves_power(step_raw):
    """8 modes on the left, 5 on the right: the channels without a partner reflect fully; S stays unitary and
    symmetric."""
    left, right = (post_process_eigenmodes(m) for m in step_raw)
    s = np.asarray(compute_interface_s_matrix(left, right[:5])[0])
    assert s.shape == (13, 13)
    assert np.max(np.abs(s.conj().T @ s - np.eye(13))) < 1e-7 and np.max(np.abs(s - s.T)) < 1e-7


def test_solver_modes_near_cutoff_are_eigenpairs():
    """The first cell of the taper of bench/bench_interface.py (0.5 um strip, window x -2 to 2, y -0.8 to 1.02 um,
    0.02 um grid), 60 modes (precision "mixed"). Review finding (2026-11-08): the solver does not return copies or
    false modes there. Each mode is an eigenpair of the assembled operator (relative residual below 1e-9; its
    eigenvalue is an ARPACK eigenvalue, shift-invert at the target); no mode is a copy of another; the conjugate
    eigenvalue of each complex mode is also an ARPACK eigenvalue (a complex pair of the real operator).
    post_process_eigenmodes keeps all 60 modes, and all decay in +z. The solver does not give the 60 eigenvalues
    nearest the target (it misses some modes above cutoff; bench/solver_vs_arpack.py gives the counts)."""
    import scipy.sparse.linalg as sla

    from meow_gpu.interface import _is_copy
    from meow_gpu.meow_backend import _target
    cs = cross_section(0.5, dx=0.02, x=(-2.0, 2.0), y=(-0.8, 1.02))
    raw = compute_modes_gpu(cs, num_modes=60, solver=GPUModeSolver(precision="mixed"), post_process=lambda ms: ms)
    p = _problem(cs)
    a_op = assemble(p).astype(complex)
    target = _target(p, None)
    scale = abs(target)
    lam_ref = sla.eigs(a_op, k=100, sigma=target, which="LM", return_eigenvectors=False)
    for m in raw:
        lam = -complex(m.neff) ** 2
        u = np.concatenate([np.asarray(m.Ex).ravel(), np.asarray(m.Ey).ravel()])
        assert np.linalg.norm(a_op @ u - lam * u) < 1e-9 * scale * np.linalg.norm(u), m.neff
        assert np.min(np.abs(lam_ref - lam)) < 1e-9 * scale, m.neff
        if abs(np.imag(m.neff)) > 1e-9 * abs(m.neff) and abs(np.real(m.neff)) > 1e-9 * abs(m.neff):
            assert np.min(np.abs(lam_ref - np.conj(lam))) < 1e-9 * scale, m.neff
    assert not any(_is_copy(raw[i], raw[j], 1e-9) for i in range(len(raw)) for j in range(i))
    got = post_process_eigenmodes(raw)
    assert len(got) == 60 and all(np.imag(m.neff) >= -1e-9 * abs(m.neff) for m in got)
