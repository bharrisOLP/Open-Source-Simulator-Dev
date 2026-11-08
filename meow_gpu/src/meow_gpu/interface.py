"""An EME interface S-matrix that conserves power: a drop-in for ``meow.compute_interface_s_matrices``.

Use it with the modes of ``post_process_eigenmodes`` (no Gram-Schmidt):

    modes = [compute_modes_gpu(cs, num_modes=20, post_process=post_process_eigenmodes) for cs in css]
    S, port_map = mw.compute_s_matrix(modes, cells=cells, interfaces_fn=compute_interface_s_matrices)

Mode matching at a step (written from the field equations; README, "A power-conserving interface"). On the left
of the interface z = z0 the field is sum_k (a_k + b_k) e_k for E_t and sum_k (a_k - b_k) h_k for H_t (forward
amplitudes a, backward b, modes e_k, h_k of the left guide); on the right the same with the modes of the right guide
and amplitudes c (forward) and d (backward). Both E_t and H_t are continuous. The bilinear form is the unconjugated
reciprocity product of the Yee grid,

    <m1, m2> = 1/2 sum (E1x H2y dA_x - E1y H2x dA_y),

with the area of each sample of the Yee cell (dA_x = dl_f(x) dl_b(y) at the Ex and Hy samples, dA_y = dl_b(x)
dl_f(y) at the Ey and Hx samples; dl_f: the cell widths, dl_b: the distances between cell centres; the weights of
the Poynting flux of the tidy3d solver and of ``meow_backend._orient``). Two eigenmodes of the discrete operator with
different propagation constants are exactly orthogonal in this form (discrete reciprocity), so it gives Gram matrices
that are diagonal to the solver tolerance; meow's own product (trapezoidal rule on the nodes) does not. With the
overlap matrix P[i, j] = <e^L_i, h^R_j> and the Gram matrices G_L, G_R:

- the E_t equation, tested with the H fields of the right modes:   P^T (a + b) = G_R^T (c + d)
- the H_t equation, tested with the E fields of the left modes:    G_L (a - b) = P (c - d)

(the choice of test fields of the conservation-of-complex-power technique, which matches E_t and takes the H_t
equation from the complex power: Safavi-Naini and MacPhie, IEEE Trans. Microw. Theory Tech. 29, 337 (1981); here
with the unconjugated product, which is the complex power for real fields). For one excitation at a time this is one
square system, K [R; T] = rhs, with K = [[P^T, -G_R^T], [G_L, P]]. With G = I and a real P (lossless guides,
propagating modes with real fields) the result is, in the singular basis P = U diag(s) V^T, a set of independent
channels with r = (1 - s^2) / (1 + s^2) and t = 2 s / (1 + s^2): r^2 + t^2 = 1 for every s, so the S-matrix is
orthogonal (power is conserved) and symmetric (reciprocal) for every truncation of the two mode sets. For the
complete set of the discrete modes the projections are exact, so the method gives the exact discrete solution (the
tests check the continuity of E_t and H_t at every sample).

meow's interface (meow 0.15, ``compute_interface_s_matrix``) tests both equations with the left modes:
(O_LR + O_RL^T) T = 2 I. It gives the exact solution for the complete set too, but with a truncated set its S-matrix
is not unitary: with 10 to 40 modes the power of the 0.5 to 2.0 um taper of photonsuite is 1.008 to 1.012 of the
input, and meow corrects that afterwards (TSVD and a passivity rule).

Limits: straight guides without PML only (the product of a bend or of a PML window needs the coordinate stretch
of the solver; ``ValueError``). Modes below cutoff (complex neff) carry no power; for them S is the matrix of the
unconjugated amplitudes. With such modes in a truncated set, P is complex and the proof above does not hold: the
power on the ports of the modes above cutoff is then not exactly conserved (README, the rows above 40 modes).
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np


def _yee_areas(mode) -> tuple[np.ndarray, np.ndarray]:
    """dA_x (at the Ex and Hy samples) and dA_y (at the Ey and Hx samples) of the grid of ``mode``."""
    mesh = mode.cs.mesh
    radius = getattr(mesh, "bend_radius", None)
    if radius is not None and np.isfinite(radius):
        raise ValueError("meow_gpu.interface: bends are not supported (use meow's interface)")
    if getattr(mesh, "angle_theta", 0.0) or getattr(mesh, "angle_phi", 0.0):
        raise ValueError("meow_gpu.interface: angled propagation is not supported (use meow's interface)")
    if tuple(getattr(mesh, "num_pml", (0, 0))) != (0, 0):
        raise ValueError("meow_gpu.interface: PML windows are not supported (use meow's interface)")
    dl = []
    for c in (np.asarray(mesh.x, float), np.asarray(mesh.y, float)):
        d = np.diff(c)
        dl.append((d, np.concatenate([d[:1], 0.5 * (d[:-1] + d[1:])])))
    (fx, bx), (fy, by) = dl
    return np.outer(fx, by), np.outer(bx, fy)


def _stack(modes, names: tuple[str, str]) -> tuple[np.ndarray, np.ndarray]:
    return tuple(np.stack([np.asarray(getattr(m, n), np.complex128).reshape(-1) for m in modes]) for n in names)


def _same_grid(modes1, modes2) -> None:
    m1, m2 = modes1[0].cs.mesh, modes2[0].cs.mesh
    if (len(m1.x) != len(m2.x) or len(m1.y) != len(m2.y) or not np.allclose(m1.x, m2.x, rtol=0, atol=1e-12)
            or not np.allclose(m1.y, m2.y, rtol=0, atol=1e-12)):
        raise ValueError("meow_gpu.interface: the two mode sets must have the same grid")


def overlap_matrix(modes1, modes2) -> np.ndarray:
    """M[i, j] = <modes1[i], modes2[j]> in the Yee product (module docstring): 1/2 sum E1 x H2 . z dA."""
    if not len(modes1) or not len(modes2):
        return np.zeros((len(modes1), len(modes2)), np.complex128)
    _same_grid(modes1, modes2)
    _yee_areas(modes2[0])                         # the checks (bend, angle, PML) for the right side too
    ax, ay = (a.reshape(-1) for a in _yee_areas(modes1[0]))
    ex, ey = _stack(modes1, ("Ex", "Ey"))
    hx, hy = _stack(modes2, ("Hx", "Hy"))
    return 0.5 * ((ex * ax) @ hy.T - (ey * ay) @ hx.T)


def yee_inner_product(mode1, mode2) -> complex:
    """<mode1, mode2> in the Yee product (unconjugated); a callable for ``meow.mode.normalize``."""
    return complex(overlap_matrix([mode1], [mode2])[0, 0])


_FIELDS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")


def _combine(modes, c: np.ndarray) -> list:
    """The modes sum_k modes[k] c[k, j] (j = 0 .. c.shape[1] - 1), each with the neff of modes[0]."""
    import meow as mw

    f = {n: np.stack([np.asarray(getattr(m, n), np.complex128) for m in modes], -1) @ c for n in _FIELDS}
    return [mw.Mode(neff=modes[0].neff, cs=modes[0].cs, **{n: f[n][..., j] for n in _FIELDS})
            for j in range(c.shape[1])]


def _real_orthonormal(cluster: list) -> list:
    """A degenerate cluster of propagating modes (one neff): the eigensolver returns any basis of the subspace, often
    with complex coefficients, so the fields are not real and the unconjugated normalization is not the power. Take
    the real basis of the subspace (from E_t) and make it orthonormal in the Yee product (symmetric orthonormalization).
    The modes stay eigenmodes. A cluster without a real basis (the residual check) stays as it is."""
    x = np.concatenate([np.stack([np.asarray(getattr(m, n), np.complex128).ravel() for m in cluster], 1)
                        for n in ("Ex", "Ey")])
    k = x.shape[1]
    y = np.linalg.svd(np.concatenate([x.real, x.imag], 1), full_matrices=False)[0][:, :k]
    c = np.linalg.lstsq(x, y, rcond=None)[0]
    if np.linalg.norm(x @ c - y) > 1e-6 * np.linalg.norm(y):
        return cluster
    real = _combine(cluster, c)
    g = overlap_matrix(real, real)
    w, v = np.linalg.eigh(0.5 * (g + g.T).real)
    if np.max(np.abs(g.imag)) > 1e-6 * np.max(np.abs(g)) or np.min(w) <= 0:
        return cluster
    return _combine(cluster, c @ (v / np.sqrt(w)) @ v.T)


def _flip(mode):
    """The same transverse profile on the other branch: neff, E_z, H_x and H_y change sign (the flip of the direction
    rule of ``meow_backend._orient``)."""
    import meow as mw

    return mw.Mode(neff=-mode.neff, cs=mode.cs, Ex=mode.Ex, Ey=mode.Ey, Ez=-np.asarray(mode.Ez),
                   Hx=-np.asarray(mode.Hx), Hy=-np.asarray(mode.Hy), Hz=mode.Hz)


def _wrong_branch(mode) -> bool:
    """Below cutoff: Im(neff) < 0 (the mode grows in +z). Above cutoff: the power flow 1/2 Re sum(E x H*) . z < 0."""
    if abs(np.imag(mode.neff)) > 1e-9 * abs(mode.neff):
        return bool(np.imag(mode.neff) < 0)
    ax, ay = _yee_areas(mode)
    ex, ey, hx, hy = (np.asarray(getattr(mode, n)) for n in ("Ex", "Ey", "Hx", "Hy"))
    return bool(np.real(np.sum(ex * np.conj(hy) * ax - ey * np.conj(hx) * ay)) < 0)


def _is_copy(a, b, degenerate_tol: float) -> bool:
    """b is a copy of a: the same neff (to degenerate_tol) and parallel E_t fields (cosine above 1 - 1e-6). Two
    different modes of a degenerate cluster are not parallel, so the rule does not remove them."""
    if abs(a.neff - b.neff) >= degenerate_tol * abs(a.neff):
        return False
    ea = np.concatenate([np.asarray(a.Ex).ravel(), np.asarray(a.Ey).ravel()])
    eb = np.concatenate([np.asarray(b.Ex).ravel(), np.asarray(b.Ey).ravel()])
    return bool(abs(np.vdot(ea, eb)) > (1.0 - 1e-6) * np.linalg.norm(ea) * np.linalg.norm(eb))


def post_process_eigenmodes(modes, max_imag_neff: float | None = None, degenerate_tol: float = 1e-9):
    """The modes for ``compute_interface_s_matrix``. No Gram-Schmidt: the modes stay eigenmodes (with their own neff),
    and the interface uses their Gram matrices. Steps:

    - The branch (``_wrong_branch``, ``_flip``): a mode below cutoff (|Im(neff)| above 1e-9 |neff|) gets the branch
      that decays in +z (Im(neff) > 0); a mode above cutoff gets the branch with power flow in +z (the rule of tidy3d
      and of ``meow_backend._orient``). Thus all modes below cutoff decay in +z. This includes both members of a
      complex pair (neff, -conj(neff)) of a lossless guide: the member with Re(neff) < 0 also decays in +z and is a
      forward mode.
    - Modes with fields that are not finite or with a self-product of almost zero (they cannot be normalized) are
      removed, and so is a copy of a kept mode (the same neff and parallel E_t fields, ``_is_copy``): a copy makes
      the interface system singular.
    - ``max_imag_neff``: remove the modes with a larger |Im neff| (1.0: the rule of meow's filter). The default keeps
      all modes: the modes below cutoff are part of the complete set.
    - The phase rule of meow (``zero_phase``), then the normalization <m, m> = 1 in the Yee product (in this order,
      so a lossless propagating mode has real fields also when its Hx is zero and zero_phase takes the phase of
      rounding noise).
    - A degenerate cluster of modes above cutoff (|neff_i - neff_j| < degenerate_tol |neff|, real neff): a real
      basis of their subspace, orthonormal in the Yee product (``_real_orthonormal``), so that |S|^2 is their power.
      A degenerate cluster below cutoff is not changed (the interface uses its Gram matrix).
    The interface needs a window without PML, so there are no PML modes to remove."""
    from meow.mode import normalize, zero_phase

    modes = [_flip(m) if _wrong_branch(m) else m for m in modes if _usable(m)]
    if max_imag_neff is not None:
        modes = [m for m in modes if abs(np.imag(m.neff)) <= max_imag_neff]
    kept = []
    for m in modes:
        if not any(_is_copy(k, m, degenerate_tol) for k in kept):
            kept.append(m)
    modes = [normalize(zero_phase(m), yee_inner_product) for m in kept]
    out, i = [], 0
    while i < len(modes):
        j = i + 1
        while j < len(modes) and abs(modes[j].neff - modes[i].neff) < degenerate_tol * abs(modes[i].neff):
            j += 1
        cluster = modes[i:j]
        if len(cluster) > 1 and abs(np.imag(cluster[0].neff)) < degenerate_tol * abs(cluster[0].neff):
            # real fields; no zero_phase after it (its phase comes from Hx, zero in a uniform-Ex mode)
            cluster = _real_orthonormal(cluster)
        out += cluster
        i = j
    return out


def _usable(mode) -> bool:
    """Finite fields and a self-product that is not almost zero against the product of the magnitudes."""
    if not np.isfinite(mode.neff) or not all(np.all(np.isfinite(np.asarray(getattr(mode, n)))) for n in _FIELDS):
        return False
    ax, ay = _yee_areas(mode)
    ex, ey, hx, hy = (np.asarray(getattr(mode, n)) for n in ("Ex", "Ey", "Hx", "Hy"))
    scale = 0.5 * np.sum(np.abs(ex * hy) * ax + np.abs(ey * hx) * ay)
    return bool(scale > 0 and abs(yee_inner_product(mode, mode)) > 1e-12 * scale)


def interface_blocks(modes1, modes2, overlap=overlap_matrix) -> tuple[np.ndarray, ...]:
    """(R_LL, T_LR, T_RL, R_RR) of the step from ``modes1`` (left) to ``modes2`` (right): column j of T_LR is the
    right-going field in the right modes for a unit amplitude of left mode j, and so on (module docstring).
    ``overlap(modes_a, modes_b)``: the matrix of the bilinear form (default: the Yee product; another form only for
    a comparison, for example meow's product in ``bench/bench_interface.py``)."""
    nl = len(modes1)
    p = overlap(modes1, modes2)
    gl = overlap(modes1, modes1)
    gr = overlap(modes2, modes2)
    k = np.block([[p.T, -gr.T], [gl, p]])
    rhs = np.block([[-p.T, gr.T], [gl, p]])      # columns: the left excitations, then the right excitations
    try:
        x = np.linalg.solve(k, rhs)
    except np.linalg.LinAlgError as exc:
        raise ValueError("meow_gpu.interface: the interface system is singular (linearly dependent modes?)") from exc
    if not np.all(np.isfinite(x)):
        raise ValueError("meow_gpu.interface: the interface solution is not finite (modes that are not finite?)")
    return x[:nl, :nl], x[nl:, :nl], x[:nl, nl:], x[nl:, nl:]


def compute_interface_s_matrix(modes1, modes2):
    """The S-matrix of one interface and its port map, in the format of ``meow.compute_interface_s_matrix``
    (ports left@0 .. left@{N_L-1}, right@0 .. right@{N_R-1}). The index order is the one of SAX, S[in, out] (the
    input port first; the transpose of the block matrix [[R_LL, T_RL], [T_LR, R_RR]]). The S-matrix is symmetric
    to the solver tolerance, so the order matters only for modes that are not orthogonal."""
    import jax.numpy as jnp

    r_ll, t_lr, t_rl, r_rr = interface_blocks(modes1, modes2)
    s = np.block([[r_ll, t_rl], [t_lr, r_rr]]).T
    ports = [f"left@{i}" for i in range(len(modes1))] + [f"right@{i}" for i in range(len(modes2))]
    return jnp.asarray(s), {p: i for i, p in enumerate(ports)}


def compute_interface_s_matrices(modes: list, **_) -> dict:
    """The interface S-matrices of a stack of mode sets: the ``interfaces_fn`` of ``meow.compute_s_matrix``."""
    return {f"i_{i}_{i + 1}": compute_interface_s_matrix(m1, m2) for i, (m1, m2) in enumerate(pairwise(modes))}


def _blocks(s, port_map: dict, n_left: int, n_right: int) -> tuple[np.ndarray, ...]:
    """(R_LL, T_LR, T_RL, R_RR) of an S-matrix in the SAX format (S[in, out]) with ports left@i and right@j."""
    m = np.asarray(s).T                       # m[out, in]: column j is the excitation of port j
    lp = [port_map[f"left@{i}"] for i in range(n_left)]
    rp = [port_map[f"right@{i}"] for i in range(n_right)]
    return m[np.ix_(lp, lp)], m[np.ix_(rp, lp)], m[np.ix_(lp, rp)], m[np.ix_(rp, rp)]


def _star(a: tuple, b: tuple) -> tuple:
    """The Redheffer star product: section a on the left, section b on the right, joined in the same modes.
    With u the right-going and v the left-going amplitudes between them: u = T_LR^a x + R_RR^a v, v = R_LL^b u."""
    r_a, t_a, tr_a, rr_a = a
    r_b, t_b, tr_b, rr_b = b
    n = rr_a.shape[0]
    u = np.linalg.solve(np.eye(n) - rr_a @ r_b, t_a)          # (I - R_RR^a R_LL^b)^-1 T_LR^a
    v = np.linalg.solve(np.eye(n) - r_b @ rr_a, tr_b)         # (I - R_LL^b R_RR^a)^-1 T_RL^b
    return r_a + tr_a @ (r_b @ u), t_b @ u, tr_a @ v, rr_b + t_b @ (rr_a @ v)


def _propagation(modes, length: float) -> tuple:
    """A cell of length ``length``: each mode gets exp(2 pi i neff length / wl) (meow's propagation)."""
    phase = np.diag(np.exp(2j * np.pi * np.array([complex(m.neff) for m in modes]) / modes[0].env.wl * length))
    zero = np.zeros_like(phase)
    return zero, phase, phase, zero


def compute_s_matrix(modes: list, cell_lengths, interfaces_fn=compute_interface_s_matrices):
    """The S-matrix of a stack of cells (propagation, interface, propagation, ...), as ``meow.compute_s_matrix``
    gives it (SAX format S[in, out]; ports left@i of the first cell and right@j of the last cell), but cascaded with
    the Redheffer star product: memory of the order of (2k)^2 for k modes. meow's SAX cascade with the klu backend
    builds a boolean matrix of (sum of the interface entries) x (twice the connected ports), about 16 n^2 k^3
    entries for n interfaces, and two integer arrays of that shape (README, "Memory of the cascade").
    ``interfaces_fn(modes)`` gives the interface S-matrices (default: the conserving interface; meow's
    ``compute_interface_s_matrices`` also works)."""
    if len(cell_lengths) != len(modes):
        raise ValueError(f"len(cell_lengths) != len(modes): {len(cell_lengths)} != {len(modes)}")
    interfaces = interfaces_fn(modes)
    total = _propagation(modes[0], cell_lengths[0])
    for i in range(1, len(modes)):
        s, pm = interfaces[f"i_{i - 1}_{i}"]
        total = _star(total, _blocks(s, pm, len(modes[i - 1]), len(modes[i])))
        total = _star(total, _propagation(modes[i], cell_lengths[i]))
    r_ll, t_lr, t_rl, r_rr = total
    ports = [f"left@{i}" for i in range(len(modes[0]))] + [f"right@{i}" for i in range(len(modes[-1]))]
    return np.block([[r_ll, t_rl], [t_lr, r_rr]]).T, {p: i for i, p in enumerate(ports)}
