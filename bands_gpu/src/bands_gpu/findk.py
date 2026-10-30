"""find_k: the wave vectors of the bands at a fixed frequency (as MPB's find-k), with the group velocity from the
Hellmann-Feynman theorem and an optional mirror parity.

    res = find_k(cell, omega=0.35, band_min=1, band_max=2, kdir=(1, 0, 0), polarization="tm")
    res.kmag              # (nb,) |k - korig| in units of 2 pi / a (NaN: no root in [kmag_min, kmag_max])
    res.group_velocity    # (nb,) d omega / dk along kdir, in units of c

Units and directions (as MPB): omega in units of 2 pi c / a; ``kdir`` and ``korig`` in units of the reciprocal
lattice vectors (as k points); k = korig + kmag * k_hat, with k_hat the Cartesian unit vector along kdir and kmag
its Cartesian length in units of 2 pi / a. A guided mode of a waveguide in a supercell: give the waveguide axis as
kdir (a grid axis with one point, or a periodic axis for a photonic-crystal waveguide). Bands are counted from 1 at
each k; band b is the b-th lowest frequency (of the selected parity). A mode with k > n_clad * omega is below the
light line of the cladding (guided); the other roots are supercell bands of the cladding.

Root finding: a safeguarded Newton method on g(k) = f_b(k) - omega (one band solve per step). The derivative is the
group velocity from the Hellmann-Feynman theorem, d(omega^2)/dk = <H| dTheta/dk |H> for a normalized eigenvector:

    Theta(q) H = -q x K (q x H),    K = F eps_inv F^-1,
    <H| dTheta/dk |H> = 2 Re <d x H| K |q x H>,    v_g = <H| dTheta/dk |H> / (2 omega)

(d: the Cartesian unit vector of k_hat; the transverse basis u(q) does not enter, because the nonzero eigenpairs of
the operator on all 3-vector fields are the transverse eigenpairs). Bands that are degenerate at k get the
eigenvalues of the derivative matrix in their subspace, in ascending order (the right derivative of the sorted
bands). The root stays in a bracket [lo, hi] (g(lo) < 0 < g(hi), for a band that rises with k): a Newton step that
leaves the bracket, or a band that does not rise, is replaced by a bisection; an end of [kmag_min, kmag_max] is
solved only when a step needs it. Bands are found from band_max down (as MPB): each band solve gives all bands up to
band_max, and these values bracket and start the lower bands. The eigenvector block of each k starts the next solve
(a warm start; off in a homogeneous cell, as in ``solve_bands``).

Parity (``parity="even"`` or ``"odd"``, mirror plane through the cell center, normal to the Cartesian axis
``mirror_axis``): the definition of MPB, E(r) = +/- R E(R r) for the polar vector E (R: the mirror), so that
H(r) = -/+ R H(R r) for the pseudovector H. Examples: in a 2-D cell, "tm" (E along z) even = E_z even in the mirror
axis; "te" (H along z) even = H_z odd. The fundamental TE-like mode of a strip waveguide along x (E mostly along y)
is odd in y. The solve uses the projector P = (1 +/- M) / 2 on the plane-wave coefficients. The mirror axis needs
an odd number of grid points: then the plane waves G and R G are both in the basis, the discrete operator commutes
with M, and the bands of the two parities together are the bands without parity. (An even grid has a Nyquist plane
without a mirror image; to leave it out changes the even bands at coarse grids. A ValueError stops this.)

The structure must be symmetric: eps at the subpixel samples r and at their exact mirror points R r must be equal
(else a ValueError). The samples of the grid are mirror images only to rounding, so a face exactly on a sample can
fall inside on one side and outside on the other; eps_inv is therefore made symmetric, (eps_inv + R eps_inv R) / 2
on mirror voxels (this changes only such voxels; ``BandEvaluator.mirror_asymmetry`` is the largest change before).
The parity subspace must hold the block of the eigensolver (num_bands + extra vectors), else a ValueError: for
example "tm" in a 2-D cell is odd in z, so its even z subspace is empty.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from bands_gpu.geometry import Cell
from bands_gpu.lobpcg import _gram, lobpcg
from bands_gpu.operator import Operator, _cross, apply, make_operator, precondition
from bands_gpu.solver import solver_device


@jax.jit
def _velocity_matrix(op: Operator, d: jax.Array, c: jax.Array) -> jax.Array:
    """<c_i| dTheta/dk |c_j> (b x b) for the Cartesian unit direction d of k (q in units of 1/a, 2 pi included):
    G + G^H with G_ij = <d x H_i| K |q x H_j>."""
    h = jnp.einsum("bmxyz,mixyz->bixyz", c, op.u.astype(c.dtype))
    a = _cross(op.q.astype(c.dtype), h)
    dd = jnp.broadcast_to(d.astype(c.dtype)[:, None, None, None], op.q.shape)
    bvec = _cross(dd, h)
    e = jnp.fft.ifftn(a, axes=(-3, -2, -1))
    if op.eps_inv.ndim == 5:
        e = jnp.einsum("ijxyz,bjxyz->bixyz", op.eps_inv.astype(c.dtype), e)
    else:
        e = op.eps_inv.astype(c.dtype) * e
    ka = jnp.fft.fftn(e, axes=(-3, -2, -1))
    g = _gram(bvec, ka)
    return g + g.conj().T


def _mirror(cell: Cell, op_q: np.ndarray, op_u: np.ndarray, axis: int):
    """The mirror map in the coefficients: T (m, m, nx, ny, nz) and the index map of the mirror axis (n -> -n)."""
    idx = (-np.arange(cell.shape[axis])) % cell.shape[axis]
    sign = np.ones(3)
    sign[axis] = -1.0
    q_m = np.take(op_q, idx, axis=1 + axis)
    if np.abs(q_m - sign[:, None, None, None] * op_q).max() > 1e-9 * max(float(np.abs(op_q).max()), 1.0):
        raise ValueError("parity: k must lie in the mirror plane (no component along the mirror axis)")
    u_m = np.take(op_u, idx, axis=2 + axis)                        # u at the mirror image R G
    # T_mm'(G) = u_m(q(G)) . (-R u_m'(q(R G))): H is a pseudovector, M H(r) = -R H(R r)
    return np.einsum("mixyz,i,nixyz->mnxyz", op_u, -sign, u_m), idx


def _check_mirror_cell(cell: Cell, axis: int):
    """The lattice vector of the mirror axis along that Cartesian axis, the others orthogonal to it, and an odd number
    of grid points on it (an even grid has a Nyquist plane, G_axis = -n/2, without a mirror image in the basis, so
    the discrete operator is not mirror symmetric there)."""
    e = cell.unit_basis
    if abs(abs(e[axis, axis]) - 1.0) > 1e-12 or any(abs(e[j, axis]) > 1e-12 for j in range(3) if j != axis):
        raise ValueError(f"parity: the lattice vector {axis + 1} must be along the Cartesian axis {axis} and the other "
                         "lattice vectors orthogonal to it")
    if cell.shape[axis] % 2 == 0:
        raise ValueError(f"parity: use an odd number of grid points along the mirror axis {axis} (an even grid has a "
                         "Nyquist plane without a mirror image)")


def _eps_at(cell: Cell, X, Y, Z) -> np.ndarray:
    """eps of the structure (with the periodic images of the objects) at the Cartesian points X, Y, Z."""
    shape = np.broadcast_shapes(np.shape(X), np.shape(Y), np.shape(Z))
    eps = np.full(shape, float(cell.eps_background))
    for obj in cell.objects:
        hit = np.zeros(shape, bool)
        for t in cell.image_shifts(obj):
            hit |= np.broadcast_to(obj.inside(X + t[0], Y + t[1], Z + t[2]), shape)
        eps = np.where(hit, obj.eps, eps)
    return eps


def _symmetrize(cell: Cell, eps_inv: np.ndarray, axis: int, sub: int) -> tuple[np.ndarray, float]:
    """Check that the structure is symmetric under the mirror of ``axis`` (through the cell center): eps at the
    subpixel samples r and at the exact mirror points R r (the coordinate negated, no rounding) must be equal, else a
    ValueError. Return (eps_inv + R eps_inv(R .) R) / 2 and the largest change. The samples of the grid are mirror
    images only to rounding: a face exactly on a sample can be inside on one side and outside on the other, which
    makes eps_inv of the two mirror voxels differ by one sample layer (``Block.inside`` uses <=)."""
    pts = cell.points(sub)
    mirrored_pts = [(-p if c == axis else p) for c, p in enumerate(pts)]
    if not np.array_equal(_eps_at(cell, *pts), _eps_at(cell, *mirrored_pts)):
        raise ValueError(f"parity: the structure is not symmetric under the mirror of axis {axis} (through the cell "
                         "center)")
    n = cell.shape[axis]
    idx = (-np.arange(n)) % n
    sign = np.ones(3)
    sign[axis] = -1.0
    if eps_inv.ndim == 5:
        mirrored = np.take(eps_inv, idx, axis=2 + axis) * (sign[:, None] * sign[None, :])[:, :, None, None, None]
    else:
        mirrored = np.take(eps_inv, idx, axis=axis)
    return 0.5 * (eps_inv + mirrored), float(np.abs(mirrored - eps_inv).max())


def _parity_dimension(t: np.ndarray, idx: np.ndarray, axis: int, sign: float) -> int:
    """The dimension of the range of P = (1 + sign M) / 2: (m N + sign tr M) / 2, where tr M sums tr T(G) over the
    plane waves with R G = G (the index map fixes them)."""
    m = t.shape[0]
    fixed = idx == np.arange(idx.size)
    tr = np.einsum("mm...->...", t)                                   # (nx, ny, nz)
    tr_m = float(np.take(tr, np.nonzero(fixed)[0], axis=axis).sum())
    return int(round((m * tr.size + sign * tr_m) / 2))


@dataclass
class FindKResult:
    kmag: np.ndarray                  # (nb,) Cartesian |k - korig|, units of 2 pi / a; NaN: no root, or max_steps
    group_velocity: np.ndarray        # (nb,) d omega / dk along k_hat at the last solve, units of c
    kpoints: np.ndarray               # (nb, 3) k in units of the reciprocal lattice vectors
    freq_error: np.ndarray            # (nb,) |f_b - omega| at the last solve of band b (before the last update)
    converged: list[bool] = field(default_factory=list)   # the root search converged (tol)
    solves: int = 0                   # band solves (each gives all bands up to band_max)
    unconverged_solves: int = 0       # band solves that did not reach the tolerance of the eigensolver
    iterations: int = 0               # eigensolver iterations, summed
    device: str = ""


class BandEvaluator:
    """Band solves at single k points with group velocities, an optional parity and a warm start between calls.

        ev = BandEvaluator(cell, num_bands=4, polarization="tm", parity="even", mirror_axis=1)
        freqs, vg = ev(k_frac, direction)   # direction: Cartesian (normalized here); freqs in units of 2 pi c / a
    """

    def __init__(self, cell: Cell, num_bands: int, polarization: str | None = None, parity: str | None = None,
                 mirror_axis: int = 1, extra: int = 4, tol: float | None = None, max_iter: int = 300,
                 dtype=jnp.complex128, sub: int = 8, seed: int = 0, warm_start: bool = True,
                 degeneracy: float | None = None):
        if dtype == jnp.complex128 and not jax.config.jax_enable_x64:
            warnings.warn("bands_gpu: complex128 needs jax.config.update('jax_enable_x64', True); using complex64",
                          stacklevel=2)
            dtype = jnp.complex64
        if parity not in (None, "even", "odd"):
            raise ValueError("parity: None, 'even' or 'odd'")
        if mirror_axis not in (0, 1, 2):
            raise ValueError("mirror_axis: 0, 1 or 2 (x, y, z)")
        single = dtype == jnp.complex64
        self.cell, self.num_bands, self.polarization = cell, num_bands, polarization
        self.parity, self.mirror_axis, self.dtype = parity, mirror_axis, dtype
        self.tol = (1e-4 if single else 1e-8) if tol is None else tol
        # bands closer than this (relative, in omega^2) are one degenerate cluster for the group velocity
        self.degeneracy = (1e-5 if single else 1e-9) if degeneracy is None else degeneracy
        self.max_iter = max_iter
        self.device = solver_device()
        self.eps_inv = cell.eps_inverse(sub)
        flat = self.eps_inv.reshape(self.eps_inv.shape[:-3] + (-1,))
        self.warm_start = warm_start and not np.all(flat == flat[..., :1])   # see solve_bands
        self.mirror_asymmetry = 0.0
        if parity is not None:
            _check_mirror_cell(cell, mirror_axis)
            self.eps_inv, self.mirror_asymmetry = _symmetrize(cell, self.eps_inv, mirror_axis, sub)
        m = 2 if polarization is None else 1
        self.block = num_bands + extra
        self.shape = (self.block, m) + tuple(cell.shape)
        self.rng = np.random.default_rng(seed)
        self.x = None
        self.solves = 0
        self.unconverged = 0
        self.iterations = 0
        self._apply = jax.jit(apply)
        self._pre = jax.jit(precondition)
        axis = mirror_axis

        def project(t, sign, idx, c):
            """P c = (c + sign M c) / 2, with (M c)_m(G) = sum_m' T_mm'(G) c_m'(R G)."""
            cm = jnp.take(c, idx, axis=c.ndim - 3 + axis)
            mc = jnp.einsum("mnxyz,...nxyz->...mxyz", t, cm)
            return 0.5 * (c + sign * mc)

        self._project = jax.jit(project)

    def solve(self, k_frac, direction=None):
        """Bands (num_bands,) at k_frac, in units of 2 pi c / a; with ``direction`` (Cartesian), also the group
        velocities along it (units of c). Returns (freqs, vg or None, converged)."""
        cell, dtype = self.cell, self.dtype
        op = make_operator(cell, k_frac, self.polarization, self.eps_inv, dtype, self.device)
        put = lambda a: jax.device_put(jnp.asarray(a), self.device)  # noqa: E731
        if self.parity is None:
            fa = lambda v: self._apply(op, v)  # noqa: E731
            fp = lambda r: self._pre(op, r)  # noqa: E731
            proj = None
        else:
            t, idx = _mirror(cell, np.asarray(op.q, float), np.asarray(op.u, float), self.mirror_axis)
            sign = 1.0 if self.parity == "even" else -1.0
            dim = _parity_dimension(t, idx, self.mirror_axis, sign)
            if dim < self.block:
                # else LOBPCG drops the extra directions and returns zero vectors with frequency 0 as converged
                raise ValueError(f"parity: the {self.parity} subspace of mirror axis {self.mirror_axis} has {dim} "
                                 f"dimensions, fewer than the {self.block} vectors of the eigensolver (num_bands + "
                                 "extra); use a finer grid, fewer bands, a smaller extra, or another parity or axis")
            t, idx = put(jnp.asarray(t, dtype)), put(idx)
            proj = lambda c: self._project(t, sign, idx, c)  # noqa: E731
            fa = lambda v: proj(self._apply(op, v))  # noqa: E731
            fp = lambda r: proj(self._pre(op, r))  # noqa: E731
        if self.x is None or not self.warm_start:
            x = jnp.asarray(self.rng.normal(size=self.shape) + 1j * self.rng.normal(size=self.shape), dtype)
            x = self._pre(op, jax.device_put(x, self.device))
        else:
            x = self.x
        if proj is not None:
            x = proj(x)
        res = lobpcg(fa, fp, x, self.num_bands, tol=self.tol, max_iter=self.max_iter,
                     scale=(2 * np.pi / max(cell.size)) ** 2 * 1e-2)
        self.x = res.vectors
        self.solves += 1
        self.iterations += res.iterations
        converged = bool(res.converged or (res.stalled and np.max(res.residuals[:self.num_bands]) < 100 * self.tol))
        self.unconverged += not converged
        lam = np.maximum(np.asarray(res.values, float), 0.0)
        freqs = np.sqrt(lam[:self.num_bands]) / (2 * np.pi)
        if direction is None:
            return freqs, None, converged
        d = np.asarray(direction, float)
        d = d / np.linalg.norm(d)
        dmat = np.asarray(_velocity_matrix(op, put(jnp.asarray(d)), res.vectors), np.complex128)
        dmat = 0.5 * (dmat + dmat.conj().T)
        # clusters of degenerate bands: the eigenvalues of the derivative in the cluster, ascending
        dl = np.real(np.diag(dmat)).copy()
        top = max(float(lam.max()), 1e-300)
        i = 0
        while i < len(lam):
            j = i + 1
            while j < len(lam) and lam[j] - lam[j - 1] <= self.degeneracy * max(lam[j], 1e-3 * top):
                j += 1
            if j - i > 1 and i < self.num_bands:
                dl[i:j] = np.linalg.eigvalsh(dmat[i:j, i:j])
            i = j
        omega = np.sqrt(lam)
        with np.errstate(divide="ignore", invalid="ignore"):
            vg = np.where(omega > 0, dl / (2 * np.where(omega > 0, omega, 1.0)), 0.0)
        return freqs, vg[:self.num_bands], converged

    __call__ = solve


def group_velocity(cell: Cell, k_frac, num_bands: int, direction, polarization: str | None = None, **kw):
    """The bands (units of 2 pi c / a) and their group velocities d omega / dk along the Cartesian ``direction``
    (units of c) at one k point (reciprocal-lattice coordinates); as MPB's compute-group-velocity-component.
    ``kw``: the options of BandEvaluator (parity, mirror_axis, tol, dtype, ...)."""
    ev = BandEvaluator(cell, num_bands, polarization, **kw)
    freqs, vg, _ = ev.solve(k_frac, direction)
    return freqs, vg


def _n_range(cell: Cell) -> tuple[float, float]:
    eps = [float(cell.eps_background)] + [float(o.eps) for o in cell.objects]
    return math.sqrt(min(eps)), math.sqrt(max(eps))


def find_k(cell: Cell, omega: float, band_min: int, band_max: int, kdir, polarization: str | None = None,
           tol: float | None = None, kmag_guess=None, kmag_min: float = 0.0, kmag_max: float | None = None,
           korig=(0.0, 0.0, 0.0), parity: str | None = None, mirror_axis: int = 1, max_steps: int = 40,
           **kw) -> FindKResult:
    """The k magnitudes (units of 2 pi / a) at which the bands band_min..band_max (1-based, of the selected parity)
    have the frequency omega (units of 2 pi c / a), along kdir (reciprocal-lattice coordinates) from korig.

    tol: relative tolerance of kmag (the last Newton step); default 1e-8 in double precision, 1e-5 in single (the
    noise of single-precision bands: 1e-8 there only adds solves). kmag_guess: a number
    (the start of the first band searched, band_max) or a list per band (band_min..band_max); default
    omega (n_min + n_max) / 2 from the permittivities of the cell. kmag_max: default 1.01 omega n_max + |korig|
    (Cartesian; |k| >= kmag - |korig|, and no band reaches omega at |k| > 1.01 omega n_max along an axis without
    lattice periodicity); along a periodic axis, give the edge of the Brillouin zone. ``kw``: the options of
    BandEvaluator (extra, tol of the eigensolver as ``band_tol``, dtype, sub, seed, warm_start)."""
    if not 1 <= band_min <= band_max:
        raise ValueError("bands: 1 <= band_min <= band_max")
    band_tol = kw.pop("band_tol", None)
    ev = BandEvaluator(cell, band_max, polarization, parity=parity, mirror_axis=mirror_axis, tol=band_tol, **kw)
    khat = cell.k_cartesian(kdir)
    if np.linalg.norm(khat) == 0:
        raise ValueError("kdir must not be zero")
    khat = khat / np.linalg.norm(khat)
    kdir_frac = cell.k_fractional(khat)                 # reciprocal-lattice coordinates of the unit vector
    korig = np.asarray(korig, float)
    if tol is None:
        tol = 1e-5 if ev.dtype == jnp.complex64 else 1e-8
    n_min, n_max = _n_range(cell)
    if kmag_max is None:
        kmag_max = 1.01 * omega * n_max + float(np.linalg.norm(cell.k_cartesian(korig)))
    nb = band_max - band_min + 1
    if kmag_guess is None:
        guesses = [None] * nb
        first = min(max(0.5 * omega * (n_min + n_max), kmag_min), kmag_max)
    elif np.ndim(kmag_guess) == 0:
        guesses, first = [None] * nb, float(kmag_guess)
    else:
        guesses, first = [float(g) for g in kmag_guess], None
        if len(guesses) != nb:
            raise ValueError("kmag_guess: one value per band (band_min..band_max)")
    cache: list[tuple[float, np.ndarray, np.ndarray]] = []    # (kmag, freqs, vg)

    def evaluate(kmag: float):
        freqs, vg, _ = ev.solve(korig + kmag * kdir_frac, khat)
        cache.append((kmag, freqs, vg))
        return freqs, vg

    out = FindKResult(np.full(nb, np.nan), np.full(nb, np.nan), np.full((nb, 3), np.nan), np.full(nb, np.nan),
                      [False] * nb)
    for b in range(band_max, band_min - 1, -1):
        j, slot = b - 1, b - band_min
        lo, hi, lo_ok, hi_ok = kmag_min, kmag_max, False, False

        def bracket(kk, g):
            nonlocal lo, hi, lo_ok, hi_ok
            if g < 0 and (not lo_ok or kk > lo) and kk < hi:
                lo, lo_ok = kk, True
            elif g > 0 and (not hi_ok or kk < hi) and kk > lo:
                hi, hi_ok = kk, True

        for kk, f, _ in cache:
            bracket(kk, f[j] - omega)
        start = guesses[slot] if guesses[slot] is not None else (first if not cache else None)
        if start is not None:
            f, v = evaluate(float(start))
            k_cur, g_cur, v_cur = float(start), f[j] - omega, v[j]
            bracket(k_cur, g_cur)
        else:                                            # the cached solve nearest to the root of this band
            kk, f, v = min(cache, key=lambda c: abs(c[1][j] - omega))
            k_cur, g_cur, v_cur = kk, f[j] - omega, v[j]
        root, ok = np.nan, False
        for _ in range(max_steps):
            if g_cur == 0.0:
                root, ok = k_cur, True
                break
            kn = k_cur - g_cur / v_cur if v_cur > 0 else np.nan
            if not lo < kn < hi:
                if g_cur > 0 and not lo_ok:              # the root is below: the sign at kmag_min is needed
                    f, v = evaluate(kmag_min)
                    if f[j] - omega > 0:
                        break                            # the band is above omega in [kmag_min, kmag_max]
                    bracket(kmag_min, f[j] - omega)
                    lo_ok = True
                    continue
                if g_cur < 0 and not hi_ok:
                    f, v = evaluate(kmag_max)
                    if f[j] - omega < 0:
                        break                            # the band is below omega in [kmag_min, kmag_max]
                    bracket(kmag_max, f[j] - omega)
                    hi_ok = True
                    continue
                kn = 0.5 * (lo + hi)
            if abs(kn - k_cur) <= tol * max(abs(kn), 1e-6) or (lo_ok and hi_ok and hi - lo <= tol * abs(kn)):
                root, ok = kn, True
                break
            f, v = evaluate(kn)
            k_cur, g_cur, v_cur = kn, f[j] - omega, v[j]
            bracket(k_cur, g_cur)
        out.kmag[slot] = root
        out.group_velocity[slot] = v_cur
        out.freq_error[slot] = abs(g_cur)
        out.converged[slot] = ok
        if np.isfinite(root):
            out.kpoints[slot] = korig + root * kdir_frac
    out.solves, out.unconverged_solves, out.iterations, out.device = ev.solves, ev.unconverged, ev.iterations, \
        str(ev.device)
    return out
