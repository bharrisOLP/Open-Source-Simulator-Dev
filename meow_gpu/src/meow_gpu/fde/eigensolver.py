"""Block generalized Davidson for the k eigenvalues of the mode operator nearest a target, on the GPU.

Work split:
    device (JAX): the operator products, the preconditioner, the orthogonalization (all large arrays);
    host (numpy): the small projected eigenproblem (m x m, m <= max_subspace) and the selection.

Preconditioner ("fast diagonalization"): for each field component, the operator without the cross terms and with
a constant reference permittivity eps_ref is separable,

    A0_x = -(eps_ref + Kx (x) I + I (x) Ky),   Kx = Dxf Dxb, Ky = Dyb Dyf   (Ex; for Ey: Dxb Dxf and Dyf Dyb),

so (A0 - theta)^-1 r costs two small dense products per side with the 1-D eigenbases (Lynch, Rice and Thomas
1964). This works on non-uniform grids and with PML. With eps_ref below neff^2 of the wanted modes, the
denominators do not go through zero. The preconditioner always runs in complex64: it is an approximation.

Precision: consumer GPUs (RDNA3: FP64 at 1/32 of the FP32 rate) run complex128 matrix products slowly. In the
"mixed" mode the subspace iteration runs in complex64 to a residual of about ``switch_tol``. Then a complex128 polish
goes to ``tol``: by default the mixed-precision refinement ``refine`` (complex128 residuals and Rayleigh-Ritz,
complex64 GMRES correction solves; see the section "Mixed-precision refinement"), or (``polish="davidson"``) a
complex128 Davidson from the complex64 Ritz vectors.

All device arrays have fixed shapes (the subspace is a buffer of ``max_subspace`` vectors), so JAX compiles each
kernel once per grid size, block size and precision.

A warm start (the modes of the previous cell of a taper, or of the previous wavelength) is the initial block.

Batched solve (``davidson_batch``): B problems with the same grid shape at the same time. The device kernels have a
leading batch axis; each member keeps its own host state (projected matrix, subspace size, stop). See the section
"Batched solve" below.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from meow_gpu.device import put, solver_device
from meow_gpu.fde.operator import DeviceOperator, FDEProblem, apply

PC_DTYPE = jnp.complex64
STALL_FLOOR = 1e-3


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Preconditioner:
    vx: jax.Array        # (2, Nx, Nx): 1-D eigenvectors along x, for Ex and Ey
    vx_inv: jax.Array
    vy: jax.Array        # (2, Ny, Ny)
    vy_inv: jax.Array
    lam: jax.Array       # (2, Nx, Ny): -(eps_ref + ax_i + ay_j)


_BASES: dict = {}           # 1-D eigenbases of the preconditioner, by stencil coefficients (they do not depend on eps)
_BASES_MAX = 16


def _bases(p: FDEProblem):
    key = tuple(np.asarray(s.main).tobytes() + np.asarray(s.off).tobytes() for s in (p.dxf, p.dxb, p.dyf, p.dyb))
    hit = _BASES.get(key)
    if hit is not None:
        return hit
    dxf, dxb, dyf, dyb = (s.dense() for s in (p.dxf, p.dxb, p.dyf, p.dyb))
    vx, vxi, vy, vyi, ax_all, ay_all = [], [], [], [], [], []
    for kx, ky in ((dxf @ dxb, dyb @ dyf), (dxb @ dxf, dyf @ dyb)):
        ax, ux = np.linalg.eig(kx)
        ay, uy = np.linalg.eig(ky)
        vx.append(ux)
        vxi.append(np.linalg.inv(ux))
        vy.append(uy)
        vyi.append(np.linalg.inv(uy))
        ax_all.append(ax)
        ay_all.append(ay)
    c = lambda a: put(np.stack(a), PC_DTYPE)  # noqa: E731
    hit = (c(vx), c(vxi), c(vy), c(vyi), np.stack(ax_all), np.stack(ay_all))
    if len(_BASES) >= _BASES_MAX:
        _BASES.pop(next(iter(_BASES)))
    _BASES[key] = hit
    return hit


def make_preconditioner(p: FDEProblem, eps_ref: float | None = None) -> Preconditioner:
    """The fast-diagonalization preconditioner. Default ``eps_ref``: the smallest real in-plane permittivity.
    The 1-D eigenbases depend only on the grid, the wavelength and the PML: they are cached."""
    if eps_ref is None:
        eps_ref = float(np.min(np.real(np.stack([p.eps_xx, p.eps_yy]))))
    vx, vxi, vy, vyi, ax, ay = _bases(p)
    lam = -(eps_ref + ax[:, :, None] + ay[:, None, :])
    return Preconditioner(vx, vxi, vy, vyi, put(lam, PC_DTYPE))


@jax.jit
def precondition(pc: Preconditioner, r: jax.Array, theta: jax.Array) -> jax.Array:
    """(A0 - theta)^-1 r for a block r of shape (b, 2, Nx, Ny), theta of shape (b,). Computed in complex64; the
    result has the dtype of r."""
    r32 = r.astype(PC_DTYPE)
    rt = jnp.einsum("cij,bcjk->bcik", pc.vx_inv, r32)
    rt = jnp.einsum("bcik,clk->bcil", rt, pc.vy_inv)
    den = pc.lam[None] - theta.astype(PC_DTYPE)[:, None, None, None]
    den = jnp.where(jnp.abs(den) < 1e-6, 1e-6, den)
    t = jnp.einsum("cij,bcjk->bcik", pc.vx, rt / den)
    t = jnp.einsum("bcik,clk->bcil", t, pc.vy)
    return t.astype(r.dtype)


# ---------------------------------------------------------------------------------------------------------------
# Fixed-shape device kernels. Blocks are (rows, 2, Nx, Ny); the products use a flat (rows, n) view.
# ---------------------------------------------------------------------------------------------------------------

def require_x64(precision: str) -> None:
    """Precision "mixed" and "double" need complex128. Without ``jax_enable_x64``, JAX runs the complex128 parts in
    complex64 without an error (a polish to the complex64 floor that cannot reach its tolerance): stop with an error."""
    if precision in ("mixed", "double") and jax.dtypes.canonicalize_dtype(jnp.complex128) != jnp.complex128:
        raise RuntimeError(f'precision "{precision}" needs complex128: call jax.config.update("jax_enable_x64", True) '
                           "before the solve")


def _flat(a: jax.Array) -> jax.Array:
    return a.reshape(a.shape[0], -1)


@jax.jit
def _gram(a: jax.Array, b: jax.Array) -> jax.Array:
    """G_ij = <a_i, b_j>. The conjugate goes on the smaller operand: G = conj(b^H a)^T when a is the larger
    one, so that XLA does not write a conjugated copy of a large buffer."""
    if a.shape[0] > b.shape[0]:
        return (_flat(b).conj() @ _flat(a).T).T.conj()
    return _flat(a).conj() @ _flat(b).T


@jax.jit
def _combine(c: jax.Array, v: jax.Array) -> jax.Array:
    """sum_i c_ij v_i: a block (k, ...) for coefficients c (m, k)."""
    return (c.T @ _flat(v)).reshape((c.shape[1],) + v.shape[1:])


@jax.jit
def _ritz(y: jax.Array, theta: jax.Array, v: jax.Array, w: jax.Array):
    """Ritz vectors x = V y, A x = W y, residuals r and their norms relative to |x| max(1, |theta|)."""
    x = _combine(y, v)
    r = _combine(y, w) - theta[:, None, None, None] * x
    xn = jnp.sqrt(jnp.sum(jnp.abs(_flat(x)) ** 2, axis=1))
    rn = jnp.sqrt(jnp.sum(jnp.abs(_flat(r)) ** 2, axis=1)) / xn / jnp.maximum(1.0, jnp.abs(theta))
    return x, r, rn


@jax.jit
def _project_out(t: jax.Array, v: jax.Array) -> jax.Array:
    """t minus its projection on the orthonormal rows of v (two passes; zero rows of v have no effect)."""
    for _ in range(2):
        t = t - _combine(_gram(v, t), v)
    return t


@partial(jax.jit, static_argnums=())
def _apply_block(op: DeviceOperator, v: jax.Array) -> jax.Array:
    return apply(op, v)


@partial(jax.jit, donate_argnums=(0,))
def _insert(buf: jax.Array, block: jax.Array, start: jax.Array) -> jax.Array:
    return jax.lax.dynamic_update_slice_in_dim(buf, block, start, axis=0)


def _svqb(t: jax.Array, drop: float) -> tuple[jax.Array, int]:
    """Orthonormalize the block t (b, ...) with SVQB (Stathopoulos and Wu 2002). The result has the same shape;
    the first ``kept`` rows are orthonormal and the other rows are zero."""
    g = np.array(_gram(t, t), dtype=np.complex128)   # host math in double (float32 underflows the guard)
    coef, kept = _svqb_coef(g, drop)
    return _combine(put(coef, t.dtype), t), kept


def _svqb_coef(g: np.ndarray, drop: float) -> tuple[np.ndarray, int]:
    """The SVQB coefficients (b, b) and the number of kept directions, from the Gram matrix g (host, complex128)."""
    b = g.shape[0]
    diag = np.real(np.diag(g))
    # rows with no norm (masked columns) keep the scale 1: 1/tiny overflows complex64 and inf * 0 = nan
    d = np.where(diag > 1e-30 * max(diag.max(), 1e-300), np.sqrt(np.maximum(diag, 0.0)), 1.0)
    gs = g / d[:, None] / d[None, :]
    s, u = np.linalg.eigh(gs)
    keep = s > drop * max(s.max(), 1e-300)
    coef = np.zeros((b, b), dtype=g.dtype)
    kept = int(keep.sum())
    if kept:
        coef[:, :kept] = (u[:, keep] / np.sqrt(s[keep])[None, :]) / d[:, None]
    return coef, kept


@dataclass
class EigResult:
    values: np.ndarray        # (k,) complex eigenvalues, sorted by distance to the target
    vectors: jax.Array        # (b, 2, Nx, Ny): the k wanted Ritz vectors first, then the extra block vectors
    residuals: np.ndarray     # (k,) relative residual norms
    iterations: int
    matvecs: int              # operator applies (vectors), in all precisions
    converged: bool
    matvecs_hi: int = 0       # the operator applies in complex128 (vectors)
    polish: str = ""          # "mixed": the complex128 polish used ("refine", "refine+davidson" or "davidson")


def _davidson_fixed(op: DeviceOperator, pc: Preconditioner, k: int, target: complex, x0: jax.Array, tol: float,
                    max_iter: int, max_subspace: int, verbose: bool, stall_iters: int = 0) -> EigResult:
    """One Davidson run in the dtype of x0. ``stall_iters`` > 0: also stop (not converged) when the largest
    residual of the k wanted pairs did not halve in that many iterations (the precision limit of complex64)."""
    dtype = x0.dtype
    b = x0.shape[0]
    big_m = max_subspace
    # SVQB drops directions whose normalized Gram eigenvalue is below ``drop``; it amplifies rounding by at most
    # 1/sqrt(drop), so that the new block stays orthogonal to the subspace to about eps / sqrt(drop).
    drop = 1e-4 if dtype == jnp.complex64 else 1e-10
    best, best_it = np.inf, 0
    x0 = put(x0)
    zeros = put(jnp.zeros((big_m,) + x0.shape[1:], dtype))
    v, w = zeros, zeros
    h = np.zeros((big_m, big_m), dtype=np.complex128)
    matvecs = 0

    def restart(block):
        nonlocal v, w, h, matvecs
        q, kept = _svqb(block, drop)
        aq = _apply_block(op, q)
        matvecs += kept
        v = _insert(jnp.zeros_like(zeros), q, 0)
        w = _insert(jnp.zeros_like(zeros), aq, 0)
        h[:] = 0
        h[:kept, :kept] = np.asarray(_gram(q, aq))[:kept, :kept]
        return kept

    m = restart(x0)
    theta = np.zeros(b, complex)
    rn = np.full(b, np.inf)
    x = x0
    it = 0
    hi = dtype == jnp.complex128

    def result(ok):
        return EigResult(theta[:k], x, rn[:k], it, matvecs, ok, matvecs if hi else 0)

    for it in range(1, max_iter + 1):
        vals, y = np.linalg.eig(h[:m, :m])
        order = np.argsort(np.abs(vals - target))[:b]
        nb = len(order)
        theta = np.zeros(b, complex)
        theta[:nb] = vals[order]
        ypad = np.zeros((big_m, b), dtype=np.complex128)
        ypad[:m, :nb] = y[:, order]
        x, r, rn_dev = _ritz(put(ypad, dtype), put(theta, dtype), v, w)
        rn = np.array(rn_dev, dtype=np.float64)    # a writable copy
        rn[nb:] = np.inf
        if verbose:
            print(f"  it {it:3d} m {m:4d} max res {rn[:k].max():.2e} theta0 {theta[0]:.12f}", flush=True)
        if np.all(rn[:k] < tol):
            return result(True)
        # stall: only near the precision floor (early plateaus are normal while the Ritz values settle)
        if rn[:k].max() < 0.5 * best:
            best, best_it = rn[:k].max(), it
        elif stall_iters and best < STALL_FLOOR and it - best_it >= stall_iters:
            return result(False)
        active = put(rn >= tol)
        t = precondition(pc, r, put(theta, dtype))
        t = jnp.where(active[:, None, None, None], t, 0)
        if m + b > big_m:
            m = restart(x)
        t = _project_out(t, v)              # two passes (CGS2); combinations of t stay orthogonal to v
        t, kept = _svqb(t, drop)
        if kept == 0:
            break
        at = _apply_block(op, t)
        matvecs += kept
        v = _insert(v, t, m)
        w = _insert(w, at, m)
        hcol = np.asarray(_gram(v, at))      # (M, b): <v_i, A t_j>
        hrow = np.asarray(_gram(t, w))       # (b, M): <t_i, A v_j>
        h[: m + kept, m: m + kept] = hcol[: m + kept, :kept]
        h[m: m + kept, :m] = hrow[:kept, :m]
        m += kept
    return result(False)


# ---------------------------------------------------------------------------------------------------------------
# Mixed-precision refinement of the eigenpairs (the "refine" polish of the "mixed" mode).
#
# Each outer step:
#   1. complex128 Rayleigh-Ritz on the subspace V = [Q, T] (Q: an orthonormal basis of the last Ritz vectors X,
#      T: the last corrections), the complex128 residuals R = A X - X Theta and their norms;
#   2. complex64 correction solves: for each pair that is not converged, ``inner`` steps of GMRES (right
#      preconditioner: the fast-diagonalization preconditioner) on the projected correction equation of
#      Jacobi-Davidson,  (I - Q Q^H)(A - theta_i)(I - Q Q^H) t_i = -r_i,  t_i orthogonal to Q. The projection on the
#      whole block deflates the near eigenvalues (a cluster) of the other block vectors;
#   3. the corrections go back to complex128, are made orthogonal to Q and orthonormal (SVQB), and get one
#      complex128 operator apply each (the only complex128 applies of the step).
# This is iterative refinement (Dongarra, Moler and Wilkinson 1983; the correction equation of Sleijpen and van der
# Vorst 1996): the complex64 solves only change the rate of convergence. The accuracy comes from the complex128
# residuals and the complex128 Rayleigh-Ritz, so the stop test is the same as that of the complex128 Davidson.
# The first step selects the b Ritz pairs nearest the target; the next steps follow each of these b pairs by its Ritz
# value. Without this, a new Ritz value (of an eigenvalue that the complex64 Davidson did not find; residual 8.6e-2)
# replaced a converged pair (GPU run, 2026-10-30, 14 GMRES steps). The operator is not Hermitian, so a Rayleigh-Ritz
# step can also give spurious Ritz values. At each step the wanted pairs are the k tracked pairs nearest the target,
# with their near neighbours (``_selection``): with the wanted set of the first step, a mode count that cuts a
# near-degenerate pair (even and odd supermodes, splitting 2.7e-8 in neff) gave the other mode of the pair (review,
# 2026-10-30).
# An identified Ritz value that no pair follows (residual below REFINE_IDENTIFIED) replaces the followed value farthest
# from the target when it is nearer the target (``_track``): an eigenvalue that the complex64 Davidson did not find
# can come in (the second mode of a near-degenerate pair of window modes, examples/chip_to_fiber_coupling.py).
# The refinement stops with ``converged=False`` when the largest residual of the stop test does not halve in
# ``REFINE_STALL`` steps (for example a tolerance below the complex128 floor; the count starts again when a pair
# enters the stop test for the first time), when no correction is left, or at the step limit; it then gives the step
# with the smallest largest residual. ``davidson`` (precision "mixed") then runs the steps of the earlier polish from
# the complex64 block (``_refine_with_fallback``).
# ---------------------------------------------------------------------------------------------------------------

REFINE_INNER = 20         # GMRES steps of each complex64 correction solve
REFINE_STALL = 3          # stop (not converged) when the largest wanted residual did not halve in this many steps
REFINE_SWITCH_TOL = 1e-3  # the complex64 Davidson of the "refine" polish stops at this residual
REFINE_IDENTIFIED = 1e-2  # a tracked pair with a residual below this is an approximate eigenpair (see _selection)
DAVIDSON_SWITCH_TOL = 2e-5  # the complex64 Davidson of the "davidson" polish stops at this residual
POLISHES = ("refine", "davidson")


def _flat3(a: jax.Array) -> jax.Array:
    return a.reshape(a.shape[0], a.shape[1], -1)


@partial(jax.jit, static_argnums=(5,))
def _correction_krylov(op: DeviceOperator, pc: Preconditioner, q: jax.Array, theta: jax.Array, rhs: jax.Array,
                       steps: int):
    """``steps`` Arnoldi steps of right-preconditioned GMRES, in the dtype of rhs (complex64), for the projected
    correction equations P (A - theta_i) P t_i = rhs_i (P = I - q q^H; one equation for each row i of rhs). The
    right preconditioner is P M_i, with M_i = ``precondition`` at theta_i.

    q: (b, 2, Nx, Ny), orthonormal rows (zero rows have no effect); the rows of rhs must be orthogonal to q; a zero
    row of rhs gives zero results. Returns the Krylov basis (steps + 1, b, ...), the Hessenberg matrices
    (b, steps + 1, steps) and the norms of the rows of rhs (b,). The solution is ``_correction_solution``.
    """
    b = rhs.shape[0]
    shape = rhs.shape[1:]
    beta = jnp.sqrt(jnp.sum(jnp.abs(_flat(rhs)) ** 2, axis=1))
    v0 = rhs / jnp.where(beta > 0, beta, 1.0).astype(rhs.dtype)[:, None, None, None]
    vb = jnp.zeros((steps + 1, b) + shape, rhs.dtype).at[0].set(v0)
    hb = jnp.zeros((b, steps + 1, steps), rhs.dtype)
    th = theta[:, None, None, None]

    def body(j, carry):
        vb, hb = carry
        z = _project_out(precondition(pc, vb[j], theta), q)
        wv = _project_out(apply(op, z) - th * z, q)
        for _ in range(2):                    # classical Gram-Schmidt, two passes; rows > j of vb are zero
            # <v_j, w> = conj(sum v_j conj(w)): the conjugate goes on the small operand
            hj = jnp.einsum("jbn,bn->bj", _flat3(vb), _flat(wv).conj()).conj()
            wv = wv - jnp.einsum("bj,jbn->bn", hj, _flat3(vb)).reshape(wv.shape)
            hb = hb.at[:, :, j].add(hj)
        hn = jnp.sqrt(jnp.sum(jnp.abs(_flat(wv)) ** 2, axis=1))
        hb = hb.at[:, j + 1, j].set(hn.astype(hb.dtype))
        ok = hn > 1e-30                       # after a breakdown the next basis vectors are zero
        scale = jnp.where(ok, hn, 1.0).astype(wv.dtype)
        vb = vb.at[j + 1].set(jnp.where(ok[:, None, None, None], wv / scale[:, None, None, None], 0))
        return vb, hb

    vb, hb = jax.lax.fori_loop(0, steps, body, (vb, hb))
    return vb, hb, beta


@jax.jit
def _correction_solution(pc: Preconditioner, q: jax.Array, theta: jax.Array, y: jax.Array,
                         vb: jax.Array) -> jax.Array:
    """t_i = P M_i (sum_j y_ij v_j,i): the GMRES solution from the coefficients y (b, steps) and the Krylov basis
    (steps + 1, b, ...) of ``_correction_krylov`` (M_i is linear, so the preconditioned basis is not stored)."""
    steps = y.shape[1]
    u = jnp.einsum("bj,jbn->bn", y, _flat3(vb[:steps])).reshape(vb.shape[1:])
    return _project_out(precondition(pc, u, theta), q)


def _gmres_coef(hb: np.ndarray, beta: np.ndarray, active: np.ndarray) -> np.ndarray:
    """The GMRES coefficients y (b, steps): the least-squares solutions of H_i y_i = beta_i e_1 (host, complex128)."""
    b, s1, s = hb.shape
    y = np.zeros((b, s), dtype=np.complex128)
    for i in np.flatnonzero(active & (beta > 0)):
        e1 = np.zeros(s1, dtype=np.complex128)
        e1[0] = beta[i]
        y[i] = np.linalg.lstsq(hb[i].astype(np.complex128), e1, rcond=None)[0]
    return y


def _track(vals: np.ndarray, prev: np.ndarray, rn_vals: np.ndarray | None = None,
           target: complex = 0.0) -> np.ndarray:
    """The indices of the Ritz values ``vals`` that continue the pairs with the values ``prev`` (one for each pair,
    in the order of prev; greedy: the nearest (pair, value) first). Pairs with no value left are left out, and the
    next pairs move one index down: this has no effect on the result, because ``_selection`` chooses the wanted
    pairs by their distance to the target, not by their index.

    rn_vals (the relative residuals of all values): then an identified value that no pair follows (residual below
    REFINE_IDENTIFIED) replaces the followed value farthest from the target, when it is nearer the target. This
    lets an eigenvalue that the complex64 Davidson did not find come in (for example the second mode of a
    near-degenerate pair), but not a spurious Ritz value (its residual is large)."""
    d = np.abs(prev[:, None] - vals[None, :])
    out = np.full(len(prev), -1)
    used = np.zeros(len(vals), dtype=bool)
    for flat in np.argsort(d, axis=None, kind="stable"):
        i, j = divmod(int(flat), len(vals))
        if out[i] < 0 and not used[j]:
            out[i], used[j] = j, True
    out = out[out >= 0]
    if rn_vals is None or len(out) == 0:
        return out
    dist = np.abs(vals - target)
    for j in np.argsort(dist, kind="stable"):
        if used[j] or not rn_vals[j] < REFINE_IDENTIFIED:
            continue
        far = int(np.argmax(dist[out]))
        if dist[j] >= dist[out[far]]:
            break
        used[out[far]], used[j] = False, True
        out[far] = j
    return out


@jax.jit
def _take_rows(xa: jax.Array, ra: jax.Array, sel: jax.Array, mask: jax.Array):
    """Rows sel of xa and ra; the rows without ``mask`` are zero."""
    m4 = mask[:, None, None, None]
    return jnp.where(m4, xa[sel], 0), jnp.where(m4, ra[sel], 0)


def _selection(theta: np.ndarray, rn: np.ndarray, nb: int, k: int, target: complex, tol: float):
    """The wanted pairs of a refinement step: the k tracked pairs (of the first nb) nearest the target, and the
    pairs that must also converge before the stop. These are the other identified pairs (residual from tol to
    REFINE_IDENTIFIED) whose Ritz value can still be nearer the target than a wanted one: |theta_j - target| minus
    the absolute residual of pair j (an eigenvalue error bound for a near-normal operator) is not more than the
    largest |theta_i - target| plus residual of the wanted pairs. So a near-degenerate pair that the complex64 Ritz
    values cannot order (an even and an odd supermode) is ordered by converged values. Returns (the indices of the
    pairs that the stop test uses, the largest of their residuals)."""
    if nb < k:
        return np.arange(nb), np.inf
    dist = np.abs(theta[:nb] - target)
    err = rn[:nb] * np.maximum(1.0, np.abs(theta[:nb]))
    order = np.argsort(dist, kind="stable")
    sel, rest = order[:k], order[k:]
    edge = np.max(dist[sel] + err[sel])
    near = rest[(rn[rest] >= tol) & (rn[rest] < REFINE_IDENTIFIED) & (dist[rest] - err[rest] <= edge)]
    used = np.concatenate([sel, near])
    return used, float(rn[used].max())


def refine(op32: DeviceOperator, op64: DeviceOperator, pc: Preconditioner, k: int, target: complex, x0: jax.Array,
           tol: float, max_steps: int, inner: int = REFINE_INNER, verbose: bool = False) -> EigResult:
    """Mixed-precision refinement of the k eigenpairs nearest ``target`` (see the section comment above).

    x0: (b, 2, Nx, Ny), the approximate block (for example the Ritz vectors of a complex64 Davidson). The first step
    selects the b Ritz pairs nearest the target; each next step follows these b pairs by their Ritz values
    (``_track``), so a new Ritz value (for example a spurious one, or an eigenvalue that the complex64 Davidson did
    not find) does not replace a pair. At each step the wanted pairs are the k tracked pairs nearest the target
    (``_selection``; not the pairs of the first step: the complex64 Ritz values cannot order a near-degenerate
    pair). The result has the complex128 residuals, all b pairs sorted by the distance to the target (the k wanted
    pairs first); ``iterations`` counts the outer steps; ``matvecs`` counts the complex128 applies and the complex64
    applies of the correction solves; ``matvecs_hi`` counts the complex128 applies. Not converged: the result of the
    step with the smallest largest residual of the stop test.
    """
    require_x64("mixed")
    b = x0.shape[0]
    big_m = 2 * b
    drop = 1e-10                                     # as the complex128 Davidson
    x0 = put(x0.astype(jnp.complex128))
    zeros = put(jnp.zeros((big_m,) + x0.shape[1:], jnp.complex128))
    q, kq = _svqb(x0, drop)
    aq = _apply_block(op64, q)
    matvecs_hi, matvecs_lo = kq, 0
    v = _insert(jnp.zeros_like(zeros), q, 0)
    w = _insert(jnp.zeros_like(zeros), aq, 0)
    h = np.zeros((big_m, big_m), dtype=np.complex128)
    h[:kq, :kq] = np.asarray(_gram(q, aq))[:kq, :kq]
    m = kq
    theta = np.zeros(b, complex)
    rn = np.full(b, np.inf)
    x = x0
    best, best_it, it = np.inf, 0, 0
    nb, seen = 0, set()                              # seen: the pairs that were in a stop test
    best_state = None                                # (theta, x, rn, nb, worst) of the step with the smallest worst

    def result(ok):
        th, xs, res, n = (theta, x, rn, nb) if ok or best_state is None else best_state[:4]
        perm = np.concatenate([np.argsort(np.abs(th[:n] - target), kind="stable"), np.arange(n, b)])
        if np.any(perm != np.arange(b)):
            xs = xs[put(perm.astype(np.int32))]
        return EigResult(th[perm][:k], xs, res[perm][:k], it, matvecs_hi + matvecs_lo, ok, matvecs_hi)

    for it in range(1, max_steps + 1):
        # 1. complex128 Rayleigh-Ritz and residuals
        vals, y = np.linalg.eig(h[:m, :m])
        yall = np.zeros((big_m, big_m), dtype=np.complex128)        # all m Ritz pairs, for _track
        yall[:m, :m] = y
        tall = np.zeros(big_m, dtype=np.complex128)
        tall[:m] = vals
        xa, ra, rna_dev = _ritz(put(yall, jnp.complex128), put(tall, jnp.complex128), v, w)
        rna = np.array(rna_dev, dtype=np.float64)
        rna[m:] = np.inf
        if it == 1:
            order = np.argsort(np.abs(vals - target))[:b]
        else:
            order = _track(vals, theta[:nb], rna[:m], target)
        nb = len(order)
        theta = np.zeros(b, complex)
        theta[:nb] = vals[order]
        ypad = np.zeros((big_m, b), dtype=np.complex128)
        ypad[:m, :nb] = y[:, order]
        sel = np.zeros(b, dtype=np.int32)
        sel[:nb] = order
        x, r = _take_rows(xa, ra, put(sel), put(np.arange(b) < nb))
        rn = np.full(b, np.inf)
        rn[:nb] = rna[order]
        used, worst = _selection(theta, rn, nb, k, target, tol)
        if verbose:
            print(f"  refine {it:3d} m {m:3d} max res {worst:.2e} ({len(used)} pairs) theta0 {theta[0]:.14f}",
                  flush=True)
            if verbose > 1:                  # the orthogonality of V, the residual and the Ritz value of each row
                orth = np.abs(np.asarray(_gram(v, v))[:m, :m] - np.eye(m)).max()
                wide = partial(np.array2string, max_line_width=100000)
                print(f"    orth {orth:.1e} rn {wide(rn, precision=1)} theta {wide(theta, precision=6)}", flush=True)
        if worst < tol:
            return result(True)
        if best_state is None or worst < best_state[4]:
            best_state = (theta, x, rn, nb, worst)
        if not seen.issuperset(used.tolist()):
            best, best_it = worst, it                # a pair is new in the stop test: the halving rule starts again
            seen.update(used.tolist())               # (at most b times)
        elif worst < 0.5 * best:
            best, best_it = worst, it
        elif it - best_it >= REFINE_STALL:
            return result(False)
        # restart basis Q = X C (orthonormal; V is orthonormal, so X^H X = Y^H Y), and A Q = W Y C
        coef, kq = _svqb_coef(ypad[:m, :nb].conj().T @ ypad[:m, :nb], drop)
        zc = np.zeros((big_m, b), dtype=np.complex128)
        zc[:m, :nb] = ypad[:m, :nb] @ coef
        hq = zc[:m].conj().T @ h[:m, :m] @ zc[:m]              # Q^H A Q
        zc_dev = put(zc)
        q, aq = _combine(zc_dev, v), _combine(zc_dev, w)
        # 2. complex64 correction solves for the pairs that are not converged
        active = rn >= tol
        rhs = jnp.where(put(active)[:, None, None, None], -r, 0).astype(jnp.complex64)
        q32, theta32 = q.astype(jnp.complex64), put(theta, jnp.complex64)
        vk, hk, beta = _correction_krylov(op32, pc, q32, theta32, rhs, inner)
        beta = np.asarray(beta, dtype=np.float64)
        yk = _gmres_coef(np.asarray(hk), beta, active)
        matvecs_lo += inner * int(np.count_nonzero(active & (beta > 0)))
        t = _correction_solution(pc, q32, theta32, put(yk, jnp.complex64), vk).astype(jnp.complex128)
        # 3. back to complex128: orthogonal to Q, orthonormal, one complex128 apply each
        t = _project_out(t, q)
        t, kt = _svqb(t, drop)
        if kt == 0:
            return result(False)
        at = _apply_block(op64, t)
        matvecs_hi += kt
        v = _insert(_insert(jnp.zeros_like(zeros), q, 0), t, kq)     # rows kq.. of q are zero
        w = _insert(_insert(jnp.zeros_like(zeros), aq, 0), at, kq)
        hcol = np.asarray(_gram(v, at))       # (2b, b): <v_i, A t_j>
        hrow = np.asarray(_gram(t, w))        # (b, 2b): <t_i, A v_j>
        h[:] = 0
        h[:kq, :kq] = hq[:kq, :kq]
        h[: kq + kt, kq: kq + kt] = hcol[: kq + kt, :kt]
        h[kq: kq + kt, :kq] = hrow[:kt, :kq]
        m = kq + kt
    return result(False)


def davidson(op: DeviceOperator | tuple[DeviceOperator, DeviceOperator], pc: Preconditioner, k: int,
             target: complex, x0: jax.Array, tol: float = 1e-10, max_iter: int = 300,
             max_subspace: int | None = None, precision: str = "mixed", switch_tol: float | None = None,
             verbose: bool = False, polish: str = "refine", inner: int = REFINE_INNER) -> EigResult:
    """The k eigenpairs of the operator nearest ``target``.

    op: the device operator in complex128 (``precision`` "double"), complex64 ("single"), or a pair
        (complex64, complex128) for "mixed".
    x0: (b, 2, Nx, Ny), b >= k: the initial block (random, or the modes of a similar problem).
    polish ("mixed" only): "refine" (the mixed-precision refinement, ``refine``) or "davidson" (a complex128
        Davidson with a subspace of 3 blocks). switch_tol: the residual at which the complex64 Davidson stops
        (default: REFINE_SWITCH_TOL or DAVIDSON_SWITCH_TOL). inner: the GMRES steps of each correction solve.
    """
    b = x0.shape[0]
    if b < k:
        raise ValueError("the initial block needs at least k vectors")
    if polish not in POLISHES:
        raise ValueError(f"polish must be one of {POLISHES}")
    require_x64(precision)
    max_subspace = max_subspace or max(4 * b, b + 2 * k)   # the host eig costs O(max_subspace^3)
    if precision == "mixed":
        op32, op64 = op
        lo = _davidson_fixed(op32, pc, k, target, x0.astype(jnp.complex64),
                             max(_switch_tol(polish, switch_tol), tol), max_iter, max_subspace, verbose,
                             stall_iters=12)
        if lo.iterations >= max_iter:
            return lo                         # no iterations left for the polish: the complex64 result, not converged
        left = max(0, max_iter - lo.iterations)
        if polish == "refine":
            hi = _refine_with_fallback(op32, op64, pc, k, target, lo.vectors, tol, left, inner, verbose,
                                       max_subspace)
        else:
            # complex128 products are slow on consumer GPUs: the polish uses a small subspace (3 blocks)
            hi = _davidson_fixed(op64, pc, k, target, lo.vectors.astype(jnp.complex128), tol, left, 3 * b, verbose)
            hi.polish = "davidson"
        hi.iterations += lo.iterations
        hi.matvecs += lo.matvecs
        return hi
    dtype = jnp.complex64 if precision == "single" else jnp.complex128
    return _davidson_fixed(op, pc, k, target, x0.astype(dtype), tol, max_iter, max_subspace, verbose,
                           stall_iters=12 if precision == "single" else 0)


FALLBACK_STALL = 30       # the complex128 stage of the fallback stops when its residual did not halve in this many
#                           iterations (only below STALL_FLOOR: a tolerance below the complex128 floor)


def _refine_with_fallback(op32: DeviceOperator, op64: DeviceOperator, pc: Preconditioner, k: int, target: complex,
                          lo_vectors: jax.Array, tol: float, left: int, inner: int, verbose: bool,
                          max_subspace: int) -> EigResult:
    """``refine`` from the complex64 block ``lo_vectors``. When it stops without convergence and iterations are left,
    the fallback runs the steps of the earlier polish from the same complex64 block: the complex64 Davidson
    continues to DAVIDSON_SWITCH_TOL, then the complex128 Davidson with a subspace of 3 blocks (with a stall rule,
    FALLBACK_STALL). The fallback does not start from the refined block: from there it cannot find an eigenvalue that
    the refined block misses (the second mode of a near-degenerate pair; review, 2026-10-30).

    Result: that of the refinement ("refine"), or (``polish`` "refine+davidson") that of the fallback when it
    converged, else the one of the two with the smaller largest residual. converged=True always means that the k
    pairs have complex128 residuals below tol. The counts include both parts."""
    hi = refine(op32, op64, pc, k, target, lo_vectors, tol, left, inner, verbose)
    hi.polish = "refine"
    rest = left - hi.iterations
    if hi.converged or rest <= 0:
        return hi
    b = lo_vectors.shape[0]
    fb = _davidson_fixed(op32, pc, k, target, lo_vectors.astype(jnp.complex64), max(DAVIDSON_SWITCH_TOL, tol), rest,
                         max_subspace, verbose, stall_iters=12)
    if fb.iterations < rest:
        lo_iters, lo_mv = fb.iterations, fb.matvecs
        fb = _davidson_fixed(op64, pc, k, target, fb.vectors.astype(jnp.complex128), tol, rest - lo_iters, 3 * b,
                             verbose, stall_iters=FALLBACK_STALL)
        fb.iterations += lo_iters
        fb.matvecs += lo_mv
    best = fb if fb.converged or np.max(fb.residuals) <= np.max(hi.residuals) else hi
    return EigResult(best.values, best.vectors.astype(jnp.complex128), best.residuals, hi.iterations + fb.iterations,
                     hi.matvecs + fb.matvecs, best.converged, hi.matvecs_hi + fb.matvecs_hi, "refine+davidson")


def _switch_tol(polish: str, switch_tol: float | None) -> float:
    if switch_tol is not None:
        return switch_tol
    return REFINE_SWITCH_TOL if polish == "refine" else DAVIDSON_SWITCH_TOL


# ---------------------------------------------------------------------------------------------------------------
# Batched solve: B problems with the same grid shape at the same time.
#
# The kernels above, vectorized with jax.vmap over a leading batch axis. The operators and the preconditioners are
# pytrees with a leading axis B on each leaf (``stack``); the blocks are (B, b, 2, Nx, Ny) and the subspace buffers
# are (B, M, 2, Nx, Ny). The small projected eigenproblems and the SVQB stay on the host, one for each member, in a
# thread pool (LAPACK releases the GIL). Each member has its own subspace size, restarts, stall counter and stop
# (the kept count of SVQB is different in each member). When a member stops, its result is kept. After that its
# rows of the block are zero: the inserts write zero rows into its buffers (rows that it does not use), and its
# result does not change. The kernels still run on these rows (fixed shapes): a batch costs as many iterations as
# its slowest member.
# ---------------------------------------------------------------------------------------------------------------

HOST_THREADS = max(1, min(8, os.cpu_count() or 1))
_POOL: ThreadPoolExecutor | None = None


def _host_map(f, args: list[tuple]) -> list:
    """[f(*a) for a in args], in the host thread pool when there is more than one item."""
    global _POOL
    if len(args) < 2 or HOST_THREADS < 2:
        return [f(*a) for a in args]
    if _POOL is None:
        _POOL = ThreadPoolExecutor(HOST_THREADS, thread_name_prefix="meow-gpu-host")
    return list(_POOL.map(lambda a: f(*a), args))


def stack(trees):
    """One pytree with a new leading batch axis, from B pytrees with the same structure and shapes (the device
    operators or the preconditioners of B problems with the same grid shape)."""
    return jax.tree.map(lambda *a: jnp.stack(a), *trees)


_gram_b = jax.jit(jax.vmap(_gram))
_combine_b = jax.jit(jax.vmap(_combine))
_ritz_b = jax.jit(jax.vmap(_ritz))
_project_out_b = jax.jit(jax.vmap(_project_out))
_apply_b = jax.jit(jax.vmap(apply))


@jax.jit
def precondition_batch(pc: Preconditioner, r: jax.Array, theta: jax.Array) -> jax.Array:
    """``precondition`` for each member: pc batched (``stack``), r (B, b, 2, Nx, Ny), theta (B, b)."""
    return jax.vmap(precondition)(pc, r, theta)


@jax.jit
def _expand_b(pc: Preconditioner, r: jax.Array, theta: jax.Array, active: jax.Array) -> jax.Array:
    """The preconditioned residuals of the active rows; the other rows are zero."""
    t = jax.vmap(precondition)(pc, r, theta)
    return jnp.where(active[:, :, None, None, None], t, 0)


@partial(jax.jit, donate_argnums=(0,))
def _insert_b(buf: jax.Array, block: jax.Array, start: jax.Array) -> jax.Array:
    """The block of member i into its buffer at the row start[i]."""
    return jax.vmap(lambda bf, bl, s: jax.lax.dynamic_update_slice_in_dim(bf, bl, s, axis=0))(buf, block, start)


@partial(jax.jit, donate_argnums=(0,))
def _reset_b(buf: jax.Array, block: jax.Array, flag: jax.Array) -> jax.Array:
    """For each member with ``flag``: the block in the first rows of the buffer, zero in the other rows. The
    buffers of the other members do not change."""
    def one(bf, bl, f):
        return jnp.where(f, jax.lax.dynamic_update_slice_in_dim(jnp.zeros_like(bf), bl, 0, axis=0), bf)
    return jax.vmap(one)(buf, block, flag)


@jax.jit
def _select_b(flag: jax.Array, new: jax.Array, old: jax.Array) -> jax.Array:
    return jnp.where(flag[:, None, None, None, None], new, old)


@jax.jit
def _h_new_b(v: jax.Array, at: jax.Array, t: jax.Array, w: jax.Array):
    """The new columns <v_i, A t_j> (B, M, b) and rows <t_i, A v_j> (B, b, M) of the projected matrices."""
    return jax.vmap(_gram)(v, at), jax.vmap(_gram)(t, w)


def _svqb_b(t: jax.Array, drop: float, members: np.ndarray) -> tuple[jax.Array, np.ndarray]:
    """SVQB of the block of each member in ``members`` (bool, (B,)); the blocks of the other members become zero."""
    g = np.array(_gram_b(t, t), dtype=np.complex128)
    coef = np.zeros(g.shape, dtype=np.complex128)
    kept = np.zeros(t.shape[0], dtype=int)
    idx = np.flatnonzero(members)
    for i, (c, n) in zip(idx, _host_map(_svqb_coef, [(g[i], drop) for i in idx])):
        coef[i], kept[i] = c, n
    return _combine_b(put(coef, t.dtype), t), kept


def _ritz_coef(h: np.ndarray, m: int, target: complex, b: int) -> tuple[np.ndarray, np.ndarray]:
    """The (at most b) Ritz values of h[:m, :m] nearest the target, and their coefficient vectors (m, nb)."""
    vals, y = np.linalg.eig(h[:m, :m])
    order = np.argsort(np.abs(vals - target))[:b]
    return vals[order], y[:, order]


@dataclass
class BatchEigResult:
    values: np.ndarray        # (B, k) complex eigenvalues of each member, sorted by distance to its target
    vectors: jax.Array        # (B, b, 2, Nx, Ny): the k wanted Ritz vectors of each member first
    residuals: np.ndarray     # (B, k) relative residual norms
    iterations: np.ndarray    # (B,) iterations of each member
    matvecs: np.ndarray       # (B,)
    converged: np.ndarray     # (B,) bool
    matvecs_hi: np.ndarray | None = None   # (B,) the complex128 applies (None: zero)
    polishes: list[str] | None = None      # (B,) the polish of each member, as EigResult.polish (None: "")

    def __len__(self) -> int:
        return len(self.iterations)

    def member(self, i: int) -> EigResult:
        """The result of member i, as ``davidson`` gives it."""
        hi = 0 if self.matvecs_hi is None else int(self.matvecs_hi[i])
        polish = "" if self.polishes is None else self.polishes[i]
        return EigResult(self.values[i], self.vectors[i], self.residuals[i], int(self.iterations[i]),
                         int(self.matvecs[i]), bool(self.converged[i]), hi, polish)


def _davidson_batch_fixed(op: DeviceOperator, pc: Preconditioner, k: int, target: np.ndarray, x0: jax.Array,
                          tol: np.ndarray, max_iter: np.ndarray, max_subspace: int, verbose: bool,
                          stall_iters: int = 0) -> BatchEigResult:
    """``_davidson_fixed`` for each member of a batch, in the dtype of x0 (B, b, 2, Nx, Ny). ``target``, ``tol``
    and ``max_iter`` have one value for each member."""
    dtype = x0.dtype
    nbat, b = x0.shape[:2]
    big_m = max_subspace
    drop = 1e-4 if dtype == jnp.complex64 else 1e-10      # as in _davidson_fixed
    target = np.broadcast_to(np.asarray(target, dtype=complex), (nbat,))
    tol = np.broadcast_to(np.asarray(tol, dtype=float), (nbat,))
    max_iter = np.broadcast_to(np.asarray(max_iter, dtype=int), (nbat,))
    x0 = put(x0)
    v = jnp.zeros((nbat, big_m) + x0.shape[2:], dtype, device=solver_device())
    w = jnp.zeros((nbat, big_m) + x0.shape[2:], dtype, device=solver_device())
    h = np.zeros((nbat, big_m, big_m), dtype=np.complex128)
    m = np.zeros(nbat, dtype=int)                          # the subspace size of each member
    matvecs = np.zeros(nbat, dtype=int)

    def restart(block, flag):
        nonlocal v, w
        q, kept = _svqb_b(block, drop, flag)
        aq = _apply_b(op, q)
        fl = put(flag)
        v = _reset_b(v, q, fl)
        w = _reset_b(w, aq, fl)
        g = np.asarray(_gram_b(q, aq))
        for i in np.flatnonzero(flag):
            n = kept[i]
            h[i] = 0
            h[i, :n, :n] = g[i, :n, :n]
            m[i] = n
            matvecs[i] += n

    restart(x0, np.ones(nbat, dtype=bool))
    values = np.zeros((nbat, k), dtype=complex)
    resid = np.full((nbat, k), np.inf)
    iters = np.zeros(nbat, dtype=int)
    conv = np.zeros(nbat, dtype=bool)
    xf = x0                                                # the vectors of the members that stopped
    done = max_iter <= 0                                   # a new (writable) array
    best, best_it = np.full(nbat, np.inf), np.zeros(nbat, dtype=int)
    rows = np.arange(b)[None, :]

    def finish(flag):
        """Keep the current Ritz pairs of the members with ``flag`` as their result; they stop."""
        nonlocal xf
        values[flag], resid[flag], iters[flag], conv[flag] = theta[flag, :k], rn[flag, :k], it, ok[flag]
        xf = _select_b(put(flag), x, xf)
        done[flag] = True

    for it in range(1, int(max_iter.max(initial=0)) + 1):
        live = np.flatnonzero(~done)
        theta = np.zeros((nbat, b), dtype=complex)
        ypad = np.zeros((nbat, big_m, b), dtype=np.complex128)
        nb = np.zeros(nbat, dtype=int)
        for i, (vals, y) in zip(live, _host_map(_ritz_coef, [(h[i], m[i], target[i], b) for i in live])):
            nb[i] = len(vals)
            theta[i, : nb[i]] = vals
            ypad[i, : m[i], : nb[i]] = y
        theta_dev = put(theta, dtype)
        x, r, rn_dev = _ritz_b(put(ypad, dtype), theta_dev, v, w)
        rn = np.array(rn_dev, dtype=np.float64)            # a writable copy
        rn[rows >= nb[:, None]] = np.inf
        rn[done] = np.inf                                  # stopped members: x = 0 and rn = 0/0
        worst = rn[:, :k].max(axis=1)
        ok = ~done & np.all(rn[:, :k] < tol[:, None], axis=1)
        # the stall rule of _davidson_fixed, for each member
        better = ~done & ~ok & (worst < 0.5 * best)
        best[better], best_it[better] = worst[better], it
        stalled = ~done & ~ok & ~better & (stall_iters > 0) & (best < STALL_FLOOR) & (it - best_it >= stall_iters)
        stop = ok | stalled
        # at the iteration limit, _davidson_fixed expands the subspace once more (its loop ends after that step) and
        # returns the Ritz pairs of this iteration: these members stop after the expansion, with the same matvecs
        limit = ~done & ~stop & (it >= max_iter)
        if verbose:
            print(f"  it {it:3d} live {len(live):3d} m {m[live].max():4d} max res {worst[live].max():.2e}",
                  flush=True)
        if stop.any():
            finish(stop)
        if done.all():
            break
        active = (rn >= tol[:, None]) & ~done[:, None]
        t = _expand_b(pc, r, theta_dev, put(active))
        full = ~done & (m + b > big_m)
        if full.any():
            restart(x, full)
        t = _project_out_b(t, v)                           # two passes (CGS2), as in _davidson_fixed
        t, kept = _svqb_b(t, drop, ~done)
        empty = ~done & (kept == 0)
        if empty.any():
            finish(empty)                                  # as the "break" of _davidson_fixed (ok is False here)
        if done.all():
            break
        at = _apply_b(op, t)
        start = put(np.minimum(m, big_m - b).astype(np.int32))   # stopped members: rows that are not used
        v = _insert_b(v, t, start)
        w = _insert_b(w, at, start)
        hcol, hrow = jax.device_get(_h_new_b(v, at, t, w))
        for i in np.flatnonzero(~done):
            mi, n = m[i], kept[i]
            h[i, : mi + n, mi: mi + n] = hcol[i, : mi + n, :n]
            h[i, mi: mi + n, :mi] = hrow[i, :n, :mi]
            m[i] += n
            matvecs[i] += n
        limit &= ~done
        if limit.any():
            finish(limit)                                  # the Ritz pairs of this iteration (x, theta, rn)
    return BatchEigResult(values, xf, resid, iters, matvecs, conv,
                          matvecs.copy() if dtype == jnp.complex128 else np.zeros(nbat, dtype=int))


def davidson_batch(op: DeviceOperator | tuple[DeviceOperator, DeviceOperator], pc: Preconditioner, k: int,
                   target, x0: jax.Array, tol=1e-10, max_iter=300, max_subspace: int | None = None,
                   precision: str = "mixed", switch_tol: float | None = None, verbose: bool = False,
                   polish: str = "refine", inner: int = REFINE_INNER) -> BatchEigResult:
    """The k eigenpairs nearest the target of each of B problems with the same grid shape, solved at the same time.

    op: a batched device operator (``stack`` of B operators) in complex128 ("double"), complex64 ("single"), or a
        pair (complex64, complex128) for "mixed".
    pc: a batched preconditioner (``stack`` of B preconditioners).
    target, tol, max_iter: one value for all members, or an array of B values.
    x0: (B, b, 2, Nx, Ny), b >= k: the initial block of each member.
    polish, switch_tol, inner: as for ``davidson``.

    Each member does the steps of ``davidson`` on its own problem (the same precision modes and stall rule); the
    device kernels run on all members together. Exception: the "refine" polish runs member after member (``refine``
    on the operator and the preconditioner of each member), after the batched complex64 Davidson.
    """
    nbat, b = x0.shape[:2]
    if b < k:
        raise ValueError("the initial block needs at least k vectors")
    if polish not in POLISHES:
        raise ValueError(f"polish must be one of {POLISHES}")
    require_x64(precision)
    max_subspace = max_subspace or max(4 * b, b + 2 * k)
    target = np.broadcast_to(np.asarray(target, dtype=complex), (nbat,))
    tol = np.broadcast_to(np.asarray(tol, dtype=float), (nbat,))
    max_iter = np.broadcast_to(np.asarray(max_iter, dtype=int), (nbat,))
    if precision == "mixed":
        op32, op64 = op
        lo = _davidson_batch_fixed(op32, pc, k, target, x0.astype(jnp.complex64),
                                   np.maximum(_switch_tol(polish, switch_tol), tol), max_iter, max_subspace, verbose,
                                   stall_iters=12)
        left = np.maximum(0, max_iter - lo.iterations)
        todo = left > 0                      # members with no iterations left keep the complex64 result
        if not todo.any():
            return lo
        if polish == "refine":
            return _refine_members(op32, op64, pc, k, target, tol, left, inner, verbose, lo, todo, max_subspace)
        hi = _davidson_batch_fixed(op64, pc, k, target, lo.vectors.astype(jnp.complex128), tol, left, 3 * b,
                                   verbose)
        keep = ~todo
        hi.values[keep] = lo.values[keep]
        hi.residuals[keep] = lo.residuals[keep]
        hi.converged[keep] = lo.converged[keep]
        hi.iterations = hi.iterations + lo.iterations
        hi.matvecs = np.where(todo, hi.matvecs + lo.matvecs, lo.matvecs)
        hi.matvecs_hi = np.where(todo, hi.matvecs_hi, 0)
        hi.polishes = ["davidson" if t else "" for t in todo]
        return hi
    dtype = jnp.complex64 if precision == "single" else jnp.complex128
    return _davidson_batch_fixed(op, pc, k, target, x0.astype(dtype), tol, max_iter, max_subspace, verbose,
                                 stall_iters=12 if precision == "single" else 0)


def _refine_members(op32, op64, pc, k, target, tol, left, inner, verbose, lo: BatchEigResult,
                    todo: np.ndarray, max_subspace: int) -> BatchEigResult:
    """``refine`` (with the fallback of ``_refine_with_fallback``) for each member with ``todo``, from the complex64
    result ``lo`` of the batch; the other members keep their complex64 result (as ``davidson`` does when no
    iterations are left)."""
    def member(tree, i):
        return jax.tree.map(lambda a: a[i], tree)

    vectors = lo.vectors.astype(jnp.complex128)
    out = BatchEigResult(lo.values.copy(), vectors, lo.residuals.copy(), lo.iterations.copy(), lo.matvecs.copy(),
                         lo.converged.copy(), np.zeros(len(lo), dtype=int), [""] * len(lo))
    rows = []
    for i in np.flatnonzero(todo):
        r = _refine_with_fallback(member(op32, i), member(op64, i), member(pc, i), k, target[i], lo.vectors[i],
                                  float(tol[i]), int(left[i]), inner, verbose, max_subspace)
        out.polishes[i] = r.polish
        out.values[i], out.residuals[i], out.converged[i] = r.values, r.residuals, r.converged
        out.iterations[i] += r.iterations
        out.matvecs[i] += r.matvecs
        out.matvecs_hi[i] = r.matvecs_hi
        rows.append((i, r.vectors))
    idx = put(np.array([i for i, _ in rows], dtype=np.int32))
    out.vectors = vectors.at[idx].set(jnp.stack([v for _, v in rows]))
    return out
