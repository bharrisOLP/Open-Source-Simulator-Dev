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
"mixed" mode the subspace iteration runs in complex64 to a residual of about ``switch_tol`` and then continues in
complex128 from the complex64 Ritz vectors to ``tol``.

All device arrays have fixed shapes (the subspace is a buffer of ``max_subspace`` vectors), so JAX compiles each
kernel once per grid size, block size and precision.

A warm start (the modes of the previous cell of a taper, or of the previous wavelength) is the initial block.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from meow_gpu.device import put
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
    b = t.shape[0]
    g = np.array(_gram(t, t), dtype=np.complex128)   # host math in double (float32 underflows the guard)
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
    return _combine(put(coef, t.dtype), t), kept


@dataclass
class EigResult:
    values: np.ndarray        # (k,) complex eigenvalues, sorted by distance to the target
    vectors: jax.Array        # (b, 2, Nx, Ny): the k wanted Ritz vectors first, then the extra block vectors
    residuals: np.ndarray     # (k,) relative residual norms
    iterations: int
    matvecs: int
    converged: bool


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
            return EigResult(theta[:k], x, rn[:k], it, matvecs, True)
        # stall: only near the precision floor (early plateaus are normal while the Ritz values settle)
        if rn[:k].max() < 0.5 * best:
            best, best_it = rn[:k].max(), it
        elif stall_iters and best < STALL_FLOOR and it - best_it >= stall_iters:
            return EigResult(theta[:k], x, rn[:k], it, matvecs, False)
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
    return EigResult(theta[:k], x, rn[:k], it, matvecs, False)


def davidson(op: DeviceOperator | tuple[DeviceOperator, DeviceOperator], pc: Preconditioner, k: int,
             target: complex, x0: jax.Array, tol: float = 1e-10, max_iter: int = 300,
             max_subspace: int | None = None, precision: str = "mixed", switch_tol: float = 2e-5,
             verbose: bool = False) -> EigResult:
    """The k eigenpairs of the operator nearest ``target``.

    op: the device operator in complex128 (``precision`` "double"), complex64 ("single"), or a pair
        (complex64, complex128) for "mixed".
    x0: (b, 2, Nx, Ny), b >= k: the initial block (random, or the modes of a similar problem).
    """
    b = x0.shape[0]
    if b < k:
        raise ValueError("the initial block needs at least k vectors")
    max_subspace = max_subspace or max(4 * b, b + 2 * k)   # the host eig costs O(max_subspace^3)
    if precision == "mixed":
        op32, op64 = op
        lo = _davidson_fixed(op32, pc, k, target, x0.astype(jnp.complex64), max(switch_tol, tol), max_iter,
                             max_subspace, verbose, stall_iters=12)
        if lo.iterations >= max_iter:
            return lo                         # no iterations left for the polish: the complex64 result, not converged
        # complex128 products are slow on consumer GPUs: the polish uses a small subspace (3 blocks)
        hi = _davidson_fixed(op64, pc, k, target, lo.vectors.astype(jnp.complex128), tol,
                             max(0, max_iter - lo.iterations), 3 * b, verbose)
        hi.iterations += lo.iterations
        hi.matvecs += lo.matvecs
        return hi
    dtype = jnp.complex64 if precision == "single" else jnp.complex128
    return _davidson_fixed(op, pc, k, target, x0.astype(dtype), tol, max_iter, max_subspace, verbose,
                           stall_iters=12 if precision == "single" else 0)

