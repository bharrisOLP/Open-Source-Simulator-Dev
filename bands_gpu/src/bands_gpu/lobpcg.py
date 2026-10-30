"""Block LOBPCG (Knyazev 2001) for the smallest eigenpairs of a Hermitian operator, on the GPU.

Device (JAX): the operator products, the preconditioner, the block products (all large arrays, fixed shapes).
Host (numpy): the small Hermitian eigenproblems (3b x 3b) and the SVQB orthogonalization coefficients.

The trial subspace S = [X, W, P] (Ritz vectors, preconditioned residuals, previous directions) is orthonormalized
with SVQB (Stathopoulos and Wu 2002) before the Rayleigh-Ritz step, so that a nearly dependent S does not break
the solve. P is the part of the new Ritz vectors orthogonal to the old ones (a stable form of the LOBPCG direction),
also made orthogonal to the new Ritz vectors. A X and A P are computed as products in each iteration (three block
products per iteration, with A W), so that no product drifts and the cancellation in P is consistent with A P.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np


def _flat(a):
    return a.reshape(a.shape[0], -1)


@jax.jit
def _gram(a, b):
    """<a_i, b_j>; the conjugate goes on the smaller block."""
    if a.shape[0] > b.shape[0]:
        return (_flat(b).conj() @ _flat(a).T).T.conj()
    return _flat(a).conj() @ _flat(b).T


@jax.jit
def _combine(c, v):
    """sum_i c_ij v_i for coefficients c (m, k): a block (k, ...)."""
    return (c.T @ _flat(v)).reshape((c.shape[1],) + v.shape[1:])


def _svqb_coef(g: np.ndarray, drop: float) -> np.ndarray:
    """Coefficients C (m x m) such that the rows of C^T V are orthonormal (Gram g of V); dropped columns are 0."""
    g = np.asarray(g, np.complex128)
    diag = np.real(np.diag(g))
    d = np.where(diag > 1e-30 * max(diag.max(), 1e-300), np.sqrt(np.maximum(diag, 0.0)), 1.0)
    gs = g / d[:, None] / d[None, :]
    gs = 0.5 * (gs + gs.conj().T)
    s, u = np.linalg.eigh(gs)
    keep = s > drop * max(s.max(), 1e-300)
    coef = np.zeros_like(g)
    k = int(keep.sum())
    coef[:, :k] = (u[:, keep] / np.sqrt(s[keep])[None, :]) / d[:, None]
    return coef


@dataclass
class Result:
    values: np.ndarray         # (b,) Ritz values, ascending
    vectors: jax.Array         # (b, ...) Ritz vectors (orthonormal)
    residuals: np.ndarray      # (b,) relative residual norms
    iterations: int
    converged: bool
    stalled: bool = False      # stopped at the precision floor (residual no longer falls): eigenvalues are accurate


def lobpcg(apply: Callable, precond: Callable, x0: jax.Array, k: int, tol: float = 1e-6, max_iter: int = 300,
           scale: float = 1.0, verbose: bool = False, stall_iters: int = 15, stall_below: float = 1e-3) -> Result:
    """The k smallest eigenpairs of the Hermitian operator ``apply`` (block x0: (b, ...), b >= k).
    ``scale``: residuals are relative to max(|theta|, scale) (eigenvalues near 0 at the Gamma point)."""
    dtype = x0.dtype
    # SVQB drop tolerance: rounding grows by at most 1/sqrt(drop); two passes restore orthonormality
    drop = 1e-5 if dtype == jnp.complex64 else 1e-10
    b = x0.shape[0]

    def orth(v, av=None):
        c = jnp.asarray(_svqb_coef(np.asarray(_gram(v, v)), drop), dtype)
        return _combine(c, v), (None if av is None else _combine(c, av))

    x, _ = orth(x0)
    ax = apply(x)
    h = np.asarray(_gram(x, ax), np.complex128)
    theta, y = np.linalg.eigh(0.5 * (h + h.conj().T))
    yc = jnp.asarray(y, dtype)
    x, ax = _combine(yc, x), _combine(yc, ax)
    p = jnp.zeros_like(x)
    ap = jnp.zeros_like(x)
    rn = np.full(b, np.inf)
    best, best_it = np.inf, 0
    it = 0
    for it in range(1, max_iter + 1):
        r = ax - jnp.asarray(theta, dtype).reshape((b,) + (1,) * (x.ndim - 1)) * x
        rn = np.sqrt(np.real(np.diag(np.asarray(_gram(r, r))))) / np.maximum(np.abs(theta), scale)
        if verbose:
            print(f"  it {it:3d} max res {rn[:k].max():.2e} theta0 {theta[0]:.10f}", flush=True)
        if np.all(rn[:k] < tol):
            return Result(theta, x, rn, it, True)
        # the residual floor of the working precision (about eps * |A| / theta) can be above tol: stop when the
        # largest residual did not halve in stall_iters iterations, once it is below stall_below
        if rn[:k].max() < 0.5 * best:
            best, best_it = rn[:k].max(), it
        elif best < stall_below and it - best_it >= stall_iters:
            return Result(theta, x, rn, it, False, stalled=True)
        w = precond(r)
        w = jnp.where(jnp.asarray(rn >= tol).reshape((b,) + (1,) * (x.ndim - 1)), w, 0)
        w = w - _combine(_gram(x, w), x)                       # W orthogonal to X (helps SVQB)
        aw = apply(w)
        s = jnp.concatenate([x, w, p])
        as_ = jnp.concatenate([ax, aw, ap])
        for _ in range(2):                                      # two SVQB passes (Hetmaniuk and Lehoucq 2006)
            coef = _svqb_coef(np.asarray(_gram(s, s)), drop)
            kept = np.any(coef != 0, axis=0)                    # dropped directions are zero vectors
            c = jnp.asarray(coef, dtype)
            s, as_ = _combine(c, s), _combine(c, as_)
        h = np.asarray(_gram(s, as_), np.complex128)
        h = 0.5 * (h + h.conj().T)
        h[~kept, ~kept] = 1e30                                  # never select a dropped (zero) direction
        vals, y = np.linalg.eigh(h)
        order = np.argsort(vals)[:b]
        yc = jnp.asarray(y[:, order], dtype)
        x_new = _combine(yc, s)
        # A x is computed again (not combined): combined products drift with rounding over many iterations, and
        # the rescaling of a small P direction magnifies the drift until A S is no longer A times S.
        ax_new = apply(x_new)
        theta = np.real(np.diag(np.asarray(_gram(x_new, ax_new))))
        ov = _gram(x, x_new)                                    # (b, b)
        p = x_new - _combine(ov, x)
        ov2 = _gram(x_new, p)                                   # keep P orthogonal to the new X (near
        p = p - _combine(ov2, x_new)                            # convergence P tends to span(X))
        # A P is a product, not a combination: P comes from a subtraction with cancellation, and its rounding
        # error must also be in A P, or the Rayleigh-Ritz matrix of the rescaled P is inconsistent and the residual
        # stalls at about sqrt(eps |A| / theta) (the one review of 2026-10-30 found this; floor 1e-6 in double)
        ap = apply(p)
        x, ax = x_new, ax_new
    return Result(theta, x, rn, it, False)
