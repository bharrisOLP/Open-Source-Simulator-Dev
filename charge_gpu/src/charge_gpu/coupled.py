"""Device solver for the linear system of the coupled Newton step of drift_diffusion.py (method="newton",
linear="device").

The system. Each Newton step solves J y = -F for the changes of psi (all cells) and of n and p (semiconductor
cells). drift_diffusion._newton scales the columns with (V_T, n, p), so y = (dpsi / V_T, dn / n, dp / p), and the
rows with 1 / max |row| (row equilibration): J_s = R J C, b = -R F. SuperLU solves this system on the CPU; this
module solves the same system on the JAX device.

Krylov method: FGMRES (flexible GMRES, right preconditioned; Saad, SIAM J. Sci. Comput. 14, 461 (1993)), restarted
after RESTART steps, Arnoldi with classical Gram-Schmidt done twice (CGS2: two matrix-vector products with the
basis, a good fit for a GPU). J_s is nonsymmetric, so CG cannot be used. The residual norm is the 2-norm of the
row-equilibrated system: each row has the max entry 1, so all rows (minority and majority carriers, Poisson) have
the same weight. J_s is on the device in ELL form (fixed number of entries per row; a gather and a row sum, no
atomic adds). One jitted call does one Arnoldi step (the
preconditioner, the product and the orthogonalization); the host keeps the small Hessenberg matrix and the Givens
rotations (one transfer of RESTART + 1 numbers per step).

Preconditioner: one block Gauss-Seidel step in quasi-Fermi variables, with one multigrid V-cycle per block.
- Change of variables (Gummel's decoupling, in linear form): x = dpsi / V_T, a = -dphi_n / V_T, b = dphi_p / V_T.
  With n = n_i exp((psi - phi_n) / V_T) and p = n_i exp((phi_p - psi) / V_T): dn / n = x + a and dp / p = -x + b,
  so y = T (x, a, b) with T = [[I, 0, 0], [E, I, 0], [-E, 0, I]] (E: the psi of the semiconductor cells).
- The blocks of G = J C T (rows unscaled):
  - G_00 = V_T (A + diag(q V (n + p) / V_T)): the Jacobian of the nonlinear Poisson equation with fixed quasi-Fermi
    levels (poisson.py), symmetric positive definite. One V-cycle of the aggregation multigrid on the full grid
    (continuity.hierarchy, the hierarchy of the Gummel Poisson step).
  - G_11 = J_nn diag(n). J_nn is the electron SG matrix with the Newton derivative of the recombination: the matrix
    M of continuity.py with dR/dn in place of the linearized R, so J_nn = -S_n E_n^-1 (Slotboom form, S_n
    symmetric positive definite) and G_11 a = -S_n (u a), u = n / E_n. One V-cycle on S_n. Holes: G_22 = S_p (v b),
    v = p / F_p.
  - G_10 and G_20 (the change of the currents with psi at fixed quasi-Fermi levels) are proportional to the
    current; G_01 = q V n and G_02 = -q V p couple the quasi-Fermi levels into the Poisson rows.
- The step: x from the Poisson block; then a from the electron rows with the term of x moved to the right side;
  then b from the hole rows with the terms of x and a. The products with G_10, G_20 and G_21 are products with the
  rows of J_s at T (x, 0, 0) and T (x, a, 0), so no block of G is assembled. The neglected couplings (upper
  triangle) are those of the Gummel iteration; FGMRES removes their error.
- The continuity blocks use the box of the semiconductor cells (continuity.Solver). Their Slotboom scaling has the
  span limit of continuity.py (continuity.SpanError above 1200 V_T): drift_diffusion._newton then uses SuperLU.

Convergence flag: after each FGMRES cycle the true residual b - J_s y (not the Givens estimate) gives the relative
residual ||b - J_s y||_2 / ||b||_2. FGMRES aims at RTOL = 1e-10 and stops there, or when a cycle does not halve the
true residual (stagnation), or after MAX_ITER steps. A solve is converged (used) when its relative residual is at
most ACCEPT = 1e-6 and all values are finite. This is an inexact Newton step (Dembo, Eisenstat and Steihaug, SIAM J.
Numer. Anal. 19, 400 (1982)): with ||F + J s|| <= eta ||F|| (here in the row-scaled norm, eta <= 1e-6) the Newton
iteration still converges, at least linearly with a rate of about eta, and the residual F of every Newton step is
computed exactly on the host. Why not RTOL as the limit: in some Newton steps far from the solution (a reverse-biased
junction started from poisson.solve) the true residual stagnates at 1e-9 to 1e-8 of ||b|| (probably a round-off
floor: the preconditioned vectors span many decades). drift_diffusion._newton does not use a solve that is not
converged: it solves that Newton step with SuperLU and counts it (DDSolution.linear_fallbacks).

Cost: the Jacobian is assembled on the host (as for SuperLU) and copied to the device per Newton step; the three
hierarchies are built on the host (the coarsening of thermal_gpu.multigrid and a 256-cell LU inverse).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
from thermal_gpu.multigrid import vcycle

from charge_gpu.continuity import Solver as ContinuitySolver
from charge_gpu.continuity import hierarchy
from charge_gpu.poisson import Q

RTOL = 1e-10        # the target: FGMRES stops when ||b - J_s y||_2 <= RTOL ||b||_2 (true residual)
ACCEPT = 1e-6       # a solve is used (converged = True) when its true relative residual is at most ACCEPT
RESTART = 60        # FGMRES steps per cycle
MAX_ITER = 300      # FGMRES steps of one solve (all cycles)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class _Ops:
    vals: jax.Array          # J_s in ELL form: (rows, K) values and column indices
    cols: jax.Array
    irs: jax.Array           # 1 / row scale: the unscaled residual is irs * z
    vt: jax.Array
    lv_psi: tuple            # multigrid of G_00 / V_T on the full grid
    ci_psi: jax.Array
    lv_n: tuple              # multigrid of S_n and S_p on the box of the semiconductor cells
    ci_n: jax.Array
    lv_p: tuple
    ci_p: jax.Array
    u_n: jax.Array           # n / E_n and p / F_p (semiconductor cells)
    u_p: jax.Array
    sflat: jax.Array         # flat cell index of each semiconductor cell (full grid)
    cell: jax.Array          # flat index of each semiconductor cell in the box


def ell(mat) -> tuple[np.ndarray, np.ndarray]:
    """(values, column indices) of a sparse matrix in ELL form: K = the max entries of a row, rows padded with
    zeros (column 0)."""
    mat = sp.csr_matrix(mat)
    counts = np.diff(mat.indptr)
    k = max(int(counts.max(initial=0)), 1)
    rows = np.repeat(np.arange(mat.shape[0]), counts)
    pos = np.arange(mat.nnz) - mat.indptr[rows]
    vals = np.zeros((mat.shape[0], k))
    cols = np.zeros((mat.shape[0], k), np.int32)
    vals[rows, pos] = mat.data
    cols[rows, pos] = mat.indices
    return vals, cols


def _rows(o: _Ops, v, lo: int, hi: int):
    """Rows lo:hi of J_s v."""
    return jnp.sum(o.vals[lo:hi] * v[o.cols[lo:hi]], axis=1)


def _matvec(o: _Ops, v):
    return jnp.sum(o.vals * v[o.cols], axis=1)


def _precond(o: _Ops, z):
    """P^-1 z: one block Gauss-Seidel step in (x, a, b) (module docstring), returned as y = T (x, a, b)."""
    shape, box = o.lv_psi[0].diag.shape, o.lv_n[0].diag.shape
    nc, ns = o.lv_psi[0].diag.size, o.cell.shape[0]
    r = z * o.irs
    x = vcycle(o.lv_psi, o.ci_psi, (r[:nc] / o.vt).reshape(shape)).ravel()
    ex = x[o.sflat]
    t1 = _rows(o, jnp.concatenate([x, ex, -ex]), nc, nc + ns)
    r1 = (z[nc:nc + ns] - t1) * o.irs[nc:nc + ns]
    grid = jnp.zeros(box[0] * box[1], z.dtype)
    a = -vcycle(o.lv_n, o.ci_n, grid.at[o.cell].set(r1).reshape(box)).ravel()[o.cell] / o.u_n
    yn = ex + a
    t2 = _rows(o, jnp.concatenate([x, yn, -ex]), nc + ns, nc + 2 * ns)
    r2 = (z[nc + ns:] - t2) * o.irs[nc + ns:]
    b = vcycle(o.lv_p, o.ci_p, grid.at[o.cell].set(r2).reshape(box)).ravel()[o.cell] / o.u_p
    return jnp.concatenate([x, yn, b - ex])


@partial(jax.jit, donate_argnums=(1, 2))
def _arnoldi(o: _Ops, basis, zs, j):
    """One FGMRES step: z = P^-1 v_j, w = J_s z, CGS2 against the basis (the rows after j are zero). Returns the
    new basis and zs, the column h of the Hessenberg matrix and the norm of the new vector."""
    z = _precond(o, basis[j])
    w = _matvec(o, z)
    h = basis @ w
    w = w - h @ basis
    h2 = basis @ w
    w = w - h2 @ basis
    beta = jnp.linalg.norm(w)
    basis = basis.at[j + 1].set(jnp.where(beta > 0, w / jnp.where(beta > 0, beta, 1.0), 0.0))
    return basis, zs.at[j].set(z), h + h2, beta


@jax.jit
def _residual(o: _Ops, b, y):
    r = b - _matvec(o, y)
    return r, jnp.linalg.norm(r)


@jax.jit
def _update(y, t, zs):
    return y + t @ zs


@dataclass
class LinearResult:
    y: np.ndarray            # the solution of J_s y = b
    iterations: int          # FGMRES steps (all cycles)
    residual: float          # ||b - J_s y||_2 / ||b||_2 (true residual)
    converged: bool          # residual <= accept (ACCEPT) and finite
    setup_s: float           # host time of the preconditioner and the transfers
    krylov_s: float          # time of the FGMRES loop


def fgmres(o: _Ops, b, rtol: float = RTOL, restart: int = RESTART, max_iter: int = MAX_ITER):
    """FGMRES for J_s y = b from y = 0 (o: the device operators). Returns y (device), the steps and the relative
    true residual. A cycle ends when the Givens estimate of the residual reaches rtol, after ``restart`` steps or at
    max_iter; then the true residual is computed. The solve stops when it is at most rtol, or when a cycle did not
    halve it (stagnation: the round-off floor of the true residual, measured at 1e-9 to 1e-8 of ||b|| in some
    far-from-solution Newton steps of a reverse-biased junction)."""
    size = b.shape[0]
    y = jnp.zeros_like(b)
    bnorm = float(jnp.linalg.norm(b))
    if bnorm == 0.0:
        return y, 0, 0.0
    r, rnorm = b, bnorm
    total = 0
    while total < max_iter:
        last = rnorm
        basis = jnp.zeros((restart + 1, size), b.dtype).at[0].set(r / rnorm)
        zs = jnp.zeros((restart, size), b.dtype)
        hess = np.zeros((restart + 1, restart))
        cs, sn = np.zeros(restart), np.zeros(restart)
        g = np.zeros(restart + 1)
        g[0] = rnorm
        k = 0
        for j in range(restart):
            basis, zs, h_d, beta_d = _arnoldi(o, basis, zs, j)
            h = np.asarray(h_d)[: j + 2].copy()
            h[j + 1] = float(beta_d)
            if not np.all(np.isfinite(h)):
                return y, total + 1, float("nan")
            for i in range(j):                                    # the earlier Givens rotations
                h[i], h[i + 1] = cs[i] * h[i] + sn[i] * h[i + 1], -sn[i] * h[i] + cs[i] * h[i + 1]
            d = np.hypot(h[j], h[j + 1])
            if d == 0.0:
                break
            cs[j], sn[j] = h[j] / d, h[j + 1] / d
            h[j], h[j + 1] = d, 0.0
            g[j], g[j + 1] = cs[j] * g[j], -sn[j] * g[j]
            hess[: j + 1, j] = h[: j + 1]
            k, total = j + 1, total + 1
            if abs(g[j + 1]) <= rtol * bnorm or total >= max_iter or float(beta_d) == 0.0:
                break
        if k == 0:
            break
        t = np.zeros(restart)
        t[:k] = sla.solve_triangular(hess[:k, :k], g[:k])
        y = _update(y, jnp.asarray(t, b.dtype), zs)
        r, rn = _residual(o, b, y)
        rnorm = float(rn)
        if not np.isfinite(rnorm) or rnorm <= rtol * bnorm or rnorm > 0.5 * last:
            break
    return y, total, rnorm / bnorm


class Solver:
    """The device solver of the Newton step for one mesh (drift_diffusion._Mesh)."""

    def __init__(self, m, device=None):
        from charge_gpu.poisson import solver_device

        self.m = m
        self.device = device or solver_device()
        self.cont = ContinuitySolver(m, self.device)
        self._put = lambda a, dt=jnp.float64: jax.device_put(jnp.asarray(a, dt), self.device)  # noqa: E731
        self.sflat = self._put(m.sflat, jnp.int32)
        self.cell = self._put(self.cont.cell, jnp.int32)

    def operators(self, js, rs, psi, n, p, tau_n, tau_p) -> _Ops:
        """The device operators of one Newton step: J_s (scaled Jacobian, rows scaled with rs) and the three
        multigrid hierarchies at (psi, n, p). Raises continuity.SpanError above the span limit."""
        from charge_gpu.drift_diffusion import srh

        m, put = self.m, self._put
        gx, gy, gc = m.host
        npf = np.zeros(m.n_cells)
        npf[m.sflat] = n + p
        lv_psi, ci_psi = hierarchy(gx, gy, gc + (Q * m.vol * npf / m.vt).reshape(m.shape), self.device)
        _, drn, drp = srh(n, p, m.n_i, tau_n, tau_p)
        gxn, gyn, exn, _, e_n = self.cont.operator(psi, "n", drn, 1.0)
        lv_n, ci_n = hierarchy(gxn, gyn, exn, self.device)
        gxp, gyp, ex_p, _, f_p = self.cont.operator(psi, "p", drp, 1.0)
        lv_p, ci_p = hierarchy(gxp, gyp, ex_p, self.device)
        vals, cols = ell(js)
        return _Ops(vals=put(vals), cols=put(cols, jnp.int32), irs=put(1.0 / rs), vt=put(m.vt), lv_psi=lv_psi,
                    ci_psi=ci_psi, lv_n=lv_n, ci_n=ci_n, lv_p=lv_p, ci_p=ci_p, u_n=put(n / e_n), u_p=put(p / f_p),
                    sflat=self.sflat, cell=self.cell)

    def solve(self, js, b, rs, psi, n, p, tau_n, tau_p, rtol: float = RTOL, accept: float = ACCEPT,
              restart: int = RESTART, max_iter: int = MAX_ITER) -> LinearResult:
        """y of J_s y = b (J_s: the scaled Jacobian of drift_diffusion._newton, scipy sparse; rs: its row scale).
        FGMRES aims at the relative residual rtol; the result is converged when it is at most ``accept``."""
        t0 = time.perf_counter()
        o = self.operators(js, rs, psi, n, p, tau_n, tau_p)
        bd = self._put(b)
        t1 = time.perf_counter()
        y, it, rel = fgmres(o, bd, rtol, restart, max_iter)
        y = np.asarray(y)
        t2 = time.perf_counter()
        ok = bool(np.isfinite(rel) and rel <= accept and np.all(np.isfinite(y)))
        return LinearResult(y=y, iterations=it, residual=float(rel), converged=ok, setup_s=t1 - t0, krylov_s=t2 - t1)
