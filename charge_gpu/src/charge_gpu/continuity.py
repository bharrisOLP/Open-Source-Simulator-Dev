"""Device solver for the linear continuity equations of the Gummel iteration (drift_diffusion.py).

At a fixed potential psi, the electron step of the Gummel iteration is the linear system M n = rhs of
drift_diffusion._continuity_matrix: Scharfetter-Gummel (SG) fluxes, the contact faces and the linearized SRH term.
M is a nonsymmetric M-matrix. The hole step is the same with p.

Symmetric form (Slotboom variables). B(-d) = exp(d) B(d), so the SG electron flux of a face a -> b is
    c (B(d) n_b - B(-d) n_a) = W_ab (u_b - u_a),    u = n / E,   E = exp((psi - psi_ref) / V_T),
    W_ab = c B(d) E_b = c B(-|d|) exp((min(psi_a, psi_b) - psi_ref) / V_T) = W_ba > 0.
So M = -S E^-1, where S is the 5-point conductance operator of the face weights W plus the diagonal of the contact
and recombination terms (times E): symmetric positive definite, the form of the operators of thermal_gpu.multigrid.
Holes: p = F v, F = exp(-(psi - psi_ref) / V_T), W_ab = c B(-|d|) exp(-(max(psi_a, psi_b) - psi_ref) / V_T), and
M_p = S_p F^-1. psi_ref is the middle of the psi range, so E and F stay within exp(+-span / (2 V_T)).

Why not plain CG on S: the weights W and the unknowns u span many decades (exp(V / V_T) across a junction: 1e16 and
more). The 2-norm of the residual of S is dominated by the rows with the largest fluxes, so a CG stopping test on it
leaves the rows of the minority carriers unconverged. The solver here therefore works in row-relative residuals:
    r_a = (b - S u)_a / (|S| |u0| + |b|)_a,
the componentwise (Oettli-Prager) backward error of row a at the start vector u0. Each row has the same weight, so
the minority-carrier rows converge to the same relative accuracy as the majority rows. In these units the operator
is not symmetric, so the Krylov method is BiCGSTAB (one jitted device loop), right-preconditioned with one V-cycle
of the aggregation multigrid of thermal_gpu on S:
    (D_r^-1 S) (MG^-1 D_r) w = D_r^-1 (b - S u0),    du = MG^-1 D_r w.
The preconditioned operator D_r^-1 S MG^-1 D_r is similar to S MG^-1 (its eigenvalues are those of the multigrid-
preconditioned SPD operator). In strong drift (a junction) the Jacobi-scaled S is nearly diagonal (the scaled
off-diagonals are about exp(-|d| / 2)); in the neutral regions S is a diffusion operator with a small recombination
term, where the multigrid removes the (L / h)^2 growth of the condition number.

Multigrid: the 2 x 2 aggregation and the Galerkin coarse operators of thermal_gpu.multigrid (its coarsening, the
private helper _coarsen, and its symmetric V-cycle). Insulator cells have no unknown: they have no faces and no
diagonal term, so they do not add to the coarse operators; a cell (or aggregate) with no semiconductor gets the
diagonal 1 and a zero residual. The coarsest operator (at most COARSEST cells) is inverted with an LU factorization
(np.linalg.inv) after a symmetric diagonal scaling. Not with a pseudo-inverse: the SVD gives the entries of the
inverse only to an absolute accuracy of about 1e-16 of the largest entry. The true entries between the two sides of
a reverse-biased junction are far smaller, so the noise moved residuals of the large rows into the small rows: the
solve diverged from 5 V reverse on the 1e18 cm^-3 junction of the tests (found by the review of this module). The
coarsest operator is an M-matrix, so its LU factors and the triangular solves of the inverse add only terms of one
sign: the small entries keep their relative accuracy, and the same solves converge (5, 10, 25 V; tests).

Stopping test of the loop: the max row-relative residual is below TOL (1e-14), or STALL iterations did not halve
its best value (the round-off floor), or MAX_ITER iterations; the result is the vector of the smallest residual.
Convergence flag: after the loop the true residual b - S u (not the recursive residual of BiCGSTAB) gives the
componentwise backward error omega = max_a |b - S u|_a / (|S| |u| + |b|)_a. Because E and F are positive
diagonals, this is also the backward error of M n = rhs (|M| |n| = |S| |u|). A solve is converged when
omega <= ``check`` (1e-11). A solve that stops above it restarts from its last vector (new row scales), at most
``restarts`` times. drift_diffusion._gummel does not use a solve that is not converged: it solves that step with
SuperLU and counts it (DDSolution.linear_fallbacks).

Range: E and F must stay in the float64 range, so the psi span may be at most SPAN_MAX = 1200 V_T (31 V at 300 K);
above it Solver.operator raises ValueError (and _gummel uses SuperLU). Tested: forward bias to 1.0 V (PIN rib) and
reverse bias to 25 V (the 1-D junction, 1e18 cm^-3, 2.5 nm cells; up to 12 V_T per cell).

Checks (tests/test_dd_device.py): the operator against the SG matrix of the SuperLU path; one solve against SuperLU
(PIN rib at 1.0 V); the Gummel solutions against poisson.solve at 0 V and against the Newton method (SuperLU) at
reverse bias (1 to 25 V) and forward bias (0.6 to 1.0 V); the fallback to SuperLU.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from thermal_gpu.multigrid import Level, _coarsen, vcycle

COARSEST = 256       # cells: dense (scaled) inverse below this size
TOL = 1e-14          # stopping test of the loop: max row-relative residual (near the round-off floor, see STALL)
CHECK = 1e-11        # convergence flag: componentwise backward error of the true residual
MAX_ITER = 400
STALL = 10           # the loop also stops after this many iterations without halving its best residual
RESTARTS = 3
SPAN_MAX = 1200.0    # max (psi_max - psi_min) / V_T: E and F stay within exp(+-600) (float64)


class SpanError(ValueError):
    """The psi span is above SPAN_MAX: the Slotboom scaling would leave the float64 range."""


def _diag5(gx, gy, extra):
    d = extra.copy()
    d[:-1] += gx
    d[1:] += gx
    d[:, :-1] += gy
    d[:, 1:] += gy
    return d


def hierarchy(gx: np.ndarray, gy: np.ndarray, extra: np.ndarray, device=None, dtype=jnp.float64,
              coarsest: int = COARSEST):
    """Multigrid levels (fine first) of the operator (gx, gy, extra) and the inverse of the coarsest operator, as
    thermal_gpu.multigrid.build_hierarchy, but: a zero diagonal (a cell or aggregate with no unknown) becomes 1,
    and the coarsest inverse is an LU inverse after a symmetric diagonal scaling (module docstring; a pseudo-inverse
    only when the LU factorization fails, for a singular coarsest operator)."""
    put = (lambda a: jax.device_put(jnp.asarray(a, dtype), device)) if device else (lambda a: jnp.asarray(a, dtype))
    levels = []
    while True:
        d = _diag5(gx, gy, extra)
        d = np.where(d > 0, d, 1.0)
        levels.append(Level(put(gx), put(gy), put(d)))
        if extra.size <= coarsest:
            break
        gx, gy, extra = _coarsen(gx, gy, extra)
    nx, ny = extra.shape
    idx = np.arange(nx * ny).reshape(nx, ny)
    a = np.diag(d.ravel())
    a[idx[:-1].ravel(), idx[1:].ravel()] -= gx.ravel()
    a[idx[1:].ravel(), idx[:-1].ravel()] -= gx.ravel()
    a[idx[:, :-1].ravel(), idx[:, 1:].ravel()] -= gy.ravel()
    a[idx[:, 1:].ravel(), idx[:, :-1].ravel()] -= gy.ravel()
    s = 1.0 / np.sqrt(d.ravel())
    a_s = s[:, None] * a * s[None, :]
    try:
        inv_s = np.linalg.inv(a_s)
    except np.linalg.LinAlgError:
        inv_s = np.linalg.pinv(a_s)
    if not np.all(np.isfinite(inv_s)):
        inv_s = np.linalg.pinv(a_s)
    return tuple(levels), put(s[:, None] * inv_s * s[None, :])


def _apply(lv: Level, v):
    out = lv.diag * v
    out = out.at[:-1].add(-lv.gx * v[1:]).at[1:].add(-lv.gx * v[:-1])
    return out.at[:, :-1].add(-lv.gy * v[:, 1:]).at[:, 1:].add(-lv.gy * v[:, :-1])


def _apply_abs(lv: Level, v):
    """|S| v (all entries of S taken positive)."""
    out = lv.diag * v
    out = out.at[:-1].add(lv.gx * v[1:]).at[1:].add(lv.gx * v[:-1])
    return out.at[:, :-1].add(lv.gy * v[:, 1:]).at[:, 1:].add(lv.gy * v[:, :-1])


@jax.jit
def _bicgstab_mg(levels, coarse_inv, active, b, x0, tol, max_iter):
    """Solve S x = b (S: levels[0]) from x0 with BiCGSTAB in row-relative residuals, right-preconditioned with the
    V-cycle. Returns x, the iterations and the componentwise backward error of the true residual."""
    lv = levels[0]
    zero = jnp.zeros_like(b)
    dr = _apply_abs(lv, jnp.abs(x0)) + jnp.abs(b)
    dr = jnp.where(active & (dr > 0), dr, 1.0)

    def op(z):
        return jnp.where(active, _apply(lv, z) / dr, 0.0)

    def prec(y):
        return jnp.where(active, vcycle(levels, coarse_inv, jnp.where(active, dr * y, 0.0)), 0.0)

    def norm(v):
        return jnp.max(jnp.abs(v))

    r0 = jnp.where(active, (b - _apply(lv, x0)) / dr, 0.0)
    one = jnp.asarray(1.0, b.dtype)

    def cond(s):
        r, it, ok, since = s[1], s[7], s[8], s[10]
        return (norm(r) > tol) & (it < max_iter) & ok & (since < STALL)

    def body(s):
        x, r, p, v, rho, alpha, omega, it, _, best, since, x_min, r_min = s
        rho_new = jnp.vdot(r0, r)
        beta = (rho_new / rho) * (alpha / omega)
        p_new = r + beta * (p - omega * v)
        ph = prec(p_new)
        v_new = op(ph)
        rv = jnp.vdot(r0, v_new)
        alpha_new = rho_new / rv
        s_vec = r - alpha_new * v_new
        sh = prec(s_vec)
        t = op(sh)
        tt = jnp.vdot(t, t)
        s_done = norm(s_vec) <= tol
        omega_new = jnp.where(s_done | (tt == 0), 0.0, jnp.vdot(t, s_vec) / jnp.where(tt == 0, 1.0, tt))
        x_new = x + alpha_new * ph + omega_new * sh
        r_new = s_vec - omega_new * t
        ok = (jnp.isfinite(alpha_new) & jnp.isfinite(omega_new) & (rho_new != 0) & (rv != 0)
              & ((omega_new != 0) | s_done) & jnp.all(jnp.isfinite(x_new)))
        x_new, r_new = jnp.where(ok, x_new, x), jnp.where(ok, r_new, r)
        rn = norm(r_new)
        better = rn < 0.5 * best
        lower = rn < r_min
        return (x_new, r_new, p_new, v_new, rho_new, alpha_new, omega_new, it + 1, ok, jnp.where(better, rn, best),
                jnp.where(better, 0, since + 1), jnp.where(lower, x_new, x_min), jnp.where(lower, rn, r_min))

    state = (zero, r0, zero, zero, one, one, one, jnp.asarray(0), jnp.asarray(True), norm(r0), jnp.asarray(0), zero,
             norm(r0))
    final = jax.lax.while_loop(cond, body, state)
    dx, it = final[11], final[7]                       # the vector of the smallest (recursive) residual
    x = x0 + dx
    res = jnp.where(active, b - _apply(lv, x), 0.0)
    den = _apply_abs(lv, jnp.abs(x)) + jnp.abs(b)
    omega_be = jnp.max(jnp.where(active, jnp.abs(res) / jnp.where(den > 0, den, 1.0), 0.0))
    return x, it, omega_be


@dataclass
class LinearResult:
    c: np.ndarray            # the carrier density (semiconductor cells, m^-3)
    iterations: int          # BiCGSTAB iterations (all restarts)
    backward_error: float    # componentwise backward error of the true residual
    converged: bool          # backward_error <= check


class Solver:
    """The device continuity solver of one mesh (drift_diffusion._Mesh). The grid of the solve is the bounding box
    of the semiconductor cells (in an SOI rib most cells are oxide: the 5 nm rib of injection.rib_pin has 146 400
    cells, its box 26 400). The grid maps are built once."""

    def __init__(self, m, device=None):
        from charge_gpu.poisson import solver_device

        self.m = m
        self.device = device or solver_device()
        ny = m.shape[1]
        si, sj = np.divmod(m.sflat, ny)
        self.i0, self.j0 = int(si.min()), int(sj.min())
        bx, by = int(si.max()) + 1 - self.i0, int(sj.max()) + 1 - self.j0
        self.shape = (bx, by)
        self.cell = (si - self.i0) * by + (sj - self.j0)          # flat index of each semiconductor cell in the box
        self.xf = m.fcol >= 0
        fi, fj = np.divmod(m.fa, ny)
        fi, fj = fi - self.i0, fj - self.j0
        self.ix = (fi * by + fj)[self.xf]                         # flat index in the (bx - 1, by) x-face grid
        self.iy = (fi * (by - 1) + fj)[~self.xf]                  # flat index in the (bx, by - 1) y-face grid
        active = np.zeros(bx * by, bool)
        active[self.cell] = True
        self.active = jax.device_put(jnp.asarray(active.reshape(bx, by)), self.device)

    def operator(self, psi, carrier: str, other, den):
        """The symmetric form of the continuity system of drift_diffusion._continuity_matrix: (gx, gy, extra) of S
        on the cell grid, b (semiconductor cells) and the scale E (n = E u) or F (p = F v)."""
        from charge_gpu.drift_diffusion import bernoulli

        m = self.m
        vt, ns = m.vt, m.n_semi
        nx, ny = self.shape
        pa, pb = psi[m.fa], psi[m.fb]
        d = (pb - pa) / vt
        ps = psi[m.sflat]
        span = (ps.max() - ps.min()) / vt
        if span > SPAN_MAX:
            raise SpanError(f"psi spans {span:.0f} V_T in the semiconductor, more than SPAN_MAX = {SPAN_MAX:.0f} V_T: "
                            "the Slotboom scale exp(+-span / 2) leaves the float64 range")
        ref = 0.5 * (ps.max() + ps.min())
        dc = (m.cpsi - psi[m.cflat]) / vt
        bneg = bernoulli(-np.abs(d))
        if carrier == "n":
            w = m.fcn * bneg * np.exp((np.minimum(pa, pb) - ref) / vt)
            scale = np.exp((ps - ref) / vt)
            diag_c = np.bincount(m.cs, m.ccn * bernoulli(-dc), ns)
            const = np.bincount(m.cs, m.ccn * bernoulli(dc) * m.cn, ns)
        elif carrier == "p":
            w = m.fcp * bneg * np.exp(-(np.maximum(pa, pb) - ref) / vt)
            scale = np.exp(-(ps - ref) / vt)
            diag_c = np.bincount(m.cs, m.ccp * bernoulli(dc), ns)
            const = np.bincount(m.cs, m.ccp * bernoulli(-dc) * m.cp, ns)
        else:
            raise ValueError(f"carrier must be 'n' or 'p', not {carrier!r}")
        gx, gy = np.zeros((nx - 1) * ny), np.zeros(nx * (ny - 1))
        gx[self.ix] = w[self.xf]
        gy[self.iy] = w[~self.xf]
        extra = np.zeros(nx * ny)
        extra[self.cell] = (diag_c + m.svol * other / den) * scale
        b = const + m.svol * m.n_i**2 / den
        return gx.reshape(nx - 1, ny), gy.reshape(nx, ny - 1), extra.reshape(nx, ny), b, scale

    def to_grid(self, values):
        """Values of the semiconductor cells on the (bx, by) grid of the solve (0 elsewhere)."""
        out = np.zeros(self.shape[0] * self.shape[1])
        out[self.cell] = values
        return out.reshape(self.shape)

    def solve(self, psi, carrier: str, other, den, c0, tol: float = TOL, max_iter: int = MAX_ITER,
              check: float = CHECK, restarts: int = RESTARTS) -> LinearResult:
        """The carrier density c of the Gummel continuity step (the solution of M c = rhs of
        drift_diffusion._continuity_matrix), from the start c0 (semiconductor cells)."""
        gx, gy, extra, b, scale = self.operator(psi, carrier, other, den)
        levels, cinv = hierarchy(gx, gy, extra, self.device)
        put = lambda a: jax.device_put(jnp.asarray(a, jnp.float64), self.device)  # noqa: E731
        bd, xd = put(self.to_grid(b)), put(self.to_grid(c0 / scale))
        total, be = 0, np.inf
        for _ in range(restarts + 1):
            xd, it, be_d = _bicgstab_mg(levels, cinv, self.active, bd, xd, tol, max_iter)
            total += int(it)
            be = float(be_d)
            if be <= check:
                break
        c = np.asarray(xd).ravel()[self.cell] * scale
        return LinearResult(c=c, iterations=total, backward_error=be, converged=bool(be <= check))
