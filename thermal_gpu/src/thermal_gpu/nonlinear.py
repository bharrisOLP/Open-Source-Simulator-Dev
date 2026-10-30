"""Steady heat conduction with a temperature-dependent conductivity kappa(T): -div(kappa(T) grad T) = q.

Method: Picard (fixed-point) iteration. Each iteration evaluates kappa at the cell temperatures of the last
iterate, builds the linear finite-volume problem of thermal_gpu.fv3d (face conductance = series form of the two
cell values) and solves it with the multigrid CG of solver3d, from the last iterate. The linear operator stays
symmetric positive definite, so CG applies (a Newton step adds the terms d kappa / dT (T_c - T_nb), which are not
symmetric). The multigrid hierarchy of the first iteration is the preconditioner of the next iterations
(reuse_mg=True; it stays symmetric positive definite, see thermal_gpu.multigrid3d; CG solves the operator of each
iteration, so only the iteration count can change). When kappa changes by a large factor, the reused hierarchy
can need 10 to 15 times more CG iterations (tests/test_nonlinear.py: a Si slab at 765 K, 143 against 14; at
1186 K, 245 against 16). So a new hierarchy is built when one solve needs more than rebuild_factor (3) times the
CG iterations of the first solve on the current hierarchy (with a floor of 2 iterations): 47 and 65 iterations in
these cases, with 3 hierarchies against 14 and 17.

Stop: max |T_k+1 - T_k| <= tol_picard * max(1 K, max |T_k+1 - T_ref|). Convergence flag: the true nonlinear residual
|b(T) - A(kappa(T)) T| / |b(T)| of the final field is at most tol, and the last linear solve converged. The unknowns
are theta = T - T_ref (as in thermal_gpu.transient3d), so the residuals are relative to the heat input and the
boundary terms G_b (T_b - T_ref), not to the absolute temperature. This is a stricter measure than the residual of
solver.solve_steady (where b holds G_b T_b with T_b = 300 K); its round-off floor is about 1e-12 to 1e-11 on the
heater stack, so the default tol is 1e-10. Picard converges linearly; the rate grows with |d ln kappa / d ln T|
times the relative temperature rise in the material. A power law kappa ~ T^-n with n > 1 has no steady solution
above a limit of the heating (thermal runaway: the Kirchhoff variable below reaches T0 / (n - 1)); then the
iteration does not converge and the flag is False.

Kirchhoff transform (the exact reference): for one material, theta(T) = T0 + integral_T0^T kappa(s) / kappa(T0) ds
gives -div(kappa0 grad theta) = q, a linear problem. Each law below has kirchhoff() and its inverse, which give
exact 1-D solutions (also for two layers in series: the flux is the same in each layer).

Materials (textbook and published values; no process data):
- Si: kappa = 148 (300 / T)^1.3 W/m/K. Source: V. Palankovski, PhD thesis, TU Wien (2000), and V. Palankovski,
  R. Quay, "Analysis and Simulation of Heterostructure Devices", Springer (2004): kappa_300 = 148, alpha = -1.3, a
  fit to the data of Glassbrenner and Slack, Phys. Rev. 134, A1058 (1964), for 300 to 800 K (the valid range of
  SI_POWER_LAW). It agrees with the table of Incropera et al., "Fundamentals of Heat and Mass Transfer", Table A.1
  (Si: 148, 98.9 and 61.9 W/m/K at 300, 400 and 600 K) to 3 % (tests/test_nonlinear.py).
- SiO2: nearly constant. The option SIO2_LINEAR: kappa = 1.4 (1 + 9.42e-4 (T - 300)), the slope of fused silica
  in Incropera Table A.2 (1.38 at 300 K, 1.51 at 400 K); valid range 300 to 600 K (at 600 K its ratio to the
  300 K value is within 2 % of the table ratio 1.75 / 1.38).
- The heater metal of thermal_gpu.structures: constant.

Valid range: each law has t_min and t_max (default 0 and infinity). The solver counts the cells of the final field
that are outside the range of their law (NonlinearResult.out_of_range) and gives a RuntimeWarning when there are
any. The converged flag refers to the discrete problem only: a field outside the range can converge, but its
kappa is then an extrapolation of the law.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from thermal_gpu.fv3d import DeviceProblem3, apply3, build3, to_device3
from thermal_gpu.multigrid3d import CYCLES, Level3, build_hierarchy3
from thermal_gpu.solver import solver_device
from thermal_gpu.solver3d import pcg3d_mg
from thermal_gpu.transient3d import reference_temperature


RANGE_SLACK = 1e-6       # K: a cell is outside the range of its law when T < t_min - RANGE_SLACK or T > t_max + ...


class _Range:
    def out_of_range(self, t) -> int:
        """The number of cells with T outside [t_min, t_max] (with RANGE_SLACK)."""
        t = np.asarray(t, float)
        return int(np.count_nonzero((t < self.t_min - RANGE_SLACK) | (t > self.t_max + RANGE_SLACK)))


@dataclass(frozen=True)
class Constant(_Range):
    k0: float
    t_min: float = 0.0
    t_max: float = np.inf

    def __call__(self, t):
        return np.full(np.shape(t), self.k0)

    def kirchhoff(self, t):
        return np.asarray(t, float)

    def inverse(self, theta):
        return np.asarray(theta, float)


@dataclass(frozen=True)
class PowerLaw(_Range):
    """kappa = k0 (t0 / T)^n, valid for t_min <= T <= t_max."""
    k0: float
    n: float
    t0: float = 300.0
    t_min: float = 0.0
    t_max: float = np.inf

    def __call__(self, t):
        return self.k0 * (self.t0 / np.asarray(t, float)) ** self.n

    def kirchhoff(self, t):
        """theta = t0 + integral_t0^T kappa / k0 dT."""
        r = np.asarray(t, float) / self.t0
        if self.n == 1:
            return self.t0 * (1 + np.log(r))
        return self.t0 + self.t0 * (r ** (1 - self.n) - 1) / (1 - self.n)

    def inverse(self, theta):
        """T of theta (nan above the runaway limit theta - t0 = t0 / (n - 1) for n > 1)."""
        u = (np.asarray(theta, float) - self.t0) / self.t0
        if self.n == 1:
            return self.t0 * np.exp(u)
        base = 1 + (1 - self.n) * u
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(base > 0, self.t0 * np.abs(base) ** (1 / (1 - self.n)), np.nan)


@dataclass(frozen=True)
class LinearLaw(_Range):
    """kappa = k0 (1 + beta (T - t0)), valid for t_min <= T <= t_max."""
    k0: float
    beta: float
    t0: float = 300.0
    t_min: float = 0.0
    t_max: float = np.inf

    def __call__(self, t):
        return self.k0 * (1 + self.beta * (np.asarray(t, float) - self.t0))

    def kirchhoff(self, t):
        u = np.asarray(t, float) - self.t0
        return self.t0 + u + 0.5 * self.beta * u * u

    def inverse(self, theta):
        """T of theta (nan where 1 + 2 beta (theta - t0) < 0: no real root)."""
        v = np.asarray(theta, float) - self.t0
        if self.beta == 0:
            return self.t0 + v
        with np.errstate(invalid="ignore"):
            return self.t0 + 2 * v / (1 + np.sqrt(1 + 2 * self.beta * v))   # the stable root


SI_POWER_LAW = PowerLaw(148.0, 1.3, 300.0, t_min=300.0, t_max=800.0)
SIO2_CONSTANT = Constant(1.4)
SIO2_LINEAR = LinearLaw(1.4, 0.13 / 1.38 / 100.0, 300.0, t_min=300.0, t_max=600.0)


class MaterialKappa:
    """kappa(T) of a structure: labels (cell array of integers) selects the law of each cell from laws."""

    def __init__(self, labels, laws):
        labels = np.asarray(labels)
        self.masks = [(labels == i) for i in range(len(laws))]
        self.laws = tuple(laws)
        if not np.all(np.logical_or.reduce(self.masks)):
            raise ValueError("each label must select one of the laws")

    def __call__(self, t):
        t = np.asarray(t, float)
        out = np.empty(t.shape)
        for mask, law in zip(self.masks, self.laws):
            out[mask] = law(t[mask])
        return out

    def out_of_range(self, t) -> int:
        """The number of cells outside the valid range of their law."""
        t = np.asarray(t, float)
        return sum(law.out_of_range(t[mask]) for mask, law in zip(self.masks, self.laws)
                   if hasattr(law, "out_of_range"))


def material_kappa(labels, laws) -> MaterialKappa:
    """kappa(T) of a structure (a callable with out_of_range): labels selects the law of each cell from laws."""
    return MaterialKappa(labels, laws)


@dataclass
class NonlinearResult:
    T: np.ndarray
    picard_iterations: int
    cg_iterations: int          # total CG iterations
    residual: float             # true nonlinear relative residual of the final field
    change: float               # max |T_k+1 - T_k| of the last Picard iteration (K)
    converged: bool
    device: str
    history: list               # max |T_k+1 - T_k| of each Picard iteration
    out_of_range: int = 0       # cells of the final field outside the valid range of their law
    hierarchies: int = 1        # multigrid hierarchies built


@jax.jit
def _residual(dp: DeviceProblem3, th):
    bn = jnp.linalg.norm(dp.b)
    return jnp.linalg.norm(dp.b - apply3(dp, th)) / jnp.where(bn > 0, bn, 1.0)


def _shifted(p, t_ref, dtype, dev):
    """The device problem in theta = T - T_ref (b - T_ref * boundary conductances; exact)."""
    dp = to_device3(p, dtype, dev)
    return DeviceProblem3(dp.gx, dp.gy, dp.gz, dp.diag, jax.device_put(jnp.asarray(p.b - t_ref * p.diag_extra,
                                                                                    dtype), dev))


def solve_steady_kappa3d(x, y, z, kappa_fn, q, bc: dict, t_init=None, tol: float = 1e-10,
                         tol_picard: float = 1e-10, max_picard: int = 100,
                         max_iter: int = 2000, cycle: str = "V", reuse_mg: bool = True,
                         rebuild_factor: float = 3.0, dtype=jnp.float64) -> NonlinearResult:
    """The steady 3-D temperature with kappa = kappa_fn(T) (a callable: cell temperatures -> cell conductivities,
    for example material_kappa(labels, laws) or one law; when it has out_of_range(T), the solver checks the valid
    range). x, y, z, q, bc: as solver3d.solve_steady3d. t_init: the first iterate (default T_ref everywhere).
    tol: the tolerance of each CG solve and of the final nonlinear residual; tol_picard: see the module docstring.
    reuse_mg=False builds a new multigrid hierarchy at each Picard iteration. With reuse_mg=True a new one is built
    when a solve needs more than rebuild_factor times the CG iterations of the first solve on the current one
    (rebuild_factor=inf: never)."""
    if cycle not in CYCLES:
        raise ValueError(f"cycle must be one of {CYCLES}, not {cycle!r}")
    dev = solver_device()
    t_ref = reference_temperature(bc)
    shape = tuple(len(e) - 1 for e in (x, y, z))
    # the state is theta = T - T_ref (as computed by CG): T = theta + T_ref only for kappa, so that the round-off of
    # T_ref (eps 300 K) does not enter the residual through the large Si conductances
    th = np.zeros(shape) if t_init is None else np.array(np.broadcast_to(t_init, shape), float) - t_ref
    levels = cinv = None
    history, total, change, lin_ok, n_hier, it_first = [], 0, np.inf, False, 0, 0
    tol_d = jnp.asarray(0.5 * tol, dtype)        # CG target tol / 2, the flag on the true residual (as transient3d)

    def kappa_of(th):
        with np.errstate(all="ignore"):
            kap = np.asarray(kappa_fn(th + t_ref), float)
        ok = np.all(np.isfinite(th)) and kap.shape == shape and np.all(np.isfinite(kap)) and np.all(kap > 0)
        return kap if ok else None

    for _ in range(max_picard):
        kap = kappa_of(th)
        if kap is None:
            break                                            # runaway or a law outside its range: not converged
        p = build3(x, y, z, kap, q, bc)
        dp = _shifted(p, t_ref, dtype, dev)
        fresh = levels is None or not reuse_mg
        if fresh:
            levels, cinv = build_hierarchy3(p.g, p.diag_extra, dev, dtype)
            n_hier += 1
        lv = (Level3(dp.gx, dp.gy, dp.gz, dp.diag, levels[0].agg),) + tuple(levels[1:])
        th_dev, it, res = pcg3d_mg(dp, lv, cinv, jax.device_put(jnp.asarray(th, dtype), dev), tol_d,
                                   jnp.asarray(max_iter), cycle)
        it = int(it)
        total += it
        if fresh:
            it_first = it
        elif it > rebuild_factor * max(it_first, 2):
            levels = None                                    # the next iteration builds a new hierarchy
        lin_ok = float(res) <= tol
        th_new = np.asarray(th_dev, float)
        change = float(np.max(np.abs(th_new - th)))
        history.append(change)
        th = th_new
        if change <= tol_picard * max(1.0, float(np.max(np.abs(th)))):
            break
    residual, outside = np.inf, 0
    kap = kappa_of(th)
    if kap is not None:
        dp = _shifted(build3(x, y, z, kap, q, bc), t_ref, dtype, dev)
        residual = float(_residual(dp, jax.device_put(jnp.asarray(th, dtype), dev)))
        if hasattr(kappa_fn, "out_of_range"):
            outside = int(kappa_fn.out_of_range(th + t_ref))
        if outside:
            warnings.warn(f"{outside} cells are outside the valid temperature range of their kappa law (T from "
                          f"{float(np.min(th)) + t_ref:.1f} to {float(np.max(th)) + t_ref:.1f} K)", RuntimeWarning,
                          stacklevel=2)
    return NonlinearResult(th + t_ref, len(history), total, residual, change, bool(lin_ok and residual <= tol),
                           str(dev), history, outside, n_hier)


class _Lift:
    """A 2-D kappa function on (nx, ny, 1) arrays, with its out_of_range when it has one."""

    def __init__(self, fn):
        self.fn = fn
        if hasattr(fn, "out_of_range"):
            self.out_of_range = lambda t: fn.out_of_range(np.asarray(t)[..., 0])

    def __call__(self, t):
        return np.asarray(self.fn(np.asarray(t)[..., 0]))[..., None]


def solve_steady_kappa2d(x, y, kappa_fn, q, bc: dict, t_init=None, **kw) -> NonlinearResult:
    """The 2-D cross-section (per unit length) as a 3-D problem with one z cell of 1 m and adiabatic z ends.
    kappa_fn maps (nx, ny) temperatures to (nx, ny) conductivities; q and t_init: (nx, ny)."""
    res = solve_steady_kappa3d(x, y, np.array([0.0, 1.0]), _Lift(kappa_fn), np.asarray(q, float)[..., None], bc,
                               None if t_init is None else np.asarray(t_init, float)[..., None], **kw)
    res.T = res.T[..., 0]
    return res
