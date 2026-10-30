"""3-D (and 2-D) transient heat conduction on the grid of thermal_gpu.fv3d, with the 3-D multigrid CG of solver3d.

    rho c_p dT/dt = div(kappa grad T) + q          (constant q from t = 0 on; kappa does not depend on T)

Semi-discrete system: M dT/dt = b - A T, with M = diag(rho_cp V) (J/K) and A, b of fv3d.build3. Each time step is
one symmetric positive definite solve (d M + A) T_new = rhs: the term d M is an extra diagonal, as in the 2-D
solver.solve_transient. Schemes (step dt_n; w = dt_n / dt_(n-1) for BDF2):

    euler   d = 1 / dt_n,    rhs = M T_n / dt_n + b                                          (order 1, L-stable)
    bdf2    d = (1 + 2 w) / ((1 + w) dt_n),
            rhs = M ((1 + w) T_n - w^2 / (1 + w) T_(n-1)) / dt_n + b                           (order 2, L-stable)
    cn      d = 2 / dt_n,    rhs = 2 M T_n / dt_n - A T_n + 2 b   (Crank-Nicolson, times 2)   (order 2, A-stable)

The first BDF2 step is an euler step. BDF2 with a variable step is zero-stable for w < 1 + sqrt(2); this module
refuses a larger step ratio. Crank-Nicolson is not L-stable: a mode with dt lambda >> 2 changes its sign at each step
and decays slowly (amplification (1 - dt lambda / 2) / (1 + dt lambda / 2), near -1). Use cn only with steps that
resolve the fast modes, or for smooth initial fields.

The unknowns are theta = T - T_ref (T_ref: the first Dirichlet temperature, else the first Robin T_inf, else 0).
The shift is exact (the rows of A sum to the boundary conductances), and the relative residual of each step then
refers to the temperature change, not to the absolute temperature. The steady state is a fixed point of each scheme.

Time steps: a constant dt (with t_end), or stages [(dt_1, n_1), (dt_2, n_2), ...]. Each stage builds one multigrid
hierarchy for its main operator. A step with an other d (the first BDF2 step, the first step after a change of dt)
uses the same hierarchy as its preconditioner; CG solves its own operator, so only the iteration count changes.
Each stage runs as one jitted lax.scan on the device (no host synchronization per step).

Monitors: boolean masks or (i, j, k) cells. The result holds the volume-weighted mean temperature of each monitor at
each time. converged is False when the true relative residual |rhs - (d M + A) theta| / |rhs| of one step is above
tol (the residual is computed after each CG loop). CG stops when its recursive residual is at most tol / 2: the true
residual can drift above the recursive one by round-off (measured: 14 %), and the margin keeps the flag from
failing on that drift. Default tol 1e-10: the residual is relative to the change of the temperature and the heat
input, not to the absolute temperature (see thermal_gpu.nonlinear for the comparison with solver.solve_steady).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from thermal_gpu.fv3d import DeviceProblem3, apply3, build3, full_diag
from thermal_gpu.multigrid3d import CYCLES, Level3, build_hierarchy3, mg_cycle3
from thermal_gpu.solver import solver_device
from thermal_gpu.solver3d import _cg

SCHEMES = ("euler", "bdf2", "cn")
BDF2_MAX_RATIO = 1.0 + np.sqrt(2.0)


@dataclass
class TransientResult3:
    times: np.ndarray        # (n_steps + 1,) s
    monitors: np.ndarray     # (n_steps + 1, n_monitors): volume-weighted mean T of each monitor (K)
    T: np.ndarray            # the final field (K)
    iterations: int          # total CG iterations
    max_step_iterations: int
    max_residual: float      # the largest true relative residual of a step
    converged: bool          # each step reached tol
    device: str
    scheme: str


def reference_temperature(bc: dict) -> float:
    """T_ref: the first Dirichlet temperature in bc, else the first Robin T_inf, else 0."""
    for kind, pos in (("dirichlet", 1), ("robin", 2)):
        for spec in bc.values():
            if spec[0] == kind:
                return float(spec[pos])
    return 0.0


def stages_from(dt, t_end=None) -> list:
    """The time-step stages [(dt, n), ...]: dt a number (with t_end; n = round(t_end / dt)) or a list of stages."""
    if np.ndim(dt) == 0:
        if t_end is None:
            raise ValueError("a constant dt needs t_end")
        n = int(round(float(t_end) / float(dt)))
        stages = [(float(dt), n)]
    else:
        if t_end is not None:
            raise ValueError("give t_end only with a constant dt")
        stages = [(float(d), int(n)) for d, n in dt]
    if not stages or any(not (np.isfinite(d) and d > 0) or n < 1 for d, n in stages):
        raise ValueError(f"each stage needs dt > 0 and at least one step: {stages}")
    return stages


def geometric_stages(dt0: float, t_end: float, per_stage: int = 20, factor: float = 2.0) -> list:
    """Stages with per_stage steps each; dt grows by factor from stage to stage, until t >= t_end."""
    if not (dt0 > 0 and t_end > 0 and per_stage >= 1 and factor >= 1):
        raise ValueError("geometric_stages needs dt0 > 0, t_end > 0, per_stage >= 1 and factor >= 1")
    stages, t, dt = [], 0.0, float(dt0)
    while t < t_end * (1 - 1e-12):
        n = int(min(per_stage, np.ceil((t_end - t) / dt - 1e-9)))
        stages.append((dt, n))
        t += n * dt
        dt *= factor
    return stages


def _monitor_index(monitors, shape, vol):
    """Flat cell indices, weights (V / sum V of the monitor) and monitor numbers of all monitor cells."""
    idx, w, seg = [], [], []
    for k, mon in enumerate(monitors):
        if isinstance(mon, tuple):
            mask = np.zeros(shape, bool)
            mask[mon] = True
        else:
            mask = np.asarray(mon, bool)
            if mask.shape != shape:
                raise ValueError(f"monitor {k}: the mask must have the shape {shape}, not {mask.shape}")
        if not mask.any():
            raise ValueError(f"monitor {k} has no cell")
        flat = np.flatnonzero(mask)
        v = vol.ravel()[flat]
        idx.append(flat)
        w.append(v / v.sum())
        seg.append(np.full(flat.size, k))
    if not idx:
        return np.zeros(0, int), np.zeros(0), np.zeros(0, int)
    return np.concatenate(idx), np.concatenate(w), np.concatenate(seg)


def _monitor_values(th, mon, n_mon):
    idx, w, seg = mon
    return jax.ops.segment_sum(th.ravel()[idx] * w, seg, num_segments=n_mon)


@partial(jax.jit, static_argnames=("n", "n_mon", "cycle", "with_a"))
def _march(gs, diag0, m, b, levels, cinv, th1, th0, coef, tol, max_iter, mon, n, n_mon, cycle, with_a):
    """n steps with the same coefficients coef = (d, c1, c2, cA, sb, w):
    (d M + A) th_new = M (c1 th_n + c2 th_(n-1)) - cA A th_n + sb b; warm start th_n + w (th_n - th_(n-1)).
    with_a (static): cA is not 0."""
    d, c1, c2, ca, sb, w = coef
    a0 = DeviceProblem3(*gs, diag0, b)
    diag = diag0 + d * m

    def step(carry, _):
        th1, th0 = carry
        rhs = m * (c1 * th1 + c2 * th0) + sb * b
        if with_a:
            rhs = rhs - ca * apply3(a0, th1)
        th2, it, res = _cg(lambda v: apply3(DeviceProblem3(*gs, diag, rhs), v),
                           lambda r: mg_cycle3(levels, cinv, r, cycle), rhs, th1 + w * (th1 - th0), tol, max_iter)
        return (th2, th1), (_monitor_values(th2, mon, n_mon), it, res)

    (th1, th0), out = jax.lax.scan(step, (th1, th0), None, length=n)
    return th1, th0, out


def _coefficients(scheme: str, dt: float, dt_prev: float | None, first: bool) -> tuple:
    """(d, c1, c2, cA, sb, w) of one step; first: the first step of the run."""
    w = 0.0 if dt_prev is None else dt / dt_prev
    if scheme == "euler" or (scheme == "bdf2" and first):
        return 1.0 / dt, 1.0 / dt, 0.0, 0.0, 1.0, w
    if scheme == "bdf2":
        return (1 + 2 * w) / ((1 + w) * dt), (1 + w) / dt, -w * w / ((1 + w) * dt), 0.0, 1.0, w
    return 2.0 / dt, 2.0 / dt, 0.0, 1.0, 2.0, w


def solve_transient3d(x, y, z, kappa, q, bc: dict, rho_cp, dt, t_end=None, scheme: str = "euler", monitors=(),
                      t0_field=None, tol: float = 1e-10, max_iter: int = 2000, cycle: str = "V",
                      dtype=jnp.float64) -> TransientResult3:
    """The 3-D transient from t0_field (default: T_ref everywhere, see the module docstring).
    x, y, z: cell edges; kappa, q, rho_cp: (nx, ny, nz) (rho_cp: volumetric heat capacity, J/m^3/K, positive);
    bc: as thermal_gpu.fv3d. dt: a constant step (with t_end) or stages [(dt, n), ...]. scheme: "euler", "bdf2"
    or "cn". monitors: boolean masks or (i, j, k) cells. cycle: the multigrid cycle, "V" or "W"."""
    if scheme not in SCHEMES:
        raise ValueError(f"scheme must be one of {SCHEMES}, not {scheme!r}")
    if cycle not in CYCLES:
        raise ValueError(f"cycle must be one of {CYCLES}, not {cycle!r}")
    stages = stages_from(dt, t_end)
    if scheme == "bdf2":
        ratios = [b[0] / a[0] for a, b in zip(stages[:-1], stages[1:])]
        if any(r >= BDF2_MAX_RATIO for r in ratios):
            raise ValueError(f"bdf2: a step ratio must be below 1 + sqrt(2); the stages have {max(ratios):.3g}")
    p = build3(x, y, z, kappa, q, bc)
    rho_cp = np.broadcast_to(np.asarray(rho_cp, float), p.shape)
    if not (np.all(np.isfinite(rho_cp)) and np.all(rho_cp > 0)):
        raise ValueError("rho_cp must be finite and positive in each cell")
    d = [np.diff(np.asarray(e, float)) for e in (x, y, z)]
    vol = d[0][:, None, None] * d[1][None, :, None] * d[2][None, None, :]
    m_host = rho_cp * vol
    t_ref = reference_temperature(bc)
    dev = solver_device()
    put = lambda a: jax.device_put(jnp.asarray(a, dtype), dev)              # noqa: E731
    gs = tuple(put(g) for g in p.g)
    diag0 = put(full_diag(p.g, p.diag_extra))
    m, b = put(m_host), put(p.b - t_ref * p.diag_extra)                      # the theta form: b - A (T_ref 1)
    th = np.zeros(p.shape) if t0_field is None else np.broadcast_to(np.asarray(t0_field, float), p.shape) - t_ref
    if not np.all(np.isfinite(th)):
        raise ValueError("t0_field must be finite")
    th1 = th0 = put(th)
    idx, w, seg = _monitor_index(monitors, p.shape, vol)
    mon = (jax.device_put(jnp.asarray(idx), dev), put(w), jax.device_put(jnp.asarray(seg), dev))
    n_mon = len(monitors)
    tol_d, mi = jnp.asarray(0.5 * tol, dtype), jnp.asarray(max_iter)        # CG target: see the module docstring
    times, vals, its, ress = [0.0], [np.asarray(_monitor_values(th1, mon, n_mon))], [], []
    t, dt_prev = 0.0, None
    for dt_s, n_s in stages:
        main = _coefficients(scheme, dt_s, dt_s, False)
        levels, cinv = build_hierarchy3(p.g, p.diag_extra + main[0] * m_host, dev, dtype)
        levels = (Level3(gs[0], gs[1], gs[2], levels[0].diag, levels[0].agg),) + levels[1:]
        runs = [(main, n_s)]
        first = _coefficients(scheme, dt_s, dt_prev, dt_prev is None)
        if first != main:
            runs = [(first, 1)] + ([(main, n_s - 1)] if n_s > 1 else [])
        for coef, n in runs:
            th1, th0, (v, it, res) = _march(gs, diag0, m, b, levels, cinv, th1, th0, put(np.array(coef)), tol_d, mi,
                                            mon, n, n_mon, cycle, coef[3] != 0)
            vals.append(np.asarray(v))
            its.append(np.asarray(it))
            ress.append(np.asarray(res))
            times.extend(t + dt_s * np.arange(1, n + 1))
            t += n * dt_s
        dt_prev = dt_s
    its, ress = np.concatenate(its), np.concatenate(ress)
    worst = float(ress.max())
    return TransientResult3(np.asarray(times), np.vstack(vals) + t_ref, np.asarray(th1) + t_ref, int(its.sum()),
                            int(its.max()), worst, bool(worst <= tol), str(dev), scheme)


def solve_transient2d(x, y, kappa, q, bc: dict, rho_cp, dt, t_end=None, scheme: str = "euler", monitors=(),
                      t0_field=None, **kw) -> TransientResult3:
    """The 2-D cross-section (per unit length) as a 3-D problem with one z cell of 1 m and adiabatic z ends: the
    same equations as thermal_gpu.solver (a z-invariant 3-D problem gives the 2-D solution). kappa, q, rho_cp,
    t0_field and the monitor masks: (nx, ny); monitor cells: (i, j). T of the result: (nx, ny)."""
    lift = lambda a: None if a is None else np.asarray(a, float)[..., None]   # noqa: E731
    mons = [(m[0], m[1], 0) if isinstance(m, tuple) else np.asarray(m, bool)[..., None] for m in monitors]
    rho = np.broadcast_to(np.asarray(rho_cp, float), np.shape(kappa))
    res = solve_transient3d(x, y, np.array([0.0, 1.0]), lift(kappa), lift(q), bc, lift(rho), dt, t_end, scheme,
                            mons, lift(t0_field), **kw)
    res.T = res.T[..., 0]
    return res


def rise_time(times, values, final: float, lo: float = 0.1, hi: float = 0.9) -> tuple:
    """The times at which values (from values[0]) first reach lo and hi of the step to final (linear
    interpolation between samples), and their difference: (t_lo, t_hi, t_hi - t_lo)."""
    times, values = np.asarray(times, float), np.asarray(values, float)
    frac = (values - values[0]) / (final - values[0])
    out = []
    for level in (lo, hi):
        k = np.flatnonzero(frac >= level)
        if k.size == 0 or k[0] == 0:
            raise ValueError(f"the curve does not cross {level} of the step after its start")
        k = k[0]
        out.append(times[k - 1] + (level - frac[k - 1]) / (frac[k] - frac[k - 1]) * (times[k] - times[k - 1]))
    return out[0], out[1], out[1] - out[0]
