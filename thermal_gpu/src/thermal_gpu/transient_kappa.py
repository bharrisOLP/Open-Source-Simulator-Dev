"""Transient heat conduction with a temperature-dependent conductivity kappa(T) and (option) a temperature-dependent
heat capacity rho c_p(T), in 3-D and 2-D, on the grid of thermal_gpu.fv3d:

    rho c_p(T) dT/dt = div(kappa(T) grad T) + q          (constant q from t = 0 on)

Semi-discrete system (enthalpy form): dH/dt = F(T), with H_c(T) = V_c integral rho c_p(s) ds (J, per cell) and
F(T) = b(T) - A(kappa(T)) T (W) of fv3d.build3, with kappa evaluated at the cell temperatures (series face
conductance of the two cell values, as in thermal_gpu.nonlinear). With a constant rho c_p, H = M T and this is the
system M dT/dt = b - A T of thermal_gpu.transient3d. The schemes of transient3d, applied to H (w = dt_n / dt_(n-1)):

    euler   d H(T_n+1) - F(T_n+1) = H(T_n) / dt_n,                                  d = 1 / dt_n
    bdf2    d H(T_n+1) - F(T_n+1) = ((1 + w) H(T_n) - w^2 / (1 + w) H(T_n-1)) / dt_n, d = (1 + 2 w) / ((1 + w) dt_n)
    cn      d H(T_n+1) - F(T_n+1) = 2 H(T_n) / dt_n + F(T_n),                          d = 2 / dt_n (times 2)

The first BDF2 step is an euler step; the limits of transient3d apply (the BDF2 step ratio below 1 + sqrt(2), Crank-
Nicolson not L-stable). The steady kappa(T) solution (nonlinear.solve_steady_kappa3d) is a fixed point of each scheme.

Nonlinear step: Picard iteration, from the extrapolation T_n + w (T_n - T_n-1):

    (d C(T_k) + A(kappa(T_k))) T_k+1 = hist + b(T_k) - d (H(T_k) - C(T_k) T_k),     C = dH/dT = diag(rho c_p(T) V)

This is a Picard step in kappa (the operator stays symmetric positive definite, so the multigrid CG of solver3d
applies) and a Newton step in H (exact when rho c_p is constant). Stop rule of each time step:
max |T_k+1 - T_k| <= tol_picard max(1 K, max |T_k+1 - T_ref|), or max_picard iterations. When kappa and rho c_p are
both constant, each step is one linear solve (the system of transient3d). converged is False when a step did not
meet the stop rule, when its true relative residual |hist + F(T) - d H(T)| / |hist + b(T) - d (H(T) - C(T) T)|, both
at the final T of the step (the denominator is the right-hand side of the linear system at that T; for a constant
rho c_p |hist + b|, the measure of transient3d), is above tol, or when kappa or rho c_p is not finite and positive in
a cell. The CG target is tol / 2 (as in transient3d).

Choose tol and tol_picard together. tol bounds the residual of each step (its backward error), tol_picard the last
change. The residual after the stop is about the Picard rate times the last change (on the Si slab of the tests,
bdf2 at tau / 20: 1.9e-6 with tol_picard = 1e-4): a loose tol_picard with a tight tol gives converged False. With
tol_picard = tol (the defaults, 1e-10) each test converged. Neither check bounds the error when the rate is near 1:
the error can be rate / (1 - rate) times the last change.

The unknowns are theta = T - T_ref (as in transient3d); T = theta + T_ref only enters the laws. H is relative to
T_ref in each cell (with expm1 and log1p, so that a small theta has no cancellation); a constant offset of H in a
cell cancels in each scheme (the coefficients of H sum to 0).

Laws (thermal_gpu.nonlinear): Constant, PowerLaw, LinearLaw, or material_kappa(labels, laws) of these (also for
rho c_p). On the device each law is k0 (t0 / T)^n (1 + beta (T - t0)) (Constant: n = beta = 0; PowerLaw: beta = 0;
LinearLaw: n = 0), selected per cell by its label. An array of cell values (or a number) is a constant law. The
whole time stage (the steps, the Picard iterations and the CG loops) runs as one jitted lax.scan on the device.

Multigrid: each stage builds one hierarchy on the host, at kappa(T) and d C(T) of the field at the start of the
stage. In each Picard iteration the fine level is the current operator and the coarse levels are those of the stage
(the cycle stays symmetric positive definite; CG solves the current operator, so only the iteration count can
change). A long stage in which kappa changes much needs more CG iterations: then split it, for example
[(dt, 50)] * 4 instead of [(dt, 200)].

Valid range: the running minimum and maximum of T in each cell over all steps are compared with the range of the
law of the cell (kappa and rho c_p). out_of_range counts the cells outside, with a RuntimeWarning (converged refers
to the discrete problem only).

Heat capacity of Si (SI_RHO_CP_LINEAR): rho = 2330 kg/m^3 and c_p = 712 J/kg/K at 300 K, 790 J/kg/K at 400 K
(Incropera et al., "Fundamentals of Heat and Mass Transfer", Table A.1), linear between them; valid range 300 to
400 K (at 600 K the line gives 946 against 867 in the table).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from thermal_gpu.fv3d import SIDES3, _hi, _lo, apply_op, build3
from thermal_gpu.multigrid3d import CYCLES, Level3, build_hierarchy3, mg_cycle3
from thermal_gpu.nonlinear import RANGE_SLACK, Constant, LinearLaw, MaterialKappa, PowerLaw
from thermal_gpu.solver import solver_device
from thermal_gpu.solver3d import _cg
from thermal_gpu.transient3d import (BDF2_MAX_RATIO, SCHEMES, TransientResult3, _coefficients, _monitor_index,
                                     _monitor_values, reference_temperature, stages_from)

SI_RHO_CP_LINEAR = LinearLaw(2330.0 * 712.0, (790.0 - 712.0) / 712.0 / 100.0, 300.0, t_min=300.0, t_max=400.0)


@dataclass
class TransientKappaResult(TransientResult3):
    picard_iterations: int = 0      # total Picard iterations (= linear solves)
    max_step_picard: int = 0        # the largest number of Picard iterations of one step
    hierarchies: int = 0            # multigrid hierarchies built (one per stage)
    out_of_range: int = 0           # cells outside the valid range of their law at some step


@dataclass(frozen=True)
class LawTable:
    """Laws per cell: labels (shape of the grid) select a row of k0, t0, n, beta, t_min, t_max (each (n_laws,))."""
    labels: np.ndarray
    rows: np.ndarray                 # (n_laws, 6): k0, t0, n, beta, t_min, t_max

    @property
    def constant(self) -> bool:
        return bool(np.all(self.rows[:, 2] == 0) and np.all(self.rows[:, 3] == 0))

    def __call__(self, t) -> np.ndarray:
        """The law values at the cell temperatures t (host, numpy; the formula of the device)."""
        k0, t0, n, beta = (self.rows[:, i][self.labels] for i in range(4))
        t = np.asarray(t, float)
        with np.errstate(all="ignore"):
            return k0 * (t0 / t) ** n * (1 + beta * (t - t0))

    def outside(self, t_lo, t_hi) -> np.ndarray:
        """Cells whose temperatures (from t_lo to t_hi) leave the valid range of their law (with RANGE_SLACK)."""
        return ((np.asarray(t_lo) < self.rows[:, 4][self.labels] - RANGE_SLACK)
                | (np.asarray(t_hi) > self.rows[:, 5][self.labels] + RANGE_SLACK))


def _row(law) -> tuple:
    if isinstance(law, Constant):
        return (law.k0, 300.0, 0.0, 0.0, law.t_min, law.t_max)
    if isinstance(law, PowerLaw):
        return (law.k0, law.t0, law.n, 0.0, law.t_min, law.t_max)
    if isinstance(law, LinearLaw):
        return (law.k0, law.t0, 0.0, law.beta, law.t_min, law.t_max)
    raise TypeError(f"a law must be Constant, PowerLaw or LinearLaw of thermal_gpu.nonlinear, not {type(law)}")


def _fit(a: np.ndarray, shape: tuple, what: str) -> np.ndarray:
    """a with the grid shape: reshaped when its shape is the grid shape without one of the length-1 axes (the
    (nx, ny) arrays of the 2-D cross-section on its (nx, ny, 1) grid), else broadcast (ValueError when it does not
    fit, for example (ny, nx) labels on an (nx, ny) grid)."""
    shape = tuple(shape)
    if any(shape[ax] == 1 and a.shape == shape[:ax] + shape[ax + 1:] for ax in range(len(shape))):
        return a.reshape(shape)
    try:
        return np.broadcast_to(a, shape)
    except ValueError:
        raise ValueError(f"{what}: the shape {a.shape} does not fit the grid {shape}") from None


def law_table(fn, shape: tuple, what: str = "kappa") -> LawTable:
    """The LawTable of fn: a law of thermal_gpu.nonlinear, material_kappa(labels, laws) of such laws, or an array
    (or a number) of constant cell values. Other callables raise TypeError (the device needs the law parameters)."""
    if isinstance(fn, MaterialKappa):
        labels = np.zeros(fn.masks[0].shape, np.int32)
        for i, mask in enumerate(fn.masks):
            labels[mask] = i
        return LawTable(np.ascontiguousarray(_fit(labels, shape, what)), np.array([_row(law) for law in fn.laws]))
    if isinstance(fn, (Constant, PowerLaw, LinearLaw)):
        return LawTable(np.zeros(shape, np.int32), np.array([_row(fn)]))
    if callable(fn):
        raise TypeError(f"{what}: give a law of thermal_gpu.nonlinear (Constant, PowerLaw, LinearLaw), "
                        "material_kappa(labels, laws) of such laws, or an array of cell values")
    values = np.asarray(fn, float)
    if not (np.all(np.isfinite(values)) and np.all(values > 0)):
        raise ValueError(f"{what}: the cell values must be finite and positive")
    uniq, inv = np.unique(_fit(values, shape, what), return_inverse=True)
    rows = np.array([(v, 300.0, 0.0, 0.0, -np.inf, np.inf) for v in uniq])
    return LawTable(inv.reshape(shape).astype(np.int32), rows)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class _DevLaw:
    labels: jax.Array
    k0: jax.Array
    t0: jax.Array
    n: jax.Array
    beta: jax.Array


def _dev_law(tab: LawTable, put, dev) -> _DevLaw:
    return _DevLaw(jax.device_put(jnp.asarray(tab.labels), dev), *(put(tab.rows[:, i]) for i in range(4)))


def _value(law: _DevLaw, t):
    """k0 (t0 / T)^n (1 + beta (T - t0)) in each cell."""
    k0, t0, n, beta = (a[law.labels] for a in (law.k0, law.t0, law.n, law.beta))
    return k0 * (t0 / t) ** n * (1 + beta * (t - t0))


def _enthalpy_density(law: _DevLaw, th, t_ref):
    """integral_T_ref^(T_ref + th) of the law (J/m^3 for rho c_p), without cancellation for a small th. The power
    law: T_b (t0 / T_b)^n expm1((1 - n) log1p(u / T_b)) / (1 - n), with the base T_b = T_ref (t0 when T_ref <= 0:
    a constant offset per cell, which cancels in each scheme)."""
    k0, t0, n, beta = (a[law.labels] for a in (law.k0, law.t0, law.n, law.beta))
    lin = th * (1 + beta * (t_ref - t0 + 0.5 * th))
    power = n != 0
    tb = jnp.where(power, jnp.where(t_ref > 0, t_ref, t0), 1.0)
    u = th + (t_ref - tb)
    a = 1 - n
    lg = jnp.log1p(jnp.where(power, u / tb, 0.0))
    g = jnp.where(a == 0, lg, jnp.expm1(a * lg) / jnp.where(a == 0, 1.0, a))
    return k0 * jnp.where(power, tb * (t0 / tb) ** n * g, lin)


@partial(jax.tree_util.register_dataclass, data_fields=["d", "qv", "bvals"], meta_fields=["sides"])
@dataclass(frozen=True)
class _Geo:
    d: tuple             # cell sizes per axis, broadcastable: (nx, 1, 1), (1, ny, 1), (1, 1, nz)
    qv: jax.Array        # q V (W)
    bvals: jax.Array     # (n_sides, 2): 1 / h (0 for dirichlet) and T_b - T_ref of each side in `sides`
    sides: tuple = ()    # indices into SIDES3 of the sides that are not adiabatic (static)


def _slab(n_side: int, shape: tuple) -> tuple:
    ax = n_side // 2
    cell = [slice(None)] * 3
    cell[ax] = slice(0, 1) if n_side % 2 == 0 else slice(shape[ax] - 1, shape[ax])
    return tuple(cell)


def _operator(geo: _Geo, kap):
    """The conductances of fv3d.build3 on the device, for kappa = kap: (g per axis, the boundary diagonal, the
    boundary part of b in theta, G_b (T_b - T_ref))."""
    shape = kap.shape
    area = [geo.d[(ax + 1) % 3] * geo.d[(ax + 2) % 3] for ax in range(3)]   # the face area normal to each axis
    r = [0.5 * geo.d[ax] / (area[ax] * kap) for ax in range(3)]            # half-cell resistances (K/W)
    g = tuple(1.0 / (r[ax][_lo(ax)] + r[ax][_hi(ax)]) for ax in range(3))
    extra = jnp.zeros(shape, kap.dtype)
    bb = jnp.zeros(shape, kap.dtype)
    for j, n_side in enumerate(geo.sides):
        ax, cell = n_side // 2, _slab(n_side, shape)
        rh = geo.bvals[j, 0] / jnp.broadcast_to(area[ax], shape)[cell]       # 1 / (h S); 0 for dirichlet
        gb = 1.0 / (r[ax][cell] + rh)
        extra = extra.at[cell].add(gb)
        bb = bb.at[cell].add(gb * geo.bvals[j, 1])
    return g, extra, bb


def _full_diag(g, extra):
    out = extra
    for ax, ga in enumerate(g):
        pad_hi, pad_lo = [(0, 0)] * 3, [(0, 0)] * 3
        pad_hi[ax], pad_lo[ax] = (0, 1), (1, 0)
        out = out + jnp.pad(ga, pad_hi) + jnp.pad(ga, pad_lo)
    return out


def _parts(geo, klaw, claw, t_ref, th):
    """kappa(T), the operator, C = rho c_p(T) V and H(T) at theta = th; ok: kappa and rho c_p finite and positive."""
    t = th + t_ref
    vol = geo.d[0] * geo.d[1] * geo.d[2]
    kap = _value(klaw, t)
    cap = _value(claw, t) * vol
    g, extra, bb = _operator(geo, kap)
    ok = jnp.all(jnp.isfinite(kap) & (kap > 0)) & jnp.all(jnp.isfinite(cap) & (cap > 0))
    return g, extra, bb, cap, _enthalpy_density(claw, th, t_ref) * vol, ok


@jax.jit
def _flux(geo, klaw, t_ref, th):
    """F(T) = b(T) - A(kappa(T)) theta (W)."""
    g, extra, bb = _operator(geo, _value(klaw, th + t_ref))
    return geo.qv + bb - apply_op(g, _full_diag(g, extra), th)


@partial(jax.jit, static_argnames=("n", "n_mon", "cycle", "agg0", "linear"))
def _march(geo, klaw, claw, t_ref, coarse, cinv, carry, coef, tol, tol_p, max_iter, max_picard, mon, n, n_mon,
           cycle, agg0, linear):
    """n steps with the coefficients coef = (d, c1, c2, cA, w): d H(T) - F(T) = c1 H_n + c2 H_n-1 + cA F_n, each
    with the Picard iteration of the module docstring. carry = (theta_n, theta_n-1, F_n, running min, running max)."""
    d, c1, c2, ca, w = coef

    def step(carry, _):
        th1, th0, f1, lo, hi = carry
        vol = geo.d[0] * geo.d[1] * geo.d[2]
        hist = vol * (c1 * _enthalpy_density(claw, th1, t_ref) + c2 * _enthalpy_density(claw, th0, t_ref)) + ca * f1

        def body(state):
            th, k, _, its = state
            g, extra, bb, cap, h, _ = _parts(geo, klaw, claw, t_ref, th)
            rhs = hist + geo.qv + bb - d * (h - cap * th)
            diag = _full_diag(g, extra) + d * cap
            levels = (Level3(g[0], g[1], g[2], diag, agg0),) + coarse
            th_new, it, _ = _cg(lambda v: apply_op(g, diag, v), lambda r: mg_cycle3(levels, cinv, r, cycle), rhs, th,
                                tol, max_iter)
            change = jnp.max(jnp.abs(th_new - th))
            done = change <= tol_p * jnp.maximum(1.0, jnp.max(jnp.abs(th_new)))
            return th_new, k + 1, done | linear, its + it

        def cond(state):
            _, k, done, _ = state
            return (~done) & (k < max_picard)

        th2, k, done, its = jax.lax.while_loop(cond, body, (th1 + w * (th1 - th0), jnp.asarray(0),
                                                            jnp.asarray(False), jnp.asarray(0)))
        g, extra, bb, cap, h, ok = _parts(geo, klaw, claw, t_ref, th2)
        f2 = geo.qv + bb - apply_op(g, _full_diag(g, extra), th2)
        den = jnp.linalg.norm(hist + geo.qv + bb - d * (h - cap * th2))
        res = jnp.linalg.norm(hist + f2 - d * h) / jnp.where(den > 0, den, 1.0)
        out = (_monitor_values(th2, mon, n_mon), its, k, jnp.where(ok, res, jnp.inf), done)
        return (th2, th1, f2, jnp.minimum(lo, th2), jnp.maximum(hi, th2)), out

    return jax.lax.scan(step, carry, None, length=n)


def solve_transient_kappa3d(x, y, z, kappa_fn, q, bc: dict, rho_cp, dt, t_end=None, scheme: str = "euler",
                            monitors=(), t0_field=None, tol: float = 1e-10, tol_picard: float = 1e-10,
                            max_picard: int = 50, max_iter: int = 2000, cycle: str = "V",
                            dtype=jnp.float64) -> TransientKappaResult:
    """The 3-D transient with kappa = kappa_fn(T) and rho c_p = rho_cp(T) (see the module docstring for the laws;
    an array is a constant). x, y, z, q, bc, dt, t_end, scheme, monitors, t0_field, cycle: as
    transient3d.solve_transient3d. tol: the step residual (CG target tol / 2); tol_picard, max_picard: the stop rule
    of the Picard iteration of each step. When kappa or rho c_p is not finite and positive at the start of a stage,
    the run stops there (the result holds the steps before; converged False)."""
    if scheme not in SCHEMES:
        raise ValueError(f"scheme must be one of {SCHEMES}, not {scheme!r}")
    if cycle not in CYCLES:
        raise ValueError(f"cycle must be one of {CYCLES}, not {cycle!r}")
    if not (max_picard >= 1 and tol > 0 and tol_picard > 0):
        raise ValueError("max_picard must be at least 1, tol and tol_picard positive")
    stages = stages_from(dt, t_end)
    if scheme == "bdf2":
        ratios = [b[0] / a[0] for a, b in zip(stages[:-1], stages[1:])]
        if any(r >= BDF2_MAX_RATIO for r in ratios):
            raise ValueError(f"bdf2: a step ratio must be below 1 + sqrt(2); the stages have {max(ratios):.3g}")
    shape = tuple(len(e) - 1 for e in (x, y, z))
    ktab, ctab = law_table(kappa_fn, shape, "kappa"), law_table(rho_cp, shape, "rho_cp")
    q = np.asarray(q, float)
    t_ref = reference_temperature(bc)
    build3(x, y, z, np.ones(shape), q, bc)                       # checks the grid, q and bc (ValueError)
    d = [np.diff(np.asarray(e, float)) for e in (x, y, z)]
    vol = d[0][:, None, None] * d[1][None, :, None] * d[2][None, None, :]
    dev = solver_device()
    put = lambda a: jax.device_put(jnp.asarray(a, dtype), dev)              # noqa: E731
    sides, bvals = [], []
    for n_side, side in enumerate(SIDES3):
        spec = bc.get(side, ("neumann",))
        if spec[0] == "dirichlet":
            sides.append(n_side)
            bvals.append((0.0, float(spec[1]) - t_ref))
        elif spec[0] == "robin":
            sides.append(n_side)
            bvals.append((1.0 / float(spec[1]), float(spec[2]) - t_ref))
    geo = _Geo(tuple(put(np.reshape(d[ax], [-1 if a == ax else 1 for a in range(3)])) for ax in range(3)),
               put(q * vol), put(np.reshape(bvals, (-1, 2))), tuple(sides))
    klaw, claw = _dev_law(ktab, put, dev), _dev_law(ctab, put, dev)
    linear = ktab.constant and ctab.constant
    tref_d = put(t_ref)
    th = np.zeros(shape) if t0_field is None else np.broadcast_to(np.asarray(t0_field, float), shape) - t_ref
    if not np.all(np.isfinite(th)):
        raise ValueError("t0_field must be finite")
    th1 = put(th)
    carry = (th1, th1, _flux(geo, klaw, tref_d, th1), th1, th1)
    idx, wts, seg = _monitor_index(monitors, shape, vol)
    mon = (jax.device_put(jnp.asarray(idx), dev), put(wts), jax.device_put(jnp.asarray(seg), dev))
    n_mon = len(monitors)
    tol_d, tolp_d = put(0.5 * tol), put(tol_picard)
    mi, mp = jnp.asarray(max_iter), jnp.asarray(max_picard)
    times, vals = [0.0], [np.asarray(_monitor_values(th1, mon, n_mon))]
    its, pics, ress, dones = [], [], [], []
    t, dt_prev, n_hier, stopped = 0.0, None, 0, False
    for dt_s, n_s in stages:
        main = _coefficients(scheme, dt_s, dt_s, False)
        t_host = np.asarray(carry[0]) + t_ref
        kap, cap = ktab(t_host), ctab(t_host) * vol
        if not (np.all(np.isfinite(kap)) and np.all(kap > 0) and np.all(np.isfinite(cap)) and np.all(cap > 0)):
            stopped = True
            break
        p = build3(x, y, z, kap, q, bc)
        levels, cinv = build_hierarchy3(p.g, p.diag_extra + main[0] * cap, dev, dtype)
        n_hier += 1
        runs = [(main, n_s)]
        first = _coefficients(scheme, dt_s, dt_prev, dt_prev is None)
        if first != main:
            runs = [(first, 1)] + ([(main, n_s - 1)] if n_s > 1 else [])
        for coef, n in runs:
            dd, c1, c2, ca, _, w = coef
            carry, (v, it, pic, res, done) = _march(geo, klaw, claw, tref_d, tuple(levels[1:]), cinv, carry,
                                                    put(np.array([dd, c1, c2, ca, w])), tol_d, tolp_d, mi, mp, mon, n,
                                                    n_mon, cycle, levels[0].agg, linear)
            vals.append(np.asarray(v))
            for lst, a in ((its, it), (pics, pic), (ress, res), (dones, done)):
                lst.append(np.asarray(a))
            times.extend(t + dt_s * np.arange(1, n + 1))
            t += n * dt_s
        dt_prev = dt_s
    cat = lambda a, dt_: np.concatenate(a) if a else np.zeros(0, dt_)      # noqa: E731
    its, pics, ress, dones = cat(its, int), cat(pics, int), cat(ress, float), cat(dones, bool)
    worst = float(ress.max()) if ress.size else 0.0
    ok = bool(not stopped and ress.size and worst <= tol and np.all(dones))
    lo, hi = np.asarray(carry[3]) + t_ref, np.asarray(carry[4]) + t_ref
    outside = int(np.count_nonzero(ktab.outside(lo, hi) | ctab.outside(lo, hi)))
    if outside:
        warnings.warn(f"{outside} cells leave the valid temperature range of their kappa or rho_cp law during the "
                      f"transient (T from {lo.min():.1f} to {hi.max():.1f} K)", RuntimeWarning, stacklevel=2)
    return TransientKappaResult(np.asarray(times), np.vstack(vals) + t_ref, np.asarray(carry[0]) + t_ref,
                                int(its.sum()), int(its.max()) if its.size else 0, worst, ok, str(dev), scheme,
                                int(pics.sum()), int(pics.max()) if pics.size else 0, n_hier, outside)


def solve_transient_kappa2d(x, y, kappa_fn, q, bc: dict, rho_cp, dt, t_end=None, scheme: str = "euler", monitors=(),
                            t0_field=None, **kw) -> TransientKappaResult:
    """The 2-D cross-section (per unit length) as a 3-D problem with one z cell of 1 m and adiabatic z ends (as
    transient3d.solve_transient2d). kappa_fn and rho_cp: laws, material_kappa with (nx, ny) labels, or (nx, ny)
    arrays; q, t0_field, monitor masks: (nx, ny); monitor cells: (i, j). T of the result: (nx, ny)."""
    lift = lambda a: None if a is None else np.asarray(a, float)[..., None]   # noqa: E731
    mons = [(m[0], m[1], 0) if isinstance(m, tuple) else np.asarray(m, bool)[..., None] for m in monitors]
    shape2 = np.shape(q)
    lift_law = lambda f: f if callable(f) else np.broadcast_to(np.asarray(f, float), shape2)[..., None]  # noqa: E731
    res = solve_transient_kappa3d(x, y, np.array([0.0, 1.0]), lift_law(kappa_fn), lift(q), bc, lift_law(rho_cp), dt,
                                  t_end, scheme, mons, lift(t0_field), **kw)
    res.T = res.T[..., 0]
    return res
