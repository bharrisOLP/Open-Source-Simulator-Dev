"""A drop-in FDE backend for meow: ``compute_modes_gpu(cs, num_modes)`` returns ``meow.Modes``.

It solves the same discrete eigenproblem as ``meow.compute_modes_tidy3d`` (the tidy3d local mode solver), with the
GPU Davidson solver of ``meow_gpu.fde``, and then applies meow's post-processing (filters and orthonormalization),
with one correction: each mode keeps its own neff (``post_process_modes``, below). The rest of meow (overlaps,
S-matrices, SAX cascade) works without a change.

The neff correction: in meow 0.15, ``Mode.__add__`` returns neff = (a.neff + b.neff) / 2, and the Gram-Schmidt step
of ``meow.fde.post_process.orthonormalize_modes`` computes ``current - c * b``. So every mode after the first gets an
average of the neff of the earlier modes, even when c is about 0 (for a 0.525 um Si strip, the first cell of the
taper of bench/bench_eme_taper.py, TM0 becomes 2.137 instead of 1.794). The EME then propagates the higher modes almost
in phase with the fundamental mode, and the mode conversion of a taper adds up coherently (a 10 um taper: |S21|^2
0.955; corrected 0.991; 3-D FDTD 0.9961 on a 20 nm grid, benchmarks/2026-10-30/fdtd-v3.jsonl).
``meow.compute_modes_tidy3d`` has the same error; the tidy3d fallback of ``compute_modes_gpu`` (bends, angled
propagation) uses the corrected post-processing too.

Warm start: ``GPUModeSolver`` keeps the eigenvectors of its last solve for each grid shape. The next solve on the
same grid (the next cell of a taper, the next wavelength) starts from them. Call ``reset()`` for a new device.

Batched solves: ``compute_modes_gpu_batch(css, num_modes, batch=B)`` solves B cross-sections with the same grid
shape at the same time (``GPUModeSolver.solve_batch``, ``meow_gpu.fde.eigensolver.davidson_batch``). The results
are the results of ``compute_modes_gpu`` (the same eigensolver steps, post-processing and direction rule). The gain
is small: on an RX 7900 XT, for the 30-cell taper of bench/bench_eme_batch.py (36k unknowns per cell), the batched
solve was about 1.2x faster in single precision and not faster in mixed precision
(benchmarks/2026-10-30/eme-batch.jsonl). The per-cell path stays the default.

Not supported yet (the call falls back to tidy3d, with a warning): bend radius and angled propagation. meow passes
diagonal permittivity only (as for its tidy3d backend).
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from meow_gpu.device import put, solver_device
from meow_gpu.fde.eigensolver import (
    davidson,
    davidson_batch,
    make_preconditioner,
    precondition,
    precondition_batch,
    stack,
)
from meow_gpu.fde.operator import FDEProblem, fields, make_problem

PRECISIONS = ("single", "mixed", "double")
TOL = {"single": 1e-5, "mixed": 1e-10, "double": 1e-10}
TOL_POWER_FLOW = 1e2 * float(np.finfo(np.float32).eps) / 10   # = tidy3d TOL_POWER_FLOW_DIRECTION
SINGLE_FLOOR = 1e-3
SINGLE_TOL_CAP = 1e-3   # the adaptive single-precision tolerance never goes above this (warning when capped)   # complex64: a stall below this residual is convergence to working precision
FLOOR_FACTOR = 10.0   # single precision: tol >= FLOOR_FACTOR * eps32 * |A| / |theta|


def post_process_modes(modes, inner_product: Callable | None = None, gm_tolerance: float = 0.01):
    """meow's default post-processing (filter PML and lossy modes, normalize, Gram-Schmidt with a drop tolerance),
    but each kept mode keeps the neff of the eigenmode that it comes from (see the module docstring)."""
    import meow as mw
    from meow.fde.post_process import filter_modes, normalize_modes
    from meow.mode import inner_product as default_inner_product

    ip = inner_product or default_inner_product
    basis = []
    for mode in normalize_modes(filter_modes(modes), ip):
        current = mode
        for b in basis:
            current = current - (ip(b, current) / ip(b, b)) * b
        norm_sq = ip(current, current)
        if abs(norm_sq) < gm_tolerance:
            continue
        current = current / np.sqrt(norm_sq)
        basis.append(mw.Mode(neff=mode.neff, cs=current.cs, Ex=current.Ex, Ey=current.Ey, Ez=current.Ez,
                             Hx=current.Hx, Hy=current.Hy, Hz=current.Hz))
    return basis


@dataclass
class SolveInfo:
    iterations: int
    matvecs: int
    converged: bool
    warm: bool
    residual: float


@dataclass
class GPUModeSolver:
    """Mode solver with a warm-start memory. ``extra``: block vectors beyond ``num_modes`` (more vectors, fewer
    iterations on clustered spectra)."""

    precision: str = "single"
    extra: int = 8
    tol: float | None = None
    max_iter: int = 500
    warm_start: bool = True
    seed: int = 0
    last_info: SolveInfo | None = None
    last_batch_info: list[SolveInfo] | None = None
    _memory: dict = field(default_factory=dict, repr=False)

    def reset(self) -> None:
        self._memory.clear()

    def solve(self, problem: FDEProblem, num_modes: int, target_neff: float | None = None):
        """Eigenpairs of one problem: (neff (k,), E (k, 3, Nx, Ny), H (k, 3, Nx, Ny)), sorted by Re(neff)."""
        self._check()
        target = _target(problem, target_neff)
        b = num_modes + self.extra
        pc = make_preconditioner(problem)
        key = (problem.shape, b)
        warm = self.warm_start and key in self._memory
        if warm:
            x0 = self._memory[key]
        else:
            x0 = precondition(pc, self._random_block(problem.shape, b), put(np.full(b, target), jnp.complex64))
        op = self._operator([problem], batched=False)
        tol, floor = self._tolerance(float(jnp.max(jnp.abs(pc.lam))), target)
        if floor is not None:
            _warn_floor(floor)
        res = davidson(op, pc, num_modes, target, x0, tol=tol, max_iter=self.max_iter, precision=self.precision)
        self._memory[key] = res.vectors
        self.last_info = self._info(res.iterations, res.matvecs, res.converged, warm, res.residuals)
        if not self.last_info.converged:
            warnings.warn(f"meow_gpu: the eigensolver did not converge (residual {np.max(res.residuals):.1e})",
                          stacklevel=2)
        neff = np.sqrt(-res.values.astype(np.complex128))
        vec = res.vectors[:num_modes].astype(jnp.complex128)
        e, h = fields(problem.device(jnp.complex128), vec, put(neff))
        return _orient(problem, neff, np.array(e), np.array(h))       # writable host copies

    def solve_batch(self, problems, num_modes: int, target_neff: float | None = None) -> list:
        """Eigenpairs of B problems with the same grid shape, solved at the same time (``davidson_batch``): a list
        of B results (neff, E, H), each as ``solve`` gives it.

        Warm start: member i starts from the last vectors of member i of the previous batch with the same grid shape
        and the same B. If there is no such batch, all members start from the memory of ``solve``. Else each member
        starts from the random block of ``solve``. After the solve, the memory of ``solve`` holds the vectors of the
        last member. ``last_batch_info`` has the SolveInfo of each member; ``last_info`` is that of the last member.
        """
        problems = list(problems)
        if not problems:
            return []
        self._check()
        shape = problems[0].shape
        if any(p.shape != shape for p in problems):
            raise ValueError("all problems of a batch need the same grid shape")
        nbat, b = len(problems), num_modes + self.extra
        targets = np.array([_target(p, target_neff) for p in problems])
        pc = stack([make_preconditioner(p) for p in problems])
        key, batch_key = (shape, b), (shape, b, nbat)
        warm = self.warm_start and (batch_key in self._memory or key in self._memory)
        if warm and batch_key in self._memory:
            x0 = self._memory[batch_key]
        elif warm:
            x0 = jnp.broadcast_to(self._memory[key], (nbat,) + self._memory[key].shape)
        else:
            x = jnp.broadcast_to(self._random_block(shape, b), (nbat, b, 2) + shape)
            x0 = precondition_batch(pc, x, put(np.repeat(targets[:, None], b, axis=1), jnp.complex64))
        op = self._operator(problems, batched=True)
        anorms = np.asarray(jnp.max(jnp.abs(pc.lam), axis=(1, 2, 3)), dtype=float)
        tols, floors = zip(*(self._tolerance(a, t) for a, t in zip(anorms, targets)))
        if any(f is not None for f in floors):
            _warn_floor(max(f for f in floors if f is not None))
        res = davidson_batch(op, pc, num_modes, targets, x0, tol=np.array(tols), max_iter=self.max_iter,
                             precision=self.precision)
        self._memory[batch_key] = res.vectors
        self._memory[key] = res.vectors[-1]
        self.last_batch_info = [self._info(int(res.iterations[i]), int(res.matvecs[i]), bool(res.converged[i]),
                                           warm, res.residuals[i]) for i in range(nbat)]
        self.last_info = self.last_batch_info[-1]
        bad = [i for i, info in enumerate(self.last_batch_info) if not info.converged]
        if bad:
            worst = max(self.last_batch_info[i].residual for i in bad)
            warnings.warn(f"meow_gpu: the eigensolver did not converge for {len(bad)} of {nbat} problems of the "
                          f"batch (largest residual {worst:.1e})", stacklevel=2)
        neff = np.sqrt(-res.values.astype(np.complex128))          # (B, k)
        vec = res.vectors[:, :num_modes].astype(jnp.complex128)
        op128 = op[1] if self.precision == "mixed" else (op if self.precision == "double" else
                                                         stack([p.device(jnp.complex128) for p in problems]))
        e, h = _fields_batch(op128, vec, put(neff))
        e, h = np.array(e), np.array(h)
        return [_orient(p, neff[i], e[i], h[i]) for i, p in enumerate(problems)]

    # -- shared steps of solve and solve_batch ------------------------------------------------------------------

    def _check(self) -> None:
        if self.precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {PRECISIONS}")

    def _random_block(self, shape: tuple[int, int], b: int) -> jax.Array:
        rng = np.random.default_rng(self.seed)
        return put(rng.normal(size=(b, 2) + shape) + 0j, jnp.complex64)

    def _operator(self, problems: list[FDEProblem], batched: bool):
        """The device operator of one problem, or (``batched``) the operator of a batch (``stack``)."""
        def one(dtype):
            return stack([p.device(dtype) for p in problems]) if batched else problems[0].device(dtype)
        if self.precision == "mixed":
            return one(jnp.complex64), one(jnp.complex128)
        return one(jnp.complex64 if self.precision == "single" else jnp.complex128)

    def _tolerance(self, anorm: float, target: float) -> tuple[float, float | None]:
        """The tolerance, and the complex64 floor when the cap applies (else None). ``anorm``: max |lambda| of the
        preconditioner (an estimate of |A|)."""
        tol = self.tol if self.tol is not None else TOL[self.precision]
        if self.precision != "single":
            return tol, None
        # the complex64 floor of the relative residual is about eps32 * |A| / |theta| (|A| grows as 1/dx^2)
        floor = FLOOR_FACTOR * float(np.finfo(np.float32).eps) * anorm / max(1.0, abs(target))
        return max(tol, min(floor, SINGLE_TOL_CAP)), (floor if floor > SINGLE_TOL_CAP else None)

    def _info(self, iterations: int, matvecs: int, converged: bool, warm: bool, residuals) -> SolveInfo:
        resid = float(np.max(residuals))
        converged = converged or (self.precision == "single" and resid < SINGLE_FLOOR)
        return SolveInfo(iterations, matvecs, converged, warm, resid)


_fields_batch = jax.jit(jax.vmap(fields))


def _target(problem: FDEProblem, target_neff: float | None) -> float:
    """The eigenvalue target -(n_target^2); default n_target: the largest index of the problem (as tidy3d)."""
    nmax = float(np.sqrt(np.max(np.abs(np.stack([problem.eps_xx, problem.eps_yy, problem.eps_zz])))))
    n_target = (target_neff if target_neff is not None else nmax) * (1 + 10 * np.finfo(float).eps)
    return -(n_target**2)


def _warn_floor(floor: float) -> None:
    warnings.warn(f"meow_gpu: the grid is fine for complex64 (floor {floor:.1e}); the tolerance is capped "
                  f"at {SINGLE_TOL_CAP:.0e}. Use precision='mixed' for accurate fields.", stacklevel=3)


def _orient(problem: FDEProblem, neff: np.ndarray, e: np.ndarray, h: np.ndarray):
    """The direction convention of tidy3d, and the sort by Re(neff) (e and h are changed in place). Power flows
    in +z when the power flow is reliable; else (modes below cutoff, where Re(Pz) is rounding noise) the passive
    branch keff >= 0. Flip H_t, E_z and neff."""
    neff = np.array(neff)
    dS = np.outer(problem.dl_f[0], problem.dl_b[1]), np.outer(problem.dl_b[0], problem.dl_f[1])
    pz = 0.5 * np.sum(e[:, 0] * h[:, 1].conj() * dS[0] - e[:, 1] * h[:, 0].conj() * dS[1], axis=(1, 2))
    reliable = np.abs(np.real(pz)) > TOL_POWER_FLOW * np.abs(pz)
    flip = (reliable & (np.real(pz) < 0)) | (~reliable & (np.imag(neff) < 0))
    e[flip, 2] *= -1
    h[flip, 0] *= -1
    h[flip, 1] *= -1
    neff[flip] *= -1
    order = np.argsort(-np.real(neff))
    return neff[order], e[order], h[order]


_DEFAULT = GPUModeSolver()


def default_solver() -> GPUModeSolver:
    return _DEFAULT


def _supported(cs) -> str | None:
    if not (np.isinf(cs.mesh.bend_radius) or np.isnan(cs.mesh.bend_radius)):
        return "bend radius"
    if cs.mesh.angle_theta != 0:
        return "angled propagation"
    return None


def compute_modes_gpu(cs, num_modes: int = 10, target_neff: float | None = None,
                      solver: GPUModeSolver | None = None, post_process: Callable | None = None):
    """``meow.Modes`` of a ``meow.CrossSection`` from the GPU solver (the signature of meow.compute_modes)."""
    post_process = post_process or post_process_modes
    why = _supported(cs)
    if why:
        return _tidy3d_fallback(cs, why, num_modes, target_neff, post_process)
    solver = solver or _DEFAULT
    neff, e, h = solver.solve(_problem(cs), num_modes, target_neff)
    return _modes(cs, neff, e, h, post_process)


def compute_modes_gpu_batch(css, num_modes: int = 10, target_neff: float | None = None,
                            solver: GPUModeSolver | None = None, post_process: Callable | None = None,
                            batch: int = 8) -> list:
    """``meow.Modes`` of each ``meow.CrossSection`` of ``css`` (a list in the order of css), with ``batch``
    cross-sections of the same grid shape in each solve (``GPUModeSolver.solve_batch``). The results are those
    of ``compute_modes_gpu`` for each cross-section.

    Order: the cross-sections of one grid shape (in the order of css) are divided into ``batch`` runs of
    neighbouring cells, and batch j solves cell j of each run. So member i of a batch starts from the modes of the
    previous cell of its run (the warm start of ``solve_batch``). When the last run is shorter, its member solves
    the last cell again (the result is not used), so that all batches have the same size (one JIT compile).
    Cross-sections that the GPU solver does not support use ``compute_modes_gpu`` (the tidy3d fallback).
    """
    if batch < 1:
        raise ValueError("batch must be 1 or more")
    post_process = post_process or post_process_modes
    solver = solver or _DEFAULT
    css = list(css)
    out: list = [None] * len(css)
    groups: dict = {}
    for i, cs in enumerate(css):
        why = _supported(cs)
        if why:
            out[i] = _tidy3d_fallback(cs, why, num_modes, target_neff, post_process)
        else:
            groups.setdefault(tuple(np.shape(cs.nx)), []).append(i)
    for idx in groups.values():
        n = len(idx)
        runs = -(-n // batch)                  # the number of batches = the length of each run
        width = -(-n // runs)                  # the batch size (<= batch)
        for j in range(runs):
            pos = [i * runs + j for i in range(width)]
            cells = [idx[min(q, n - 1)] for q in pos]
            results = solver.solve_batch([_problem(css[c]) for c in cells], num_modes, target_neff)
            for q, c, (neff, e, h) in zip(pos, cells, results):
                if q < n:
                    out[c] = _modes(css[c], neff, e, h, post_process)
    return out


def _tidy3d_fallback(cs, why: str, num_modes: int, target_neff: float | None, post_process: Callable):
    """The tidy3d CPU solver, with a warning that points at the caller of the public function."""
    import meow as mw

    warnings.warn(f"meow_gpu: {why} is not supported yet; using the tidy3d CPU solver", stacklevel=3)
    return mw.compute_modes_tidy3d(cs, num_modes=num_modes, target_neff=target_neff, post_process=post_process)


def _problem(cs) -> FDEProblem:
    return make_problem(cs.nx**2, cs.ny**2, cs.nz**2, cs.mesh.x, cs.mesh.y, cs.env.wl,
                        num_pml=tuple(cs.mesh.num_pml))


def _modes(cs, neff, e, h, post_process: Callable):
    import meow as mw

    modes = [mw.Mode(cs=cs, Ex=e[i, 0], Ey=e[i, 1], Ez=e[i, 2], Hx=h[i, 0], Hy=h[i, 1], Hz=h[i, 2],
                     neff=complex(neff[i])) for i in range(len(neff))]
    return post_process(modes)


def device_summary() -> str:
    return str(solver_device())
