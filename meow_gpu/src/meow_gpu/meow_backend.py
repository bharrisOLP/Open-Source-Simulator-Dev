"""A drop-in FDE backend for meow: ``compute_modes_gpu(cs, num_modes)`` returns ``meow.Modes``.

It solves the same discrete eigenproblem as ``meow.compute_modes_tidy3d`` (the tidy3d local mode solver), with the
GPU Davidson solver of ``meow_gpu.fde``, and then applies the same meow post-processing (filters and
orthonormalization). The rest of meow (overlaps, S-matrices, SAX cascade) works without a change.

Warm start: ``GPUModeSolver`` keeps the eigenvectors of its last solve for each grid shape. The next solve on the
same grid (the next cell of a taper, the next wavelength) starts from them. Call ``reset()`` for a new device.

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
from meow_gpu.fde.eigensolver import davidson, make_preconditioner, precondition
from meow_gpu.fde.operator import FDEProblem, fields, make_problem

PRECISIONS = ("single", "mixed", "double")
TOL = {"single": 1e-5, "mixed": 1e-10, "double": 1e-10}
TOL_POWER_FLOW = 1e2 * float(np.finfo(np.float32).eps) / 10   # = tidy3d TOL_POWER_FLOW_DIRECTION
SINGLE_FLOOR = 1e-3
SINGLE_TOL_CAP = 1e-3   # the adaptive single-precision tolerance never goes above this (warning when capped)   # complex64: a stall below this residual is convergence to working precision
FLOOR_FACTOR = 10.0   # single precision: tol >= FLOOR_FACTOR * eps32 * |A| / |theta|


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
    _memory: dict = field(default_factory=dict, repr=False)

    def reset(self) -> None:
        self._memory.clear()

    def solve(self, problem: FDEProblem, num_modes: int, target_neff: float | None = None):
        """Eigenpairs of one problem: (neff (k,), E (k, 3, Nx, Ny), H (k, 3, Nx, Ny)), sorted by Re(neff)."""
        if self.precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {PRECISIONS}")
        nmax = float(np.sqrt(np.max(np.abs(np.stack([problem.eps_xx, problem.eps_yy, problem.eps_zz])))))  # as tidy3d
        n_target = (target_neff if target_neff is not None else nmax) * (1 + 10 * np.finfo(float).eps)
        target = -(n_target**2)
        b = num_modes + self.extra
        pc = make_preconditioner(problem)
        key = (problem.shape, b)
        warm = self.warm_start and key in self._memory
        if warm:
            x0 = self._memory[key]
        else:
            rng = np.random.default_rng(self.seed)
            x0 = put(rng.normal(size=(b, 2) + problem.shape) + 0j, jnp.complex64)
            x0 = precondition(pc, x0, put(np.full(b, target), jnp.complex64))
        if self.precision == "mixed":
            op = (problem.device(jnp.complex64), problem.device(jnp.complex128))
        else:
            op = problem.device(jnp.complex64 if self.precision == "single" else jnp.complex128)
        tol = self.tol if self.tol is not None else TOL[self.precision]
        if self.precision == "single":
            # the complex64 floor of the relative residual is about eps32 * |A| / |theta| (|A| grows as 1/dx^2)
            anorm = float(jnp.max(jnp.abs(pc.lam)))
            floor = FLOOR_FACTOR * float(np.finfo(np.float32).eps) * anorm / max(1.0, abs(target))
            if floor > SINGLE_TOL_CAP:
                warnings.warn(f"meow_gpu: the grid is fine for complex64 (floor {floor:.1e}); the tolerance is capped "
                              f"at {SINGLE_TOL_CAP:.0e}. Use precision='mixed' for accurate fields.", stacklevel=2)
            tol = max(tol, min(floor, SINGLE_TOL_CAP))
        res = davidson(op, pc, num_modes, target, x0, tol=tol, max_iter=self.max_iter, precision=self.precision)
        self._memory[key] = res.vectors
        resid = float(np.max(res.residuals))
        converged = res.converged or (self.precision == "single" and resid < SINGLE_FLOOR)
        self.last_info = SolveInfo(res.iterations, res.matvecs, converged, warm, resid)
        if not converged:
            warnings.warn(f"meow_gpu: the eigensolver did not converge (residual {np.max(res.residuals):.1e})",
                          stacklevel=2)
        neff = np.sqrt(-res.values.astype(np.complex128))
        vec = res.vectors[:num_modes].astype(jnp.complex128)
        e, h = fields(problem.device(jnp.complex128), vec, put(neff))
        e, h = np.array(e), np.array(h)          # writable host copies
        # the direction convention of tidy3d: power flows in +z when the power flow is reliable; else (modes below
        # cutoff, where Re(Pz) is rounding noise) the passive branch keff >= 0. Flip H_t, E_z and neff.
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
    import meow as mw
    from meow.fde.post_process import post_process_modes

    post_process = post_process or post_process_modes
    why = _supported(cs)
    if why:
        warnings.warn(f"meow_gpu: {why} is not supported yet; using the tidy3d CPU solver", stacklevel=2)
        return mw.compute_modes_tidy3d(cs, num_modes=num_modes, target_neff=target_neff)
    solver = solver or _DEFAULT
    problem = make_problem(cs.nx**2, cs.ny**2, cs.nz**2, cs.mesh.x, cs.mesh.y, cs.env.wl,
                           num_pml=tuple(cs.mesh.num_pml))
    neff, e, h = solver.solve(problem, num_modes, target_neff)
    modes = [mw.Mode(cs=cs, Ex=e[i, 0], Ey=e[i, 1], Ez=e[i, 2], Hx=h[i, 0], Hy=h[i, 1], Hz=h[i, 2],
                     neff=complex(neff[i])) for i in range(len(neff))]
    return post_process(modes)


def device_summary() -> str:
    return str(solver_device())
