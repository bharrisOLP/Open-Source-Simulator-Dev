"""Band structures: the lowest bands at each k point, with a warm start along the k path.

    freqs = solve_bands(cell, kpoints, num_bands=8, polarization="tm")     # (nk, num_bands), in units of 2 pi c / a

k points are fractions of the reciprocal lattice vectors (as in MPB). ``interpolate(points, n)`` makes a path.
"""

from __future__ import annotations

import os
import warnings

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from bands_gpu.geometry import Cell
from bands_gpu.lobpcg import lobpcg
from bands_gpu.operator import apply, make_operator, precondition


def interpolate(points, n: int) -> np.ndarray:
    """Like MPB's interpolate: n points between each pair of corner points."""
    points = np.asarray(points, float)
    out = []
    for a, b in zip(points[:-1], points[1:]):
        for t in np.arange(n + 1) / (n + 1):
            out.append(a + t * (b - a))
    out.append(points[-1])
    return np.asarray(out)


def solver_device() -> jax.Device:
    """The first GPU (ROCm or CUDA), else the CPU. BANDS_GPU_DEVICE=cpu forces the CPU (a same-algorithm
    baseline)."""
    if os.environ.get("BANDS_GPU_DEVICE", "").lower() != "cpu":
        try:
            return jax.devices("gpu")[0]
        except RuntimeError:
            pass
    return jax.devices("cpu")[0]


@dataclass
class BandResult:
    freqs: np.ndarray                 # (nk, num_bands)
    iterations: list[int] = field(default_factory=list)
    converged: list[bool] = field(default_factory=list)   # tol reached, or stalled at the precision floor
    stalled: list[bool] = field(default_factory=list)
    device: str = ""


def solve_bands(cell: Cell, kpoints, num_bands: int, polarization: str | None = None, extra: int = 4,
                tol: float = 1e-6, max_iter: int = 300, dtype=jnp.complex128, sub: int = 8,
                seed: int = 0, warm_start: bool = True) -> BandResult:
    if dtype == jnp.complex128 and not jax.config.jax_enable_x64:
        warnings.warn("bands_gpu: complex128 needs jax.config.update('jax_enable_x64', True); using complex64",
                      stacklevel=2)
        dtype = jnp.complex64
    device = solver_device()
    eps_inv = cell.eps_inverse(sub)
    m = 2 if polarization is None else 1
    b = num_bands + extra
    shape = (b, m) + tuple(cell.shape)
    rng = np.random.default_rng(seed)
    x = None
    out = BandResult(np.zeros((len(kpoints), num_bands)))
    fapply = jax.jit(apply)
    fpre = jax.jit(precondition)
    for i, k in enumerate(kpoints):
        op = make_operator(cell, k, polarization, eps_inv, dtype, device)
        if x is None or not warm_start:
            # random coefficients weighted to low |q| (the low bands live there)
            x = jax.device_put(jnp.asarray(rng.normal(size=shape) + 1j * rng.normal(size=shape), dtype), device)
            x = fpre(op, x)
        res = lobpcg(lambda v: fapply(op, v), lambda r: fpre(op, r), x, num_bands, tol=tol, max_iter=max_iter,
                     scale=(2 * np.pi / max(cell.size)) ** 2 * 1e-2)
        x = res.vectors                                   # warm start for the next k point
        out.freqs[i] = np.sqrt(np.maximum(res.values[:num_bands], 0.0)) / (2 * np.pi)
        out.iterations.append(res.iterations)
        # a stall counts as converged only near tol (the review found stalls declared converged at 5e-3)
        out.converged.append(bool(res.converged or (res.stalled and np.max(res.residuals[:num_bands]) < 100 * tol)))
        out.stalled.append(res.stalled)
    out.device = str(device)
    return out
