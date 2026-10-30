"""Yee-grid finite differences with a stretched-coordinate PML, as 1-D stencil coefficients.

Each 1-D derivative operator is bidiagonal. We keep it as two coefficient vectors, so that the GPU can apply it as a
stencil (no sparse matrix):

    forward  (E -> H positions):  (Df v)[i] = main[i] * v[i] + off[i] * v[i+1]      (v[N] = 0: PEC at the far end)
    backward (H -> E positions):  (Db v)[i] = main[i] * v[i] + off[i] * v[i-1]      (v[-1] = 0)

The boundary and PML conventions are those of the tidy3d local mode solver (which meow uses), so that both solvers
have the same discrete eigenproblem:
    - PEC on all four sides, or PMC on the near side of an axis with a PMC symmetry;
    - polynomial PML (order 3, kappa 1 -> 3, sigma_max 2 in units of avg_speed / (eta_0 dl));
    - all derivatives are normalized by k0 = 2 pi / wavelength.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PML_ORDER = 3
PML_KAPPA_MIN = 1.0
PML_KAPPA_MAX = 3.0
PML_SIGMA_MAX = 2.0


@dataclass(frozen=True)
class Stencil1D:
    """A bidiagonal 1-D derivative: ``main`` on the diagonal, ``off`` on the super- (forward) or sub-diagonal
    (backward). ``forward`` gives the direction."""

    main: np.ndarray
    off: np.ndarray
    forward: bool

    @property
    def n(self) -> int:
        return self.main.size

    def dense(self) -> np.ndarray:
        """The N x N matrix (for the preconditioner and for tests)."""
        m = np.diag(self.main.astype(np.complex128))
        if self.n > 1:
            if self.forward:
                m += np.diag(self.off[:-1], 1)
            else:
                m += np.diag(self.off[1:], -1)
        return m


def grid_steps(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Primal steps (between the corners, for forward derivatives) and dual steps (between the cell centers, for
    backward derivatives; the first dual step is the first primal step)."""
    dl_f = np.diff(np.asarray(coords, dtype=np.float64))
    dl_b = np.concatenate([dl_f[:1], 0.5 * (dl_f[:-1] + dl_f[1:])])
    return dl_f, dl_b


def s_value(dl: float, step: float, k0: float, avg_speed: complex) -> complex:
    """Stretching factor s = kappa + i sigma / (omega eps_0). ``step`` is the depth into the PML (0 at its inner edge,
    1 at the boundary). With sigma = sigma_max * avg_speed / (eta_0 dl) * step^m, sigma / (omega eps_0) reduces to
    sigma_max * avg_speed * step^m / (k0 dl)."""
    kappa = PML_KAPPA_MIN + (PML_KAPPA_MAX - PML_KAPPA_MIN) * step**PML_ORDER
    return kappa + 1j * PML_SIGMA_MAX * avg_speed * step**PML_ORDER / (k0 * dl)


def s_factors(dls: np.ndarray, n_pml: int, near_pml: bool, k0: float, speeds: tuple[complex, complex],
              forward: bool) -> np.ndarray:
    """The stretching factors at the H (forward) or E (backward) positions of one axis."""
    n = dls.size
    s = np.ones(n, dtype=np.complex128)
    if n_pml == 0:
        return s
    for i in range(n):
        if forward:
            if i <= n_pml - 1 and near_pml:
                s[i] = s_value(dls[0], (n_pml - i - 0.5) / n_pml, k0, speeds[0])
            elif i >= n - n_pml:
                s[i] = s_value(dls[-1], (i - (n - n_pml) + 0.5) / n_pml, k0, speeds[1])
        else:
            if i < n_pml and near_pml:
                s[i] = s_value(dls[0], (n_pml - i) / n_pml, k0, speeds[0])
            elif i > n - n_pml:
                s[i] = s_value(dls[-1], (i - (n - n_pml)) / n_pml, k0, speeds[1])
    return s


def pml_speeds(eps_diag: np.ndarray, npml: tuple[int, int]) -> np.ndarray:
    """Relative speed of light in the four PML regions (x-, x+, y-, y+): 1 / sqrt(mean eps), with mu = 1. The mean
    is over the three diagonal components. ``eps_diag`` has the shape (3, Nx, Ny)."""
    nx, ny = eps_diag.shape[1:]

    def mean(a: np.ndarray) -> complex:
        return complex(np.mean(a)) if a.size else 1.0

    avg = np.array([
        mean(eps_diag[:, : npml[0], :]),
        mean(eps_diag[:, nx - npml[0] + 1:, :]),
        mean(eps_diag[:, :, : npml[1]]),
        mean(eps_diag[:, :, ny - npml[1] + 1:]),
    ])
    return 1 / np.sqrt(avg)


def axis_stencils(coords: np.ndarray, k0: float, n_pml: int = 0, symmetry: int = 0,
                  speeds: tuple[complex, complex] = (1.0, 1.0)) -> tuple[Stencil1D, Stencil1D]:
    """Forward and backward derivatives along one axis, with PML and normalized by k0. ``symmetry`` of the near
    side: 0 none (PEC wall, PML if n_pml > 0), 1 PMC symmetry, -1 PEC symmetry (no near-side PML for both)."""
    dl_f, dl_b = grid_steps(coords)
    n = dl_f.size
    pmc = symmetry == 1
    near_pml = symmetry == 0
    if n == 1:                                   # a 1-D cross-section: no derivative along this axis
        z = np.zeros(1, dtype=np.complex128)
        return Stencil1D(z, z, True), Stencil1D(z, z, False)
    f_main = -1.0 / dl_f
    f_off = 1.0 / dl_f
    if not pmc:
        f_main = f_main.copy()
        f_main[0] = 0.0
    b_main = 1.0 / dl_b
    b_off = -1.0 / dl_b
    b_main = b_main.copy()
    b_main[0] = 2.0 / dl_b[0] if pmc else 0.0
    sf = s_factors(dl_f, n_pml, near_pml, k0, speeds, forward=True)
    sb = s_factors(dl_b, n_pml, near_pml, k0, speeds, forward=False)
    fwd = Stencil1D((f_main / sf / k0).astype(np.complex128), (f_off / sf / k0).astype(np.complex128), True)
    bwd = Stencil1D((b_main / sb / k0).astype(np.complex128), (b_off / sb / k0).astype(np.complex128), False)
    return fwd, bwd
