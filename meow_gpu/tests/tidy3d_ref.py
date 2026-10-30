"""The tidy3d matrix of the diagonal mode solver, built with tidy3d's own derivative and PML functions (reference)."""
import numpy as np
import scipy.sparse as sp
from tidy3d.components.mode.derivatives import create_d_matrices, create_s_matrices
from tidy3d.constants import C_0


def tidy3d_matrix(eps_xx, eps_yy, eps_zz, x, y, wavelength, num_pml=(0, 0), symmetry=(0, 0)):
    nx, ny = eps_xx.shape
    n = nx * ny
    freq = C_0 / wavelength
    omega = 2 * np.pi * freq
    k0 = omega / C_0
    eps_tensor = np.zeros((3, 3, n), dtype=complex)
    mu_tensor = np.zeros((3, 3, n), dtype=complex)
    for i, e in enumerate((eps_xx, eps_yy, eps_zz)):
        eps_tensor[i, i] = np.asarray(e, dtype=complex).ravel()
        mu_tensor[i, i] = 1.0
    dl_f = [np.diff(c) for c in (x, y)]
    dl_b = [np.hstack((d[0], (d[:-1] + d[1:]) / 2)) for d in dl_f]
    dls = (dl_f, dl_b)
    dmin_pmc = [s == 1 for s in symmetry]
    d = create_d_matrices((nx, ny), dls, dmin_pmc)
    s = create_s_matrices(omega, (nx, ny), num_pml, dls, eps_tensor, mu_tensor, np.array(symmetry) == 0)
    dxf, dxb, dyf, dyb = [S.dot(D) / k0 for S, D in zip(s, d)]
    iez = sp.spdiags(1 / eps_tensor[2, 2], [0], n, n)
    eye = sp.eye(n)
    p_mu = sp.bmat([[None, eye], [-eye, None]])
    p_partial = sp.bmat([[-dxf @ iez @ dyb, dxf @ iez @ dxb], [-dyf @ iez @ dyb, dyf @ iez @ dxb]])
    q_ep = sp.bmat([[None, sp.spdiags(eps_tensor[1, 1], [0], n, n)], [sp.spdiags(-eps_tensor[0, 0], [0], n, n), None]])
    q_partial = sp.bmat([[-dxb @ dyf, dxb @ dxf], [-dyb @ dyf, dyb @ dxf]])
    return (p_mu @ (q_ep + q_partial) + p_partial @ q_ep).tocsr()


class _Captured(Exception):
    pass


def tidy3d_solver_matrix(eps_xx, eps_yy, eps_zz, x, y, wavelength, num_pml=(0, 0), symmetry=(0, 0),
                         bend_radius=None, bend_axis=None, plane_center=(0.0, 0.0)):
    """The matrix that tidy3d's own ``compute_modes`` gives to its eigensolver (captured at the call of
    ``EigSolver.solver_eigs``; the eigensolver does not run). It includes tidy3d's coordinate transform (bend),
    its PML and its media; nothing is built here. Media without a good conductor: no tidy3d preconditioner."""
    from types import SimpleNamespace

    from tidy3d.components.mode.solver import EigSolver, compute_modes

    captured = {}

    def capture(cls, mat, *args, **kwargs):
        captured["mat"] = mat
        raise _Captured

    ms = SimpleNamespace(num_modes=1, target_neff=None, num_pml=num_pml, filter_pol=None, angle_theta=0.0,
                         angle_phi=0.0, bend_radius=bend_radius, precision="double", bend_axis=bend_axis,
                         track_freq="central", group_index_step=False)
    z = np.zeros_like(eps_xx)
    original = EigSolver.__dict__["solver_eigs"]
    EigSolver.solver_eigs = classmethod(capture)
    try:
        compute_modes(eps_cross=[eps_xx, z, z, z, eps_yy, z, z, z, eps_zz], coords=[x, y], freq=C_0 / wavelength,
                      mode_spec=ms, precision="double", symmetry=symmetry, plane_center=plane_center)
    except _Captured:
        pass
    finally:
        EigSolver.solver_eigs = original
    return sp.csr_matrix(captured["mat"])
