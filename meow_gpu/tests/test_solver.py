"""The GPU eigensolver finds the tidy3d modes (same grid) in each precision, and a warm start converges faster."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.constants import c
from structures import strip

from meow_gpu.fde.operator import make_problem
from meow_gpu.meow_backend import GPUModeSolver

K = 4


@pytest.fixture(scope="module")
def case():
    exx, eyy, ezz, x, y = strip(dx=0.04, dy=0.04, w_dom=2.4, h_dom=1.6)
    from tidy3d.components.mode.solver import compute_modes
    ms = SimpleNamespace(num_modes=K, target_neff=None, num_pml=(0, 0), filter_pol=None, angle_theta=0.0,
                         angle_phi=0.0, bend_radius=None, precision="double", bend_axis=None, track_freq="central",
                         group_index_step=False)
    z = 0 * exx
    out = compute_modes(eps_cross=[exx, z, z, z, eyy, z, z, z, ezz], coords=[x, y], freq=c / (1.55e-6),
                        mode_spec=ms, precision="double")
    n_ref = np.sort(np.real(out[1].squeeze()))[::-1]
    return make_problem(exx, eyy, ezz, x, y, 1.55), n_ref


@pytest.mark.parametrize("precision,tol", [("double", 1e-9), ("mixed", 1e-9), ("single", 2e-5)])
def test_neff_matches_tidy3d(case, precision, tol):
    p, n_ref = case
    s = GPUModeSolver(precision=precision, extra=4)
    neff, e, h = s.solve(p, K)
    assert s.last_info.converged
    assert np.max(np.abs(neff.real - n_ref)) < tol


def test_warm_start_is_faster(case):
    p, _ = case
    s = GPUModeSolver(precision="double", extra=4)
    s.solve(p, K)
    cold = s.last_info.iterations
    s.solve(p, K)
    assert s.last_info.warm and s.last_info.iterations < cold / 2
