"""The matrix-free operator and its assembled form equal the tidy3d mode-solver matrix (machine precision)."""
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse.linalg as spl
from tidy3d_ref import tidy3d_matrix

from meow_gpu.fde.operator import apply, assemble, make_problem


@pytest.mark.parametrize("pml,sym", [((0, 0), (0, 0)), ((4, 3), (0, 0)), ((4, 3), (1, 0)), ((0, 5), (0, -1)),
                                     ((3, 3), (1, 1))])
def test_operator_equals_tidy3d(pml, sym):
    rng = np.random.default_rng(1)
    nx, ny = 23, 17
    x = np.cumsum(np.r_[0, rng.uniform(0.02, 0.05, nx)])
    y = np.cumsum(np.r_[0, rng.uniform(0.02, 0.05, ny)])
    eps = [rng.uniform(2, 12, (nx, ny)) + 1j * rng.uniform(0, 0.1, (nx, ny)) for _ in range(3)]
    p = make_problem(*eps, x, y, 1.55, pml, sym)
    ref = tidy3d_matrix(*eps, x, y, 1.55, pml, sym)
    assert spl.norm(assemble(p) - ref) / spl.norm(ref) < 1e-14
    v = rng.normal(size=(3, 2, nx, ny)) + 1j * rng.normal(size=(3, 2, nx, ny))
    av = np.asarray(apply(p.device(), jnp.asarray(v)))
    want = np.stack([(ref @ vi.ravel()).reshape(2, nx, ny) for vi in v])
    assert np.linalg.norm(av - want) / np.linalg.norm(want) < 1e-14
