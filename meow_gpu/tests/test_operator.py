"""The matrix-free operator and its assembled form equal the tidy3d mode-solver matrix (machine precision)."""
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse.linalg as spl
from tidy3d_ref import tidy3d_matrix, tidy3d_solver_matrix

from meow_gpu.fde.operator import apply, assemble, make_problem


def _random_case(seed=1, nx=23, ny=17):
    rng = np.random.default_rng(seed)
    x = np.cumsum(np.r_[0, rng.uniform(0.02, 0.05, nx)])
    y = np.cumsum(np.r_[0, rng.uniform(0.02, 0.05, ny)])
    eps = [rng.uniform(2, 12, (nx, ny)) + 1j * rng.uniform(0, 0.1, (nx, ny)) for _ in range(3)]
    return rng, eps, x, y


def _check(p, ref, rng):
    nx, ny = p.shape
    assert spl.norm(assemble(p) - ref) / spl.norm(ref) < 1e-14
    v = rng.normal(size=(3, 2, nx, ny)) + 1j * rng.normal(size=(3, 2, nx, ny))
    av = np.asarray(apply(p.device(), jnp.asarray(v)))
    want = np.stack([(ref @ vi.ravel()).reshape(2, nx, ny) for vi in v])
    assert np.linalg.norm(av - want) / np.linalg.norm(want) < 1e-14


@pytest.mark.parametrize("pml,sym", [((0, 0), (0, 0)), ((4, 3), (0, 0)), ((4, 3), (1, 0)), ((0, 5), (0, -1)),
                                     ((3, 3), (1, 1))])
def test_operator_equals_tidy3d(pml, sym):
    rng, eps, x, y = _random_case()
    p = make_problem(*eps, x, y, 1.55, pml, sym)
    ref = tidy3d_matrix(*eps, x, y, 1.55, pml, sym)
    _check(p, ref, rng)


def test_straight_operator_equals_the_tidy3d_solver_matrix():
    """The reference of the bend tests (the matrix captured from tidy3d's compute_modes) equals the reference of
    the straight tests."""
    rng, eps, x, y = _random_case()
    ref = tidy3d_matrix(*eps, x, y, 1.55, (4, 3), (1, 0))
    got = tidy3d_solver_matrix(*eps, x, y, 1.55, (4, 3), (1, 0))
    assert spl.norm(got - ref) / spl.norm(ref) < 1e-15
    _check(make_problem(*eps, x, y, 1.55, (4, 3), (1, 0)), got, rng)


# Radii from 1.2 um (a strong transform: a_e from about 0.7 to 1.6 over the domain of about 0.8 x 0.6 um) to 50 um,
# both bend axes, a negative radius (the centre of the bend on the low side of the normal axis), PML and
# symmetries. plane_center: the middle of the coordinates (0.4, 0.3), or 0 (the domain on one side of the plane
# centre).
@pytest.mark.parametrize("radius,axis,center,pml,sym", [
    (5.0, 1, (0.4, 0.3), (0, 0), (0, 0)),
    (5.0, 0, (0.4, 0.3), (0, 0), (0, 0)),
    (1.2, 1, (0.4, 0.3), (4, 3), (0, 0)),
    (1.2, 0, (0.4, 0.3), (4, 3), (0, 0)),
    (-3.0, 1, (0.4, 0.3), (4, 3), (0, 1)),
    (50.0, 0, (0.0, 0.0), (3, 3), (1, 1)),
    (10.0, 1, (0.0, 0.0), (0, 5), (-1, 0)),
])
def test_bend_operator_equals_tidy3d(radius, axis, center, pml, sym):
    rng, eps, x, y = _random_case(seed=2)
    p = make_problem(*eps, x, y, 1.55, pml, sym, bend_radius=radius, bend_axis=axis, plane_center=center)
    ref = tidy3d_solver_matrix(*eps, x, y, 1.55, pml, sym, bend_radius=radius, bend_axis=axis, plane_center=center)
    straight = tidy3d_solver_matrix(*eps, x, y, 1.55, pml, sym)
    assert spl.norm(ref - straight) / spl.norm(ref) > 1e-4         # the bend changes the matrix
    _check(p, ref, rng)


def test_bend_needs_its_centre_outside_the_domain():
    _, eps, x, y = _random_case()
    with pytest.raises(ValueError, match="centre of the bend"):
        make_problem(*eps, x, y, 1.55, bend_radius=0.3, bend_axis=1, plane_center=(0.4, 0.3))
    with pytest.raises(ValueError, match="bend_axis"):
        make_problem(*eps, x, y, 1.55, bend_radius=5.0)
    with pytest.raises(ValueError, match="not 0"):
        make_problem(*eps, x, y, 1.55, bend_radius=0.0, bend_axis=1)
