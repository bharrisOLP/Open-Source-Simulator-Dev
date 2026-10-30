"""Checks of the band solver: exact homogeneous bands, the exact 1-D Bragg stack, and the published MPB square
lattice of rods (MPB Python tutorial: eps 12, r = 0.2 a, TM gap 0.28262 to 0.41933)."""
import sys
from pathlib import Path

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bands_gpu.geometry import Block, Cell, Cylinder  # noqa: E402
from bands_gpu.solver import interpolate, solve_bands  # noqa: E402


def test_homogeneous_bands_are_free_photon_lines():
    eps = 2.25
    cell = Cell(size=(1, 1, 1), shape=(8, 8, 8), eps_background=eps)
    k = np.array([0.1, 0.2, 0.05])
    res = solve_bands(cell, [k], num_bands=6)
    g = np.array([[i, j, l] for i in range(-2, 3) for j in range(-2, 3) for l in range(-2, 3)])
    # each plane wave k + G gives two modes (two polarizations) at |k + G| / n
    exact = np.sort(np.repeat(np.linalg.norm(k + g, axis=1), 2))[:6] / np.sqrt(eps)
    assert np.allclose(res.freqs[0], exact, atol=1e-6)


def bragg_exact(k, n1, d1, n2, d2, num):
    """Frequencies (units c/a, a = d1 + d2) of a 1-D stack at Bloch k (units 2 pi / a), normal incidence:
    cos(2 pi k) = cos(w n1 d1) cos(w n2 d2) - (n1/n2 + n2/n1)/2 sin(w n1 d1) sin(w n2 d2), w = 2 pi f."""
    f = np.linspace(1e-6, 2.5, 200001)
    w = 2 * np.pi * f
    rhs = np.cos(w * n1 * d1) * np.cos(w * n2 * d2) - 0.5 * (n1 / n2 + n2 / n1) * np.sin(w * n1 * d1) * np.sin(w * n2 * d2)
    g = rhs - np.cos(2 * np.pi * k)
    roots = f[:-1][np.sign(g[:-1]) != np.sign(g[1:])]
    return roots[:num]


def test_bragg_stack_matches_the_exact_dispersion():
    n1, n2, d1 = 3.0, 1.0, 0.25
    cell = Cell(size=(1, 1, 1), shape=(256, 1, 1), eps_background=n2**2,
                objects=(Block(size=(d1, np.inf, np.inf), eps=n1**2),))
    k = 0.3
    res = solve_bands(cell, [[k, 0, 0]], num_bands=6, polarization="te")
    exact = bragg_exact(k, n1, d1, n2, 1 - d1, 3)
    # 1-D, normal incidence: TE and TM coincide, so each exact frequency appears once per polarization
    assert np.allclose(res.freqs[0][:3], exact, rtol=2e-3)


@pytest.mark.slow
def test_square_lattice_rods_tm_against_an_independent_solver():
    """Square lattice of rods (eps 12, r = 0.2 a), TM: band 1 at M and band 2 at X. Reference: the converged
    values of an independent finite-difference solver (tests/fd_tm_reference.py; n = 128..512: 0.28066, 0.41716).
    The MPB tutorial gives 0.28262 and 0.41933 at resolution 32; that is about 0.7 % above the converged values."""
    cell = Cell(size=(1, 1, 1), shape=(64, 64, 1), objects=(Cylinder(radius=0.2, eps=12.0),))
    res = solve_bands(cell, [[0.5, 0.5, 0], [0.5, 0, 0]], num_bands=2, polarization="tm", extra=3)
    assert all(res.converged)
    assert res.freqs[0, 0] == pytest.approx(0.28066, abs=2e-4)
    assert res.freqs[1, 1] == pytest.approx(0.41716, abs=2e-4)
    assert res.freqs[0, 0] == pytest.approx(0.282623, rel=0.01)      # MPB tutorial, resolution 32
    assert res.freqs[1, 1] == pytest.approx(0.419335, rel=0.01)


def test_band_path_converges_with_warm_start():
    cell = Cell(size=(1, 1, 1), shape=(32, 32, 1), objects=(Cylinder(radius=0.2, eps=12.0),))
    ks = interpolate([[0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0], [0, 0, 0]], 2)
    warm = solve_bands(cell, ks, num_bands=4, polarization="tm")
    cold = solve_bands(cell, ks, num_bands=4, polarization="tm", warm_start=False)
    assert all(warm.converged) and all(cold.converged)
    assert np.allclose(warm.freqs, cold.freqs, atol=1e-6)
    assert sum(warm.iterations) < sum(cold.iterations)      # measured: 284 against 341 (k steps of 1/3 segment)


def test_averaging_tensor_at_a_planar_interface():
    """A voxel half filled by a slab with its normal along x: eps_inv has the harmonic mean <1/eps> along x (E
    normal to the interface) and 1/<eps> along y and z (E parallel)."""
    cell = Cell(size=(1, 1, 1), shape=(16, 1, 1), eps_background=1.0,
                objects=(Block(size=(0.5, np.inf, np.inf), eps=12.0),))     # edges at voxel centers (half-filled voxels)
    t = cell.eps_inverse(sub=16)
    i = int(np.argmax(np.abs(t[0, 0, :, 0, 0] - t[1, 1, :, 0, 0])))                    # an interface voxel
    assert t[0, 0, i, 0, 0] == pytest.approx(0.5 * (1 + 1 / 12), rel=1e-12)
    assert t[1, 1, i, 0, 0] == pytest.approx(1 / 6.5, rel=1e-12)
    assert t[2, 2, i, 0, 0] == pytest.approx(1 / 6.5, rel=1e-12)
    assert abs(t[0, 1, i, 0, 0]) < 1e-12


def test_independent_fd_reference_converges():
    """The independent reference itself (tests/fd_tm_reference.py) at n = 128: 0.28066 and 0.41716 within 2e-4."""
    from fd_tm_reference import fd_tm_bands
    assert fd_tm_bands(128, (0.5, 0.5))[0] == pytest.approx(0.28066, abs=2e-4)
    assert fd_tm_bands(128, (0.5, 0.0))[1] == pytest.approx(0.41716, abs=2e-4)
