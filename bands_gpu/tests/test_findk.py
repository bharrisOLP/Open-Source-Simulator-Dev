"""Checks of find_k and of the group velocity (Hellmann-Feynman):

- a homogeneous medium: k = n omega and v_g = 1/n exactly (2-D TE and TM, an oblique kdir in a triangular cell, the
  degenerate polarization pair of 3-D);
- a symmetric dielectric slab waveguide in a supercell, TE (E_z) and TM (H_z), even and odd modes: against the exact
  dispersion of the periodic slab array at Bloch k_y = 0 (the supercell itself; transfer matrix) and of the isolated
  slab (transcendental equation); the difference of the two exact values is the coupling error of the supercell;
- the group velocity against central differences of omega(k) (rods TE and TM, spheres in 3-D) and against the
  derivative of the exact slab dispersion;
- consistency: omega_b(find_k(omega)) = omega with solve_bands (the code path without parity);
- a line-defect photonic-crystal waveguide (TM, a periodic waveguide axis): find_k, solve_bands and an independent
  finite-difference solver (tests/fd_tm_reference.py) at the found k;
- parity: the projector, the 3-D path (a strip waveguide, mirror y and z), the controls (even grid, asymmetric
  structure, k off the mirror plane, a parity subspace smaller than the block), and a face on a subpixel sample;
- the defaults: kmag_max with korig, the root tolerance in single precision.
"""
import sys
from pathlib import Path

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax.numpy as jnp  # noqa: E402

from bands_gpu.findk import BandEvaluator, _mirror, find_k, group_velocity  # noqa: E402
from bands_gpu.geometry import TRIANGULAR, Block, Cell, Cylinder, Sphere  # noqa: E402
from bands_gpu.operator import make_operator  # noqa: E402
from bands_gpu.solver import solve_bands  # noqa: E402
from slab_reference import slab_isolated as _iso, supercell_exact as _sc  # noqa: E402

# Slab: a high-index core (n = 3.5, about silicon) in a low-index cladding (n = 1.45, about silica), thickness 0.5 a,
# in a supercell of period 5 a; frequency 0.5 (2 pi c / a): two TE and two TM guided modes.
N1, N2, D, L, OMEGA = 3.5, 1.45, 0.5, 5.0, 0.5


def slab_isolated(pol: str, even: bool) -> float:
    return _iso(pol, even, N1, N2, D, OMEGA)


def supercell_exact(pol: str, kguess: float) -> tuple[float, float]:
    return _sc(pol, kguess, N1, N2, D, OMEGA, L)


def slab_cell(n: int = 1025) -> Cell:
    return Cell(size=(1, L, 1), shape=(1, n, 1), eps_background=N2**2,
                objects=(Block(size=(np.inf, D, np.inf), eps=N1**2),))


def test_homogeneous_medium_gives_k_equal_n_omega():
    n, omega = 1.5, 0.2
    for pol in ("te", "tm"):
        cell = Cell(size=(1, 1, 1), shape=(8, 8, 1), eps_background=n**2)
        r = find_k(cell, omega, 1, 1, kdir=(1, 0.5, 0), polarization=pol, kmag_guess=0.1)
        assert r.converged == [True] and r.kmag[0] == pytest.approx(n * omega, abs=1e-10)
        assert r.group_velocity[0] == pytest.approx(1 / n, abs=1e-10)
    # oblique cell, kdir in reciprocal-lattice coordinates (towards M = b2 / 2): k = kmag b2 / |b2|
    tri = Cell(size=(1, 1, 1), shape=(12, 12, 1), basis=TRIANGULAR, eps_background=n**2)
    r = find_k(tri, omega, 1, 1, kdir=(0, 1, 0), polarization="te", kmag_guess=0.05)
    assert r.kmag[0] == pytest.approx(n * omega, abs=1e-10)
    assert np.allclose(tri.k_cartesian(r.kpoints[0]), [0, n * omega, 0], atol=1e-10)
    # 3-D: the two polarizations are degenerate (the derivative is diagonalized in the pair)
    cube = Cell(size=(1, 1, 1), shape=(6, 6, 6), eps_background=n**2)
    r = find_k(cube, omega, 1, 2, kdir=(1, 2, 0.5), kmag_guess=0.12)
    assert np.allclose(r.kmag, n * omega, atol=1e-10) and np.allclose(r.group_velocity, 1 / n, atol=1e-10)


@pytest.mark.parametrize("pol,solver_pol", [("E", "tm"), ("H", "te")])
def test_slab_waveguide_against_the_exact_dispersion(pol, solver_pol):
    """Even and odd guided modes (parity of MPB: E_z even = "even"; H_z even = "odd"), bands 1 and 2 without parity.
    Errors of kmag at 1025 points (205 per a): against the exact supercell (discretization) and against the
    isolated slab (with the coupling of the supercell); see the README."""
    cell = slab_cell()
    par = {"E": ("even", "odd"), "H": ("odd", "even")}[pol]           # (fundamental, first odd-order) of E_z / H_z
    exact_iso = [slab_isolated(pol, True), slab_isolated(pol, False)]
    found = []
    for parity, k_iso in zip(par, exact_iso):
        r = find_k(cell, OMEGA, 1, 1, kdir=(1, 0, 0), polarization=solver_pol, parity=parity, sub=64)
        assert r.converged == [True] and r.unconverged_solves == 0
        k_sc, vg_sc = supercell_exact(pol, k_iso)
        assert abs(r.kmag[0] - k_sc) / k_sc < 6e-5                         # discretization: max 4.4e-5
        assert abs(r.kmag[0] - k_iso) / k_iso < 8e-5                       # with the coupling of the supercell
        assert r.group_velocity[0] == pytest.approx(vg_sc, rel=2e-4)       # against the exact slope: max 1.0e-4
        found.append(r.kmag[0])
        # consistency: the band solver without parity gives omega at the found k
        f = solve_bands(cell, [r.kpoints[0]], 2, solver_pol, sub=64, tol=1e-9).freqs[0]
        assert np.min(np.abs(f - OMEGA)) < 1e-10
    full = find_k(cell, OMEGA, 1, 2, kdir=(1, 0, 0), polarization=solver_pol, sub=64)
    assert np.allclose(full.kmag, sorted(found, reverse=True), rtol=1e-9)


def test_supercell_coupling_error_of_the_slab_check():
    """The exact supercell (K = 0) against the isolated slab: the error that the period L = 5 adds (README)."""
    errs = {}
    for pol in ("E", "H"):
        for even in (True, False):
            k_iso = slab_isolated(pol, even)
            errs[pol, even] = abs(supercell_exact(pol, k_iso)[0] - k_iso) / k_iso
    assert max(errs[("E", True)], errs[("E", False)], errs[("H", True)]) < 1e-9
    assert 1e-6 < errs[("H", False)] < 5e-5        # TM1 (n_eff 1.59, near the cladding) decays slowly


def _richardson(cell, k, d, pol, nb, h=2e-3):
    """Central differences of omega(k) along the Cartesian d, extrapolated (h, h/2): error O(h^4)."""
    dk = cell.k_fractional(np.asarray(d) / np.linalg.norm(d))

    def cd(s):
        fp, _ = group_velocity(cell, k + s * dk, nb, d, pol, tol=1e-10)
        fm, _ = group_velocity(cell, k - s * dk, nb, d, pol, tol=1e-10)
        return (fp - fm) / (2 * s)

    return (4 * cd(h / 2) - cd(h)) / 3


def test_group_velocity_against_finite_differences():
    """Rods (eps 12, r = 0.2 a), TE and TM, a general k and a direction not along k (the full eps_inv tensor in TE);
    spheres (eps 12, r = 0.3 a, 3-D, 10^3)."""
    rods = Cell(size=(1, 1, 1), shape=(32, 32, 1), objects=(Cylinder(radius=0.2, eps=12.0),))
    k, d = np.array([0.3, 0.1, 0.0]), np.array([0.6, 0.8, 0.0])
    for pol in ("te", "tm"):
        _, vg = group_velocity(rods, k, 4, d, pol, tol=1e-10)
        assert np.abs(vg - _richardson(rods, k, d, pol, 4)).max() < 1e-9      # measured 2.6e-11 (TE), 4.1e-11 (TM)
    spheres = Cell(size=(1, 1, 1), shape=(10, 10, 10), objects=(Sphere(radius=0.3, eps=12.0),))
    k, d = np.array([0.2, 0.1, 0.05]), np.array([1.0, -0.5, 0.3])
    _, vg = group_velocity(spheres, k, 4, d, None, tol=1e-10)
    assert np.abs(vg - _richardson(spheres, k, d, None, 4)).max() < 1e-9       # measured 2.2e-11


def line_defect(n: int = 15, width: int = 7) -> Cell:
    """Square lattice of rods (eps 12, r = 0.2 a; the MPB tutorial crystal) with one row removed: a supercell of
    1 x width cells, waveguide axis x (periodic), n points per a (odd n * width for the y parity)."""
    rods = tuple(Cylinder(radius=0.2, eps=12.0, center=(0.0, float(j), 0.0))
                 for j in range(-(width // 2), width // 2 + 1) if j != 0)
    return Cell(size=(1, width, 1), shape=(n, n * width, 1), objects=rods)


def test_line_defect_waveguide():
    """The guided TM band in the gap (even E_z: band 4 of the even parity, band 7 of all bands) at omega = 0.34.
    Independent check: the finite-difference TM solver at the found k (n = 64 per a)."""
    from fd_tm_reference import fd_tm_rect

    cell, omega = line_defect(), 0.34
    r = find_k(cell, omega, 4, 4, kdir=(1, 0, 0), polarization="tm", parity="even", kmag_max=0.5, kmag_guess=0.2)
    assert r.converged == [True] and 0.1 < r.kmag[0] < 0.25
    k = r.kpoints[0]
    full = solve_bands(cell, [k], 8, "tm", tol=1e-9)
    assert abs(full.freqs[0, 6] - omega) < 1e-9                              # band 7 without parity
    # group velocity: Hellmann-Feynman against central differences of the even band 4
    ev = BandEvaluator(cell, 4, "tm", parity="even", tol=1e-10)
    fp, _, _ = ev.solve(k + [1e-3, 0, 0])
    fm, _, _ = ev.solve(k - [1e-3, 0, 0])
    assert r.group_velocity[0] == pytest.approx((fp[3] - fm[3]) / 2e-3, abs=1e-5)   # measured 9.0e-7 (O(h^2))
    rods = [(0.0, float(j), 0.2, 12.0) for j in (-3, -2, -1, 1, 2, 3)]
    fd = fd_tm_rect((64, 448), (1.0, 7.0), rods, (k[0], 0.0), num=8)
    assert abs(fd[6] - omega) < 5e-4          # measured 2.8e-4 (FD at 64 per a; at 128 per a also 2.8e-4)


def test_parity_projector_and_its_controls():
    """P = (1 + s M) / 2 is Hermitian and idempotent, the two parities are orthogonal; errors for an even grid, an
    asymmetric structure and a k off the mirror plane."""
    cell = line_defect(n=5)
    op = make_operator(cell, (0.3, 0, 0), "te", dtype=jnp.complex128)
    t, idx = _mirror(cell, np.asarray(op.q), np.asarray(op.u), 1)
    rng = np.random.default_rng(1)
    shape = (1,) + tuple(cell.shape)
    a, b = (rng.normal(size=shape) + 1j * rng.normal(size=shape) for _ in range(2))
    mfun = lambda c: np.einsum("mnxyz,nxyz->mxyz", t, np.take(c, idx, axis=2))  # noqa: E731
    assert np.allclose(mfun(mfun(a)), a, atol=1e-13)                           # M^2 = 1
    assert np.vdot(b, mfun(a)) == pytest.approx(np.vdot(mfun(b), a), abs=1e-12)  # M Hermitian (unitary, M^2 = 1)
    even, odd = 0.5 * (a + mfun(a)), 0.5 * (b - mfun(b))
    assert abs(np.vdot(even, odd)) < 1e-12
    with pytest.raises(ValueError, match="odd number"):
        BandEvaluator(line_defect(n=16), 2, "tm", parity="even")
    shifted = Cell(size=(1, 3, 1), shape=(1, 31, 1), objects=(Block(size=(np.inf, 0.5, np.inf), eps=4.0,
                                                                     center=(0, 0.3, 0)),))
    with pytest.raises(ValueError, match="not symmetric"):
        BandEvaluator(shifted, 2, "tm", parity="even")
    with pytest.raises(ValueError, match="mirror plane"):
        BandEvaluator(slab_cell(31), 2, "tm", parity="even").solve((0.3, 0.1, 0))


def test_parity_subspace_too_small_is_an_error():
    """The parity subspace must hold the eigensolver block (num_bands + extra): "tm" in a 2-D cell is odd in z (its
    even z subspace is empty), a 7-point 1-D cell has even and odd subspaces of 4 and 3. Without the check, the
    solve gave frequencies 0 as converged. "te" is even in z: the even z subspace is all of it."""
    cell = slab_cell(255)
    with pytest.raises(ValueError, match="0 dimensions"):
        BandEvaluator(cell, 2, "tm", parity="even", mirror_axis=2).solve((1.2, 0, 0))
    small = slab_cell(7)
    for parity, dim in (("even", 4), ("odd", 3)):
        with pytest.raises(ValueError, match=f"{dim} dimensions"):
            BandEvaluator(small, 2, "tm", parity=parity).solve((1.2, 0, 0))
    f_even_z, v_even_z, _ = BandEvaluator(cell, 3, "te", parity="even", mirror_axis=2).solve((1.2, 0, 0), (1, 0, 0))
    f_all, v_all, _ = BandEvaluator(cell, 3, "te").solve((1.2, 0, 0), (1, 0, 0))
    assert np.allclose(f_even_z, f_all, rtol=1e-12) and np.allclose(v_even_z, v_all, rtol=1e-8)


def test_parity_with_a_face_on_a_subpixel_sample():
    """A symmetric slab with its faces exactly on subpixel samples (thickness 0.325 to 0.675 in a 5 a cell, 25
    points, sub 8): the samples of the grid are mirror images only to rounding, so eps_inv of the two face voxels
    differed and the old check gave a false "not symmetric". Now: the structure is checked at exact mirror points,
    and eps_inv is made symmetric (only in the face voxels). With that eps_inv, the even and odd bands together are
    the bands without parity."""
    for d in [round(0.325 + 0.05 * i, 6) for i in range(8)]:      # faces at y = +/- d / 2, on samples
        cell = Cell(size=(1, L, 1), shape=(1, 25, 1), eps_background=N2**2,
                    objects=(Block(size=(np.inf, d, np.inf), eps=N1**2),))
        plain = BandEvaluator(cell, 4, "tm")
        even = BandEvaluator(cell, 4, "tm", parity="even", extra=2)
        odd = BandEvaluator(cell, 4, "tm", parity="odd", extra=2)
        assert even.mirror_asymmetry > 1e-3                        # the face-on-sample case occurs at each d
        changed = np.any(np.abs(even.eps_inv - plain.eps_inv) > 1e-15, axis=(0, 1))
        assert changed.sum() == 2                                 # only the voxels at the two faces
        plain.eps_inv = even.eps_inv                               # the same (symmetric) operator without parity
        f_all, _, _ = plain.solve((1.0, 0, 0))
        f_par = np.sort(np.concatenate([even.solve((1.0, 0, 0))[0], odd.solve((1.0, 0, 0))[0]]))[:4]
        assert np.abs(f_par - f_all).max() < 1e-12                   # measured 8.9e-16
    # the check is at the resolution of the samples (here 0.025 a): a shift of 0.01 a moves no sample across a face
    # (the sampled structure is symmetric), a shift of 0.02 a moves one sample layer
    for shift, ok in ((0.01, True), (0.02, False)):
        shifted = Cell(size=(1, L, 1), shape=(1, 25, 1), eps_background=N2**2,
                       objects=(Block(size=(np.inf, 0.5, np.inf), eps=N1**2, center=(0, shift, 0)),))
        if ok:
            assert BandEvaluator(shifted, 2, "tm", parity="even").mirror_asymmetry < 1e-14
        else:
            with pytest.raises(ValueError, match="not symmetric"):
                BandEvaluator(shifted, 2, "tm", parity="even")


def test_strip_waveguide_parity_in_3d():
    """The 3-D parity path (two coefficients per plane wave, T a 2 x 2 matrix): a strip (n 3.48, 0.5 x 0.22) in a
    cladding (n 1.444), cell 1 x 3 x 2, grid 1 x 25 x 15, k = 1.5 along x. Mirror y and mirror z: M^2 = 1, M
    Hermitian; the even and odd bands together are the bands without parity; v_g against central differences; band
    1 (the TE-like mode, E mostly along y) is odd in y and even in z."""
    strip = Cell(size=(1, 3, 2), shape=(1, 25, 15), eps_background=1.444**2,
                 objects=(Block(size=(np.inf, 0.5, 0.22), eps=3.48**2),))
    k, nb = np.array([1.5, 0.0, 0.0]), 6
    full = solve_bands(strip, [k], nb, None, tol=1e-10).freqs[0]
    op = make_operator(strip, k, None, dtype=jnp.complex128)
    rng = np.random.default_rng(2)
    shape = (2,) + tuple(strip.shape)
    a, b = (rng.normal(size=shape) + 1j * rng.normal(size=shape) for _ in range(2))
    band1 = {}
    for axis in (1, 2):
        t, idx = _mirror(strip, np.asarray(op.q), np.asarray(op.u), axis)
        mfun = lambda c: np.einsum("mnxyz,nxyz->mxyz", t, np.take(c, idx, axis=1 + axis))  # noqa: E731,B023
        assert np.abs(mfun(mfun(a)) - a).max() < 1e-13
        assert abs(np.vdot(b, mfun(a)) - np.vdot(mfun(b), a)) < 1e-11
        res = {}
        for parity in ("even", "odd"):
            ev = BandEvaluator(strip, nb, None, parity=parity, mirror_axis=axis, tol=1e-10)
            f, v, ok = ev.solve(k, (1, 0, 0))
            assert ok
            fp, _, _ = ev.solve(k + [1e-4, 0, 0])
            fm, _, _ = ev.solve(k - [1e-4, 0, 0])
            assert np.abs(v - (fp - fm) / 2e-4).max() < 1e-7          # measured 1.1e-8 (O(h^2))
            res[parity] = f
        merged = np.sort(np.concatenate([res["even"], res["odd"]]))[:nb]
        assert np.abs(merged - full).max() < 1e-12                    # measured 1.4e-15 (y), 1.6e-15 (z)
        band1[axis] = "even" if res["even"][0] < res["odd"][0] else "odd"
    assert band1 == {1: "odd", 2: "even"}


def test_korig_and_single_precision_defaults():
    """The default kmag_max includes |korig| (homogeneous n 1.5, omega 0.2, korig (-0.1, 0, 0): the root kmag 0.4
    is above 1.01 omega n = 0.303); the default root tolerance follows the precision (single: 1e-5)."""
    hom = Cell(size=(1, 1, 1), shape=(8, 8, 1), eps_background=2.25)
    r = find_k(hom, 0.2, 1, 1, kdir=(1, 0, 0), korig=(-0.1, 0, 0), polarization="te", kmag_guess=0.2)
    assert r.converged == [True] and r.kmag[0] == pytest.approx(0.4, abs=1e-10)
    cell = slab_cell(255)
    ref = find_k(cell, OMEGA, 1, 1, kdir=(1, 0, 0), polarization="tm", parity="even")
    single = find_k(cell, OMEGA, 1, 1, kdir=(1, 0, 0), polarization="tm", parity="even", dtype=jnp.complex64)
    assert single.converged == [True] and single.solves <= 4          # measured 3
    assert single.kmag[0] == pytest.approx(ref.kmag[0], rel=1e-6)          # measured 1.0e-7


def test_no_root_and_warm_start():
    """A band that stays above omega gives NaN; the warm start saves eigensolver iterations (README)."""
    cell = slab_cell(255)
    r = find_k(cell, 0.05, 2, 2, kdir=(1, 0, 0), polarization="tm", kmag_max=0.3)
    assert np.isnan(r.kmag[0]) and r.converged == [False]
    warm = find_k(cell, OMEGA, 1, 2, kdir=(1, 0, 0), polarization="tm")
    cold = find_k(cell, OMEGA, 1, 2, kdir=(1, 0, 0), polarization="tm", warm_start=False)
    assert np.allclose(warm.kmag, cold.kmag, rtol=1e-9)
    assert warm.iterations < cold.iterations
