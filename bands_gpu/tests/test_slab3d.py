"""3-D checks with slabs (the full vector operator, two coefficients per plane wave, the z mirror parity, the eps
sampling of the JAX path):

- a uniform dielectric slab (n 3.5, thickness 0.5 a, in n 1.45; the slab of test_findk with its normal along z) in a
  3-D cell with a small triangular in-plane lattice: the guided modes TE0, TE1, TM0, TM1 at omega 0.5 against the
  exact dispersion of the periodic slab array (transfer matrix, tests/slab_reference.py) and against the 2-D cell of
  the same grid along the normal (the 3-D operator has the 1-D problem as its in-plane G = 0 block);
- the photonic-crystal slab of the MPB example hole-slab (after S. G. Johnson et al., "Guided modes in photonic
  crystal slabs", Phys. Rev. B 60, 5751 (1999)): a triangular lattice of air holes (r 0.3 a) in a slab (eps 12,
  thickness 0.5 a), supercell height 4 a, the even (TE-like) bands at Gamma, M and K, in the triangular primitive cell
  against the same crystal in the rectangular supercell (1 x sqrt(3) x 4, two holes; folded k points).
"""
import sys
from pathlib import Path

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bands_gpu import sampling  # noqa: E402
from bands_gpu.findk import BandEvaluator, find_k  # noqa: E402
from bands_gpu.geometry import TRIANGULAR, Block, Cell, Cylinder  # noqa: E402
from bands_gpu.solver import TRIANGULAR_POINTS  # noqa: E402
from slab_reference import slab_isolated, supercell_exact  # noqa: E402

N1, N2, D, L, OMEGA = 3.5, 1.45, 0.5, 5.0, 0.5
S3 = np.sqrt(3.0)


def test_uniform_slab_in_3d_against_the_exact_dispersion_and_the_2d_cell():
    """z-even (MPB): TE0 (band 1) and TM1 (band 2); z-odd: TM0 (band 1) and TE1 (band 2). 1025 points along the
    normal (the face at 1/4 of a voxel, so sub 8 gives the exact fill fraction), in-plane lattice 0.25 a (triangular,
    4 x 4 points: the folded modes are above omega). kmag against the exact supercell: the discretization error of
    the 2-D slab check (README: 1.2e-5 to 4.4e-5); against the 2-D cell: the root tolerance (1e-8)."""
    slab = (Block(size=(np.inf, np.inf, D), eps=N1**2),)
    cell = Cell(size=(0.25, 0.25, L), shape=(4, 4, 1025), basis=TRIANGULAR, eps_background=N2**2, objects=slab)
    assert sampling.use_jax(cell, 8, "auto")                        # the eps sampling of this cell runs in JAX
    flat = Cell(size=(1, L, 1), shape=(1, 1025, 1), eps_background=N2**2,
                objects=(Block(size=(np.inf, D, np.inf), eps=N1**2),))
    modes = {"even": (("E", True, "tm", "even"), ("H", False, "te", "even")),      # TE0, TM1
             "odd": (("H", True, "te", "odd"), ("E", False, "tm", "odd"))}         # TM0, TE1
    err_exact, err_2d, err_vg = [], [], []
    for parity, pair in modes.items():
        r = find_k(cell, OMEGA, 1, 2, kdir=(1, 0, 0), parity=parity, mirror_axis=2)
        assert r.converged == [True, True] and r.unconverged_solves == 0
        for b, (pol, even, pol2d, parity2d) in enumerate(pair):
            k_sc, vg_sc = supercell_exact(pol, slab_isolated(pol, even, N1, N2, D, OMEGA), N1, N2, D, OMEGA, L)
            r2 = find_k(flat, OMEGA, 1, 1, kdir=(1, 0, 0), polarization=pol2d, parity=parity2d)
            err_exact.append(abs(r.kmag[b] - k_sc) / k_sc)
            err_vg.append(abs(r.group_velocity[b] - vg_sc) / vg_sc)
            err_2d.append(abs(r.kmag[b] - r2.kmag[0]) / r2.kmag[0])
    print("kmag against the exact supercell", err_exact, "v_g", err_vg, "against 2-D", err_2d)
    assert max(err_exact) < 6e-5                                      # the 2-D check: max 4.4e-5
    assert max(err_vg) < 2e-4
    assert max(err_2d) < 1e-7


def hole_slab(cell_kind: str, n: int, nz: int) -> Cell:
    """The MPB hole-slab crystal: air holes r 0.3 a in an eps 12 slab of thickness 0.5 a, supercell height 4 a."""
    slab = Block(size=(np.inf, np.inf, 0.5), eps=12.0)
    if cell_kind == "triangular":
        return Cell(size=(1, 1, 4), shape=(n, n, nz), basis=TRIANGULAR,
                    objects=(slab, Cylinder(radius=0.3, height=0.5, eps=1.0)))
    return Cell(size=(1, S3, 4), shape=(n, round(S3 * n), nz),
                objects=(slab, Cylinder(radius=0.3, height=0.5, eps=1.0),
                         Cylinder(radius=0.3, height=0.5, eps=1.0, center=(0.5, S3 / 2, 0))))


def even_bands(cell: Cell, kpoints, num: int) -> np.ndarray:
    ev = BandEvaluator(cell, num, None, parity="even", mirror_axis=2, tol=1e-8)
    out = []
    for k in kpoints:
        f, _, ok = ev.solve(k)
        assert ok
        out.append(f)
    return np.array(out)


@pytest.mark.slow
def test_hole_slab_even_bands_triangular_against_the_supercell():
    """The z-even (TE-like) bands of the hole slab at Gamma, M, K: the triangular cell (16 x 16 x 65) against the
    rectangular supercell (16 x 28 x 65; its bands at k are the triangular bands at k and k + (0, 1/2)). The gap of
    the even guided modes: band 1 at K to band 2 at M. In the supercell, M folds onto Gamma, where the even bands are
    0 and a degenerate pair of radiation modes of the supercell (0.177); so the gap edges there are band 2 at K and
    band 5 at M. Measured: folded bands against the supercell 2.8e-4 (6 bands); gap 0.264781 to 0.344200 (triangular),
    0.264824 to 0.344142 (supercell)."""
    gmk = [TRIANGULAR_POINTS[p] for p in ("Gamma", "M", "K")]
    tri = hole_slab("triangular", 16, 65)
    rect = hole_slab("rectangular", 16, 65)
    num = 6
    ft = even_bands(tri, gmk + [np.add(k, (0, 0.5, 0)) for k in gmk], num)
    fr = even_bands(rect, [rect.k_fractional(tri.k_cartesian(k)) for k in gmk], num)
    folded = np.sort(np.concatenate([ft[:3], ft[3:]], axis=1), axis=1)[:, :num]
    gap_tri = np.array([ft[2, 0], ft[1, 1]])
    gap_rect = np.array([fr[2, 1], fr[1, 4]])
    print("hole slab: folded against the supercell", np.abs(folded - fr).max(), "gap", gap_tri, gap_rect)
    assert np.abs(folded - fr).max() < 2e-3
    assert np.abs(gap_tri - gap_rect).max() < 2e-3
    assert gap_tri[1] > gap_tri[0]                                     # a gap of the even guided modes
    # guided: below the light line of air at M and K (|k| = 1/sqrt(3) and 2/3 in units of 2 pi / a)
    assert gap_tri[0] < 2 / 3 and gap_tri[1] < 1 / S3


def sampled_thickness(height: float, nz: int, sub: int = 8) -> float:
    """The slab thickness that the subpixel samples see: (samples inside the slab) / sub x dz, in a column of a cell of
    the given height (the slab of the hole slab, without the hole)."""
    cell = Cell(size=(1, 1, height), shape=(1, 1, nz), objects=(Block(size=(np.inf, np.inf, 0.5), eps=12.0),))
    return float(np.count_nonzero(cell.eps_samples(sub) == 12.0)) / sub * height / nz


def test_hole_slab_faces_on_a_subpixel_sample():
    """The grid of the test above and of bench/bench_hole_slab.py (height 4 a, 4 n + 1 points, sub 8) puts each slab
    face exactly on a subpixel sample (z = 0.25 is 7/16 of a voxel below a voxel center), so the samples see
    a face voxel as 0 or 2/16 full in place of 1/16: the slab is dz / 8 too thick or too thin. With the height
    (4 n + 1) / n (dz = 1 / n; `hole_slab_fdtd.bands_cell`) the faces are on voxel faces (n a multiple of 4) and the
    samples give the exact thickness. The bands of the two grids at n = 16 differ by more than 4e-4; a thicker slab
    gives lower frequencies. Measured (test print): band 1 at K 0.264781 (height 4) against 0.265368, band 2 at M
    0.344200 against 0.344869."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import hole_slab_fdtd as hs

    n = 16
    dz = 4.0 / (4 * n + 1)
    t_old, t_new = sampled_thickness(4.0, 4 * n + 1), sampled_thickness((4 * n + 1) / n, 4 * n + 1)
    print("sampled thickness: height 4", t_old, "height (4 n + 1) / n", t_new, "dz / 8", dz / 8)
    assert abs(abs(t_old - 0.5) - dz / 8) < 1e-12
    assert abs(t_new - 0.5) < 1e-12
    old = hs.even_bands(hs.bands_cell(n, None, 4.0), ("K", "M"), num=4)
    new = hs.even_bands(hs.bands_cell(n, None), ("K", "M"), num=4)
    d = np.array([old["K"][0][0] - new["K"][0][0], old["M"][0][1] - new["M"][0][1]])
    print("band 1 at K, band 2 at M: height 4", old["K"][0][0], old["M"][0][1], "exact faces", new["K"][0][0],
          new["M"][0][1], "difference", d)
    assert np.all(np.abs(d) > 4e-4) and np.all(np.sign(d) == -np.sign(t_old - 0.5))
