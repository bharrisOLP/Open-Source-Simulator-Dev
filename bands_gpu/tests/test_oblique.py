"""Checks of oblique lattices: the lattice algebra, eps sampling with periodic images across oblique faces, the
interface normal in sheared and non-cubic voxels (against the normals of the fit and of the code before oblique
cells), exact empty-lattice bands (2-D triangular, 3-D hexagonal and fcc), a rotated and a sheared description of
the square lattice (with the order of convergence), and triangular lattices of rods (TM) and air holes (TE) against
the same structures in a rectangular supercell (and, for TM, an independent finite-difference solver)."""
import dataclasses
import itertools
import sys
from pathlib import Path

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bands_gpu.geometry import TRIANGULAR, Block, Cell, Cylinder, Sphere  # noqa: E402
from bands_gpu.operator import make_operator  # noqa: E402
from bands_gpu.solver import TRIANGULAR_POINTS, solve_bands, triangular_path  # noqa: E402

S3 = np.sqrt(3.0)
FCC = ((0, 1, 1), (1, 0, 1), (1, 1, 0))           # primitive vectors of the fcc lattice (cube edge 1: size 1/sqrt(2))
GMK = [TRIANGULAR_POINTS[p] for p in ("Gamma", "M", "K")]


def free_photon(cell: Cell, k_frac, num: int, per_g: int) -> np.ndarray:
    """The exact bands of a homogeneous cell: |k + G| / sqrt(eps), G = sum_i n_i b_i (per_g modes per G)."""
    kc = cell.k_cartesian(k_frac)
    rng = [range(-4, 5) if n > 1 else (0,) for n in cell.shape]
    g = np.array([np.array(n, float) @ cell.reciprocal for n in itertools.product(*rng)])
    return np.sort(np.repeat(np.linalg.norm(kc + g, axis=1), per_g))[:num] / np.sqrt(cell.eps_background)


def first_moment_eps_inverse(cell: Cell, sub: int = 8) -> np.ndarray:
    """eps_inv with the normal of the code before oblique cells: the direction of the first moment sum_i m_i d_i
    (d_i = a_i / n_i, the voxel edges). For comparisons only."""
    s = [sub if n > 1 else 1 for n in cell.shape]
    nx, ny, nz = cell.shape
    e = cell.eps_samples(sub).reshape(nx, s[0], ny, s[1], nz, s[2])
    mean_eps, mean_inv = e.mean(axis=(1, 3, 5)), (1 / e).mean(axis=(1, 3, 5))
    offs = [(np.arange(k) + 0.5) / k - 0.5 for k in s]
    shapes = [(1, -1, 1, 1, 1, 1), (1, 1, 1, -1, 1, 1), (1, 1, 1, 1, 1, -1)]
    m = [(e * o.reshape(sh)).mean(axis=(1, 3, 5)) for o, sh in zip(offs, shapes)]
    d = cell.lattice / np.array(cell.shape, float)[:, None]
    v = sum(m[i][None] * d[i][:, None, None, None] for i in range(3))
    norm = np.sqrt((v**2).sum(axis=0))
    iface = (mean_inv * mean_eps > 1 + 1e-12) & (norm > 0)
    n = np.where(iface[None], v / np.where(norm > 0, norm, 1.0)[None], 0.0)
    nn = n[:, None] * n[None, :]
    return mean_inv * nn + (1 / mean_eps) * (np.eye(3)[:, :, None, None, None] - nn)


class FirstMomentCell(Cell):
    """A cell with the normal of the code before oblique cells (for comparisons)."""

    def eps_inverse(self, sub: int = 8, normal: str = "object") -> np.ndarray:
        return first_moment_eps_inverse(self, sub)


class FitCell(Cell):
    """A cell with the fitted gradient as the normal in all voxels (for comparisons)."""

    def eps_inverse(self, sub: int = 8, normal: str = "object") -> np.ndarray:
        return Cell.eps_inverse(self, sub, normal="fit")


def angle_to_radial(cell: Cell, t: np.ndarray, center) -> np.ndarray:
    """Angles (degrees) between the normal of eps_inv (the eigenvector of the larger in-plane eigenvalue, <1/eps>) and
    the radial direction from the nearest image of a rod center, at the voxel centers of all interface voxels of a
    2-D rectangular cell."""
    X, Y, _ = cell.points(1)
    p = np.stack(np.broadcast_arrays(X[:, :, 0], Y[:, :, 0]), axis=-1)
    size = np.array(cell.size[:2], float)
    d = (p - np.asarray(center, float) + 0.5 * size) % size - 0.5 * size
    m2 = np.moveaxis(t[:2, :2, :, :, 0], (0, 1), (-2, -1))
    iface = np.abs(m2[..., 0, 1]) + np.abs(m2[..., 0, 0] - m2[..., 1, 1]) > 1e-12
    _, v = np.linalg.eigh(m2[iface])
    n, r = v[:, :, 1], d[iface]
    # arctan2 of |cross| and |dot| (arccos near 1 loses half of the digits)
    return np.degrees(np.arctan2(np.abs(n[:, 0] * r[:, 1] - n[:, 1] * r[:, 0]), np.abs(np.sum(n * r, axis=-1))))


def test_lattice_reciprocal_vectors_and_k_points():
    tri = Cell(size=(1, 1, 1), shape=(8, 8, 1), basis=TRIANGULAR)
    assert np.allclose(tri.lattice @ tri.reciprocal.T, np.eye(3), atol=1e-15)
    assert np.allclose(tri.reciprocal[:2, :2], [[1, -1 / S3], [0, 2 / S3]], atol=1e-15)
    # M: center of a zone edge (|M| = 1/sqrt(3)); K: a zone corner (|K| = 2/3), on the edge through M (K . M^ = |M|)
    m, k = tri.k_cartesian(TRIANGULAR_POINTS["M"]), tri.k_cartesian(TRIANGULAR_POINTS["K"])
    assert np.linalg.norm(m) == pytest.approx(1 / S3, rel=1e-15)
    assert np.linalg.norm(k) == pytest.approx(2 / 3, rel=1e-15)
    assert k @ m / np.linalg.norm(m) == pytest.approx(1 / S3, rel=1e-15)
    assert np.allclose(tri.k_fractional(tri.k_cartesian([0.3, -0.2, 0.1])), [0.3, -0.2, 0.1], atol=1e-15)
    path = triangular_path(3)
    assert len(path) == 13
    assert np.allclose(path[4], TRIANGULAR_POINTS["M"]) and np.allclose(path[8], TRIANGULAR_POINTS["K"])
    rect = Cell(size=(1, 1.5, 2), shape=(8, 8, 8))
    assert np.array_equal(rect.reciprocal, np.diag([1, 1 / 1.5, 0.5]))
    # an unused axis must be orthogonal to the used ones; te/tm need an in-plane lattice and k_z = 0
    with pytest.raises(ValueError):
        Cell(size=(1, 1, 1), shape=(8, 8, 1), basis=((1, 0, 0), (0, 1, 0), (0, 0.5, 1)))
    with pytest.raises(ValueError):
        Cell(size=(1, 1, 1), shape=(8, 8, 1), basis=((1, 0, 0), (2, 0, 0), (0, 0, 1)))
    with pytest.raises(ValueError):
        make_operator(Cell(size=(1, 1, 1), shape=(8, 8, 1), basis=TRIANGULAR), [0.1, 0.2, 0.3], "tm")
    # an object that is infinite along a direction that is not a lattice vector: its images would not be complete
    rod = (Cylinder(radius=0.1, eps=2.0),)
    slab = (Block(size=(np.inf, 0.3, np.inf), eps=2.0),)
    with pytest.raises(ValueError):
        Cell(size=(np.sqrt(0.5),) * 3, shape=(8, 8, 8), basis=FCC, objects=rod).eps_samples(2)
    c, s = np.cos(0.5), np.sin(0.5)
    with pytest.raises(ValueError):
        Cell(size=(1, 1, 1), shape=(8, 8, 1), basis=((c, s, 0), (-s, c, 0), (0, 0, 1)), objects=slab).eps_samples(2)
    # rods along a3 = z, and a slab along a1 = x, do not change along that lattice vector: no error
    Cell(size=(1, 1, 1.3), shape=(8, 8, 8), basis=TRIANGULAR, objects=rod).eps_samples(2)
    Cell(size=(1, 1, 1), shape=(8, 8, 1), basis=TRIANGULAR, objects=slab).eps_samples(2)


def test_samples_have_periodic_images_across_oblique_faces():
    """A rod moved from the cell center to a corner or an edge center of the triangular cell (a half-cell shift)
    gives the same samples, rolled by half the grid; the fill fraction is pi r^2 / (sqrt(3) / 2). In the fcc cell
    (strongly sheared), a sphere has its volume fraction at the center, a corner and a face; a block and a short
    cylinder give the same samples when they move by a lattice vector."""
    n, sub = 32, 8
    kw = {"size": (1, 1, 1), "shape": (n, n, 1), "basis": TRIANGULAR}
    center = Cell(**kw, objects=(Cylinder(radius=0.3, eps=2.0),)).eps_samples(sub)
    lat = Cell(**kw).lattice
    for shift, roll in ((0.5 * lat[0] + 0.5 * lat[1], (n * sub // 2, n * sub // 2)), (0.5 * lat[0], (n * sub // 2, 0))):
        moved = Cell(**kw, objects=(Cylinder(radius=0.3, eps=2.0, center=tuple(shift)),)).eps_samples(sub)
        assert np.mean(moved != np.roll(center, roll, axis=(0, 1))) == 0.0
    assert np.mean(center == 2.0) == pytest.approx(np.pi * 0.09 / (S3 / 2), rel=1e-3)

    kw = {"size": (np.sqrt(0.5),) * 3, "shape": (16, 16, 16), "basis": FCC}
    lat = Cell(**kw).lattice
    for c in ((0, 0, 0), tuple(0.5 * lat.sum(axis=0)), tuple(0.5 * lat[0])):
        e = Cell(**kw, objects=(Sphere(radius=0.3, eps=2.0, center=c),)).eps_samples(4)
        assert np.mean(e == 2.0) == pytest.approx(4 / 3 * np.pi * 0.3**3 / 0.25, rel=2e-3)
    # face positions off the planes of samples (a face on such a plane makes the result depend on rounding)
    for obj in (Block(size=(0.3017, 0.4013, 0.4991), eps=2.0), Cylinder(radius=0.2, height=0.4937, eps=2.0)):
        vol = np.prod(obj.size) if isinstance(obj, Block) else np.pi * obj.radius**2 * obj.height
        e0 = Cell(**kw, objects=(obj,)).eps_samples(4)
        e1 = Cell(**kw, objects=(dataclasses.replace(obj, center=tuple(lat[0] - lat[2])),)).eps_samples(4)
        assert np.mean(e0 != e1) == 0.0
        assert np.mean(e0 == 2.0) == pytest.approx(vol / 0.25, rel=0.02)   # flat faces: staircase error ~1/N (1.1 %)


def test_interface_normal_in_sheared_and_non_cubic_voxels():
    """Where one object boundary (one periodic image) crosses a voxel, the normal is the analytic normal of the
    object at the voxel center; where several cross, it is the fitted gradient 12 sum_i m_i n_i b_i.
    (1) A slab with its faces along x in the triangular cell: the normal is exactly y in the sheared voxels, also
    with the fit (a planar interface; the first-moment direction sum_i m_i d_i is a2, 30 degrees off).
    (2) A rod across the faces of a non-cubic cell: the normal is radial (from the nearest image of the rod center)
    in all interface voxels. (3) Two rods that nearly touch: in the voxels that both boundaries cross, the tensor is
    the one of the fit; elsewhere it is not."""
    slab = Cell(size=(1, 1, 1), shape=(16, 16, 1), basis=TRIANGULAR,
                objects=(Block(size=(np.inf, 0.3, np.inf), eps=12.0),))
    for normal in ("object", "fit"):
        t = slab.eps_inverse(8, normal=normal)
        iface = np.abs(t[1, 1] - t[0, 0]) > 1e-9
        assert iface.sum() >= 32                   # each of the two faces crosses a full row of 16 voxels along a1
        assert np.abs(t[0, 1]).max() < 1e-14 and np.all(t[1, 1][iface] > t[0, 0][iface])

    center = (0.4, 0.6)                                         # the rod crosses the faces x = 1/2 and y = 3/4
    cell = Cell(size=(1, 1.5, 1), shape=(24, 20, 1), objects=(Cylinder(radius=0.3, eps=12.0, center=(*center, 0)),))
    angle = angle_to_radial(cell, cell.eps_inverse(8), center)
    assert angle.size >= 30 and angle.max() < 1e-10

    two = Cell(size=(1, 1, 1), shape=(32, 32, 1), objects=(Cylinder(radius=0.2, eps=12.0, center=(-0.205, 0, 0)),
                                                           Cylinder(radius=0.2, eps=12.0, center=(0.205, 0, 0))))
    _, count, _ = two._sample(8, track=True)
    t_obj, t_fit = two.eps_inverse(8), two.eps_inverse(8, normal="fit")
    assert (count == 2).sum() >= 2                              # the gap (0.01) is smaller than a voxel (1/32)
    assert np.array_equal(t_obj[..., count == 2], t_fit[..., count == 2])
    assert np.abs(t_obj[..., count == 1] - t_fit[..., count == 1]).max() > 1e-3


def test_non_cubic_voxels_against_a_cubic_reference():
    """Rod r = 0.3 a at (0.1, 0.2) in a 1 x 1.5 cell, 24 x 20 voxels (aspect 1.8), TE, 4 bands at X, M and (0.2, 0.1);
    reference: 96 x 144 (cubic voxels). Max error (measured): object normal 2.34e-3, fit 4.12e-3, the first-moment
    direction of the code before oblique cells 8.53e-3. Angle to the radial direction over all 40 interface voxels:
    mean 0 (object), 9.13 degrees (fit), 20.44 degrees (first moment)."""
    obj = (Cylinder(radius=0.3, eps=12.0, center=(0.1, 0.2, 0)),)
    ks = [[0.5, 0, 0], [0.5, 0.5, 0], [0.2, 0.1, 0]]
    ref = solve_bands(Cell(size=(1, 1.5, 1), shape=(96, 144, 1), objects=obj), ks, 4, "te", tol=1e-9).freqs
    err, angle = {}, {}
    for cls in (Cell, FitCell, FirstMomentCell):
        cell = cls(size=(1, 1.5, 1), shape=(24, 20, 1), objects=obj)
        res = solve_bands(cell, ks, 4, "te", tol=1e-9)
        assert all(res.converged)
        err[cls] = np.abs(res.freqs - ref).max()
        angle[cls] = angle_to_radial(cell, cell.eps_inverse(8), (0.1, 0.2))
    assert all(a.size == 40 for a in angle.values())
    assert angle[Cell].max() < 1e-10
    assert angle[FitCell].mean() == pytest.approx(9.13, abs=0.01)
    assert angle[FirstMomentCell].mean() == pytest.approx(20.44, abs=0.01)
    assert err[Cell] == pytest.approx(2.34e-3, rel=0.01)
    assert err[FitCell] == pytest.approx(4.12e-3, rel=0.01)
    assert err[FirstMomentCell] == pytest.approx(8.53e-3, rel=0.01)


@pytest.mark.parametrize("pol", ["te", "tm"])
def test_empty_triangular_lattice_is_the_folded_free_photon_lines(pol):
    cell = Cell(size=(1, 1, 1), shape=(12, 12, 1), eps_background=2.25, basis=TRIANGULAR)
    ks = GMK + [[0.17, -0.31, 0.0]]
    # the default warm start: solve_bands turns it off in a homogeneous cell (the operator is diagonal in G, so the
    # warm block from K kept its plane waves and converged at once to wrong bands at the last k point)
    res = solve_bands(cell, ks, 7, pol, tol=1e-10)
    exact = np.array([free_photon(cell, k, 7, 1) for k in ks])
    assert all(res.converged)
    assert np.abs(res.freqs - exact).max() < 1e-12                          # measured 2e-14
    # eps 2.25 (n = 1.5): Gamma 0, |b| / n (six-fold); M |M| / n (two-fold), then 2/3; K |K| / n = 4/9 (three-fold)
    assert np.allclose(exact[:3, :3], [[0, 4 / (3 * S3), 4 / (3 * S3)], [2 / (3 * S3)] * 2 + [2 / 3], [4 / 9] * 3])


def test_empty_3d_oblique_lattices_are_free_photon_lines():
    """The hexagonal lattice (triangular, c = 1.3) and the fcc lattice, at a general k: two modes per G."""
    k = [0.13, -0.21, 0.07]
    for cell in (Cell(size=(1, 1, 1.3), shape=(8, 8, 8), eps_background=2.25, basis=TRIANGULAR),
                 Cell(size=(np.sqrt(0.5),) * 3, shape=(8, 8, 8), eps_background=2.25, basis=FCC)):
        res = solve_bands(cell, [k], 8, tol=1e-10)
        assert res.converged[0]
        assert np.abs(res.freqs[0] - free_photon(cell, k, 8, 2)).max() < 1e-12   # measured 6e-16


def test_rotated_square_lattice_equals_the_rectangular_cell():
    """The square lattice with its basis rotated (2-D: 0.5 rad about z; 3-D: 0.7 rad about (1, 2, 3)) and a rod or a
    sphere at the center is the same crystal, rotated. Its bands at the same k (reciprocal-lattice coordinates) must
    equal the rectangular cell to rounding: q, the transverse basis and eps_inv are general in direction. With the
    basis given as the identity, the numbers are the same bit for bit."""
    rod = (Cylinder(radius=0.2, eps=12.0),)
    ks = [[0.5, 0, 0], [0.5, 0.5, 0], [0.2, 0.1, 0]]
    c, s = np.cos(0.5), np.sin(0.5)
    rect = Cell(size=(1, 1, 1), shape=(32, 32, 1), objects=rod)
    rot = Cell(size=(1, 1, 1), shape=(32, 32, 1), objects=rod, basis=((c, s, 0), (-s, c, 0), (0, 0, 1)))
    eye = Cell(size=(1, 1, 1), shape=(32, 32, 1), objects=rod, basis=((1, 0, 0), (0, 1, 0), (0, 0, 1)))
    assert np.array_equal(rect.eps_inverse(), eye.eps_inverse())
    for pol in ("tm", "te"):
        ref = solve_bands(rect, ks, 4, pol, tol=1e-10).freqs
        assert np.abs(solve_bands(rot, ks, 4, pol, tol=1e-10).freqs - ref).max() < 1e-10     # measured 2e-15
        assert np.array_equal(solve_bands(eye, ks, 4, pol, tol=1e-10).freqs, ref)

    ax = np.array([1.0, 2.0, 3.0]) / np.sqrt(14.0)
    kx = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    rmat = np.eye(3) + np.sin(0.7) * kx + (1 - np.cos(0.7)) * kx @ kx
    sph = (Sphere(radius=0.3, eps=12.0),)
    k3 = [[0.5, 0.25, 0.1], [0.5, 0.5, 0.5]]
    ref = solve_bands(Cell(size=(1, 1, 1), shape=(12, 12, 12), objects=sph), k3, 4, tol=1e-10).freqs
    got = solve_bands(Cell(size=(1, 1, 1), shape=(12, 12, 12), objects=sph, basis=tuple(map(tuple, rmat.T))), k3, 4,
                      tol=1e-10).freqs
    assert np.abs(got - ref).max() < 1e-10                                                  # measured 4e-16


def test_sheared_square_lattice_converges_to_the_rectangular_cell():
    """The square lattice of rods described with a1 = (1, 0), a2 = (1, 1): other voxels (sheared) and other plane
    waves, so equal only in the limit. Max difference over 4 bands at X, M and a general k (measured):
    TM 2.2e-3 (n = 16) and 2.4e-4 (n = 64); TE 1.67e-3 and 5.7e-4 (the order of TE: the next test)."""
    rod = (Cylinder(radius=0.2, eps=12.0),)
    ks = [[0.5, 0, 0], [0.5, 0.5, 0], [0.2, 0.1, 0]]
    diff = {}
    for pol in ("tm", "te"):
        for n in (16, 64):
            rect = Cell(size=(1, 1, 1), shape=(n, n, 1), objects=rod)
            sh = Cell(size=(1, np.sqrt(2), 1), shape=(n, n, 1), objects=rod, basis=((1, 0, 0), (1, 1, 0), (0, 0, 1)))
            a = solve_bands(rect, ks, 4, pol, tol=1e-8)
            b = solve_bands(sh, [sh.k_fractional(rect.k_cartesian(k)) for k in ks], 4, pol, tol=1e-8)
            assert all(a.converged) and all(b.converged)
            diff[pol, n] = np.abs(a.freqs - b.freqs).max()
    assert diff["tm", 64] < 4e-4 and diff["tm", 64] < diff["tm", 16] / 4
    assert diff["te", 64] < 7e-4 and diff["te", 64] < diff["te", 16] / 2.5


@pytest.mark.slow
def test_sheared_te_converges_at_second_order():
    """TE error of the sheared square lattice (rods r = 0.2 a, eps 12; 4 bands at X, M and (0.2, 0.1)) against the
    rectangular cell at 192 x 192. Measured, n = 64 and 128: object normal 1.21e-3 and 2.52e-4 (ratio 4.8, second
    order); the fit in all voxels 2.08e-3 and 8.38e-4 (ratio 2.5, about first order: the bias of the fit at a curved
    step in a sheared voxel)."""
    rod = (Cylinder(radius=0.2, eps=12.0),)
    ks = [[0.5, 0, 0], [0.5, 0.5, 0], [0.2, 0.1, 0]]
    ref_cell = Cell(size=(1, 1, 1), shape=(192, 192, 1), objects=rod)
    ref = solve_bands(ref_cell, ks, 4, "te", tol=1e-10).freqs
    err = {}
    for cls in (Cell, FitCell):
        for n in (64, 128):
            sh = cls(size=(1, np.sqrt(2), 1), shape=(n, n, 1), objects=rod, basis=((1, 0, 0), (1, 1, 0), (0, 0, 1)))
            res = solve_bands(sh, [sh.k_fractional(ref_cell.k_cartesian(k)) for k in ks], 4, "te", tol=1e-10)
            assert all(res.converged)
            err[cls, n] = np.abs(res.freqs - ref).max()
    assert err[Cell, 64] / err[Cell, 128] > 3.5
    assert err[FitCell, 64] / err[FitCell, 128] < 3.0
    assert err[Cell, 128] < 3.5e-4 and err[Cell, 128] < err[FitCell, 128] / 2


def triangular_and_supercell(n: int, pol: str, radius: float, eps_rod: float, bg: float, num: int = 4):
    """Bands at Gamma, M, K of the triangular cell (n x n) with a rod at each lattice point, folded into the
    rectangular supercell (1 x sqrt(3), two rods: (0, 0) and (1/2, sqrt(3)/2)), and the supercell itself
    (n x round(sqrt(3) n)) at the same k. The supercell has the extra reciprocal vector (0, 1/sqrt(3)) = (0, 1/2) in
    triangular coordinates, so its bands at k are the union of the triangular bands at k and k + (0, 1/2, 0)."""
    tri = Cell(size=(1, 1, 1), shape=(n, n, 1), basis=TRIANGULAR, eps_background=bg,
               objects=(Cylinder(radius=radius, eps=eps_rod),))
    rect = Cell(size=(1, S3, 1), shape=(n, round(S3 * n), 1), eps_background=bg,
                objects=(Cylinder(radius=radius, eps=eps_rod),
                         Cylinder(radius=radius, eps=eps_rod, center=(0.5, S3 / 2, 0))))
    ft = solve_bands(tri, GMK + [np.add(k, (0, 0.5, 0)) for k in GMK], num, pol, tol=1e-8)
    kr = [rect.k_fractional(tri.k_cartesian(k)) for k in GMK]
    fr = solve_bands(rect, kr, num, pol, tol=1e-8)
    assert all(ft.converged) and all(fr.converged)
    folded = np.sort(np.concatenate([ft.freqs[:3], ft.freqs[3:]], axis=1), axis=1)[:, :num]
    return ft.freqs[:3], folded, fr.freqs, kr


def gap_edges(tri_freqs):
    """Band 1 at K and band 2 at M of the triangular bands (rows Gamma, M, K)."""
    return np.array([tri_freqs[2, 0], tri_freqs[1, 1]])


def supercell_gap_edges(rect_freqs):
    """The same two bands in the supercell: at K, band 2 (the folded k + (0, 1/2) has a lower band); at M, band 3
    (M folds onto Gamma, which adds the band 0)."""
    return np.array([rect_freqs[2, 1], rect_freqs[1, 2]])


@pytest.mark.slow
def test_triangular_rods_tm_gap_against_supercell_and_finite_differences():
    """Rods eps 12, r = 0.2 a, TM: the gap from band 1 at K to band 2 at M. n = 64: 0.27448 and 0.44527; the
    supercell 0.27453 and 0.44530; the finite-difference supercell (128 x 222) 0.27443 and 0.44520."""
    from fd_tm_reference import fd_tm_rect
    t32 = triangular_and_supercell(32, "tm", 0.2, 12.0, 1.0)[0]
    t64, folded, rect, kr = triangular_and_supercell(64, "tm", 0.2, 12.0, 1.0)
    assert np.abs(gap_edges(t32) - gap_edges(t64)).max() < 5e-4                    # measured 2.1e-4
    assert gap_edges(t64) == pytest.approx([0.27448, 0.44527], abs=5e-5)
    assert np.abs(folded - rect).max() < 4e-4                      # all 4 bands at Gamma, M, K; measured 2.3e-4
    assert np.abs(gap_edges(t64) - supercell_gap_edges(rect)).max() < 1e-4         # measured 5e-5
    rodlist = [(0.0, 0.0, 0.2, 12.0), (0.5, S3 / 2, 0.2, 12.0)]
    fd = np.array([fd_tm_rect((128, 222), (1, S3), rodlist, k[:2]) for k in kr])
    assert np.abs(folded - fd).max() < 4e-4                        # measured 2.5e-4
    assert np.abs(gap_edges(t64) - supercell_gap_edges(fd)).max() < 1.5e-4         # measured 7e-5


@pytest.mark.slow
def test_triangular_holes_te_gap_against_the_supercell():
    """Air holes r = 0.3 a in eps 12, TE (E in the plane: the anisotropic averaging in sheared voxels): the gap from
    band 1 at K to band 2 at M. n = 64: 0.20706 and 0.27440; the supercell 0.20707 and 0.27451."""
    t32 = triangular_and_supercell(32, "te", 0.3, 1.0, 12.0)[0]
    t64, folded, rect, _ = triangular_and_supercell(64, "te", 0.3, 1.0, 12.0)
    assert np.abs(gap_edges(t32) - gap_edges(t64)).max() < 5e-4                    # measured 1.6e-4
    assert gap_edges(t64) == pytest.approx([0.20706, 0.27440], abs=5e-5)
    assert np.abs(folded - rect).max() < 2e-4                      # measured 1.1e-4
    assert np.abs(gap_edges(t64) - supercell_gap_edges(rect)).max() < 2e-4         # measured 1.1e-4
