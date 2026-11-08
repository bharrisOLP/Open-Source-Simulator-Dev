"""Checks of the JAX path of the eps sampling (bands_gpu/sampling.py) against the NumPy reference (Cell.voxel_averages):
the boundary counts and the analytic normals (equal), the fill fractions <eps>, the unit normals and the averaged
tensors (1e-12) on the test cells of the package (rods, air holes, oblique and non-cubic cells, a slab with its faces
on samples, fcc spheres, blocks and cylinders, a diamond lattice, a 3-D slab, symmetric voxels with eps values that
are not exact in binary), on the solver device (the GPU) and on the CPU device of JAX; the inside test at samples on
the surface of a cylinder and a sphere (the same decision as NumPy, so no fused multiply-add on the device); the
moment bound in symmetric voxels; the slabs of voxels; the bands with each path; the dispatch (with the runs of
objects of one kind)."""
import dataclasses
import functools
import math

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from bands_gpu import sampling  # noqa: E402
from bands_gpu.geometry import TRIANGULAR, Block, Cell, Cylinder, Sphere  # noqa: E402
from bands_gpu.solver import solve_bands, solver_device  # noqa: E402

S3 = np.sqrt(3.0)
FCC = ((0, 1, 1), (1, 0, 1), (1, 1, 0))
FCC_SIZE = (np.sqrt(0.5),) * 3
C, S = np.cos(0.5), np.sin(0.5)


def cells() -> dict[str, tuple[Cell, int]]:
    """The test cells (with the subpixel count): the cells of test_bands, test_oblique and test_findk, and the 3-D
    cells of the slab and diamond checks."""
    rod = (Cylinder(radius=0.2, eps=12.0),)
    return {
        "rods 64": (Cell(size=(1, 1, 1), shape=(64, 64, 1), objects=rod), 8),
        "holes, triangular 64": (Cell(size=(1, 1, 1), shape=(64, 64, 1), basis=TRIANGULAR, eps_background=12.0,
                                      objects=(Cylinder(radius=0.3, eps=1.0),)), 8),
        "rods, sheared 32": (Cell(size=(1, np.sqrt(2), 1), shape=(32, 32, 1), objects=rod,
                                  basis=((1, 0, 0), (1, 1, 0), (0, 0, 1))), 8),
        "rods, rotated 32": (Cell(size=(1, 1, 1), shape=(32, 32, 1), objects=rod,
                                  basis=((C, S, 0), (-S, C, 0), (0, 0, 1))), 8),
        "rod across faces, 24 x 20": (Cell(size=(1, 1.5, 1), shape=(24, 20, 1),
                                           objects=(Cylinder(radius=0.3, eps=12.0, center=(0.4, 0.6, 0)),)), 8),
        "two rods 0.01 apart": (Cell(size=(1, 1, 1), shape=(32, 32, 1),
                                     objects=(Cylinder(radius=0.2, eps=12.0, center=(-0.205, 0, 0)),
                                              Cylinder(radius=0.2, eps=12.0, center=(0.205, 0, 0)))), 8),
        "slab, triangular 16": (Cell(size=(1, 1, 1), shape=(16, 16, 1), basis=TRIANGULAR,
                                     objects=(Block(size=(np.inf, 0.3, np.inf), eps=12.0),)), 8),
        **{f"slab, faces on samples, d {d}": (Cell(size=(1, 5, 1), shape=(1, 25, 1), eps_background=1.45**2,
                                                   objects=(Block(size=(np.inf, d, np.inf), eps=3.5**2),)), 8)
           for d in [round(0.325 + 0.05 * i, 6) for i in range(8)]},
        "slab 1-D, 1025": (Cell(size=(1, 5, 1), shape=(1, 1025, 1), eps_background=1.45**2,
                                objects=(Block(size=(np.inf, 0.5, np.inf), eps=3.5**2),)), 8),
        "strip 3-D": (Cell(size=(1, 3, 2), shape=(1, 25, 15), eps_background=1.444**2,
                           objects=(Block(size=(np.inf, 0.5, 0.22), eps=3.48**2),)), 8),
        "spheres, simple cubic 16": (Cell(size=(1, 1, 1), shape=(16, 16, 16), objects=(Sphere(radius=0.3, eps=12.0),)),
                                     8),
        "sphere, fcc 16": (Cell(size=FCC_SIZE, shape=(16, 16, 16), basis=FCC,
                                objects=(Sphere(radius=0.3, eps=2.0, center=(0.1, 0.05, 0.0)),)), 4),
        "block and cylinder, fcc 16": (Cell(size=FCC_SIZE, shape=(16, 16, 16), basis=FCC,
                                            objects=(Block(size=(0.3017, 0.4013, 0.4991), eps=2.0),
                                                     Cylinder(radius=0.2, height=0.4937, eps=5.0,
                                                              center=(0.2, 0.1, 0.0)))), 4),
        "diamond, fcc 12": (Cell(size=FCC_SIZE, shape=(12, 12, 12), basis=FCC,
                                 objects=(Sphere(radius=0.25, eps=11.56, center=(0.125,) * 3),
                                          Sphere(radius=0.25, eps=11.56, center=(-0.125,) * 3))), 8),
        "hole slab, triangular 3-D": (Cell(size=(1, 1, 4), shape=(16, 16, 33), basis=TRIANGULAR,
                                           objects=(Block(size=(np.inf, np.inf, 0.6), eps=12.0),
                                                    Cylinder(radius=0.3, height=0.6, eps=1.0))), 4),
        **symmetric_cells(),
    }


SI, SIO2 = 3.48**2, 1.444**2                      # eps values that are not exact in binary


def symmetric_cells() -> dict[str, tuple[Cell, int]]:
    """Cells with voxels that are symmetric along a lattice vector, with eps values that are not exact in binary: the
    first moments there are zero in exact arithmetic, and the sums give rounding residues (the review case: before
    the moment bound, the two paths gave tensors 7.8e-2 apart and TE bands 7.77e-4 apart). Each object is centered on
    a grid point, so the analytic normal there is zero (the fit is used)."""
    return {
        "two Si rods 0.01 apart": (Cell(size=(1, 1, 1), shape=(32, 32, 1), eps_background=SIO2,
                                        objects=(Cylinder(radius=0.2, eps=SI, center=(-0.205, 0, 0)),
                                                 Cylinder(radius=0.2, eps=SI, center=(0.205, 0, 0)))), 8),
        "small centered Si sphere": (Cell(size=(1, 1, 1), shape=(8, 8, 8), eps_background=SIO2,
                                          objects=(Sphere(radius=0.3 / 8, eps=SI),)), 8),
        "thin centered Si slab": (Cell(size=(1, 1, 1), shape=(16, 1, 1), eps_background=SIO2,
                                       objects=(Block(size=(0.3 / 16, np.inf, np.inf), eps=SI),)), 8),
        "thin centered Si rod": (Cell(size=(1, 1, 1), shape=(16, 16, 1), eps_background=SIO2,
                                      objects=(Cylinder(radius=0.4 / 16, eps=SI),)), 8),
    }


CELLS = cells()


@functools.lru_cache(maxsize=None)
def reference(name: str, normal: str) -> dict[str, np.ndarray]:
    cell, sub = CELLS[name]
    return cell.voxel_averages(sub, normal)


def compare(name: str, normal: str, device, **kw) -> dict[str, float]:
    """The largest differences of the JAX path against the reference (asserts the counts and the analytic normals
    equal)."""
    cell, sub = CELLS[name]
    ref = reference(name, normal)
    got = sampling.voxel_averages(cell, sub, normal, device=device, **kw)
    assert set(got) == set(ref)
    assert np.array_equal(got["count"], ref["count"])
    assert np.array_equal(got["analytic"], ref["analytic"])
    eps_max = max([cell.eps_background] + [o.eps for o in cell.objects])
    return {"eps_inv": float(np.abs(got["eps_inv"] - ref["eps_inv"]).max()),
            "mean_eps": float(np.abs(got["mean_eps"] - ref["mean_eps"]).max() / eps_max),
            "normal": float(np.abs(got["normal"] - ref["normal"]).max())}


DEVICES = {"solver": solver_device, "cpu": lambda: jax.devices("cpu")[0]}


def compare_cells(cell: Cell, sub: int, **kw) -> dict[str, float]:
    """As ``compare``, for a cell that is not in CELLS (on the solver device)."""
    ref = cell.voxel_averages(sub)
    got = sampling.voxel_averages(cell, sub, device=solver_device(), **kw)
    assert np.array_equal(got["count"], ref["count"]) and np.array_equal(got["analytic"], ref["analytic"])
    return {k: float(np.abs(got[k] - ref[k]).max()) for k in ("eps_inv", "mean_eps", "normal")}


@pytest.mark.parametrize("where", list(DEVICES))
def test_jax_sampling_equals_the_numpy_reference(where):
    """All test cells, both normals: tensors, unit normals and <eps> / max eps within 1e-12 (measured: the largest
    difference is printed with pytest -s)."""
    device = DEVICES[where]()
    worst = {}
    for name in CELLS:
        for normal in ("object", "fit"):
            d = compare(name, normal, device)
            worst[name, normal] = d
            assert max(d.values()) < 1e-12, (name, normal, d)
    print(where, device, {k: max(d[k] for d in worst.values()) for k in ("eps_inv", "mean_eps", "normal")})
    # the cells have interfaces of each kind: one boundary (the analytic normal), two boundaries (the fit)
    ref = reference("two rods 0.01 apart", "object")
    assert (ref["count"] == 2).sum() >= 2 and (ref["count"] == 1).sum() > 50
    for name in ("slab, triangular 16", "slab, faces on samples, d 0.325", "hole slab, triangular 3-D"):
        assert (reference(name, "object")["count"] >= 1).sum() > 0


def test_slabs_of_voxels_give_the_same_averages():
    """A small sample budget: slabs of 3 voxel planes and a last slab of 1 (fcc, 16 planes); also a 1-D cell, whose
    first used axis is y."""
    sub = CELLS["sphere, fcc 16"][1]
    plane = 16 * 16 * sub**3
    d = compare("sphere, fcc 16", "object", solver_device(), max_samples=3 * plane + 5)
    assert max(d.values()) < 1e-12
    d = compare("slab 1-D, 1025", "object", solver_device(), max_samples=1000)
    assert max(d.values()) < 1e-12


@pytest.mark.parametrize("kind", ["cylinder", "sphere"])
def test_inside_at_samples_on_the_surface_is_the_numpy_decision(kind):
    """Points on the surface to rounding (x^2 + y^2 (+ z^2) within a few units in the last place of r^2): the device
    must give the decision of NumPy at each point. A fused multiply-add (one rounding for x * x + y * y) changes
    many of these decisions."""
    rng = np.random.default_rng(3)
    n = 200_000
    u = rng.normal(size=(3, n))
    if kind == "cylinder":
        u[2] = rng.uniform(-0.1, 0.1, n)
        u[:2] /= np.hypot(u[0], u[1])
        obj = Cylinder(radius=0.3, eps=2.0, center=(0.01, -0.02, 0.0), height=0.5)
    else:
        u /= np.linalg.norm(u, axis=0)
        obj = Sphere(radius=0.3, eps=2.0, center=(0.01, -0.02, 0.03))
    x = (obj.center[0] + obj.radius * u[0], obj.center[1] + obj.radius * u[1],
         obj.center[2] + (obj.radius * u[2] if kind == "sphere" else u[2]))
    ref = obj.inside(*x)
    assert 0.2 < ref.mean() < 0.8                                    # the points are on the surface to rounding
    p = jnp.asarray(sampling._params(obj))
    fn = jax.jit(lambda p, x, y, z: sampling._inside(kind, p, x, y, z))
    for device in DEVICES.values():
        got = np.asarray(fn(*jax.device_put((p, *map(jnp.asarray, x)), device())))
        assert np.array_equal(got, ref)


def raw_moments(cell: Cell, sub: int, order: np.ndarray) -> np.ndarray:
    """The first moments of each voxel without the bound, summed one sample after the other in the given order of the
    N samples of a voxel (an order of the sums other than the one of the reference)."""
    s = [sub if n > 1 else 1 for n in cell.shape]
    nx, ny, nz = cell.shape
    e = cell.eps_samples(sub).reshape(nx, s[0], ny, s[1], nz, s[2]).transpose(0, 2, 4, 1, 3, 5)
    e = e.reshape(nx, ny, nz, -1)[..., order]
    o = np.meshgrid(*[(np.arange(k) + 0.5) / k - 0.5 for k in s], indexing="ij")
    return np.stack([np.cumsum(e * oi.ravel()[order], axis=-1)[..., -1] / e.shape[-1] for oi in o])


def test_moment_bound_makes_symmetric_voxels_the_same_in_both_paths():
    """In each symmetric cell, a voxel has moments that are rounding residues (nonzero in one of three orders of the
    sums, at most the bound): they count as zero, so the voxel is isotropic in both paths, and eps_inv and the bands of the two
    paths agree to 1e-12 (eps_inv) and 1e-12 (bands). A true moment of these voxels is at least
    |eps_1 - eps_2| / (2 s N), far above the bound."""
    for name, (cell, sub) in symmetric_cells().items():
        bound = cell.moment_bound(sub)
        n = math.prod(sub if k > 1 else 1 for k in cell.shape)
        orders = [np.arange(n), np.arange(n)[::-1], np.random.default_rng(5).permutation(n)]
        residue = np.max([np.abs(raw_moments(cell, sub, o)).max(axis=0) for o in orders], axis=0)
        assert np.any((residue > 0) & (residue <= bound)), name           # the case occurs in this cell
        true_min = abs(SI - SIO2) / (2 * sub * n)
        assert np.all((residue <= bound) | (residue > 1e3 * bound)) and true_min > 1e3 * bound
        for where in DEVICES:
            got = sampling.voxel_averages(cell, sub, device=DEVICES[where]())
            ref = cell.voxel_averages(sub)
            assert np.abs(got["eps_inv"] - ref["eps_inv"]).max() < 1e-12, (name, where)
            res = (residue > 0) & (residue <= bound) & (ref["count"] != 1)
            res |= (residue > 0) & (residue <= bound) & ((ref["analytic"] ** 2).sum(axis=0) == 0)
            assert np.all(ref["normal"][:, res] == 0) and np.all(got["normal"][:, res] == 0)
    bands = {}
    for name, ks, pol, nb in (("two Si rods 0.01 apart", [[0.5, 0, 0], [0.5, 0.5, 0]], "te", 4),
                              ("small centered Si sphere", [[0.3, 0.1, 0.2]], None, 4),
                              ("thin centered Si slab", [[0.3, 0, 0]], "te", 4),
                              ("thin centered Si rod", [[0.5, 0.5, 0]], "te", 4)):
        cell = symmetric_cells()[name][0]
        a = solve_bands(cell, ks, nb, pol, tol=1e-10, sampling="numpy")
        b = solve_bands(cell, ks, nb, pol, tol=1e-10, sampling="jax")
        assert all(a.converged) and all(b.converged)
        bands[name] = float(np.abs(a.freqs - b.freqs).max())
    print("bands, NumPy against JAX", bands)
    assert max(bands.values()) < 1e-12


def test_bands_with_the_jax_sampling_equal_the_numpy_bands():
    """Diamond (fcc, two spheres) at 12^3 and the triangular air holes at 32 x 32 (TE): the bands of the two paths
    agree to 1e-12."""
    diamond = dataclasses.replace(CELLS["diamond, fcc 12"][0], shape=(10, 10, 10))
    ks = [[0.5, 0.5, 0.5], [0.5, 0.0, 0.5]]
    a = solve_bands(diamond, ks, 4, tol=1e-10, sampling="numpy")
    b = solve_bands(diamond, ks, 4, tol=1e-10, sampling="jax")
    assert all(a.converged) and all(b.converged)
    assert np.abs(a.freqs - b.freqs).max() < 1e-12
    holes = dataclasses.replace(CELLS["holes, triangular 64"][0], shape=(32, 32, 1))
    ks = [[0.0, 0.5, 0.0], [-1 / 3, 1 / 3, 0.0]]
    a = solve_bands(holes, ks, 4, "te", tol=1e-10, sampling="numpy")
    b = solve_bands(holes, ks, 4, "te", tol=1e-10, sampling="jax")
    assert np.abs(a.freqs - b.freqs).max() < 1e-12


def test_dispatch():
    """"auto": JAX from AUTO_MIN_WORK inside tests (samples times images) times the runs of objects of one kind up,
    for supported cells; "jax" with an unsupported cell (an override of the sampling, an unknown object): an error.
    A run of objects of one kind is one scan (the slabs and the image lists of different lengths: the reference)."""

    class FitCell(Cell):
        def eps_inverse(self, sub: int = 8, normal: str = "object") -> np.ndarray:
            return Cell.eps_inverse(self, sub, normal="fit")

    class ShiftedCell(Cell):                         # other sample positions: the JAX path must not be used
        def axes(self, sub: int = 1) -> list[np.ndarray]:
            return [a + 1e-3 for a in Cell.axes(self, sub)]

    @dataclasses.dataclass(frozen=True)
    class Ball(Sphere):
        pass

    rod = (Cylinder(radius=0.2, eps=12.0),)
    big = Cell(size=(1, 1, 1), shape=(512, 512, 1), objects=rod)                 # 4096^2 samples, 9 images
    small = dataclasses.replace(big, shape=(256, 256, 1))
    assert sampling.work(big, 8) == 4096**2 * 9 >= sampling.AUTO_MIN_WORK > sampling.work(small, 8)
    assert sampling.use_jax(big, 8, "auto") and not sampling.use_jax(small, 8, "auto")
    assert sampling.use_jax(small, 8, "jax") and not sampling.use_jax(big, 8, "numpy")
    for cell in (FitCell(size=big.size, shape=big.shape, objects=rod),
                 ShiftedCell(size=big.size, shape=big.shape, objects=rod),
                 Cell(size=big.size, shape=big.shape, objects=(Ball(radius=0.2, eps=12.0),))):
        assert not sampling.supported(cell) and not sampling.use_jax(cell, 8, "auto")
        with pytest.raises(ValueError):
            sampling.voxel_averages(cell, 8)
    fit = FitCell(size=(1, 1, 1), shape=(16, 16, 1), objects=rod)
    assert np.array_equal(sampling.eps_inverse(fit, 8), Cell.eps_inverse(fit, 8, normal="fit"))
    # runs: alternating kinds compile once per object, so "auto" asks for more work
    mixed = Cell(size=(1, 1, 1), shape=(256, 256, 1), objects=(
        Block(size=(0.1, 0.6, np.inf), eps=4.0, center=(-0.25, 0.1, 0)), Cylinder(radius=0.05, eps=9.0),
        Block(size=(0.1, 0.6, np.inf), eps=4.0, center=(0.25, 0.1, 0))))
    assert sampling.runs(mixed) == 3 and sampling.work(mixed, 8) == 2048**2 * 27
    assert not sampling.use_jax(mixed, 8, "auto")                    # 1.1e8 < 3e8
    c = np.arange(10) - 4.5
    many = Cell(size=(10, 10, 1), shape=(80, 80, 1),
                objects=tuple(Cylinder(radius=0.2, eps=12.0, center=(x, y, 0)) for x in c for y in c))
    assert sampling.runs(many) == 1 and sampling.use_jax(many, 8, "auto")
    d = compare_cells(mixed, 8, max_samples=200_000)               # slabs of 12 planes (and a last of 4)
    assert max(d.values()) < 1e-12
    d = compare_cells(many, 8)
    assert max(d.values()) < 1e-12
    with pytest.raises(ValueError):
        sampling.use_jax(big, 8, "gpu")
    with pytest.raises(ValueError):
        sampling.voxel_averages(small, 8, normal="first")
