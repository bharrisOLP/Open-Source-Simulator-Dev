# bands-gpu

Photonic band structures on the GPU, with the method of [MPB](https://github.com/NanoComp/mpb): a transverse
plane-wave basis, the operator curl eps^-1 curl applied with FFTs, anisotropic subpixel averaging of eps, and a
block eigensolver with a warm start along the k path. Written from the published method in JAX (no MPB code).
First target: AMD GPUs with ROCm (tested on a Radeon RX 7900 XT, gfx1100).

```python
import jax; jax.config.update("jax_enable_x64", True)        # complex128 (else complex64, with a warning)
from bands_gpu.geometry import Cell, Cylinder
from bands_gpu.solver import interpolate, solve_bands

cell = Cell(size=(1, 1, 1), shape=(32, 32, 1), objects=(Cylinder(radius=0.2, eps=12.0),))
ks = interpolate([[0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0], [0, 0, 0]], 4)     # Gamma - X - M - Gamma
res = solve_bands(cell, ks, num_bands=8, polarization="tm")                 # res.freqs: (nk, 8), units 2 pi c / a
```

Oblique lattices: give the directions of the lattice vectors with `basis` (rows; normalized, as in MPB) and their
lengths with `size`. k points are in units of the reciprocal lattice vectors, as in MPB.

```python
from bands_gpu.geometry import TRIANGULAR                                   # a1 = (1, 0), a2 = (1/2, sqrt(3)/2), a3 = z
from bands_gpu.solver import TRIANGULAR_POINTS, triangular_path             # Gamma (0, 0), M (0, 1/2), K (-1/3, 1/3)

cell = Cell(size=(1, 1, 1), shape=(64, 64, 1), basis=TRIANGULAR, eps_background=12.0,
            objects=(Cylinder(radius=0.3, eps=1.0),))                       # air holes; objects are Cartesian
res = solve_bands(cell, triangular_path(8), num_bands=6, polarization="te")  # Gamma - M - K - Gamma
cell.lattice, cell.reciprocal, cell.k_cartesian(TRIANGULAR_POINTS["K"])    # a_i, b_i (a_i . b_j = delta_ij), k
```

## Method

- **Lattice.** Lattice vectors a_i = size_i basis_i / |basis_i| (rows of `Cell.lattice`), reciprocal vectors b_i
  with a_i . b_j = delta_ij (`Cell.reciprocal`). The grid is uniform in the lattice coordinates; the FFT frequencies
  n_i of each axis give the plane waves G = 2 pi (n1 b1 + n2 b2 + n3 b3), and q = k + G is Cartesian. The
  transverse basis, the products and eps_inv are Cartesian, so the operator is the same for any direction of q.
  Objects are Cartesian; the sampling tests each object at all lattice translations that can reach the cell (from
  the extent of the object along each b_i), so objects continue across the oblique faces. An object that is
  infinite along b_i must not change along a_i (else a ValueError). Without `basis` the cell is rectangular, with the
  same q and samples bit for bit as before oblique cells (for the normal, see **eps**).
- **Basis and operator.** H(r) = sum_G h(G) exp(i (k+G).r) with h(G) orthogonal to k+G (two coefficients per G
  in 3-D; one per G for the TE or TM polarization of a 2-D cell). Theta h = -q x FFT[eps_inv . IFFT[q x h]]:
  six FFTs per product.
- **eps.** Supersampled in each voxel; at an interface eps_inv = <1/eps> n n^T + (1/<eps>)(I - n n^T) (the
  averaging of MPB and Meep; Kottke et al. 2008). The normal n (as MPB): where exactly one object boundary (one
  periodic image of one object) crosses the voxel, n is the analytic normal of that object at the voxel center
  (radial for a sphere or a cylinder side; the nearest face for a block or a cylinder cap). Where several
  boundaries cross the voxel, n is the gradient of eps that fits the first moments m_i = <eps o_i> of the voxel
  (o_i: offsets in voxel fractions along a_i): n ~ sum_i m_i n_i b_i. The fit is exact for a linear eps and for a
  planar interface (a slab in the triangular cell: n = y, where the first-moment direction sum_i m_i d_i is
  30 degrees off). At a curved step in a sheared voxel the fit has a bias: with the fit in all voxels, the TE error
  of a sheared square lattice falls at about first order; with the object normal, at second order (the table in
  "Checks of oblique lattices"). Before oblique cells, the normal was the direction of the first moment (in a cubic
  voxel the same direction as the fit). Changes on the rectangular path: TM bands, 1-D stacks and homogeneous cells
  are the same bit for bit (the normal of a rod is in the plane and TM uses only eps_inv_zz; the normal of a slab is
  exactly x). TE and 3-D bands with curved objects change, and are more accurate (same table).
- **Eigensolver.** Block LOBPCG (Knyazev 2001) with the preconditioner 1/(|k+G|^2 <eps_inv>), two-pass SVQB
  orthogonalization of [X, W, P], P kept orthogonal to X, and A X and A P computed as products in each iteration.
  Two lessons: combined products drift until the solve diverges near convergence; and A P must be a product,
  because P comes from a subtraction with cancellation (else the residual stalls near sqrt(eps |A| / theta); the
  one review of this commit found it). A stall detector stops a solve whose residual no longer falls; it counts
  as converged only within 100 x tol. The block of one k point starts the next one (rod lattice, 10 k points,
  double precision: 284 iterations against 341 without the warm start).

## Checks (`tests/test_bands.py`)

| Case | Reference | Agreement |
|---|---|---|
| homogeneous eps = 2.25, 3-D, k = (0.1, 0.2, 0.05) | free-photon lines, exact | 1e-6 |
| 1-D Bragg stack (n = 3 / 1, d = 0.25 / 0.75) | exact transfer-matrix dispersion | 2e-3 relative (256 points) |
| square lattice of rods (eps 12, r = 0.2 a), TM, band 1 at M and band 2 at X, 64 x 64 | an independent finite-difference solver (`tests/fd_tm_reference.py`), converged: 0.28066 and 0.41716 | 2e-4 |

The MPB tutorial gives 0.28262 and 0.41933 for the rod lattice at resolution 32. Both this solver and the
independent finite-difference solver converge to values about 0.7 % lower (0.28066, 0.41716), so the tutorial
numbers carry the discretization error of resolution 32. This solver gives 0.28105 and 0.41729 at resolution 32.

## Checks of oblique lattices (`tests/test_oblique.py`)

| Case | Reference | Agreement |
|---|---|---|
| empty triangular lattice (eps 2.25), TE and TM, 7 bands at Gamma, M, K and a general k, default warm start (off in a homogeneous cell) | folded free-photon lines \|k + G\| / sqrt(eps), exact | 2e-14 |
| empty hexagonal (c = 1.3 a) and fcc lattices, 3-D, 8 bands at a general k | free-photon lines, exact | 6e-16 |
| square lattice of rods with a rotated basis (2-D TE and TM, 0.5 rad); simple cubic spheres, basis rotated 0.7 rad about (1, 2, 3) | the rectangular cell (the same crystal, rotated) | 2e-15 (2-D), 4e-16 (3-D) |
| square lattice of rods with the basis (1, 0), (1, 1): sheared voxels, other plane waves | the rectangular cell at the same n | TM 2.2e-3 (16 x 16), 2.4e-4 (64 x 64); TE 1.67e-3, 5.7e-4 |
| the same sheared lattice, TE, n = 64 and 128 (order of convergence) | the rectangular cell at 192 x 192 | object normal 1.21e-3, 2.52e-4 (ratio 4.8); fit in all voxels 2.08e-3, 8.38e-4 (ratio 2.5) |
| rod r = 0.3 a at (0.1, 0.2) in a 1 x 1.5 cell, 24 x 20 voxels (aspect 1.8), TE, 4 bands at X, M, (0.2, 0.1) | 96 x 144 (cubic voxels) | object normal 2.34e-3; fit 4.12e-3; first moment (before oblique cells) 8.53e-3 |
| normal in all 40 interface voxels of that cell | the radial direction at the voxel center | mean angle 0 (object), 9.13 degrees (fit), 20.44 degrees (first moment) |
| two rods 0.01 apart (32 x 32): voxels that both boundaries cross | the tensor of the fit | the same bit for bit |
| rod at the center, a corner or an edge center of the triangular cell; sphere, block, cylinder in the fcc cell | rolled or translated samples (the same bit for bit); volume fractions | rod 4.5e-4, sphere 5.8e-4 (the same at the three positions), block 1.1 %, cylinder 0.2 % (staircase of flat faces) |
| triangular lattice of rods (eps 12, r = 0.2 a), TM, gap from band 1 at K to band 2 at M | the same crystal in a rectangular supercell (1 x sqrt(3), two rods, folded k points); an independent finite-difference solver on the supercell | see below |
| triangular lattice of air holes (r = 0.3 a, eps 12), TE, gap from band 1 at K to band 2 at M | the rectangular supercell | see below |

The supercell has the extra reciprocal vector (0, 1/2) (in triangular coordinates), so its bands at k are the
union of the triangular bands at k and at k + (0, 1/2). Gap edges (band 1 at K, band 2 at M) against resolution:

| Grid (triangular n x n; supercell n x round(sqrt(3) n)) | Rods, TM | Holes, TE |
|---|---|---|
| triangular 16 | 0.27512, 0.44526 | 0.20736, 0.27501 |
| triangular 32 | 0.27469, 0.44530 | 0.20717, 0.27456 |
| triangular 64 | 0.27448, 0.44527 | 0.20706, 0.27440 |
| triangular 128 | 0.274438, 0.445244 | 0.207036, 0.274389 |
| supercell 64 | 0.27453, 0.44530 | 0.20707, 0.27451 |
| supercell 128 | 0.274438, 0.445246 | 0.207037, 0.274408 |
| finite differences, supercell 128 x 222 / 256 x 443 | 0.27443, 0.44520 / 0.274423, 0.445217 | - |

At 64 the triangular cell and the supercell agree to 2.3e-4 (rods) and 1.2e-4 (holes) over 4 bands at Gamma, M
and K (the tests use 64); at 128 the gap edges agree to 2e-6 (rods) and 1.9e-5 (holes). The rods are TM, so the
normal does not change them; the holes are TE, with the object normal (with the fit, the holes gave 0.27448 at 64
and agreed to 3.2e-5 at 128).

Order of convergence with the two normals (`bench/normal_order.py`, CPU): square lattice of rods (r 0.2 a, eps
12), TE, max error over 4 bands at X, M and (0.2, 0.1), against the rectangular cell at 256 x 256 (object normal).

| n | rectangular, object normal | rectangular, fit (in a cubic voxel: the normal before oblique cells) | sheared (1, 0), (1, 1), object normal | sheared, fit |
|---|---|---|---|---|
| 32 | 2.60e-3 | 2.90e-3 | 3.81e-3 | 5.86e-3 |
| 64 | 6.86e-4 | 9.10e-4 | 1.25e-3 | 2.12e-3 |
| 128 | 1.18e-4 | 2.73e-4 | 2.98e-4 | 8.77e-4 |
| 192 | 4.66e-5 | 1.11e-4 | 2.04e-4 | 5.62e-4 | The tests marked `slow`
(the two gap tests and the order test) take 5 to 16 s each on the CPU.

## GPU timings (RX 7900 XT against the same algorithm on the 16 CPU threads of the same PC)

Single precision, full k paths (`bench/bench_bands.py`; the JSON rows are in `benchmarks/2026-10-30/`). After the
A P fix (about 50 iterations per k point, all k points converged; `run-20261030-0626.jsonl`): rods 128 x 128, 16 k
points, GPU 5.2 s against CPU 60.3 s (11.7x); spheres 48^3, 13 k points, GPU 17.6 s (no CPU run).

The table below used the solver before the A P fix (`bands-before-ap-fix.jsonl`): about 300 iterations per k point
on both devices, and most runs stopped at the iteration limit before all k points converged. The ratios compare the
same work on the two devices, but they are not timings of the current solver.

| Case | Speed-up of the GPU |
|---|---|
| rods, 64 x 64, TM, 16 k points | 4.8x |
| rods, 128 x 128 | 11x |
| spheres (simple cubic, r 0.3 a, eps 12), 32^3, both polarizations, 13 k points | 12x |
| spheres, 48^3 | 15.6x |

Oblique cell (`bench/bench_bands.py --case tri-holes --n 128`: triangular lattice of air holes, TE, Gamma - M - K -
Gamma, 16 k points, 8 bands, single precision, object normal; rows in `benchmarks/2026-10-30/bands-oblique.jsonl`):
GPU 5.64 s (all k points converged, 54 iterations per k point) against 89.3 s on the 16 CPU threads. The CPU run
did not converge all k points in the iteration limit (65 iterations per k point), so the ratio of the times (15.8x)
compares unequal work; per iteration the GPU is 13x faster (6.5 ms against 85 ms per k point and iteration). Gap
edges 0.207036 and 0.274389 on both devices (the double-precision values at 128 of the check table). A first GPU
run (32.5 s) shared the GPU with another test run and is not used.

## Known limits

- The normal of the averaging: where several object boundaries cross a voxel (objects that touch or nearly
  touch), the fitted gradient is the normal; in a sheared voxel it has the bias of the fit there. A thin feature
  exactly centered in a voxel gets no normal (scalar averaging there).
- Above n = 128 the TE error of the sheared square lattice falls slowly with both normals (object normal: 2.98e-4 at
  128, 2.04e-4 at 192; `bench/normal_order.py`). The sheared TM bands, which do not use the normal, also converge
  irregularly at these grids (seen in the review). The cause is not known.
- Objects: cylinders (axis along z), spheres, blocks (edges along x, y, z), in Cartesian coordinates. An object
  that is infinite along b_i (for example a cylinder of infinite height when b_i has a z component) must not change
  along a_i: rods along z with a3 = z are correct; in an fcc cell, give the cylinder a finite height (else a
  ValueError).
- The eps sampling runs on the CPU. A strongly sheared cell needs more images: a sphere of r = 0.3 in the fcc
  primitive cell is tested at 125 lattice translations (27 in a cubic cell).
- In a 2-D cell, the lattice vector of the unused axis must be orthogonal to the other two; TE/TM need the lattice
  vectors 1 and 2 in the xy plane and k_z = 0 (else a ValueError).
- Warm start: in a homogeneous cell (the same eps_inv in all voxels) the operator and the preconditioner are
  diagonal in G, so a block of plane waves from the last k point never gains other plane waves, and converged at
  once to wrong bands (the triangular lattice, K to a general k). `solve_bands` therefore turns the warm start off
  there (test: the empty triangular lattice with the default warm start). A very weak contrast couples the plane
  waves only weakly; if bands look wrong there, use `warm_start=False`.
- No `find_k` (waveguide modes at fixed frequency) yet; no field output except `operator.fields_h`.
