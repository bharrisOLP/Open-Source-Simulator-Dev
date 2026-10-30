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

Waveguide modes at a fixed frequency (as MPB's `find-k`), and group velocities:

```python
import numpy as np
from bands_gpu.geometry import Block
from bands_gpu.findk import find_k, group_velocity

slab = Cell(size=(1, 5, 1), shape=(1, 1025, 1), eps_background=1.45**2,     # supercell along y, waveguide axis x
            objects=(Block(size=(np.inf, 0.5, np.inf), eps=3.5**2),))
res = find_k(slab, omega=0.5, band_min=1, band_max=1, kdir=(1, 0, 0), polarization="tm", parity="even")
res.kmag, res.group_velocity, res.kpoints              # |k| (units 2 pi / a), d omega / dk (units c), k (as MPB)
freqs, vg = group_velocity(cell, [0.3, 0.1, 0], num_bands=4, direction=(1, 0, 0), polarization="te")  # one k
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
- **find_k** (`bands_gpu/findk.py`). The k along a direction at which the bands band_min..band_max have the
  frequency omega. Units as MPB: omega in 2 pi c / a; `kdir` and `korig` in reciprocal-lattice coordinates; k =
  korig + kmag k_hat, with kmag the Cartesian length in units of 2 pi / a. Root finding: Newton on g(k) = f_b(k) -
  omega, one band solve per step, inside a bracket g(lo) < 0 < g(hi) that every solve narrows; a step out of the
  bracket (or a band that does not rise) becomes a bisection, and an end of [kmag_min, kmag_max] is solved only when
  a step needs it. Default kmag_max = 1.01 omega n_max + |korig| (no band reaches omega at |k| > 1.01 omega n_max
  along an axis without lattice periodicity, and |k| >= kmag - |korig|); along a periodic axis (a photonic-crystal
  waveguide), give the zone edge. Default tolerance of kmag (relative): 1e-8 in double precision, 1e-5 in single
  (the noise of single-precision bands). The bands are found from
  band_max down (as MPB); each solve gives all bands up to band_max, and these values bracket and start the lower
  bands. The eigenvector block of the last k starts the next solve (warm start; slab at 255 points, TE0 and TE1:
  99 eigensolver iterations against 150 without it (`test_no_root_and_warm_start`); line-defect waveguide:
  111 against 167 (`bench/bench_find_k.py --case pcw`, with and without `--cold`)).
- **Group velocity** (Hellmann-Feynman, as MPB): v_g = <H| dTheta/dk |H> / (2 omega), with <H| dTheta/dk |H> =
  2 Re <d x H| F eps_inv F^-1 |q x H> (d: the unit vector of the derivative). The derivative is taken on the
  operator on all 3-vector fields, whose nonzero eigenpairs are the transverse ones, so the k dependence of the
  transverse basis does not enter. Degenerate bands get the eigenvalues of the derivative matrix in their subspace,
  in ascending order (the right derivative of the sorted bands).
- **Parity** (`parity="even"` or `"odd"`, mirror plane through the cell center normal to `mirror_axis`): the
  definition of MPB, E(r) = +/- R E(R r) (E polar, R the mirror), so H(r) = -/+ R H(R r) (H a pseudovector). In a
  2-D cell: "tm" (E_z) even = E_z even (the TE0 slab mode); "te" (H_z) even = H_z odd (TM0 has H_z even, so it is
  "odd"); the TE-like mode of a strip along x is odd in y. The eigensolver works in the range of the projector
  P = (1 +/- M) / 2 on the plane-wave coefficients ((M c)(G) = T(G) c(R G), T from the transverse bases at G and
  R G). The mirror axis needs an odd grid: then G and R G are both in the basis and the discrete operator commutes
  with M (an even grid has a Nyquist plane without a mirror image; a ValueError). The structure must be symmetric
  at the subpixel samples: eps at each sample r and at its exact mirror point R r must be equal (else a ValueError;
  a shift smaller than the sample spacing that moves no sample across a face passes). The samples of the grid are
  mirror images only to rounding, so a face exactly on a sample can be inside on one side and outside on the other;
  eps_inv is therefore made symmetric, (eps_inv + R eps_inv R) / 2 on mirror voxels, which changes only such face
  voxels (`BandEvaluator.mirror_asymmetry` gives the largest change). Also a ValueError: a k with a component along
  the mirror axis, and a parity subspace with fewer dimensions than the eigensolver block (num_bands + extra), for
  example "tm" with the z mirror (a 2-D TM field is odd in z) or too few grid points; without this check the solve
  returned frequencies 0 as converged.

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

## Checks of find_k (`tests/test_findk.py`, 12 tests, 30 s on the GPU computer)

| Case | Reference | Agreement |
|---|---|---|
| homogeneous eps 2.25, omega 0.2: 2-D TE and TM (kdir (1, 0.5)), triangular cell (kdir b2), 3-D (the degenerate pair), start 0.05 to 0.12 | k = n omega, v_g = 1 / n, exact | 4.4e-16 (2 solves each) |
| slab n 3.5 / 1.45, d 0.5 a, period 5 a, omega 0.5, 1025 points, with parity: TE0, TE1 (E_z), TM0, TM1 (H_z) | the exact supercell at Bloch K = 0 (transfer matrix) | kmag 1.17e-5, 4.38e-5, 2.72e-5, 1.34e-5 (relative); 3 to 6 solves |
| the same | the isolated slab (transcendental equations) | 1.17e-5, 4.38e-5, 2.72e-5, 4.18e-5 |
| the same, group velocity (Hellmann-Feynman) | the slope of the exact supercell dispersion | 1.0e-5, 6.0e-5, 1.1e-5, 1.0e-4 (relative) |
| the same, bands 1 and 2 without parity | the two parity results | 2.8e-16 (TE), 1.5e-16 (TM) |
| consistency: solve_bands (no parity) at the found k | omega | 2.2e-16 to 6.7e-16 |
| v_g of rods (eps 12, r 0.2 a, 32 x 32), TE and TM, 4 bands at k = (0.3, 0.1), d = (0.6, 0.8); spheres (eps 12, r 0.3 a, 10^3), 4 bands | central differences of omega(k), Richardson (h 2e-3, 1e-3) | 2.6e-11 (TE), 4.1e-11 (TM), 2.2e-11 (3-D) |
| line-defect waveguide: rods (eps 12, r 0.2 a), one row removed, 1 x 7 supercell, 15 x 105, TM, omega 0.34, even band 4, kmag_max 0.5 | solve_bands band 7 at the found k (0.189185) | 3.9e-16 |
| the same | an independent finite-difference TM solver (`tests/fd_tm_reference.py`) at the found k, 64 and 128 points per a | omega 2.8e-4 (the FD values at 64 and 128 agree to 1e-6) |
| the same, v_g = 0.39839 | central difference (h 1e-3) | 9.0e-7 |
| strip (n 3.48, 0.5 x 0.22, in n 1.444), cell 1 x 3 x 2, grid 1 x 25 x 15, 3-D vectors (T a 2 x 2 matrix), k 1.5, 6 bands; mirror y and mirror z | the bands without parity (even and odd together) | 1.4e-15 (y), 1.6e-15 (z); band 1 (TE-like) odd in y, even in z |
| the same, v_g of each parity | central difference (h 1e-4) | 1.1e-8 |
| slab with its faces exactly on subpixel samples (d 0.325 to 0.675, 25 points, sub 8), even and odd | the bands without parity of the same (symmetrized) eps_inv | 8.9e-16; 2 voxels changed, by 0.049 to 0.179 |
| homogeneous, korig (-0.1, 0, 0), default kmag_max | kmag 0.4, exact | 2.2e-16 |
| slab, 255 points, TE0, single precision, default tolerance | double precision | kmag 1.0e-7; 3 solves |

The supercell couples the slab to its images: the exact supercell and the isolated slab differ by less than 2e-11
for TE0, TE1 and TM0, and by 2.84e-5 for TM1 (n_eff 1.588, near the cladding index 1.45; it decays slowly). The
derivative of the discrete operator is exact (the Richardson check, 2e-11); the errors of v_g against the exact
slab are the discretization error of the band solver.

Discretization error of the slab (`bench/bench_find_k.py --case slab --n N --sub S`, double precision, on the GPU
computer; the rows are in `benchmarks/2026-10-30/bands-findk.jsonl`). Relative error of kmag against the exact
supercell, TE0, TE1, TM0, TM1, with `--sub 256`:

| N (points in 5 a) | TE0 | TE1 | TM0 | TM1 |
|---|---|---|---|---|
| 257 | -7.29e-5 | -1.75e-4 | -2.37e-4 | -2.00e-5 |
| 513 | -7.51e-5 | -2.92e-4 | -1.86e-4 | -1.14e-4 |
| 1025 | -1.17e-5 | -4.38e-5 | -2.72e-5 | -1.34e-5 |
| 2049 | -5.74e-6 | -2.38e-5 | -1.36e-5 | -1.02e-5 |
| 4097 | +7.16e-7 | +5.31e-6 | +1.87e-6 | +4.68e-6 |

Most of the irregular convergence comes from the subpixel samples, not from the averaging formula. The fill fraction
of a face voxel is a multiple of 1/sub, so a flat face at a general position moves by up to dx / (2 sub): with the
default `sub = 8`, a flat face is placed only to 1/16 of a voxel. The bench gives the position of the face in its
voxel, the error of the slab thickness that the samples see, and the error against the exact supercell with that
sampled thickness (largest of the 4 modes):

| N | face in voxel | sub | sampled thickness - d (a) | error against the exact supercell | error against the exact supercell of the sampled thickness |
|---|---|---|---|---|---|
| 257 | 0.85 | 256 | +6.1e-5 | 2.4e-4 | 2.8e-4 |
| 513 | 0.65 | 256 | -3.05e-5 | 2.9e-4 | 2.4e-4 |
| 1025 | 0.25 | 8, 256 | 0 | 4.4e-5 | 4.4e-5 |
| 2049 | 0.45 | 8 | +2.4e-4 | 4.0e-4 | 1.8e-5 |
| 2049 | 0.45 | 256 | -3.8e-6 | 2.4e-5 | 1.7e-5 |
| 2049 | 0.45 | 2048 | +9.5e-7 | 1.6e-5 | 1.7e-5 |
| 4097 | 0.85 | 8 | +6.1e-5 | 1.0e-4 | 6.9e-7 |
| 4097 | 0.85 | 256 | +3.8e-6 | 5.3e-6 | 1.1e-6 |
| 4097 | 0.85 | 2048 | +2.4e-7 | 7.7e-7 | 1.2e-6 |

At 1025 points the face is at exactly 1/4 of a voxel, so 8 samples give the exact fill fraction (sub 8 and sub 256
agree to 10 digits); that is a property of this grid, not of sub 8. With sub 8, the errors at 2049 and 4097 points
are larger than at 1025 and have the other sign: the slab is 2.4e-4 a and 6.1e-5 a too thick. Against the exact
supercell of the sampled thickness, the error does not depend on sub (2049: 1.7e-5 to 1.8e-5; 4097: 6.9e-7 to
1.2e-6) and falls from 2.8e-4 (257) to 1.2e-6 (4097), 240x over 16x in N (a second order gives 256x), but not
evenly at each step (the coefficient changes with the position of the face in its voxel). For a flat face at a
general position, use a larger sub, or put the face at the center or the edge of a voxel.

Small bench cases of find_k (`bench/bench_find_k.py`, one JSON line per case, on the RX 7900 XT, device rocm:0;
double precision unless noted; the rows are in `benchmarks/2026-10-30/bands-findk.jsonl`). The double-precision
results are the same as on the CPU to the digits shown.

| Case | Result | Solves | Eigensolver iterations |
|---|---|---|---|
| `--case slab --n 1025` (4 modes) | the errors above | 3, 3, 4, 6 | 55, 53, 79, 121 |
| `--case pcw --n 15` | kmag 0.189185, v_g 0.398385 | 4 | 111 |
| `--case pcw --n 15 --cold` (no warm start) | kmag the same to 3e-15, v_g to 4e-11 | 4 | 167 |
| `--case pcw --n 15 --precision single` | kmag 0.18919, v_g 0.39840 | 3 | 46 |
| `--case strip --n 16 --precision single` (Si 0.5 x 0.22 in silica, wavelength 1.55, 3-D vectors, 49 x 32) | n_eff 2.4315, group index 4.020 | 2 | 26 |

These cases are small (1-D, 15 x 105, 49 x 32 points): the GPU is not loaded, and the times in the rows are not a
speed-up measurement. The strip case has no reference value (a coarse grid; a timing case).

Larger cases (`benchmarks/2026-10-30/bands-findk.jsonl`; time of the second call, single precision unless noted):

| Case | GPU (RX 7900 XT) | CPU (16 threads, `BANDS_GPU_DEVICE=cpu`) | Result |
|---|---|---|---|
| `--case strip --n 64` | 0.28 s | 1.71 s (6.1x) | n_eff 2.4443, group index 4.050 (27 iterations on both) |
| `--case pcw --n 31` | 0.58 s (80 iterations) | 0.93 s (56 iterations) | kmag 0.18975 on both (unequal iterations: 2.3x per iteration) |
| `--case slab --n 4097` (default sub 8), double precision | 4.05 s | not run | errors +1.6e-5 to +1.0e-4 (the slab is 6.1e-5 a too thick at sub 8; see the sub table above) |
| `--case slab --n 4097 --sub 256`, double precision | 4.25 s | not run | errors +7.2e-7 to +5.3e-6 |

The strip value moves from n_eff 2.4315 at n 16 to 2.4443 at n 64 (grid convergence, no reference).

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
- No field output except `operator.fields_h`.
- find_k finds one root per band in [kmag_min, kmag_max]. The bracket assumes that the band rises with k at the
  root (a forward mode); a band with several roots there (a folded photonic-crystal band) gives one of them, so give
  a range with one root. Roots of supercell bands of the cladding are also found: a mode is guided only if
  kmag > n_clad omega. Band solves that do not reach the eigensolver tolerance are counted in
  `unconverged_solves`. kmag is NaN (and `converged` False) when the band does not cross omega in the range, or
  when the search did not converge in `max_steps` (default 40) solves.
- A backward mode (a band that falls with k at the root, for example the folded band of a photonic-crystal
  waveguide past its maximum) gives NaN: the bracket needs g < 0 below the root and g > 0 above it (from the
  code; no test).
- Parity: one mirror plane at a time, through the cell center, normal to a lattice vector along a Cartesian axis,
  with an odd number of grid points on that axis. The symmetry check is at the resolution of the subpixel samples:
  a structure that is asymmetric by less than the sample spacing, and moves no sample across a face, passes.
- The subpixel samples place a flat face to dx / (2 sub) only (1/16 of a voxel with the default sub 8). For
  converged waveguide numbers, use a larger sub, or put the faces at voxel centers or edges (the slab table).
- Degenerate bands: bands closer than 1e-9 (relative, in omega^2; 1e-5 in single precision) count as one cluster
  for the group velocity. Bands that are nearly degenerate but not in a cluster get the Hellmann-Feynman value of
  each vector, which is correct only for a separated band.
