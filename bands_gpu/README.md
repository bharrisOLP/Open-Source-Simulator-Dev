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

## Method

- **Basis and operator.** H(r) = sum_G h(G) exp(i (k+G).r) with h(G) orthogonal to k+G (two coefficients per G
  in 3-D; one per G for the TE or TM polarization of a 2-D cell). Theta h = -q x FFT[eps_inv . IFFT[q x h]]:
  six FFTs per product.
- **eps.** Supersampled in each voxel; at an interface eps_inv = <1/eps> n n^T + (1/<eps>)(I - n n^T) (the
  averaging of MPB and Meep; Kottke et al. 2008), with n from the first moment of eps in the voxel.
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

## GPU timings (RX 7900 XT against the same algorithm on the 16 CPU threads of the same PC)

Single precision, full k paths (`bench/bench_bands.py`). These runs used the solver before the A P fix (more
iterations, the same on both devices): the speed-ups hold, the absolute times are now lower.

| Case | Speed-up of the GPU |
|---|---|
| rods, 64 x 64, TM, 16 k points | 4.8x |
| rods, 128 x 128 | 11x |
| spheres (simple cubic, r 0.3 a, eps 12), 32^3, both polarizations, 13 k points | 12x |
| spheres, 48^3 | 15.6x |

## Known limits

- The normal of the averaging comes from the first moment of eps in the voxel: a thin feature centered in a voxel
  gets no normal (scalar averaging there).
- Rectangular cells only (no oblique lattices yet). Objects: cylinders, spheres, blocks.
- No `find_k` (waveguide modes at fixed frequency) yet; no field output except `operator.fields_h`.
