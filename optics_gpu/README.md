# optics-gpu

Physical optics through lens systems for photonics (the coupling use case of ZEMAX POP: a waveguide or fiber
mode, through lenses, into a fiber), on the GPU with PyTorch (ROCm or CUDA) and real ray tracing with
[Optiland](https://github.com/HarrisonKramer/optiland) (MIT).

```python
from optics_gpu.pop import gaussian, overlap_efficiency, propagate_through

img = propagate_through(optic, source_field, dx_um, wavelength_um, x_img_um, y_img_um)   # optic: an Optiland Optic
eta = overlap_efficiency(img.E, gaussian(x_img_um, y_img_um, 5.2))                      # into SMF-28 (MFD 10.4 um)
img = propagate_through(..., trace_device="cuda")    # all steps on the GPU (the device pipeline, float64)

from optics_gpu.pop import image_planes, image_sum, lateral_scan, overlap_efficiency_device, trace_bundle
bundle = trace_bundle(optic, source_field, dx_um, wavelength_um, device="cuda")   # trace one time
E = image_sum(bundle, x_img_um, y_img_um, dz=20.0)       # the field 20 um after the image surface, on the GPU
stack = image_planes(bundle, x_img_um, y_img_um, np.linspace(-60, 60, 25))      # 25 planes in one pass (Z, Nx, Ny)
eta_z = overlap_efficiency_device(stack, fiber_mode, power=p0)                  # one eta per plane
stack = image_planes(bundle, x_img_um, y_img_um, dzs, method="nufft")  # uniform grid: non-uniform FFT, tol 1e-12

from optics_gpu.analysis import fft_mtf, fft_psf, huygens_psf, pupil_wavefront, zernike_fit
wf = pupil_wavefront(optic, field=(0.0, 0.0), n_pupil=256, device="cuda")   # OPD map (waves), 256 x 256 pupil grid
fit = zernike_fit(wf, 36, "fringe")      # or "noll"; fit.coeffs in waves
psf = fft_psf(wf, grid_size=1024)        # psf.psf: peak 1 for a perfect pupil (Strehl units); psf.dx in um
mtf = fft_mtf(wf, psf)                   # mtf.tangential, mtf.sagittal against mtf.freq_tan, mtf.freq_sag (cycles/mm)
I = huygens_psf(optic, 1.55, x_img_um, y_img_um, device="cuda")   # the second PSF method: the image sum

from optics_gpu.analysis import image_quality       # fields x wavelengths x planes in one batch ("Batches")
res = image_quality(optic, [(0, 0), (0, 1)], [0.4861, 0.5876, 0.6563], dz=[-50, 0, 50], weights=[1, 2, 1])
```

## Method (`src/optics_gpu/pop.py`)

1. Angular spectrum of the source (FFT): one direction (L, M) per sample.
2. One real ray per direction from the object point, traced by Optiland to the image surface: image-side
   directions, positions and optical path W. The trace runs on Optiland's NumPy backend (the default,
   `trace_device=None`) or on its PyTorch backend on `"cpu"` or `"cuda"` (see "Ray trace on the GPU").
3. Image field E(x, y) = sum_r A_r sqrt(J_r) df exp(i k (W_r + L'_r (x - x_r) + M'_r (y - y_r))), with
   J = |d(L', M')/d(L, M)| (power conservation per ray tube). The sum is separable in x and y: one matrix
   product on the GPU, in chunks of rays (`ray_chunk`; by default (Nx + Ny) x chunk = 2^25). The sum keeps one
   chunk of the x and y factors in memory: (Nx + Ny) x chunk complex128 values, 512 MB by default. This is the
   sum of the NumPy pipeline; the device pipeline uses `image_planes` (see "Focus scans in one pass").
4. Coupling efficiency by the normalized overlap with the receiving mode.

Valid for a source that is small compared with the isoplanatic patch of the system (true for micrometer modes).
Scalar fields; no obliquity factor (moderate NA); object and image spaces are air. **Give the stop a physical
aperture** (`RadialAperture`): the EPD setting of Optiland aims its ray generator but does not clip rays, and
without a physical aperture a point source overfills the lens (found in the singlet check: correlation 0.30 without it, 0.998 with it).

## Ray trace on the GPU (`trace_rays`, `optiland_torch` in `src/optics_gpu/pop.py`)

Optiland 0.6.3 keeps one global backend (`optiland.backend.set_backend("numpy" | "torch")`). Its PyTorch backend
has a global device (`set_device("cpu" | "cuda" | "mps")`) and precision (`set_precision`, default float32). An
`Optic` keeps the arrays of the backend that was active when it was made: a NumPy-built optic does not trace under
the PyTorch backend (TypeError, ndarray times Tensor). Thus, for `trace_device="cpu"` or `"cuda"`, `trace_rays`:

1. sets the PyTorch backend, the device and float64 for the duration of the trace, and sets the earlier backend,
   device and precision again after it (also after an error). PyTorch on ROCm reports an AMD GPU as `"cuda"`.
   Apple `"mps"` is refused: it has no float64. Float32 is not used: its relative step (6e-8) on an optical path
   of 40 mm is 2.4e-6 mm, that is 0.01 rad of phase at 1.55 um for each rounding.
2. makes a copy of the optic under that backend (`Optic.to_dict`, then `Optic.from_dict`), so the caller keeps
   a NumPy optic;
3. traces up to 25 of the rays through the original and through the copy, and stops with a RuntimeError if they
   differ by more than 1e-9 (mm or direction cosine), for example for a surface that the dictionary does not keep.
   The original goes through the NumPy backend if it keeps NumPy arrays. An optic made under Optiland's PyTorch
   backend in float32 gives float32 check rays: a ValueError names that cause;
4. traces all rays through the copy (`torch.no_grad`). `trace_rays` returns the Optiland rays with tensors on the
   device; `torch_tracer` gives the same launcher for many calls (one copy and one check).

`ImageField.trace_device` gives the device of the trace (`"numpy"`, `"cpu"` or `"cuda:0"`).

## The device pipeline (`trace_bundle`, `image_sum` in `src/optics_gpu/pop.py`)

`propagate_through` has two pipelines. `trace_device=None` is the NumPy reference (unchanged): angular spectrum,
trace, Jacobian and weights in NumPy, then the image sum on the GPU. `trace_device="cuda"` (or `"cpu"`) is the
device pipeline: all steps run on that device in float64, and only the image field comes back to the host.

1. Angular spectrum with `torch.fft` (the same formulas and order of operations as the NumPy version). The
   bounding box of the kept directions needs 4 scalars on the host.
2. Trace, Jacobian and weights in blocks of rows of the direction grid, about `trace_chunk` rays each (default
   `TRACE_RAYS` = 2^21). Each block traces one more row on each side (a halo), so that its central differences
   are those of the full grid. `_np_gradient0` uses the formulas of `np.gradient` (edge order 1; NumPy takes the
   `fftshift(fftfreq)` axis as non-uniform, because its steps differ in the last bits). The start values of the
   rays are made on the device (no copy from the host).
3. Only the kept rays stay in memory: a `RayBundle` of L', M', N' and the weights a (40 bytes per ray), in blocks.
   `RayBundle.power` is the power of the kept rays by Parseval (sum of |A df|^2 / df, in um^2 times the field
   squared): it does not depend on an image grid or on dz.
4. `image_sum(bundle, x, y, dz)`: E = sum_r a_r exp(i k (L'_r x + M'_r y + N'_r dz)). One plane wave per ray, so
   `dz` is the angular-spectrum propagation in image space: a focus scan needs one trace. `image_planes(bundle, x,
   y, dzs)` computes all planes of a focus scan in one pass over the rays (see "Focus scans in one pass");
   `image_sum` is one plane of it. `overlap_efficiency_device` takes the stack (Z, Nx, Ny) and gives one eta per
   plane. `lateral_scan` gives the coupling into a Gaussian mode moved to every point of a 2-D grid of lateral
   offsets with two matrix products (the Gaussian is separable).

The data that come to the host inside the device pipeline are the 25 check rays of the copy of the optic and
scalars: the bounding box, and about 50 scalars per block from Optiland's own tests in its trace (observed).

In a focus scan, use one power for all planes (`power=` of `overlap_efficiency_device` and `lateral_scan`). The
power on a finite image grid changes with dz: wide-angle power leaves the grid, and power that is off the grid at
the image surface comes onto it. The FFT method of `examples/chip_to_fiber_coupling.py` keeps the power of its
periodic grid. In the example below, eta normalized by the power on the grid at each plane differed from the FFT
method by 7.8e-4 at dz = 60 um; normalized by p0, the power on the grid at the image surface (as the FFT method),
by 5.9e-11. That agreement shows that the two propagation methods agree with the same normalization. It does not
show that p0 is the total power (see the example).

## Focus scans in one pass (`image_planes` in `src/optics_gpu/pop.py`)

`image_planes(bundle, x, y, dzs, ray_chunk=None, dtype=torch.complex128)` gives the fields of Z planes, a (Z, Nx, Ny)
complex128 tensor, in one pass over the rays. `image_sum` is one plane of it. For each chunk of rays:

1. The weights of each plane, w_zr = a_r exp(i (k L'_r xc + k M'_r yc + k N'_r dz_z)), in float64. xc and yc are
   the centers of the grid axes (the phase reference). The large phases (k N' dz, and k W in a) stay in float64.
2. The grid factors exp(i k L'_r (x - xc)) (Nx x chunk) and exp(i k M'_r (y - yc)) (Ny x chunk), in `dtype`.
3. The x factors times the weights of all planes make one (Z Nx x chunk) matrix. Its product with the y factors
   sums groups of `GROUP_RAYS` = 1024 rays, and the groups are added in complex128. For Z Nx Ny below
   `GROUP_OUTPUTS` = 2^20, one batched matrix product computes all groups of the chunk. For a larger output, the
   matrix product has enough parallel work: complex128 adds the whole chunk into E in place (`addmm_`, float64
   needs no groups), and complex64 adds one product per group.

The arithmetic of the matrix products is the same as for one sum per plane (8 Nx Ny floating-point operations
per ray and plane). The gain is that the GPU gets more parallel work: Z Nx rows instead of Nx, and one matrix per
group. The earlier sum gave only a 128 x 128 output per product for the GPU, and one long sum over all rays of the
chunk. The grid factors are computed one time for all planes. In complex128 the batched sum does 0.92e12
operations per second (the 1e6-ray rows below: 5.28e5 rays x 8 x 128^2 in 0.075 s per plane); complex64 is 8.5x
faster.

`dtype=torch.complex64` does steps 2 and 3 in float32. `PHASE32_MAX` = 256 rad limits the largest grid phase
theta = k max|L'| max|x - xc| (and the same in y): above it, a ValueError stops the sum (this check reads one
scalar from the GPU). The guard limits the phase, not the error. Two errors come from float32 (u = 2^-24):
- the rounding of the grid phases: at most about 3 theta u for the x factor and the same for the y factor;
- the float32 sum of a group of n rays: at most about n u of the sum of |terms|.
To first order, |E64 - E128| <= about (`GROUP_RAYS` + 6 `PHASE32_MAX`) u sum_r |a_r| = 1.5e-4 sum_r |a_r|. The
size of the error depends on the field. Rounding errors of random sign stay far below the bound. One plane wave
(2e4 equal rays at a grid phase of 243 rad, all with the same rounding errors) gave 3.3e-5 of sum_r |a_r| on the
GPU (`test_complex64_error_of_one_plane_wave_is_within_the_bound`).

For the tested focused fields, the coupling efficiency with complex64 differs from complex128 by at most 3.3e-7
(a uniform pupil of NA 0.6 at a grid phase of 243 rad, with one power for all planes, as in a focus scan), and by
8.3e-8 or less in the other checks. The bench differs by 6.4e-9 at 1e7 rays, and the focus scan of the example by
9.1e-8 (row 2 of `benchmarks/2026-11-07/optics-sym-x.jsonl`). These values hold only for these fields. The default
stays complex128.

The default chunk keeps the temporaries of one chunk (the weights, the factors and the products) at most
`SUM_ELEMENTS` complex128 values (512 MB). `_planes_memory` gives the model: per ray, the largest of the three
steps of a chunk (the weights need 1.5 Z complex128 values per ray before they are cast to `dtype`); fixed, the sum
of the groups (Z Nx Ny complex128 values) or, for complex64 without batched groups, the product of one group; and
6 MB for the rounding of the PyTorch caching allocator. The result E (Z Nx Ny complex128 values) is outside that
budget. Exception: for complex64 with E above 1 GB, the product of one group alone is above the budget; then the
chunk is `GROUP_RAYS` rays, and the peak is above the budget.

## The non-uniform FFT (`image_planes(..., method="nufft")` in `src/optics_gpu/pop.py`)

`image_planes` and `image_sum` take `method="nufft"` and a tolerance `tol` (default 1e-12, from 1e-12 to 1e-2).
On a uniform grid x_i = x_0 + i h_x, y_j = y_0 + j h_y the image sum is a type-1 non-uniform FFT:

    E_z(m, n) = sum_r c_zr exp(i 2 pi (m f_r + n g_r)),   f_r = L'_r h_x / lambda,   g_r = M'_r h_y / lambda,
    c_zr = a_r exp(i k (L'_r x_ref + M'_r y_ref + N'_r dz_z)),

with the modes m = i - Nx // 2 and n = j - Ny // 2 counted from the grid point (x_ref, y_ref), and the weights c in
float64 (the phase reference, as in the direct sum). f_r and g_r are cycles per grid step. On the grid points,
exp(i 2 pi m f) is periodic in f with period 1, so only f mod 1 matters: a grid that does not resolve the rays
(more than 0.5 cycles per step) gives the same aliased values as the direct sum.

1. Spread: each ray adds c_zr phi_x phi_y to w x w points of a periodic fine grid of n1 x n2 points (n1 the
   smallest even 2^a 3^b 5^c of at least 2 Nx and 2 w; the same for n2), at the position (f_r mod 1) n1,
   (g_r mod 1) n2 in fine-grid cells. The kernel is the "exponential of semicircle" (ES)
   phi(z) = exp(beta (sqrt(1 - z^2) - 1)), |z| <= 1, with beta = 2.30 w (Barnett, Magland and af Klinteberg, SIAM
   J. Sci. Comput. 41 (2019) C479). The adds use `index_add_` in complex128.
2. One inverse FFT (unnormalized) per plane.
3. Deconvolution: each mode is divided by psi_x(m) psi_y(n), the Fourier transform of the kernel (Gauss-Legendre
   quadrature with 128 nodes in t, with z = sin t: then the integrand is smooth; in z, the square-root ends of phi
   make the rule converge slowly). Against mpmath, psi agrees to 6.4e-15 (test).

The cost per ray and plane is w^2 complex additions (w = 15 for tol 1e-12, 9 for 1e-6), against 8 Nx Ny
floating-point operations for the direct sum; then one FFT of n1 x n2 points per plane. The kernel values (2 w
per ray) are computed one time for each batch of planes.

The kernel width is `nufft_width(tol)`: w = ceil(log10(1 / tol)) + 3, two points more than the rule of the
paper, which is for a relative l2 error. Here tol is a bound of the largest error relative to sum_r |a_r|, the
size of the field of all rays in phase. The worst case is one ray at the worst position in its fine-grid cell on
both axes, on a grid whose fine grid is exactly 2 Nx (the edge modes reach |m| / n1 = 1/4, and the x and y errors
add). In that case on a 128 x 128 grid (test), the width of the paper gives 14.3 tol (tol 1e-12) and 14.6 tol
(1e-9), the width plus 1 gives 1.89 tol and 1.68 tol, and this width gives 0.26 tol and 0.16 tol (0.10 tol at
1e-6, 0.08 tol at 1e-2). For many rays the tested errors are 0.014 tol or less (Checks below). The range stops at
1e-12, near the float64 rounding of the sum. Relative to the largest |E|, the error is larger by
sum_r |a_r| / max |E|: about 1 at a focus, more out of focus.

The float64 phase reference limits both methods: each ray has an error of about eps |k (L'_r x_ref + M'_r y_ref +
N'_r dz)| (eps = 2.2e-16), so far from the axis or at a large dz the error per ray can be above tol 1e-12 (for
example, a phase of 1e4 rad gives about 2e-12), for the direct sum as for the non-uniform FFT.

The GPU does many atomic adds into the same fine-grid points: for a small grid and a small NA all rays fall into
a few hundred points. Two measures spread these adds:
- The rays go in the order r s mod n (s coprime with n, near 0.618 n), so rays that are next to each other in
  that order are far apart on the direction grid.
- The spread adds into up to `NUFFT_COPIES` = 16 copies of the fine grid (the rays in turn). The copies are added
  before the FFT.
The order of the atomic adds changes from run to run, so two runs on the GPU can differ by rounding.

Memory: the planes go in batches of zb planes, with 6 n1 n2 zb at most `SUM_ELEMENTS` (the FFT step needs a few
times the fine grids of the batch); the copies use at most `SUM_ELEMENTS` / 8; the chunk of rays fills the rest
(w^2 (zb + 2) + 6 w + 3 zb complex128 values per ray). A test checks the peak against the budget. The result E is
outside the budget, as for the direct sum. Exception: if 6 n1 n2 is above `SUM_ELEMENTS` (an image grid above about
1150 x 1150 points), a batch is one plane, the chunk is at least `NUFFT_MIN_CHUNK` = 4096 rays, and the peak is
above the budget: the fine grid of one plane, its FFT step, and 4096 times the values per ray. From about
2800 x 2800 points the fine grid alone is above the budget; before the one review the chunk then fell to one ray.

Limits: the grid must be uniform on each axis (|x_i - (x_0 + i h)| <= 64 eps max|x|, so `np.arange` and
`np.linspace` grids pass); for another grid, `image_planes` gives a UserWarning and uses the direct sum. The
non-uniform FFT computes in complex128 only (`dtype=torch.complex64` with `method="nufft"` gives a ValueError).

**Timings.** GPU (RX 7900 XT; `.venv-optics`, PyTorch 2.14.1 for ROCm 7.2; with the shared limits of the
computer: `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` 4). Rows 1 to 24 of
`benchmarks/2026-11-07/optics-nufft.jsonl` (kernel widths w 15, 12 and 9 for tol 1e-12, 1e-9 and 1e-6), from the
repository folder on the GPU computer:

    bash tools/gpu_run.sh optics_gpu python bench/bench_nufft.py --rays 1e5 1e6 1e7 \
        --image 128 256 512 1024 --planes 1 21

A point source through the singlet (NA 0.09), image grid pitch 1 um, planes from dz = -100 to +100 um. Times are
per plane: `image_planes` with the direct sum (complex128, complex64) and with the non-uniform FFT on all rays,
best of 2 runs. \* = `direct_scaled`: the direct sum was timed on every s-th ray (`direct_rays_timed`, 8835 to
2.65e6 rays) and scaled to all rays (its time is linear in the rays); 13 of the 24 rows. The last column is
`speedup_c128_1e-12` / `speedup_c64_1e-12` (direct time over the time of the non-uniform FFT at tol 1e-12).

| Rays traced (kept) | Grid | Planes | Direct complex128 | Direct complex64 | NUFFT tol 1e-12 | NUFFT 1e-9 | NUFFT 1e-6 | NUFFT faster than complex128 / complex64 |
|---|---|---|---|---|---|---|---|---|
| 1e5 (5.3e4) | 128^2 | 1 | 11.4 ms | 2.68 ms | 8.64 ms | 6.61 ms | 5.85 ms | 1.32x / 0.31x |
| 1e5 (5.3e4) | 128^2 | 21 | 7.53 ms | 0.846 ms | 1.13 ms | 0.811 ms | 0.549 ms | 6.65x / 0.75x |
| 1e5 (5.3e4) | 256^2 | 1 | 36.3 ms | 4.38 ms | 8.06 ms | 6.53 ms | 5.79 ms | 4.50x / 0.54x |
| 1e5 (5.3e4) | 256^2 | 21 | 29.3 ms | 3.23 ms | 1.20 ms | 0.868 ms | 0.607 ms | 24.33x / 2.69x |
| 1e5 (5.3e4) | 512^2 | 1 | 130 ms | 14.0 ms | 8.80 ms | 7.40 ms | 6.70 ms | 14.81x / 1.59x |
| 1e5 (5.3e4) | 512^2 | 21 | 116 ms\* | 12.5 ms\* | 1.95 ms | 1.56 ms | 1.21 ms | 59.24x / 6.38x |
| 1e5 (5.3e4) | 1024^2 | 1 | 499 ms | 54.4 ms | 12.5 ms | 11.3 ms | 10.5 ms | 39.74x / 4.33x |
| 1e5 (5.3e4) | 1024^2 | 21 | 461 ms\* | 49.0 ms\* | 4.12 ms | 3.43 ms | 2.85 ms | 111.87x / 11.89x |
| 1e6 (5.28e5) | 128^2 | 1 | 108 ms | 13.5 ms | 31.2 ms | 22.4 ms | 15.2 ms | 3.47x / 0.43x |
| 1e6 (5.28e5) | 128^2 | 21 | 74.9 ms | 8.76 ms | 9.90 ms | 6.61 ms | 3.94 ms | 7.57x / 0.88x |
| 1e6 (5.28e5) | 256^2 | 1 | 358 ms | 43.6 ms | 26.3 ms | 18.8 ms | 12.9 ms | 13.63x / 1.66x |
| 1e6 (5.28e5) | 256^2 | 21 | 292 ms\* | 32.4 ms\* | 9.97 ms | 6.60 ms | 3.92 ms | 29.26x / 3.25x |
| 1e6 (5.28e5) | 512^2 | 1 | 1.29 s | 146 ms | 26.3 ms | 19.0 ms | 13.4 ms | 49.08x / 5.56x |
| 1e6 (5.28e5) | 512^2 | 21 | 1.15 s\* | 125 ms\* | 13.4 ms | 9.10 ms | 5.74 ms | 86.37x / 9.36x |
| 1e6 (5.28e5) | 1024^2 | 1 | 4.96 s\* | 538 ms\* | 29.2 ms | 21.9 ms | 16.3 ms | 169.78x / 18.38x |
| 1e6 (5.28e5) | 1024^2 | 21 | 4.59 s\* | 487 ms\* | 23.7 ms | 16.3 ms | 10.7 ms | 193.53x / 20.53x |
| 1e7 (5.3e6) | 128^2 | 1 | 1.08 s | 140 ms | 330 ms | 238 ms | 165 ms | 3.28x / 0.43x |
| 1e7 (5.3e6) | 128^2 | 21 | 752 ms\* | 88.7 ms\* | 104 ms | 71.7 ms | 44.8 ms | 7.22x / 0.85x |
| 1e7 (5.3e6) | 256^2 | 1 | 3.59 s\* | 433 ms\* | 262 ms | 179 ms | 117 ms | 13.72x / 1.65x |
| 1e7 (5.3e6) | 256^2 | 21 | 2.93 s\* | 326 ms\* | 113 ms | 77.4 ms | 47.4 ms | 25.90x / 2.89x |
| 1e7 (5.3e6) | 512^2 | 1 | 13.0 s\* | 1.48 s\* | 259 ms | 176 ms | 111 ms | 50.03x / 5.70x |
| 1e7 (5.3e6) | 512^2 | 21 | 11.6 s\* | 1.26 s\* | 167 ms | 113 ms | 68.5 ms | 69.43x / 7.56x |
| 1e7 (5.3e6) | 1024^2 | 1 | 49.9 s\* | 5.35 s\* | 259 ms | 177 ms | 112 ms | 192.34x / 20.63x |
| 1e7 (5.3e6) | 1024^2 | 21 | 46.1 s\* | 4.98 s\* | 255 ms | 173 ms | 108 ms | 180.51x / 19.50x |

At tol 1e-12 the non-uniform FFT is faster than the direct complex128 sum in all 24 rows (1.32x to 194x). It is
slower than the direct complex64 sum on all 128^2 rows (0.31x to 0.88x) and on 256^2 with one plane at 1e5 rays
(0.54x); it is faster in the other 17 rows (1.59x to 20.6x). On the rays of the direct timing, the largest field
difference from the direct complex128 sum is 1.8e-14 of sum_r |a_r| at tol 1e-12, 2.7e-12 at 1e-9 and 2.2e-9 at
1e-6 (`err_*`); the largest coupling difference into a Gaussian mode of radius 8 um is 2.5e-15, 5.2e-13 and
2.4e-10 (`d_eta_*`).

The example (row 25 of the same file; 10877 rays, 25 planes, 128 x 128 points):

    bash tools/gpu_run.sh optics_gpu python examples/coupling_scans_on_device.py \
        --mode-file ../results/optics_gpu/sin_tip_mode_sym_x.npz

Its focus scan takes 0.362 ms per plane with the non-uniform FFT, 1.64 ms with the direct complex128 sum (4.5x
slower) and 0.175 ms with complex64 (the non-uniform FFT is 2.1x slower than complex64); eta differs from
complex128 by at most 2.2e-14.

**The default stays `method="direct"`.** A non-uniform FFT default needs a rule for `dtype=torch.complex64`
(see Next).

## Image quality: OPD, Zernike, PSF and MTF (`src/optics_gpu/analysis.py`)

The analyses of lens-design tools for one field point and one wavelength, on one PyTorch device in float64
(`device="cuda"` or `"cpu"`: the same code). Optiland's PyTorch backend traces a checked copy of the optic (as in
"Ray trace on the GPU"; 6 check rays against the original), with Optiland's ray aiming: one ray per pupil point.

1. Pupil grid: N x N points `np.linspace(-1, 1, N)` of the normalized entrance-pupil coordinates (Px, Py); the
   points with Px^2 + Py^2 <= 1 are traced (51040 rays for N = 256), in blocks of `trace_chunk` rays. The arrays
   are indexed [ix, iy], as the image fields of this package (Optiland: [iy, ix]).
2. OPD map (`pupil_wavefront`): the reference sphere has its center at the image point of the chief ray and the
   radius R from that point to the paraxial exit-pupil point. Each ray goes back from the image surface along its
   direction to the sphere. W = (P_chief - P_ray) / lambda in waves, with P the optical path from the object to
   the sphere (W > 0: a shorter path than the chief ray). For an object at infinity with an angle field, the phase
   of the incident plane wave across the launch plane is added. Each sample also gives its point on the reference
   sphere (`xp`, `yp`, `zp` in mm).
3. Zernike fit (`zernike_fit`): unweighted least squares over the traced points with an intensity above 0, in
   rho and phi of (Px, Py) (phi = atan2(Py, Px); m >= 0: cos(m phi), m < 0: sin(|m| phi)). The coefficients are in
   waves.
   - `"fringe"` (the default; 36 terms by default): the Fringe (University of Arizona) numbering
     j = (1 + (n + |m|)/2)^2 - 2|m| + floor((1 - sgn m)/2), not normalized (Z4 = 2 rho^2 - 1,
     Z9 = 6 rho^4 - 6 rho^2 + 1). For j <= 36 this is the 37-term Fringe set of lens-design tools. Term 37 is
     (n, m) = (6, 6) here (as Optiland) and (12, 0) in that set.
   - `"noll"`: the numbering of Noll (J. Opt. Soc. Am. 66, 207 (1976)), normalized to unit RMS over the unit disk
     (sqrt(n + 1) for m = 0, sqrt(2 (n + 1)) else; Z4 = sqrt(3) (2 rho^2 - 1)).

   The normal matrix A^T A and A^T W (points x terms: the large part) are made on the device. The terms x terms
   system is solved on the host (NumPy `lstsq`, the columns scaled to unit norm; `fit.cond` is its condition
   number: 2.33 for the singlet with 37 Fringe terms).
4. FFT PSF (`fft_psf`): the N x N pupil function P = sqrt(I) exp(-i 2 pi W) (I: the ray intensity, 0 for a clipped
   ray) in the middle of an M x M grid of zeros (default M = 2 N; (M - N) // 2 zeros before), and
   PSF = |fftshift(fft2(P))|^2 / K^2, K = the number of samples with |P| > 0. Thus a uniform pupil without
   aberration has the peak 1, and the peak of an aberrated PSF is its Strehl ratio. The image sampling is
   dx = lambda F (N - 1) / M, with the working F-number F = 1 / (2 sqrt(<(n sin a)^2>)) of the four marginal rays
   (a: the angle to the chief ray; the rays that are not clipped). The window M dx = lambda F (N - 1) is set by N;
   the padding M / N sets the sampling (M = 2 N: about 2 samples per lambda F). The sample M // 2 is the image
   point of the chief ray.
5. MTF (`fft_mtf`): |fftshift(fft2(PSF))| normalized at zero frequency; tangential along y (Py), sagittal along x
   (Px), M // 2 values each. A lag of one pupil sample is the frequency n |D(+1) - D(-1)| / ((N - 1) lambda)
   (D: the direction of the marginal ray at the pupil coordinate +1 or -1, on a flat image surface normal to z),
   so the cutoff 2 NA / lambda is at the lag N - 1. M >= 2 N - 1 keeps the autocorrelation of the pupil from
   wrapping. `diffraction_limited_mtf` gives the MTF of a uniform circular pupil without aberration,
   (2 / pi) (acos s - s sqrt(1 - s^2)), s = nu / (2 NA / lambda) (Goodman, Introduction to Fourier Optics, 3rd
   ed., eq. 6-32), and `airy` the Airy pattern (2 J1(v) / v)^2, v = 2 pi NA r / lambda (Born and Wolf, Principles
   of Optics, section 8.5.2).
6. Huygens PSF (`huygens_psf`), the second method: the image sum of the device pipeline for a point source on the
   axis (one sample of a 64 x 64 source of 1 um pitch, padded 16 times: a flat angular spectrum), normalized as
   |E|^2 / (sum_r |a_r|)^2 (the peak for all rays in phase), on any grid of image points. It sums plane waves in
   the image-side ray directions with the power weights sqrt(J). It does not use the pupil grid or the reference
   sphere of items 1 to 5. It needs an object at a finite distance on the axis (an object at infinity gives a
   ValueError) and a physical stop. `return_bundle=True` also returns the `RayBundle` (its `n_rays`).

Items 2 to 5 follow the conventions of Optiland 0.6.3 (its `Wavefront` "chief_ray" strategy, `ZernikeFit`,
`ScalarFFTPSF` and `ScalarFFTMTF`), so that the results compare one to one. Parts of the code follow Optiland's
code (MIT; the notice is at the top of `analysis.py` and in `THIRD_PARTY_NOTICES.md`): `pupil_wavefront` follows
`ChiefRayStrategy.compute_wavefront_data` step by step (the reference sphere, the optical path to it, the restored
launch-plane phase, the mask of valid rays), `get_working_FNO` and `_get_mtf_frequency_steps`; `_path_to_sphere`
follows `SphericalReference.path_length`; `fft_psf` follows the padding and normalization of `ScalarFFTPSF`.
Two differences:
- The PSF here has the peak 1 for a perfect pupil (Optiland: 100).
- The index of image space. This code uses the medium before the image surface (image space) at the traced
  wavelength for the optical path to the reference sphere, the working F-number and the MTF frequency step.
  Optiland 0.6.3 uses the medium after the image surface (air by default): at the primary wavelength for the OPD,
  at the traced wavelength for the working F-number and the MTF frequency step. The results are the same when the
  two media are the same and, for the OPD, that medium is not dispersive or the traced wavelength is the primary
  one (all comparisons with Optiland here: air before and after the image surface). For an image in a medium (an
  immersed image) with air after the image surface they differ, also without dispersion: the OPD, and F, the PSF
  step dx and the MTF axis by the factor n. The test with a dispersive image space below shows the OPD part; no
  test compares F or the MTF axis for an immersed image.

`fft_mtf` refuses a grid M below 2 N - 1 (ValueError): the full autocorrelation of the pupil needs 2 N - 1 points,
and the bound keeps it free of wrap. The returned values (up to the cutoff) stay exact down to M = 2 N - 3 (a check
in the review `reviews/2026-11-08-0e8cf1910c4b.md`: at M = 30 and N = 16 they agree with M = 64 to 4.2e-17).

`systems.ideal_relay` is the perfect lens of the checks: two equal ideal thin lenses (f 20 mm) in a 4f
arrangement, the stop on lens 1 with a physical aperture of radius 2 mm, the image surface at the back focus or
moved by dz. Optiland's thin lens adds the optical path f - sqrt(h^2 + f^2) at the height h, so all rays of the
axis point meet in one point with equal optical paths. The ray at the pupil radius rho has tan u = rho r / f on
both sides (not the sine condition), and the image-side NA is sin u = r / sqrt(r^2 + f^2) = 0.0995.

**Pure defocus.** For the image surface dz after the focus, a spherical wave on the reference sphere (radius R)
has the exact OPD W(u) = dz (1 - cos u) + sqrt(R^2 - dz^2 sin^2 u) - R; the exit pupil of `ideal_relay` is 2 f
after lens 2, so R = f - dz. With tan u = t rho, W = dz t^2 rho^2 / 2 - 3 dz t^4 rho^4 / 8 - dz^2 t^2 rho^2 / (2 R)
+ ... The Fringe coefficient c4 is half the sum of the rho^2 and rho^4 coefficients (rho^4 = Z9 / 6 + Z4 / 2 +
Z1 / 3), and NA^2 = t^2 / (1 + t^2). Against the small-NA form W020 = NA^2 dz / 2 (c4 = W020 / 2; Noll:
W020 / (2 sqrt(3))) this gives c4 / (W020 / 2) - 1 = t^2 / 4 - dz / R to first order.

| Case | Reference | Agreement (observed; test tolerance) |
|---|---|---|
| perfect lens: `ideal_relay`, on the axis, pupil 256, grid 1024 (CPU, GPU) | OPD 0; R = f; F = 1 / (2 NA); the Airy pattern at r < 3.11 lambda / NA (5 dark rings); the diffraction-limited MTF | max \|W\| 0 (1e-9 waves); R (1e-9 mm); F and the frequency step (1e-12 relative); peak 1 (1e-12); PSF 3.0e-4 of the peak (1e-3); MTF tangential and sagittal 3.5e-4 (1e-3) |
| Huygens PSF (image sum) of the same lens on 49 x 49 samples of the FFT PSF (dx 1.94 um), directions up to 0.15 (CPU, GPU) | the Airy pattern; the FFT PSF | peak 1 (1e-9); 8.1e-4 (2e-3); 6.6e-4 (2e-3) |
| pure defocus: `ideal_relay` with dz = +20 and -20 um (r 2 mm, t 0.1) and +20 um (r 1 mm, t 0.05), pupil 128, Fringe and Noll, 36 terms (CPU, GPU) | the exact W(u) above and its Zernike fit on the same points; W020 = NA^2 dz / 2 = 0.0639, -0.0639, 0.0161 waves | map 1.4e-11 waves (1e-9); all 36 coefficients (1e-9 waves); c4 / (W020 / 2) - 1 = 1.48e-3, 3.48e-3, -3.77e-4 (Fringe and Noll), against t^2 / 4 - dz / R = 1.50e-3, 3.50e-3, -3.75e-4 (1e-4) |
| `zernike_basis`, 45 terms, 500 random points of the unit disk | Optiland's `ZernikeFringe` and `ZernikeNoll` | the same (n, m); values within the tolerance 1e-11 (the observed value is not printed) |
| 28 terms, midpoint rule on 2000 x 2000 points of the unit disk | Noll: <Z_i Z_j> = delta_ij; Fringe: diagonal 1 / (n + 1) (m = 0), 1 / (2 (n + 1)) (m != 0) | 2.5e-4 (Noll), 2.7e-5 (Fringe) (1e-3) |
| the singlet on the axis, the singlet with an object height of 1 mm, and the singlet with the object at infinity and a 2-degree field (OPD peak to valley 4.07, 5.23, 2.61 waves; Strehl 0.028, 0.033, 0.046), pupil 128, grid 256 (CPU, GPU) | Optiland (NumPy): `Wavefront` (uniform, 128); `ZernikeFit` on its points (Fringe and Noll, 37 terms); `ScalarFFTPSF`; `ScalarFFTMTF` | OPD GPU 8.9e-16, 4.6e-12, 2.3e-12 waves, CPU 9.2e-12, 3.2e-11, 6.9e-12 (1e-10); the same intensities; R (1e-12 relative); the points on the sphere (1e-9 mm); Zernike 8.0e-13 waves or less (1e-11); PSF 5.1e-14 or less (1e-11); MTF 5.2e-13 or less (1e-11); frequency axes (1e-12 relative) |
| the singlet, Fringe, 37 terms on the uniform pupil of 128 | Optiland's `ZernikeOPD` (15 hexapolar rings) | 9.9e-7 waves (1e-5); c4 2.02, c9 0.71, c16 0.019 waves; RMS residual of the fit 1.6e-7 waves |
| the singlet, Huygens PSF (image sum) on 80 x 80 points of 1 um, and on 41 x 41 samples of the FFT PSF (pupil 256, grid 1024, dx 2.28 um) (CPU, GPU) | Optiland's `HuygensPSF` (pupil 128); the FFT PSF | correlation 0.999 and 0.997 (0.995); peak 0.0245 against 0.0269 (Optiland) and 0.0245 against 0.0280 (FFT PSF) (ratio 0.85 to 1) |
| pupil weights of the singlet PSF: Debye sums over the pupil grid (pupil 256; the traced OPD and the image-side direction cosines; 80 x 80 points of 1 um) with the same weight for each pupil sample and with sqrt(J) (GPU) | the image sum on the same points | J max / min 1.57 (1.4 to 1.8); peaks 0.0287 and 0.0235: the sqrt(J) peak 18.2 % lower (12 % to 25 %); the image sum 0.0245, nearer to the sqrt(J) sum, 4.3 % above it (below 8 %) |
| an image surface in a dispersive medium (AbbeMaterial n_d 1.4, V_d 20, after the singlet), traced at 0.4861 um with the primary wavelength 0.5876 um, pupil 64 (CPU) | the definition with Optiland's NumPy trace and `SphericalReference.path_length` at the index of that medium at 0.4861 um (1.42) | 8.8e-11 waves (1e-9); the same definition at the index at 0.5876 um (1.4) differs by 2.58 waves, Optiland's `Wavefront` (air after the image surface, primary wavelength) by 49.3 waves (both above 1e-3) |
| the singlet, pupil 200 in blocks of 7000 rays, grid 512 (GPU) | the same code on the CPU | OPD 1.8e-11 waves, Zernike 1.2e-13 waves, PSF 2.3e-14, MTF 2.1e-13 (1e-10) |
| pupil 64 in blocks of 500 rays (CPU) | one block | bit for bit |
| bad inputs: device "mps", `n_pupil` 1, `trace_chunk` 0, grid below N, kind "standard", fewer points than terms; `fft_mtf` with a grid of 30 for N = 16 (below 2 N - 1; 31 passes); `huygens_psf` with an object at infinity | | ValueError |

The three PSFs of the singlet have the same shape (correlation 0.997 to 0.999) but different peaks: 0.0245 (the
image sum), 0.0269 (Optiland's Huygens PSF) and 0.0280 (the FFT PSF). The main cause is the amplitude weight of
the pupil samples (found in the review `reviews/2026-11-08-1f9108bcd8e3.md`; shown by
`test_singlet_huygens_peak_depends_on_the_pupil_weights`):
- A Debye sum over the pupil grid (pupil 256; the traced OPD and the image-side direction cosines of each ray; the
  80 x 80 points of 1 um) gives the peak 0.0287 with the same weight for each entrance-pupil sample (the
  convention of the FFT PSF and of Optiland) and 0.0235 with the energy-conserving weight sqrt(J),
  J = |d(L', M') / d(Px, Py)|. J changes by 1.57x across the pupil of this singlet. The weight alone moves the
  peak by 18 %.
- The image sum (sqrt(J) weights by construction, up to the object-side factor of a flat angular spectrum) gives
  0.0245, 4 % above the Debye sum with sqrt(J). This 4 % stays without a cause.

Thus the "Strehl ratio" of `fft_psf` (and of Optiland) assumes the same amplitude for each entrance-pupil sample
and a linear map from the pupil coordinates to the image-side direction cosines; the image sum conserves the energy
of each ray tube. For the perfect lens J is constant, and the methods agree to 8.1e-4 of the peak.

**Timings** (`bench/bench_analysis.py`). Optiland 0.6.3 on its NumPy backend on the CPU of the GPU computer
(`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` 4) against this code on the GPU (RX 7900 XT;
`.venv-optics`, PyTorch 2.14.1 for ROCm 7.2), on the axis, pupil N x N, grid M = 2 N, best of 2 runs after one
untimed run. Rows in `benchmarks/2026-11-08/optics-analysis.jsonl`, from the repository folder on the GPU computer:

    bash tools/gpu_run.sh optics_gpu python bench/bench_analysis.py --pupil 64 128 256 512 1024 2048 --huygens
    bash tools/gpu_run.sh optics_gpu python bench/bench_analysis.py --system relay --pupil 256 1024

Each Optiland call is timed alone: `ScalarFFTPSF` and `ScalarFFTMTF` trace their own wavefront (and
`ScalarFFTMTF` its own PSF). The last two columns are `speedup_psf` (`ScalarFFTPSF` over `pupil_wavefront` +
`fft_psf`) and `speedup_mtf` (`ScalarFFTMTF` over `pupil_wavefront` + `fft_psf` + `fft_mtf`).

| System | N (rays) | Optiland: Wavefront / ZernikeFit / ScalarFFTPSF / ScalarFFTMTF | GPU: `pupil_wavefront` / `zernike_fit` / `fft_psf` / `fft_mtf` | Faster: PSF / MTF |
|---|---|---|---|---|
| singlet | 64 (3096) | 3.74 / 3.68 / 4.22 / 8.43 ms | 42.6 / 4.57 / 0.284 / 0.159 ms | 0.10x / 0.20x |
| singlet | 128 (12644) | 6.09 / 11.1 / 6.58 / 11.6 ms | 41.8 / 4.38 / 0.277 / 0.130 ms | 0.16x / 0.28x |
| singlet | 256 (51040) | 14.0 / 37.2 / 16.9 / 22.5 ms | 42.6 / 9.79 / 0.414 / 0.235 ms | 0.39x / 0.52x |
| singlet | 512 (205012) | 41.2 / 173 / 60.4 / 75.7 ms | 47.4 / 34.0 / 1.88 / 1.02 ms | 1.23x / 1.51x |
| singlet | 1024 (821904) | 198 / 830 / 275 / 336 ms | 51.3 / 135 / 6.25 / 3.93 ms | 4.78x / 5.46x |
| singlet | 2048 (3290904) | 797 / 4483 / 1368 / 1696 ms | 102 / 503 / 11.6 / 8.98 ms | 12.0x / 13.8x |
| relay | 256 (51040) | 11.9 / 44.3 / 15.3 / 22.9 ms | 42.1 / 10.1 / 0.433 / 0.268 ms | 0.36x / 0.54x |
| relay | 1024 (821904) | 186 / 849 / 255 / 345 ms | 47.5 / 128 / 4.02 / 2.74 ms | 4.95x / 6.35x |

- **Slower up to N = 256.** `pupil_wavefront` has a fixed cost: 41.8 to 42.6 ms for 3096 to 51040 rays (N 64 to
  256), and 102 ms for 3.29e6 rays. Up to N = 256 this cost is larger than Optiland's whole FFT PSF or FFT MTF,
  so the GPU path is slower (0.10x to 0.39x for the PSF, 0.20x to 0.54x for the MTF). The cost is in the trace
  step; the rows do not split it into its parts (the copy of the optic and its check rays, the small traces, the
  launches of Optiland's PyTorch trace). These rows are of the code of 66da5d3: "Batch timings" below gives the
  parts of this cost, the changes that cut it, and new rows (the PSF faster from N = 256, the MTF from N = 128).
- **Faster from N = 512.** The PSF and the MTF are faster on the GPU from N = 512 on: 1.23x and 1.51x at N = 512,
  12.0x and 13.8x at N = 2048. In these rows the crossover is between N = 256 and N = 512. The trace step alone
  is still slower at N = 512 (47.4 ms against 41.2 ms) and faster from N = 1024 (3.85x at 1024, 7.78x at 2048).
- The FFT steps after the trace take 0.28 to 11.6 ms (`fft_psf`) and 0.13 to 8.98 ms (`fft_mtf`) on the GPU.
  `zernike_fit` (37 Fringe terms) is slower than Optiland's `ZernikeFit` at N = 64 (4.57 ms against 3.68 ms) and
  faster from N = 128 (2.5x) to N = 2048 (8.9x).
- Differences from Optiland in all 8 rows: OPD at most 8.9e-16 waves, Zernike coefficients 2.8e-13 waves, PSF
  3.3e-16 (Strehl units), MTF 7.8e-16 (0 for the OPD and the coefficients of the perfect lens, Strehl 1.0).

Huygens PSF (rows 7 to 9, singlet, 80 x 80 points of 1 um): Optiland's `HuygensPSF` takes 92.1, 322 and 1074 ms
with pupil 64, 128 and 256; `huygens_psf` takes 13.9, 13.8 and 13.5 ms (6.7x, 23x and 79x). These are two methods
with different sampling: the image sum uses its own direction grid (a 64 x 64 source padded 16 times), not the
pupil N, so its time is the same in the three rows. Correlation 0.9984 to 0.9985; peak 0.0245 against 0.0271 to
0.0269. These rows do not have the fields that the bench logs now (the clock time, the versions of Optiland and
PyTorch, the numba thread count and threading layer, the ray counts; the time 10:17 to 10:27 EST is from the
index `benchmarks/2026-11-08/README.md`). Optiland's `HuygensPSF` traces 3096, 12644 and 51040 rays: its pupil
grid is the same "uniform" grid of N as in the FFT rows, whose `rays` field gives these counts for N = 64, 128 and
256. The image sum keeps the same rays in the three rows: 7505 (the `image_sum_rays` field of a one-row check run
in the review `reviews/2026-11-08-0e8cf1910c4b.md`; not in a logged row). Optiland's
`HuygensPSF` runs numba kernels: these rows log `threads` 16 (PyTorch, after that call), so it probably ran with
16 numba threads (the probable cause in the review record: numba's OpenMP layer sets the shared thread count to the
CPU count), not with the 4 threads of the other rows.

## Batches: fields, wavelengths, through focus, polychromatic (`src/optics_gpu/analysis.py`, items 7 to 10)

```python
from optics_gpu.analysis import defocus, fft_mtfs, fft_psfs, image_quality, polychromatic_psf, pupil_wavefronts
fcd = [0.4861, 0.5876, 0.6563]
wfs = pupil_wavefronts(optic, [(0, 0), (0, 0.5), (0, 1)], fcd, n_pupil=256, device="cuda")   # wfs[field][wl]
planes = [defocus(wfs[0][1], dz) for dz in np.linspace(-100, 100, 9)]   # um along z: no new trace
psfs = fft_psfs(planes, 512)              # all planes in one batched FFT; fft_mtfs(planes, psfs) for the MTFs
poly = polychromatic_psf(wfs[2], weights=[1, 2, 1])   # one common grid; poly.shift: the lateral color (um)
res = image_quality(optic, fields, fcd, dz=[-50, 0, 50], weights=[1, 2, 1], n_pupil=256, device="cuda")
res.psfs[f][i][z].psf, res.mtfs[f][i][z].tangential, res.poly_psfs[f][z].psf, res.poly_mtfs[f][z].sagittal
```

**The fixed cost of one call.** The code of 66da5d3 made a new PyTorch copy of the optic in each `pupil_wavefront`
call (`Optic.to_dict`, then `Optic.from_dict`) and traced three times through the copy: the 6 check rays, the chief
ray with the four marginal rays, and the pupil rays. Each Optiland `trace_generic` call runs the paraxial ray
aiming, and the aiming builds the paraxial path of the optic (`SurfaceGroup.build_paraxial_path`: the frames of
all surfaces, with many small device-to-host copies) three times; the surface trace of a few rays also has a fixed
cost. `bench/profile_wavefront.py` makes these steps again with Optiland's functions and times each one on the GPU
(the numbers are in "Batch timings" below). Changes (item 7 of the module docstring):
1. One trace for each block of rays of one wavelength: the check rays and the 5 rays of each field go into the
   first block with the pupil rays, and the rays of all fields at one wavelength go into the same blocks
   (`pupil_wavefronts`). `pupil_wavefront` is `pupil_wavefronts` for one field and one wavelength.
2. The PyTorch copy is kept for each state of the optic and each device (`COPY_CACHE` = 8 copies, the least
   recently used goes first; the key is the SHA-256 of the pickled `Optic.to_dict()`; `clear_copy_cache()` forgets
   them). Nothing changes the copy, so its paraxial path is made one time and kept on the copy (Optiland rebuilds
   it on each call because the geometry of an optic can change; the key changes when the optic changes).
3. The check rays still go through the original optic (NumPy, one trace for each wavelength with all fields) in
   each call. Thus a change of the optic that its dictionary does not keep still stops the call with a
   RuntimeError, as before.

**Through focus** (`defocus(wf, dz)`, item 8). The image plane moves by dz (um) along +z. Each ray of the pupil
grid is a straight line in image space from its point on the reference sphere; `defocus` finds its point on the
moved plane and its optical path back to the new reference sphere (center: the chief-ray point on the moved plane;
radius: the distance from there to the exit-pupil point). This is the calculation of a new trace to the moved image
surface, without the trace, when the exit-pupil point stays where it is. Optiland's paraxial exit pupil is computed
in the medium after the image surface. It stays fixed only when that medium has the index of image space at the
primary wavelength (for example air on both sides). Else it moves with the image surface by dz (1 - n_after /
n_image), a new trace uses another reference sphere, and `defocus` refuses the wavefront (a ValueError;
`exit_pupil_fixed` False). For an image space in a glass, give the image surface the same glass as its material.
To first order in dz the change is the plane-wave (Debye) defocus phase
n dz (1 - D . D_c) / (N_c lambda); the full form also keeps the change of the reference sphere (for the perfect lens
the rest is -dz^2 sin^2 u / (2 R') at the edge of the pupil, R' the radius of the new sphere). This rest has the
same sign for both signs of dz, so the through-focus PSF of `defocus` is not symmetric about the focus, while the
plane-wave image sum (`image_sum`, `huygens_psf` with dz) is symmetric (the checks below give the size). `defocus`
needs a flat image surface normal to z (else a ValueError) and keeps the intensities, F and the MTF step of the
wavefront.

**Polychromatic PSF and MTF** (item 9). `polychromatic_psf(wfs, weights)` gives sum_l v_l PSF_l(x - s_l) on one
grid: PSF_l in Strehl units, s_l the chief-ray point of wavelength l minus that of the reference wavelength (lateral
color), v_l = w_l K_l^2 / (lambda_l F_l)^2 normalized to the sum 1 (w_l: the spectral power). With these weights
each wavelength has the energy w_l sum(I_l), and a system without aberration and lateral color has the peak 1. The
resampling is exact, not an interpolation: a matrix Fourier transform (two matrix products for each wavelength;
Soummer et al., Opt. Express 15, 15935 (2007)) evaluates the Fraunhofer sum of the pupil function at the points of
the common grid. On the grid of `fft_psf` it gives the FFT PSF. The grid must stay inside half the period
lambda_l F_l (N - 1) of each wavelength (else a ValueError); the default step is the largest that does so.
`polychromatic_mtf(wfs, psfs, weights)` gives |sum_l u_l OTF_l(nu) exp(-i 2 pi nu s_l)| with u_l = w_l sum(I_l)
(normalized), OTF_l the complex OTF of the FFT PSF of each wavelength, linearly interpolated between its lags;
tangential along y, sagittal along x; the default frequencies are those of `fft_mtf` of the reference wavelength.

**One batch** (`image_quality`, item 10): `pupil_wavefronts`, `defocus` for each plane, `fft_psfs` and `fft_mtfs`
(the FFTs of all fields, wavelengths and planes in one batched `torch.fft.fft2` call, split above `FFT_ELEMENTS`
= 2^25 complex values, 512 MB), and with weights `polychromatic_psf` and `polychromatic_mtf` for each field and
plane. Memory: a wavefront keeps 8 float64 and 2 bool N x N arrays (66 N^2 bytes; 4.3 MB for N = 256), each plane
of `defocus` 4 new float64 arrays, and each PSF M^2 float64 values.

Checks (`tests/test_analysis_batch.py`; the values are test prints on the GPU computer, CPU and GPU, unless one is
named):

| Case | Reference | Agreement (observed; test tolerance) |
|---|---|---|
| `systems.dispersive_singlet` (N-BK7 n_d and V_d), 3 fields x 3 wavelengths (F, d, C), pupil 48, one batch in blocks of 1000 rays (rays of two fields in a block); the singlet with the object at infinity, 2 angle fields, pupil 40 | `pupil_wavefront` for each field and wavelength | bit for bit: all tensors and numbers of each wavefront (the lateral color F to C at 1 mm: 0.266 um) |
| `fft_psfs`, `fft_mtfs`: 9 wavefronts, pupil 64, grid 160 | `fft_psf`, `fft_mtf` of each | CPU: bit for bit; GPU: PSF 6.9e-18, MTF 1.1e-16 (1e-14) |
| `defocus` by -50 and +50 um: the singlet on the axis, at the object height 1 mm, and with the object at infinity (2 degrees); pupil 96 (OPD peak to valley 2.2 to 5.0 waves) | `pupil_wavefront` of the optic with the image surface moved by dz (a new trace) | OPD 2.8e-11, 2.3e-11, 4.6e-12 waves (1e-9); the same intensities and valid rays; radius (1e-12 relative); center 1.1e-16 mm or less (1e-12); points on the sphere 1.1e-14 mm or less (1e-9); FFT PSF 3.0e-13 or less (1e-9) |
| `defocus` by -80 and +80 um with an image space in a glass (n 1.4, `IdealMaterial`), the image surface with the index 1.4 also after it; on the axis and at the object height 1 mm, 1.55 um, pupil 64 (OPD peak to valley 3.7 to 5.0 waves) | `pupil_wavefront` of the optic with the image surface moved by dz (a new trace) | OPD 4.1e-11 waves (1e-9); radius and exit pupil: 0 (1e-12 relative, 1e-9 mm) |
| the same optic with air after the image surface (Optiland's default) | Optiland's exit pupil of the moved optic | `exit_pupil_fixed` False and `defocus` gives a ValueError; the exit pupil moves by 0.0229 mm for dz = 80 um, dz (1 - 1 / 1.4) (1e-6 relative) |
| the perfect lens (`ideal_relay`) through focus: dz = -150, 0, 60, 150 um, pupil 256, grid 1024 | the exact spherical-wave OPD of the moved plane; the image sum (`huygens_psf` with the same dz); the FFT PSF and a Debye sum with the first-order phase | see "Through focus of the perfect lens" below |
| `polychromatic_psf` of one wavelength (default grid and step) | `fft_psf`; `polychromatic_mtf` against `fft_mtf` | PSF 3.1e-17 (1e-13); MTF 6.3e-17 (1e-12) |
| the dispersive singlet on the axis, pupil 64, a common grid of 100 points (step 1.30 um) | Optiland's `MMDFTPSF` of each wavelength (the same pupil sampling, pixel pitch and 98 points) | 7.2e-13 (CPU), 1.2e-12 (GPU) (1e-11) |
| the dispersive singlet at 1 mm (lateral color 0.266 um), weights 1, 2, 1, pupil 64, grid 128 | the weighted sum of the per-call FFT PSFs, each moved by s_l by trigonometric interpolation (NumPy: the DFT of the samples, then the trigonometric polynomial; exact for M >= 2 N - 1), with v_l made from the per-call data | 1.0e-17 (CPU), 1.4e-17 (GPU) (1e-12); the weights (1e-12 relative) |
| the perfect lens at 1.31, 1.55, 1.65 um, weights 1, 2, 1, pupil 256, grid 1024 | sum_l v_l Airy(r; NA, lambda_l), v_l ~ w_l / lambda_l^2; sum_l u_l MTF_dl(nu; NA, lambda_l), u_l = w_l / sum(w) | peak 1 (1e-12); PSF 2.9e-4 inside r < 51.6 um of the window of +-839 um (1e-3); MTF 3.1e-4, both axes (1e-3) |
| one wavefront twice, the second moved by (2, 3) um | (PSF(x) + PSF(x - s)) / 2; the MTF times abs(cos(pi nu s)) on each axis | PSF 1.4e-17 (1e-13); MTF 5.6e-17 and 1.1e-16 (1e-12) |
| `polychromatic_mtf`, the sign of the lateral-color phase: two PSFs of one wavelength (the singlet on the axis; the second with 0.4 waves of Fringe Z7 and Z8 and moved by (2, 3) um), pupil 64 | the OTF of PSF_A(x) + PSF_B(x - s) on the period of the FFT PSF (B moved by trigonometric interpolation, then the DFT) | 1.6e-15 (1e-5); against the same reference with -s: 0.0178 (above 1e-3) |
| `fft_psfs` and `fft_mtfs` with `FFT_ELEMENTS` set to 2 planes: 5 wavefronts in 3 calls (CPU, GPU) | one call | 0 (1e-14) |
| `image_quality`: the dispersive singlet, 3 fields x 3 wavelengths x 3 planes (-40, 0, 40 um), weights 1, 2, 1, pupil 48 | the per-call functions for each field, wavelength and plane | the same wavefronts (bit for bit); CPU: 0; GPU: PSF 6.9e-18, MTF 5.6e-17, polychromatic PSF 0, polychromatic MTF 5.6e-17 (1e-14) |
| the copy of the optic | a second call; a changed thickness | the same object and the same wavefront; a new copy and a new OPD |
| `COPY_CACHE` set to 2 with 3 states of one optic; `_state_key` None; a cpu and a cuda copy of one optic; `image_quality(mtf=False)` | | the first copy goes and the last two stay; no copy is kept and the wavefront is the same (bit for bit); two copies (OPD within 1e-10); no MTFs, the same PSFs and polychromatic PSFs |
| bad inputs: `defocus` without ray data or with a curved image surface; no field; two pupil samplings in one batch; `fft_mtfs` with a grid below 2 N - 1; a window outside half the period; negative weights; a weight count that is not the wavelength count; negative frequencies; no plane | | ValueError |

**Through focus of the perfect lens** (`test_through_focus_of_the_perfect_lens`; test prints, the same on the CPU
and the GPU): `ideal_relay` (NA 0.0995), pupil 256, grid 1024, 49 x 49 samples of the FFT PSF (1.94 um), dz = -150,
0, 60 and 150 um (W020 = NA^2 dz / 2 = -0.48, 0, 0.19, 0.48 waves).
- The OPD of `defocus` agrees with the exact OPD of a spherical wave on the reference sphere of the moved plane to
  9.2e-12 waves or less (1e-9). The first-order phase n dz (1 - D . D_c) / (N_c lambda) differs from it by
  3.56e-3, 5.8e-4 and 3.62e-3 waves at -150, 60 and 150 um: the second-order rest dz^2 sin^2 u / (2 R') at the edge
  of the pupil (3.6e-3 at 150 um; the test bound is 1.1 times this value).
- The image sum (`huygens_psf` with the same dz) is a plane-wave (Debye) sum. The FFT PSF of the pupil with the
  first-order phase only agrees with it to 6.6e-4 (in focus), 6.8e-4 (60 um) and 1.13e-3 (-150 and 150 um), and a
  Debye sum over the pupil rays (the first-order phase, the traced image-side directions, the weight J of each
  sample) to 6.6e-4, 6.3e-4 and 1.11e-3 (tolerance 2e-3). Both are symmetric in dz: the peaks at -150 and 150 um
  are 0.438 (FFT) and 0.439 (image sum) on both sides.
- The FFT PSF of `defocus` (the reference sphere of the moved plane, as a trace to the moved image surface) has the
  peaks 0.432 at -150 um and 0.444 at 150 um, and differs from the image sum by 7.0e-3 and 4.9e-3 (tolerance 1e-2;
  5.5e-4 at 60 um). The cause is the second-order rest above: it has the same sign for both signs of dz, so it adds
  to the defocus before the focus and removes from it after the focus. The two models differ at the order dz / R
  (0.75 % at 150 um here); `defocus` keeps the result of a trace to the moved surface (the check against the moved
  surface above).

**Batch timings.** RX 7900 XT with `.venv-optics` (PyTorch 2.14.1 for ROCm 7.2, Optiland 0.6.3); Optiland on its
NumPy backend on the CPU of the same computer. The rows are in `benchmarks/2026-11-08/` (logged 17:54 to 18:23 EST,
in the GPU queue from 17:34; after the tests of this change: 176 passed), from the repository folder on the GPU
computer, in this order (the files `optics-profile.out`, `optics-batch.jsonl` and `optics-analysis-2.jsonl`):

    bash tools/gpu_run.sh optics_gpu python bench/profile_wavefront.py --pupil 64 256 --profile
    bash tools/gpu_run.sh optics_gpu python bench/bench_batch.py --pupil 64 128 256 512
    bash tools/gpu_run.sh optics_gpu python bench/bench_analysis.py --pupil 64 128 256 512 1024 2048

The fixed cost (`optics-profile.out`; the singlet on the axis, the median of 5 runs, ms; N = 64 / 256). The steps of
the 66da5d3 call, made again with Optiland's functions:

| Step | N = 64 | N = 256 |
|---|---|---|
| exit-pupil point, `to_dict`, NumPy trace of the 6 check rays, `Optic.from_dict` | 0.34, 0.017, 1.31, 1.04 | 0.35, 0.017, 1.30, 1.07 |
| trace of the 6 check rays through the copy, with the comparison | 13.5 | 13.3 |
| trace of the chief ray and the 4 marginal rays | 12.5 | 12.5 |
| trace of the pupil rays (3096 / 51040 rays) | 13.0 | 13.4 |
| sum of these steps | 41.7 | 41.9 |
| inside one trace: Optiland's paraxial ray aiming of 5 rays; the surface trace of 5 rays | 8.97; 2.75 | 8.93; 2.76 |
| `pupil_wavefront` now: the kept copy; a new copy (after `clear_copy_cache`) | 9.09; 12.9 | 9.96; 13.4 |

The sum of the steps (41.7 and 41.9 ms) agrees with the 66da5d3 rows above (42.6 ms at N = 64 and 256). The
present call is 4.58x and 4.20x faster with the kept copy (3.23x and 3.13x with a new copy). In the cProfile table
of 10 ray aimings on a new copy (10.1 ms each in cProfile), `build_paraxial_path` has 30 calls (3 for each aiming)
and 86 ms, 85 % of the time; the table also shows 900 `.cpu()` calls and 710 `.item()` calls. One present call
(the kept copy) makes 88 device-to-host reads (`host_reads`; the 66da5d3 count is not in a logged row).

Per call against Optiland (`optics-analysis-2.jsonl`, the singlet on the axis, grid 2 N, best of 2 runs, ms), with
`pupil_wavefront` of the 66da5d3 rows (`optics-analysis.jsonl`) for comparison:

| N (rays) | `pupil_wavefront` 66da5d3 / now (faster) | Optiland: Wavefront / ScalarFFTPSF / ScalarFFTMTF | Faster than Optiland: PSF / MTF (66da5d3 rows) |
|---|---|---|---|
| 64 (3096) | 42.6 / 10.7 (3.97x) | 3.72 / 4.13 / 8.32 | 0.37x / 0.74x (0.10x / 0.20x) |
| 128 (12644) | 41.8 / 10.3 (4.06x) | 5.65 / 6.75 / 11.4 | 0.64x / 1.07x (0.16x / 0.28x) |
| 256 (51040) | 42.6 / 11.7 (3.65x) | 14.2 / 16.5 / 23.3 | 1.36x / 1.89x (0.39x / 0.52x) |
| 512 (205012) | 47.4 / 11.9 (3.97x) | 45.7 / 61.7 / 75.3 | 4.36x / 5.05x (1.23x / 1.51x) |
| 1024 (821904) | 51.3 / 21.3 (2.40x) | 191 / 275 / 345 | 10.9x / 12.3x (4.78x / 5.46x) |
| 2048 (3290904) | 102 / 77.6 (1.32x) | 802 / 1330 / 1723 | 14.9x / 17.5x (12.0x / 13.8x) |

"Faster" is `speedup_psf` (ScalarFFTPSF over `pupil_wavefront` + `fft_psf`) and `speedup_mtf` (ScalarFFTMTF over
`pupil_wavefront` + `fft_psf` + `fft_mtf`), recomputed from the times of the rows. The differences from Optiland
are at most: OPD 8.9e-16 waves, Zernike 2.8e-13 waves, PSF 1.4e-17, MTF 3.3e-16.
- **Still slower** than Optiland on one call: the PSF at N = 64 and 128 (0.37x, 0.64x) and the MTF at N = 64
  (0.74x). The trace step alone is slower up to N = 128 (`pupil_wavefront` 10.7 and 10.3 ms against Optiland's
  Wavefront 3.72 and 5.65 ms: 0.35x and 0.55x) and faster from N = 256 (1.22x). `zernike_fit` is slower at N = 64
  (4.59 against 3.65 ms, 0.80x), as before. The present call has one Optiland trace on the GPU (the ray aiming
  and the surface trace); the rows do not split the 9.1 to 11.9 ms of the present call (N up to 512) into its parts.
- **The threads of these rows.** The first row logs `threads` 4 and the numba layer "not initialized"; rows 2 to 6
  log `threads` 16 and the layer "omp" (numba started its threads in the first row). The CPU parts of rows 2 to 6
  (Optiland, and the host steps of this code) thus could use 16 threads; the 66da5d3 rows do not log this field.

The batch (`optics-batch.jsonl`; `systems.dispersive_singlet`, 3 fields x 3 wavelengths x 5 planes (dz = -100,
-50, 0, 50, 100 um) = 45 planes, grid 2 N, best of 2 runs, ms):

| N | `image_quality` without weights / with weights 1, 2, 1 | one call for each plane: a new trace / `defocus` | Optiland: ScalarFFTPSF + ScalarFFTMTF / ScalarFFTMTF alone | without weights faster than: a new trace / `defocus` / Optiland / ScalarFFTMTF alone |
|---|---|---|---|---|
| 64 | 41.2 / 68.4 | 431 / 103 | 576 / 383 | 10.5x / 2.50x / 14.0x / 9.29x |
| 128 | 42.2 / 74.2 | 485 / 106 | 836 / 516 | 11.5x / 2.51x / 19.8x / 12.2x |
| 256 | 54.7 / 130 | 457 / 110 | 1833 / 1040 | 8.36x / 2.01x / 33.5x / 19.0x |
| 512 | 119 / 486 | 534 / 152 | (not run) | 4.51x / 1.29x / - / - |

- "A new trace" is `pupil_wavefront` of the optic with the image surface moved by dz, `fft_psf` and `fft_mtf` for
  each of the 45 planes (9.6 to 11.9 ms per plane); "`defocus`" is 9 `pupil_wavefront` calls, then `defocus`,
  `fft_psf` and `fft_mtf` for each plane. The batch gives the same numbers as the per-call path with a new trace
  (PSF 1.5e-12, MTF 9.1e-12 or less) and as Optiland (PSF 3.5e-12, MTF 9.1e-12 or less, N up to 256).
- The polychromatic PSF and MTF (with weights; 15 field points and planes) add 27.2, 32.0, 75.2 and 368 ms. At
  N = 512 they are 76 % of the batch, and the batch with weights (486 ms) is only 1.10x faster than the per-call
  path with a new trace, which has no polychromatic results. This part (a matrix Fourier transform of each
  wavelength on the common grid, and the OTFs) is the slowest part of the batch at large N; Optiland has no
  polychromatic PSF to compare.
- One `pupil_wavefront` call of this system (field 0, wavelength d) takes 9.31 to 11.5 ms with the kept copy and
  13.5 to 16.1 ms with a new copy (N = 64 to 512). The peak memory of the process is 1.91 to 2.76 GiB
  (`max_rss_gb`).

## Checks (`tests/test_*.py`: `pop`, `trace_backends`, `device_pipeline`, `image_planes`, `nufft`, `sources`)

The checks of `tests/test_analysis.py` are in the section "Image quality" above.

| Case | Reference | Agreement (observed; test tolerance) |
|---|---|---|
| 4f system of two ideal thin lenses (f 2 mm / 5 mm), 6 um Gaussian (half-angle 0.082 rad) | magnification 2.5; analytic Gaussian-mismatch coupling | image radius 0.3 % (0.5 %); coupling 1e-3 (2e-3) |
| the same at half-angle 0.247 rad | the image error grows as theta^2 | ratio within 25 % of (0.247/0.082)^2 |
| power through the lossless 4f system | integral of \|E\|^2 at the source | (1 %) |
| point source through a plano-convex singlet at 2f-2f, strong spherical aberration (6 mm stop) | Optiland's scalar Huygens PSF (independent method) | correlation 0.998 (0.99) |
| PyTorch-backend trace (float64) of 101 x 101 directions, \|L\|, \|M\| <= 0.09, through the 4f system and the singlet (the stop clips about half of them), on the CPU and on the GPU | Optiland's NumPy-backend trace of the same rays | GPU: 0. CPU: positions 7e-15 mm, directions 9e-16, OPL 4e-14 mm, intensity 0 (1e-12 mm and 1e-12); the same NaN rays |
| the 4f coupling into 3 fiber radii and the singlet PSF field, traced by PyTorch (CPU and GPU) | the same with the NumPy trace | eta: 4.4e-16 (1e-12); field: GPU 1.2e-14, CPU 5.6e-12 relative (1e-9) (the PyTorch trace goes through the device pipeline and its image sum, `image_planes`); the NumPy values are the same as before this change: eta 0.99998, 0.95324, 0.93319 |
| image sum in 12 chunks of 401 rays (4f, 4501 rays) | one sum | 9e-15 relative (1e-12) |
| peak GPU memory of the image sum, 4f, chunks of 2048 rays, 128 x 128 image | one chunk of the x and y factors, (128 + 128) x 2048 complex128 values | 1.08x (below 1.3x) |
| `ray_chunk` 0 and -1 | (an all-zero field before the one review) | ValueError |
| an optic made under Optiland's PyTorch backend in float32 | the precision of the check rays | ValueError that names float32 |
| a NumPy optic while Optiland's PyTorch backend (float32) is active | the NumPy trace | the check rays run on NumPy; passes (1e-12) |
| a PyTorch copy of the optic with the image surface moved by 10 um | the check rays of `trace_rays` | RuntimeError |
| `_np_gradient0` on a uniform axis, the `fftshift(fftfreq)` axis and a random axis, along both axes (CPU, GPU) | `np.gradient` | (1e-13 relative) |
| device pipeline: the 4f coupling into 3 fiber radii and the singlet PSF field, in one block and in blocks of about 500 rays (CPU, GPU) | the NumPy pipeline | field, GPU: 9.8e-15 and 8.0e-15 (4f, one block and blocks), 1.2e-14 and 1.3e-14 (singlet); CPU: 8.6e-15 (4f), 5.6e-12 (singlet) (4f 1e-12, singlet 1e-9); eta 4.4e-16 (1e-12); the same ray counts |
| device PSF of the singlet, blocks of about 4000 rays | Optiland's Huygens PSF | correlation (0.99) |
| `trace_bundle` and `image_sum` on the GPU, 4f, point source, 5.5e4 rays in 8 blocks, watched by a `TorchFunctionMode` | no large tensor made on the CPU; arrays to the host only of the 25 check rays; scalars bounded per block | 20 arrays of 25 values (at most 40); 421 scalars (at most 60 per block + 20) |
| the watch itself: 10 forms of a copy of 30 values to the host (`cpu`, `to` with a device name, a device, a keyword, a dtype or a CPU tensor, `tolist`, `numpy`, `copy_`, `Tensor.cpu`) and 4 forms of a scalar | | each form is seen |
| `image_sum(dz=d)` against the NumPy pipeline with the image surface moved by d: 4f, Gaussian with a curved wavefront (R 1.5 mm), d = -80 and +80 um; singlet PSF, d = -60 and +60 um (CPU, GPU) | Optiland's trace to the moved surface | 4f 2.1e-12, 2.9e-12 (1e-10); singlet 2.2e-11, 2.6e-11 (GPU), 3.0e-11 (CPU) (1e-9); with the opposite sign of dz 1.0, 1.7, 2.1 (above 0.5); eta(-80 um) 0.9959, eta(+80 um) 0.9875 |
| `RayBundle.power`, 4f, 6 um Gaussian (CPU, GPU) | the source power; the power of the image field on the 96 um grid | -1e-12 (1e-9); 5.5e-5 (1e-4) |
| peak GPU memory of `trace_bundle`, 4f, point source, 2.45e5 rays traced, 1.93e5 kept | blocks of 4096 rays against one block of all rays | 53 bytes per traced ray (below 64) against 249 (difference above 150) |
| focus scan by `image_sum(dz)`, 4f image of a 6 um Gaussian into a 15 um Gaussian mode, dz -300 to 400 um (CPU, GPU) | 1 / (1 + (lambda dz / (2 pi w^2))^2); the FFT angular-spectrum propagation of the dz = 0 field | 1.4e-3 (2e-3); FFT: 1.5e-6 (1e-5), 2.7e-7 with the power at dz = 0 (1e-6) |
| `lateral_scan`, offsets -10 to 8 um in x and y | exp(-s^2 / w^2) times the centered value; `overlap_efficiency` of the moved mode | 8.1e-4 (2e-3); 3e-16 (1e-12) |
| bad inputs of the device pipeline ("mps", `trace_chunk` 0, no kept direction, `ray_chunk` 0) | | ValueError |
| `image_planes`: 3000 random rays in 2 blocks, a non-uniform x grid, a y grid off the axis, 4 planes; groups of 256 rays, chunks of 1000 and 700 rays (short last groups); batched groups and the path for large outputs (`GROUP_OUTPUTS` patched to 1) (CPU, GPU) | the plane-wave sum in NumPy (all rays at one time, no phase reference) | complex128 4.1e-14 (1e-12); complex64 5.9e-7 (1e-5); `image_sum` and the per-plane sum: the same |
| focus scan of 21 planes (dz -80 to 80 um) in one pass: the 4f image of a Gaussian with a curved wavefront, the singlet PSF (CPU, GPU) | one sum per plane with the sum of the NumPy pipeline | field: GPU 1.0e-14 (4f), 1.3e-14 (singlet); CPU 2.0e-15, 2.8e-15 (1e-12); eta of the stack against eta per plane 1.4e-14 or less (1e-12) |
| complex64 against complex128, 9 planes: the 4f image with a curved wavefront, the singlet PSF, a Gaussian focus with rays up to the direction cosine 0.6; 3 planes: a uniform pupil of NA 0.6 (equal weights, 7.1e4 rays); the last two on a 201 x 201 um grid (grid phase 243 rad) (CPU, GPU) | complex128 | GPU: \|d eta\| with the power of each plane / with one power: 3.7e-9 / 8.3e-8 (4f), 1.2e-8 / 1.6e-8 (singlet), 1.2e-8 / 1.2e-8 (Gaussian focus), 9.9e-8 / 3.3e-7 (uniform pupil) (1e-6); field 1.5e-6, 1.1e-6, 4.0e-7, 1.8e-6 (2e-5) |
| complex64, one plane wave: 2e4 equal rays (L' 0.5, M' 0.3), grid +-120 um (grid phase 243 rad) (CPU, GPU) | the bound (`GROUP_RAYS` + 6 `PHASE32_MAX`) 2^-24 sum \|a\| = 1.5e-4 sum \|a\| | GPU 3.3e-5, CPU 2.1e-5 of sum \|a\| |
| complex64 with a grid phase of 316 rad (above `PHASE32_MAX`) | | ValueError; complex128 gives a finite field |
| peak GPU memory of `image_planes` minus E, default chunk, several chunks (GPU): 41 planes 128 x 128 (budget 2^22 values; complex128, complex64), 21 planes 128 x 128 (2^25; both), 9 planes 512 x 256 (2^22; both), 41 planes 512 x 512 complex64, 101 planes 512 x 512 complex128, 1000 planes on a 1 x 1 grid complex64 (2^25) | the budget, `SUM_ELEMENTS` complex128 values | 0.67, 0.63; 0.97, 0.95; 0.91, 0.74; 0.82; 0.99; 0.96 of the budget (0.5 to 1, no tolerance). Before the one review: 1.13 (41 x 512 x 512 complex64), 1.45 (101 x 512 x 512), 1.28 (1000 x 1 x 1), in the review record |
| `polarized_mode` (the source of the coupling example): two synthetic pure polarizations of different shapes, mixed by a unitary, a 45-degree and a non-unitary 2 x 2 matrix, dneff 1e-7 and 2.1e-4; two orthogonal diagonal modes | the pure fields; for orthogonal parts the power split 0.5 / 0.5 and the mean neff | field and fraction: 9.6e-17 (1e-12) for x and y; part powers and neff (1e-12) |
| `polarized_mode`, inputs that have no defined source: a pair with dneff above the tolerance (not mixed); equal fractions without degeneracy; two x-polarized supermodes (dneff 1e-4); the same mode twice, also with 1e-9 EY added (near-singular power matrix); a source below `min_fraction` | | not mixed; ValueError in the other cases |
| bad inputs of `image_planes` (`dtype` float64, `ray_chunk` 0, a 2-D `x_img`, `y_img` or `dz`, for example a meshgrid array) | | ValueError; an empty list of planes gives a (0, Nx, Ny) result |
| `image_planes(method="nufft")`: 3000 random rays in 2 blocks, uniform grids of 24 x 17 points not centered on 0, 4 planes, one chunk and chunks of 500 rays; rays to 0.3 on a 0.9 um grid, and rays to 0.6 on a 2 um grid (0.77 cycles per step, aliased) (CPU, GPU) | the plane-wave sum in NumPy | max \|dE\| / sum \|a\| for tol 1e-2, 1e-6, 1e-9, 1e-12: 2.2e-5, 2.7e-9, 4.4e-12, 5.4e-15 (0.9 um); 1.5e-5, 1.8e-9, 3.0e-12, 4.3e-15 (2 um) (tol) |
| one ray, 20 random directions up to 0.6, 101 x 101 grid of pitch 0.5 um (CPU, GPU) | exp(i k (L x + M y)) | for tol 1e-2, 1e-6, 1e-9, 1e-12: 2.4e-4, 3.7e-8, 9.4e-11, 1.3e-13 (tol) |
| one ray at the worst position: 128 x 128 grid of pitch 2 um (fine grid exactly 2 Nx), L = M, 64 positions in one fine-grid cell and the two worst directions of the one review (CPU, GPU) | exp(i k (L x + M y)) | worst / tol for tol 1e-2, 1e-6, 1e-9, 1e-12: 0.08, 0.10, 0.16, 0.26 (1); one point narrower (the width before the one review): 1.68 (1e-9), 1.89 (1e-12) (above 1); the width of Barnett et al.: 14.58, 14.31 (above 1) |
| `_es_transform` (128 Gauss-Legendre nodes in t, z = sin t) at the modes 0 to n / 4, the widths of tol 1e-2 to 1e-12 | mpmath (tanh-sinh, 30 digits) | 6.4e-15 relative (1e-14); the rule in z with 128 nodes (before the one review): 8.6e-11 at w 5, 7.0e-15 at w 15 (above 1e-12 at w 5) |
| the non-uniform FFT in focus scans: the 4f image with a curved wavefront and the singlet PSF (41 planes), a Gaussian focus with rays to 0.6 (9 planes), a uniform pupil of NA 0.6 (3 planes; the last two on a 201 x 201 um grid) (CPU, GPU) | `image_planes`, direct sum, complex128 | GPU, field per sum \|a\| at tol 1e-12: 1.4e-14, 5.8e-15, 1.2e-14, 1.3e-14 (1e-12); \|d eta\| with the power of each plane or of the middle plane: at most 2.4e-14 at tol 1e-12 (1e-11), 3.1e-13 at tol 1e-9 (1e-8) |
| the 4f coupling of a 6 um Gaussian into 3 fiber radii, `image_sum(method="nufft")` (CPU, GPU) | the direct sum; the analytic Gaussian mismatch | GPU \|d eta\|: 4.0e-11 or less at tol 1e-6 (1e-8); 1.1e-16 or less at tol 1e-12 (1e-12); eta 0.99998, 0.95324, 0.93319 (analytic: 2e-3) |
| grids of the non-uniform FFT: `np.linspace` axes, one-point axes, a descending axis; a non-uniform axis (CPU, GPU) | the NumPy sum; the direct sum | 1e-12 of sum \|a\| and no warning; UserWarning and the direct field bit for bit |
| the non-uniform FFT in batches of 3 planes, with 3 copies, with one copy, in chunks of 31, 164 and 97 rays; one plane of the fine grid above the budget (chunks of `NUFFT_MIN_CHUNK` rays) (`SUM_ELEMENTS`, `ALLOC_SLACK`, `NUFFT_COPIES` patched) (CPU, GPU) | the default: one batch, 16 copies, one chunk | GPU 1.9e-16, 2.0e-16, 2.8e-16, 2.4e-16 of sum \|a\| (1e-14) |
| `_nufft_steps` with the default budget: Nx 2048 (fine grid 4096^2), Nx 4096 (8192^2), 1 and 21 planes | | 21333 rays; `NUFFT_MIN_CHUNK` = 4096 rays (1 ray before the one review) |
| peak GPU memory of the non-uniform FFT minus E (GPU): 1 and 21 planes 128 x 128 (budget 2^22), 41 planes 256 x 256 and 1 plane 512 x 512 (2^23), 5 planes 1024 x 1024 (2^25) | the budget | 0.60, 0.83, 0.80, 0.63, 0.65 (0.25 to 1) |
| bad inputs of the non-uniform FFT: tol 0, 1e-13, 0.1, NaN; method "fft"; complex64; `ray_chunk` 0 | | ValueError |

All tests ran on the GPU computer (RX 7900 XT; `.venv-optics` from `tools/env/optics_rocm.sh`, PyTorch 2.14.1 for
ROCm 7.2): 154 passed (`tests/test_analysis.py`: 27, `tests/test_device_pipeline.py`: 18,
`tests/test_image_planes.py`: 30, `tests/test_nufft.py`: 48, `tests/test_sources.py`: 14). The observed values of
the rows from "PyTorch-backend trace" down, and of the image-quality checks, come from the same computer. The
NumPy pipeline gives the same values as before the device pipeline: eta 0.99998, 0.95324, 0.93319 in the 4f check.

The Gaussian-beam formulas are textbook results for two Gaussian beams of equal radius (for example Saleh and
Teich, Fundamentals of Photonics, chapter 3). The focus check differs by 1.4e-3 because the image of the
ideal-lens model is not an exact Gaussian (see below).

Optiland's "paraxial" thin lens maps ray slopes (tan), not direction cosines (sin), so it does not meet the sine
condition: a beam with half-angle theta images about theta^2/2 too wide. That is a property of the ideal-lens
model, not of this method (the second check shows the theta^2 law).

No test yet varies the sqrt(J) weight in isolation (in the 4f system J is constant; the aberrated singlet varies it
only in part).

## Timings (`bench/bench_trace.py`)

The aberrated singlet (default) or the 2.5x lens pair (`--system 4f`), 1e4 to 1e7 rays on a square grid of
directions; one JSON line per case: the trace time with NumPy (CPU), PyTorch on the CPU and PyTorch on the GPU, and
the largest difference of each PyTorch trace from the NumPy trace.

GPU (RX 7900 XT, gfx1100; `.venv-optics` from `tools/env/optics_rocm.sh`; best of 2 runs; rows in
`benchmarks/2026-10-30/optics-trace.jsonl`), against two CPU traces on the same computer: NumPy on one thread (the
NumPy operations and the numba kernels of Optiland's NumPy backend are serial) and PyTorch on 8 threads
(`"threads": 8` in the rows, `torch.get_num_threads()`):

| System | Rays | NumPy, CPU | PyTorch, CPU | PyTorch, GPU | GPU result to the host |
|---|---|---|---|---|---|
| singlet | 1e4 | 0.0012 s | 0.0037 s | 0.0103 s | 0.0001 s |
| singlet | 1e5 | 0.0117 s | 0.0057 s | 0.0111 s | 0.0004 s |
| singlet | 1e6 | 0.091 s | 0.070 s | 0.019 s | 0.011 s |
| singlet | 1e7 | 1.356 s | 1.974 s | 0.186 s | 0.118 s |
| 4f | 1e5 | 0.0047 s | 0.0045 s | 0.0099 s | 0.0005 s |
| 4f | 1e6 | 0.077 s | 0.041 s | 0.017 s | 0.008 s |
| 4f | 1e7 | 1.018 s | 1.426 s | 0.138 s | 0.101 s |

At 1e7 rays the GPU trace is 7.3x (singlet) and 7.4x (4f) faster than NumPy on one thread, and 10.6x and 10.3x
faster than PyTorch on 8 CPU threads. With the copy of the result to the host, the GPU is 4.5x (singlet) and 4.3x
(4f) faster than NumPy. Up to about 1e5 rays the GPU is not faster, because each trace has a fixed cost (the copy
of the optic, the check rays, the kernel launches): at 1e5 rays it is equal to NumPy (singlet, 0.0111 s and
0.0117 s) or 2.1x slower (4f). The crossover is between 1e5 and 1e6 rays.

The GPU trace differs from the NumPy trace by 0 (positions and OPL, both systems, all ray counts). The PyTorch CPU
trace differs from NumPy by up to 1e-14 mm (positions) and 7e-14 mm (OPL).

## End-to-end timings (`bench/bench_pipeline.py`)

One image field (angular spectrum, trace, Jacobian and weights, image sum, the field to the host) of a point
source through the singlet (or `--system 4f`), 1e5 to 1e7 rays, `--image` x `--image` points: the NumPy pipeline
against the device pipeline (with its parts `bundle_s` = `trace_bundle` and `sum_s` = `image_sum`), and the
largest field difference of the device pipeline from the NumPy field. Each case has one untimed run first: the
first FFT of a new grid size compiles its rocFFT kernels (about 2 s, observed).

GPU (RX 7900 XT; rows in `benchmarks/2026-10-30/optics-pipeline.jsonl`; 128 x 128 image points; the NumPy
pipeline traces with NumPy and sums the image on the GPU, the device pipeline does both on the GPU):

| System | Rays traced | NumPy pipeline | Device pipeline (trace and weights + image sum) | Field difference |
|---|---|---|---|---|
| singlet | 1e5 | 0.040 s | 0.034 s (0.010 + 0.024) | 1.5e-15 |
| singlet | 1e6 | 0.412 s | 0.252 s (0.017 + 0.235) | 1.2e-15 |
| singlet | 1e7 | 4.58 s | 2.48 s (0.12 + 2.36) | 1.2e-14 |
| 4f | 1e5 | 0.050 s | 0.044 s (0.009 + 0.035) | 5.9e-16 |
| 4f | 1e6 | 0.509 s | 0.362 s (0.014 + 0.348) | 5.8e-16 |
| 4f | 1e7 | 5.42 s | 3.59 s (0.09 + 3.49) | 6.6e-15 |

The field difference is `max_rel_dE` of the rows: the largest |E_device - E_NumPy| relative to the largest |E|
(the GPU trace itself differs from the NumPy trace by 0; the sums run in a different order). The device pipeline
is 1.40x to 1.85x faster from 1e6 rays on. From 1e6 rays on, the image sum is 93 % to 97 % of the time of the
device pipeline; at 1e5 rays it is 70 % (singlet) and 80 % (4f). These rows are from before `image_planes`: the
image sum of one plane is now 2.1x to 2.2x faster (`single_s` against `per_plane_s` in the next section).

## Focus-scan timings (`bench/bench_image_sum.py`)

A point source through the singlet (as in `bench/bench_pipeline.py`, 128 x 128 image points, pitch 1 um), traced
one time; Z planes evenly from dz = -100 to +100 um. Time per plane: one sum per plane with the earlier method
(`per_plane_s`, the sum of the NumPy pipeline: one complex128 matrix product per chunk), `image_sum` (one plane of
`image_planes`, complex128), and `image_planes` for all Z planes in complex128 and complex64. The per-plane
methods are timed on 3 planes (on 1 for Z = 1); `image_planes` on all planes (best of 2 runs).

GPU (RX 7900 XT; `bench/bench_image_sum.py --rays 1e5 1e6 1e7 --planes 1 21 41`, run on 2026-10-30; rows in
`benchmarks/2026-10-30/optics-image-sum.jsonl`):

| Rays traced (kept) | Planes | One sum per plane | `image_sum` | `image_planes`, complex128 | `image_planes`, complex64 | Faster: complex128 / complex64 (`speedup_c128` / `speedup_c64`) |
|---|---|---|---|---|---|---|
| 1e5 (5.3e4) | 1 | 24.0 ms | 11.6 ms | 11.7 ms | 1.49 ms | 2.05x / 16.09x |
| 1e5 (5.3e4) | 21 | 23.7 ms | 11.5 ms | 7.55 ms | 0.85 ms | 3.14x / 27.99x |
| 1e5 (5.3e4) | 41 | 23.8 ms | 11.4 ms | 7.55 ms | 0.89 ms | 3.15x / 26.64x |
| 1e6 (5.3e5) | 1 | 236 ms | 109 ms | 108 ms | 13.8 ms | 2.17x / 17.06x |
| 1e6 (5.3e5) | 21 | 236 ms | 108 ms | 74.9 ms | 8.81 ms | 3.14x / 26.73x |
| 1e6 (5.3e5) | 41 | 236 ms | 109 ms | 74.9 ms | 8.77 ms | 3.15x / 26.84x |
| 1e7 (5.3e6) | 1 | 2.36 s | 1.08 s | 1.08 s | 0.142 s | 2.18x / 16.67x |
| 1e7 (5.3e6) | 21 | 2.36 s | 1.08 s | 0.752 s | 0.0883 s | 3.14x / 26.76x |
| 1e7 (5.3e6) | 41 | 2.36 s | 1.08 s | 0.752 s | 0.0880 s | 3.14x / 26.83x |

A focus scan of 41 planes at 1e7 rays takes 31 s in complex128 and 3.6 s in complex64, against 97 s with one sum
per plane. Agreement with one sum per plane (all rows): the field differs by at most 3.3e-14 (complex128) and
5.6e-7 (complex64) relative to the largest |E|; the coupling into a Gaussian mode of radius 8 um by at most
5.8e-16 (complex128) and 3.2e-9 (complex64). Between complex64 and complex128 over all planes, with one power for
all planes (the 1e7 rows only): 6.4e-9.

## Example: focus and lateral scans on the GPU (`examples/coupling_scans_on_device.py`)

The synthetic system of `examples/chip_to_fiber_coupling.py` (a SiN tip mode from meow-gpu, the 2.5x lens pair,
SMF-28). The rays are traced one time; then a focus scan of 25 planes (dz -60 to 60 um, `image_planes`) and a
lateral scan of 25 x 25 offsets (-3 to 3 um, `lateral_scan`) at the best plane run on the GPU. Two runs, because
meow-gpu (JAX) and Optiland with PyTorch are in two environments (`.venv-rocm` and `.venv-optics`); from the
repository folder on the GPU computer (the logged runs of 2026-11-07, with `OMP_NUM_THREADS`,
`OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` 4 and `XLA_PYTHON_CLIENT_MEM_FRACTION` 0.4, because another session
used the computer):

    bash tools/gpu_run.sh optics_gpu ../.venv-rocm/bin/python examples/coupling_scans_on_device.py --save-mode --pol x --mode-file ../results/optics_gpu/sin_tip_mode_sym_x.npz
    bash tools/gpu_run.sh optics_gpu python examples/coupling_scans_on_device.py --mode-file ../results/optics_gpu/sin_tip_mode_sym_x.npz

`--pol x` (the default, the usual TE source) or `y` sets the polarization of the source. The first run uses the
default polish of meow-gpu; `--polish davidson` gives the earlier polish (see below). Without `--mode-file`, the
mode file is `results/optics_gpu/sin_tip_mode.npz`. Rows in `benchmarks/2026-11-07/optics-sym-x.jsonl` (row 1:
the mode solve, row 2: the optics run).

**The core.** The grid has 0.05 um cells in a 14 x 14 um window, and the core (0.3 x 0.3 um) is 6 x 6 cells. The
example builds it from integer cell indices, so it is symmetric. Each permittivity sample gets the mean over the
cell around it: a sample on a face of the core (the Yee positions on cell corners) gets half core and half
cladding, one on a corner of the core a quarter. The rule matters, because 6 cells resolve the core poorly. In
development runs on the GPU computer (not in the log), neff was 1.44507 with the samples on the faces in the
cladding, 1.45079 with them in the core (the one review), and 1.44719 with the mean. Row 1 shows the symmetry: the
centroid of the source on its Yee positions is (6.5e-12, 4.8e-12) um, and its largest mirror asymmetry is 2.1e-11
of its maximum.

Before 2026-11-07 the core test used the coordinates of `np.arange(-7.0, 7.0 + 1e-9, 0.05)` with `|x| < 0.15`.
The rounding put the samples on one face into the core and those on the other face into the cladding
(g[143] = 0.1499999999999746 < 0.15, but g[137] = -0.15000000000002434). That core kept the x-y swap symmetry but
lost both mirror symmetries: its two fundamental modes split by 2.06e-4 into two diagonal polarizations (equal
power in Ex and Ey), and the mode moved off the axis (centroid (-0.0101, 0.0235) um on the Yee positions). The
earlier code took the field component with the larger maximum of mode 0; the two maxima were equal by the swap
symmetry, so rounding chose: Ey on 2026-10-30, Ex in the polish runs of 2026-11-07 (a source turned by 90
degrees). These values are in row 1 of `benchmarks/2026-11-07/optics-pol-x.jsonl` and in
`optics-polish.jsonl`; the one review of 2026-11-07 found the staircase error.

**The source mode.** The two fundamental modes of the symmetric square core are degenerate, and the solver can
give any basis of that pair. In row 1, modes 0 and 1 have the same neff 1.4471879 (difference 3.2e-14), x power
fractions |Ex|^2 / (|Ex|^2 + |Ey|^2) of 0.112 and 0.658, and an overlap |P01| / sqrt(P00 P11) of 0.713: a rotated
pair that is not orthogonal. "Mode 0" is therefore not a defined source. `optics_gpu.sources.polarized_mode`
selects it:
- modes 0 and 1 near-degenerate (|dneff| < `DEGENERATE_TOL` = 1e-3): the combination of the two with the largest
  power fraction in the chosen component (a 2 x 2 generalized eigenvalue problem), normalized. A ValueError stops
  it if the two modes are nearly the same field, or if they are not one x-like and one y-like field (for example
  two supermodes of one polarization);
- else the one of the two with the larger fraction; if the two fractions are equal to 1e-6, a ValueError stops it
  (the choice would depend on rounding);
- the example also refuses a source with less than 0.9 of its power in the chosen component (`MIN_FRACTION`).
Row 1: the source has 0.99848 of its power in Ex. The parts have the relative powers 0.695 and 1.811: their sum is
not 1, because the two modes are not orthogonal (the cross term). The x and y sources are transposes of each
other, so x and y give the same efficiencies in this rotationally symmetric system, and the lateral scan is
transposed. `tests/test_sources.py` checks the selection with synthetic fields.

**Output of row 2.** Mode radius 3.04 um (the second-moment radius of |Ex|^2 along x of the interpolated source);
13689 rays traced and 10877 kept; eta 0.8866 at the image surface, the best plane at dz = 0, eta 0.6665 at
dz = +-60 um; the 1 dB full width of the lateral scan along x 5.39 um (linear interpolation between the scan points;
5.0 um is the span of the 0.25 um scan points within 1 dB); the best lateral point at (0, 0). The source centroid is
(0, 0) um and the image centroid (-0.0001, -0.0001) um (to the 4 decimals of the output). Against the host
reference (the NumPy pipeline, the FFT focus scan and NumPy overlaps): 5.4e-11 (focus), 1.3e-15 (lateral). Time per
scan point: 1.61 ms per focus plane with `image_planes` in complex128 and 0.172 ms in complex64, against 4.99 ms
with one sum per plane (the earlier method, in the same run); 0.55 us per lateral point. The focus scan in one pass
differs from one sum per plane by 6.7e-16, complex64 from complex128 by 9.1e-8.

The FFT focus scan of the host reference took 0.31 ms per plane in row 2, and 8.8 to 8.9 ms in the rows of
2026-10-30 (the last rows of `benchmarks/2026-10-30/optics-pipeline.jsonl` and `optics-image-sum.jsonl`), with
the same environment (`.venv-optics`, installed on 2026-10-30 before those runs). Row 2 ran with 4 CPU threads,
the rows of 2026-10-30 without a limit. The cause of the difference is not found, so this README does not compare
the time of the FFT focus scan with the GPU times.

The efficiencies use p0, the power on the 64 um image grid at the image surface, for all planes. p0 is not the
total power, and the normalization of eta is uncertain by about 5e-4 here (`n_pad` 4):
- The power on the grid is between 0.99888 x p0 and 1.000434 x p0 over the 25 planes. The grid power increases
  above p0 at some planes, so at least 4.3e-4 of the power of the ray-sum field is off the grid at dz = 0: by that
  measure, eta 0.8866 is high by about 4e-4.
- The Parseval power of the kept rays (`RayBundle.power`) is 0.999648 x p0; normalized by it, eta at the image
  surface is 0.8869. In the 4f check with a Gaussian source, the grid power and `RayBundle.power` agree to
  5.5e-5; for this mode with its wide-angle part they differ by 3.5e-4 (for the source of 2026-10-30 with `n_pad`
  8: 5.6e-5, one development run).
These values are in row 2 (`grid_power_over_p0_min_max`, `parseval_power_over_p0`,
`eta_at_focus_parseval_power`).

**The mode solve.** It uses the default polish of meow-gpu (`"refine"`). For this weakly guided mode the
refinement alone does not converge, and the fallback of meow-gpu (since commit b495484) finishes the solve:
`polish_used` "refine+davidson", converged, residual 6.4e-11 after 216 iterations, 6.3 s with the JAX compilation
of a new process (row 1). Before that fix, the default polish stopped at a residual of 3.2e-10 after 255
iterations without its converged flag, and this example used the earlier polish. The earlier complex128 Davidson
polish stays available with `--polish davidson`. A comparison of the two polishes is in
`benchmarks/2026-11-07/optics-polish.jsonl` (rows 1 and 2: the mode solves with the default polish and with
`--polish davidson`; rows 3 and 4: the optics runs with each mode file). Those rows, and the rows of
`optics-pol-x.jsonl`, used the asymmetric core and the earlier source definitions, so their neff and efficiencies
are not those of this section. In the polish rows the default polish converged with a residual of 5.1e-11 after
282 iterations in 12.2 s, and `--polish davidson` with 7.2e-11 after 286 iterations in 8.5 s (faster there). The
two polishes gave the same mode: neff 1.447296249827 (difference 4.1e-14), and the saved mode fields (the major
component, normalized) differed by 1.7e-11 (one comparison of the two mode files on the GPU computer; not in the
log). The optics runs with the two mode files gave the same efficiencies.

## Known limits

- The image field of the angular-spectrum method is periodic, with period lambda / dL' in image space (dL': the
  direction step after the system). Keep the image grid smaller than that period: in the example (`n_pad` 4) the
  period is 240 um.
- The device pipeline keeps 40 bytes per kept ray (the bundle) and the box of the spectrum (17 bytes per traced
  ray); the trace temporaries are bounded by one block. The spectrum itself (complex128) is on the device during
  step 1.
- The image-quality analyses (`analysis.py`) use scalar fields (no polarization). The FFT PSF needs a sampled
  pupil phase (W changes by well below 0.5 waves between neighboring pupil samples) and a geometric spot inside the
  window lambda F (N - 1). The MTF frequency step assumes a flat image surface normal to z (Optiland projects on
  the tangent plane at the chief ray). The orientation of an off-axis FFT PSF follows Optiland (checked against
  Optiland, not against the image sum). `huygens_psf` takes only an object point on the axis at a finite distance.
- Batches (`pupil_wavefronts`, `defocus`, `polychromatic_psf`, `polychromatic_mtf`): a batch gives the numbers of
  one call for each field and wavelength because the ray computations are elementwise; an optic whose surface
  intersections iterate until all rays of a call converge can differ to its iteration tolerance (not tested here:
  the systems of the tests have analytic intersections). `defocus` needs a flat image surface normal to z, one
  homogeneous medium after the last surface, and the same index on both sides of the image surface at the primary
  wavelength (else a ValueError). The polychromatic MTF reads each OTF at nu / (its frequency step),
  linearly interpolated between lags (not tested against a finer pupil except on the perfect lens), and the
  lateral-color phase uses nu in cycles/mm: the frequency step of item 6 (from the marginal-ray directions) and
  the FFT frequency 1 / (M dx) (from F of item 5) differ slightly for an aberrated or off-axis field (not
  quantified). The copy cache keeps up to 8 PyTorch copies of optics; the key needs a dictionary that pickles (else
  no copy is kept).
- The medium after the image surface (older than the batches; review `reviews/2026-11-08-de7c55fed642.md`).
  Optiland 0.6.3 computes the paraxial values at the image surface in the medium after that surface (air by
  default), a medium that the light does not enter. `pupil_wavefront` takes the center of its reference sphere from
  Optiland's paraxial exit pupil (item 2), as Optiland does, so its radius depends on that medium. Test prints of
  `test_defocus_with_a_glass_image_space` (image space n 1.4): `image_solve` puts the image surface 38.3 mm after
  the last lens surface with air after the image surface and 53.2 mm with the index 1.4 after it (1.39 times); the
  reference-sphere radii are 29.3 and 56 mm; with air after it, the exit pupil moves with the image surface by
  dz (1 - 1 / 1.4). The comparisons with Optiland do not see this (Optiland uses the same exit pupil). For an image
  space in a medium, give the image surface the same medium as its material; `defocus` refuses the other case.

## Next

- Batched Zernike fits (one normal matrix for all fields and wavelengths) and a batched Huygens PSF (the image
  sum of several wavelengths in one pass).
- The non-uniform FFT as the default for uniform grids in complex128 (a method "auto"). The logged rows
  (`benchmarks/2026-11-07/optics-nufft.jsonl`) show it faster than the direct complex128 sum in all bench rows and
  in the example, and slower than complex64 on 128 x 128 grids; "auto" needs a rule for complex64.
- A float32 spread for the non-uniform FFT (with `dtype=torch.complex64`). It needs a time comparison of float32
  and float64 atomic adds on the GPU, and a bound for the float32 sums of many rays in one fine-grid point.
- `propagate_through` and `bench/bench_pipeline.py` with the complex64 option of the image sum.
