# optics-gpu

Physical optics through lens systems for photonics (the coupling use case of ZEMAX POP: a waveguide or fiber
mode, through lenses, into a fiber), on the GPU with PyTorch (ROCm or CUDA) and real ray tracing with
[Optiland](https://github.com/HarrisonKramer/optiland) (MIT).

```python
from optics_gpu.pop import gaussian, overlap_efficiency, propagate_through

img = propagate_through(optic, source_field, dx_um, wavelength_um, x_img_um, y_img_um)   # optic: an Optiland Optic
eta = overlap_efficiency(img.E, gaussian(x_img_um, y_img_um, 5.2))                      # into SMF-28 (MFD 10.4 um)
img = propagate_through(..., trace_device="cuda")    # all steps on the GPU (the device pipeline, float64)

from optics_gpu.pop import image_sum, lateral_scan, overlap_efficiency_device, trace_bundle
bundle = trace_bundle(optic, source_field, dx_um, wavelength_um, device="cuda")   # trace one time
E = image_sum(bundle, x_img_um, y_img_um, dz=20.0)       # the field 20 um after the image surface, on the GPU
```

## Method (`src/optics_gpu/pop.py`)

1. Angular spectrum of the source (FFT): one direction (L, M) per sample.
2. One real ray per direction from the object point, traced by Optiland to the image surface: image-side
   directions, positions and optical path W. The trace runs on Optiland's NumPy backend (the default,
   `trace_device=None`) or on its PyTorch backend on `"cpu"` or `"cuda"` (see "Ray trace on the GPU").
3. Image field E(x, y) = sum_r A_r sqrt(J_r) df exp(i k (W_r + L'_r (x - x_r) + M'_r (y - y_r))), with
   J = |d(L', M')/d(L, M)| (power conservation per ray tube). The sum is separable in x and y: one matrix
   product on the GPU, in chunks of rays (`ray_chunk`; by default (Nx + Ny) x chunk = 2^25). The sum keeps one
   chunk of the x and y factors in memory: (Nx + Ny) x chunk complex128 values, 512 MB by default.
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
4. `image_sum(bundle, x, y, dz)`: E = sum_r a_r exp(i k (L'_r x + M'_r y + N'_r dz)), in chunks of rays as
   before. One plane wave per ray, so `dz` is the angular-spectrum propagation in image space: a focus scan needs
   one trace and one image sum per plane. `lateral_scan` gives the coupling into a Gaussian mode moved to every
   point of a 2-D grid of lateral offsets with two matrix products (the Gaussian is separable).

The data that come to the host inside the device pipeline are the 25 check rays of the copy of the optic and
scalars: the bounding box, and about 50 scalars per block from Optiland's own tests in its trace (observed).

In a focus scan, use one power for all planes (`power=` of `overlap_efficiency_device` and `lateral_scan`). The
power on a finite image grid changes with dz: wide-angle power leaves the grid, and power that is off the grid at
the image surface comes onto it. The FFT method of `examples/chip_to_fiber_coupling.py` keeps the power of its
periodic grid. In the example below, eta normalized by the power on the grid at each plane differed from the FFT
method by 7.8e-4 at dz = 60 um; normalized by p0, the power on the grid at the image surface (as the FFT method),
by 5.9e-11. That agreement shows that the two propagation methods agree with the same normalization. It does not
show that p0 is the total power (see the example).

## Checks (`tests/test_pop.py`, `tests/test_trace_backends.py`, `tests/test_device_pipeline.py`)

| Case | Reference | Agreement (observed; test tolerance) |
|---|---|---|
| 4f system of two ideal thin lenses (f 2 mm / 5 mm), 6 um Gaussian (half-angle 0.082 rad) | magnification 2.5; analytic Gaussian-mismatch coupling | image radius 0.3 % (0.5 %); coupling 1e-3 (2e-3) |
| the same at half-angle 0.247 rad | the image error grows as theta^2 | ratio within 25 % of (0.247/0.082)^2 |
| power through the lossless 4f system | integral of \|E\|^2 at the source | (1 %) |
| point source through a plano-convex singlet at 2f-2f, strong spherical aberration (6 mm stop) | Optiland's scalar Huygens PSF (independent method) | correlation 0.998 (0.99) |
| PyTorch-backend trace (float64) of 101 x 101 directions, \|L\|, \|M\| <= 0.09, through the 4f system and the singlet (the stop clips about half of them), on the CPU and on the GPU | Optiland's NumPy-backend trace of the same rays | GPU: 0. CPU: positions 7e-15 mm, directions 9e-16, OPL 4e-14 mm, intensity 0 (1e-12 mm and 1e-12); the same NaN rays |
| the 4f coupling into 3 fiber radii and the singlet PSF field, traced by PyTorch (CPU and GPU) | the same with the NumPy trace | eta: 0 (1e-12); field: GPU 0, CPU 3.6e-12 relative (1e-9); the NumPy values are the same as before this change: eta 0.99998, 0.95324, 0.93319 |
| image sum in 12 chunks of 401 rays (4f, 4501 rays) | one sum | 9e-15 relative (1e-12) |
| peak GPU memory of the image sum, 4f, chunks of 2048 rays, 128 x 128 image | one chunk of the x and y factors, (128 + 128) x 2048 complex128 values | 1.08x (below 1.3x) |
| `ray_chunk` 0 and -1 | (an all-zero field before the one review) | ValueError |
| an optic made under Optiland's PyTorch backend in float32 | the precision of the check rays | ValueError that names float32 |
| a NumPy optic while Optiland's PyTorch backend (float32) is active | the NumPy trace | the check rays run on NumPy; passes (1e-12) |
| a PyTorch copy of the optic with the image surface moved by 10 um | the check rays of `trace_rays` | RuntimeError |
| `_np_gradient0` on a uniform axis, the `fftshift(fftfreq)` axis and a random axis, along both axes (CPU, GPU) | `np.gradient` | (1e-13 relative) |
| device pipeline: the 4f coupling into 3 fiber radii and the singlet PSF field, in one block and in blocks of about 500 rays (CPU, GPU) | the NumPy pipeline | field, GPU: 7e-16 (one block), 1.3e-14 (blocks); CPU: 9e-15 (4f), 5.6e-12 (singlet) (4f 1e-12, singlet 1e-9); eta 4e-16 (1e-12); the same ray counts |
| device PSF of the singlet, blocks of about 4000 rays | Optiland's Huygens PSF | correlation (0.99) |
| `trace_bundle` and `image_sum` on the GPU, 4f, point source, 5.5e4 rays in 8 blocks, watched by a `TorchFunctionMode` | no large tensor made on the CPU; arrays to the host only of the 25 check rays; scalars bounded per block | 20 arrays of 25 values (at most 40); 421 scalars (at most 60 per block + 20) |
| the watch itself: 10 forms of a copy of 30 values to the host (`cpu`, `to` with a device name, a device, a keyword, a dtype or a CPU tensor, `tolist`, `numpy`, `copy_`, `Tensor.cpu`) and 4 forms of a scalar | | each form is seen |
| `image_sum(dz=d)` against the NumPy pipeline with the image surface moved by d: 4f, Gaussian with a curved wavefront (R 1.5 mm), d = -80 and +80 um; singlet PSF, d = -60 and +60 um (CPU, GPU) | Optiland's trace to the moved surface | 4f 2.1e-12, 2.9e-12 (1e-10); singlet 2.2e-11, 2.6e-11 (GPU), 3.0e-11 (CPU) (1e-9); with the opposite sign of dz 1.0, 1.7, 2.1 (above 0.5); eta(-80 um) 0.9959, eta(+80 um) 0.9875 |
| `RayBundle.power`, 4f, 6 um Gaussian (CPU, GPU) | the source power; the power of the image field on the 96 um grid | -1e-12 (1e-9); 5.5e-5 (1e-4) |
| peak GPU memory of `trace_bundle`, 4f, point source, 2.45e5 rays traced, 1.93e5 kept | blocks of 4096 rays against one block of all rays | 53 bytes per traced ray (below 64) against 249 (difference above 150) |
| focus scan by `image_sum(dz)`, 4f image of a 6 um Gaussian into a 15 um Gaussian mode, dz -300 to 400 um (CPU, GPU) | 1 / (1 + (lambda dz / (2 pi w^2))^2); the FFT angular-spectrum propagation of the dz = 0 field | 1.4e-3 (2e-3); FFT: 1.5e-6 (1e-5), 2.7e-7 with the power at dz = 0 (1e-6) |
| `lateral_scan`, offsets -10 to 8 um in x and y | exp(-s^2 / w^2) times the centered value; `overlap_efficiency` of the moved mode | 8.1e-4 (2e-3); 3e-16 (1e-12) |
| bad inputs of the device pipeline ("mps", `trace_chunk` 0, no kept direction, `ray_chunk` 0) | | ValueError |

All tests ran on the GPU computer (RX 7900 XT; `.venv-optics` from `tools/env/optics_rocm.sh`, PyTorch 2.14.1 for
ROCm 7.2): 35 passed (`tests/test_device_pipeline.py`: 18). The observed values of the rows from "PyTorch-backend
trace" down come from the same computer. The NumPy pipeline gives the same values as before the device pipeline:
eta 0.99998, 0.95324, 0.93319 in the 4f check.

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
device pipeline (the next target); at 1e5 rays it is 70 % (singlet) and 80 % (4f).

## Example: focus and lateral scans on the GPU (`examples/coupling_scans_on_device.py`)

The synthetic system of `examples/chip_to_fiber_coupling.py` (a SiN tip mode from meow-gpu, the 2.5x lens pair,
SMF-28). The rays are traced one time; then a focus scan of 25 planes (dz -60 to 60 um, `image_sum`) and a
lateral scan of 25 x 25 offsets (-3 to 3 um, `lateral_scan`) at the best plane run on the GPU. Two runs, because
meow-gpu (JAX) and Optiland with PyTorch are in two environments (`.venv-rocm` and `.venv-optics`); from the
repository folder on the GPU computer:

    bash tools/gpu_run.sh optics_gpu ../.venv-rocm/bin/python examples/coupling_scans_on_device.py --save-mode
    bash tools/gpu_run.sh optics_gpu python examples/coupling_scans_on_device.py

Output of one run (GPU computer, 2026-10-30): neff 1.4473, mode radius 2.71 um, 13689 rays traced and 10877 kept;
eta 0.8888 at the image surface (the earlier example: 0.889), the best plane at dz = 0, eta 0.6667 at
dz = +-60 um; the 1 dB full width of the lateral scan 5.30 um (linear interpolation between the scan points; 5.25 um
is the span of the 0.25 um scan points within 1 dB). Against the host reference (the NumPy pipeline, the FFT focus
scan and NumPy overlaps): 5.9e-11 (focus), 1.4e-15 (lateral). Time per scan point (the last row of
`optics-pipeline.jsonl`): 5.0 ms per focus plane on the GPU against 8.9 ms for the NumPy FFT focus scan; 0.8 us per
lateral point.

The efficiencies use p0, the power on the 64 um image grid at the image surface, for all planes. p0 is not the
total power, and the normalization of eta is uncertain by about 5e-4 here (`n_pad` 4):
- The power on the grid is 1.000455 x p0 at dz = +-30 um and 0.998826 x p0 at dz = +-60 um. The grid power
  increases first, so at least 4.6e-4 of the power of the ray-sum field is off the grid at dz = 0: by that
  measure, eta 0.8888 is high by about 4e-4.
- The Parseval power of the kept rays (`RayBundle.power`) is 0.999645 x p0; normalized by it, eta at the image
  surface is 0.8891. In the 4f check with a Gaussian source, the grid power and `RayBundle.power` agree to
  5.5e-5; for this mode with its wide-angle part they differ by 3.6e-4 (with `n_pad` 8: 5.6e-5, one run).
These values are in the output of the example (`grid_power_over_p0_min_max`, `parseval_power_over_p0`,
`eta_at_focus_parseval_power`); the `n_pad` 8 value is from one development run, not in the output.

The mode solve uses the complex128 Davidson polish of meow-gpu (`polish="davidson"`): with the default polish
(`"refine"`), the solver stopped at a residual of 3.2e-10 after 255 iterations without its converged flag for
this weakly guided mode, on the GPU computer.

The lateral scan has its maximum at x = -0.25 um, not 0: the two points -0.25 and 0 give the same eta (0.8888).
The cause is the source, not the pipeline: the intensity centroid of the interpolated source mode is at
x = +0.0507 um (y = -0.0006 um), and that of the image at x = -0.1235 um (y = +0.0012 um). The lens pair inverts
the image (magnification -2.5), so the expected image offset is -0.127 um. (The source is the mode as the
earlier example interpolates it: the major field component on the cell centers of the mode-solver grid.)

## Known limits

- The image field of the angular-spectrum method is periodic, with period lambda / dL' in image space (dL': the
  direction step after the system). Keep the image grid smaller than that period: in the example (`n_pad` 4) the
  period is 240 um.
- The device pipeline keeps 40 bytes per kept ray (the bundle) and the box of the spectrum (17 bytes per traced
  ray); the trace temporaries are bounded by one block. The spectrum itself (complex128) is on the device during
  step 1.

## Next

- A larger image grid or many planes in one image sum (a batch of dz values in one matrix product).
