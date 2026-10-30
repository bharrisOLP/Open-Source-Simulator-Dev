# optics-gpu

Physical optics through lens systems for photonics (the coupling use case of ZEMAX POP: a waveguide or fiber
mode, through lenses, into a fiber), on the GPU with PyTorch (ROCm or CUDA) and real ray tracing with
[Optiland](https://github.com/HarrisonKramer/optiland) (MIT).

```python
from optics_gpu.pop import gaussian, overlap_efficiency, propagate_through

img = propagate_through(optic, source_field, dx_um, wavelength_um, x_img_um, y_img_um)   # optic: an Optiland Optic
eta = overlap_efficiency(img.E, gaussian(x_img_um, y_img_um, 5.2))                      # into SMF-28 (MFD 10.4 um)
img = propagate_through(..., trace_device="cuda")    # the ray trace on the GPU (Optiland's PyTorch backend, float64)
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
4. traces all rays through the copy (`torch.no_grad`). The tensors come back to NumPy for the Jacobian and the
   weights (NumPy, as before), and then go to the GPU for the image sum.

`ImageField.trace_device` gives the device of the trace (`"numpy"`, `"cpu"` or `"cuda:0"`).

## Checks (`tests/test_pop.py`, `tests/test_trace_backends.py`)

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

All tests ran on the GPU computer (RX 7900 XT; `.venv-optics` from `tools/env/optics_rocm.sh`, PyTorch 2.14.1 for
ROCm 7.2): 17 passed. The observed values of the rows from "PyTorch-backend trace" down come from the same
computer.

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

## Next

- The photonics example: a waveguide mode (from meow-gpu) into a fiber through a lens pair, with a focus scan on
  the GPU.
- The Jacobian and the weights on the GPU (today NumPy), so that a GPU trace does not go back to the host.
