# optics-gpu

Physical optics through lens systems for photonics (the coupling use case of ZEMAX POP: a waveguide or fiber
mode, through lenses, into a fiber), on the GPU with PyTorch (ROCm or CUDA) and real ray tracing with
[Optiland](https://github.com/HarrisonKramer/optiland) (MIT).

```python
from optics_gpu.pop import gaussian, overlap_efficiency, propagate_through

img = propagate_through(optic, source_field, dx_um, wavelength_um, x_img_um, y_img_um)   # optic: an Optiland Optic
eta = overlap_efficiency(img.E, gaussian(x_img_um, y_img_um, 5.2))                      # into SMF-28 (MFD 10.4 um)
```

## Method (`src/optics_gpu/pop.py`)

1. Angular spectrum of the source (FFT): one direction (L, M) per sample.
2. One real ray per direction from the object point, traced by Optiland to the image surface: image-side
   directions, positions and optical path W.
3. Image field E(x, y) = sum_r A_r sqrt(J_r) df exp(i k (W_r + L'_r (x - x_r) + M'_r (y - y_r))), with
   J = |d(L', M')/d(L, M)| (power conservation per ray tube). The sum is separable in x and y: one matrix
   product on the GPU.
4. Coupling efficiency by the normalized overlap with the receiving mode.

Valid for a source that is small compared with the isoplanatic patch of the system (true for micrometer modes).
Scalar fields; no obliquity factor (moderate NA); object and image spaces are air. **Give the stop a physical aperture** (`RadialAperture`): the EPD
setting of Optiland aims its ray generator but does not clip rays, and without a physical aperture a point source
overfills the lens (found in the singlet check: correlation 0.30 without it, 0.998 with it).

## Checks (`tests/test_pop.py`)

| Case | Reference | Agreement (observed; test tolerance) |
|---|---|---|
| 4f system of two ideal thin lenses (f 2 mm / 5 mm), 6 um Gaussian (half-angle 0.082 rad) | magnification 2.5; analytic Gaussian-mismatch coupling | image radius 0.3 % (0.5 %); coupling 1e-3 (2e-3) |
| the same at half-angle 0.247 rad | the image error grows as theta^2 | ratio within 25 % of (0.247/0.082)^2 |
| power through the lossless 4f system | integral of \|E\|^2 at the source | (1 %) |
| point source through a plano-convex singlet at 2f-2f, strong spherical aberration (6 mm stop) | Optiland's scalar Huygens PSF (independent method) | correlation 0.998 (0.99) |

Optiland's "paraxial" thin lens maps ray slopes (tan), not direction cosines (sin), so it does not meet the sine
condition: a beam with half-angle theta images about theta^2/2 too wide. That is a property of the ideal-lens
model, not of this method (the second check shows the theta^2 law).

No test yet varies the sqrt(J) weight in isolation (in the 4f system J is constant; the aberrated singlet varies it
only in part).

## Next

- The photonics example: a waveguide mode (from meow-gpu) into a fiber through a lens pair, with a focus scan on
  the GPU.
- Optiland's own ray trace on the GPU (its PyTorch backend) for large ray counts; benchmarks on ROCm.
