# Upstream issues found in this work

Errors and limits of upstream packages that this repository found and works around. Nothing here was reported
upstream. The user decides if and where to report them.

## meow 0.15: Gram-Schmidt changes the neff of the higher modes

- **Where:** `meow/mode.py`, `Mode.__add__` returns `neff = 0.5 * (self.neff + other.neff)`.
  `meow/fde/post_process.py`, `orthonormalize_modes` computes `current - (inner_product(b, current) / ...) * b`.
- **Effect:** after the default post-processing (`compute_modes_tidy3d`, and every backend that uses
  `post_process_modes`), each mode after the first has an average of the neff of the earlier modes, also when the
  projection coefficient is about 0. Example: a 0.525 x 0.22 um Si strip in SiO2 at 1.55 um (20 nm grid), TM0 1.794 becomes 2.137
  (the mean of TM0 and TE0). The EME propagates the higher modes with these wrong phases, so the mode conversion
  of a taper adds up coherently: a linear taper 0.5 -> 2.0 um over 10 um gives |S21|^2 = 0.955; with the
  correction 0.991 (120 cells), and the 3-D FDTD gives 0.9945 (20 nm grid). More modes make the error larger
  (20 modes: 0.947); with the correction the result does not change with 20 modes (0.9899 against 0.9900).
- **Fix in this repository:** `meow_gpu.meow_backend.post_process_modes` (the default of `compute_modes_gpu`) does
  the same filter, normalization and Gram-Schmidt, and keeps the neff of each mode. Test:
  `meow_gpu/tests/test_meow_backend.py::test_post_processing_keeps_the_neff_of_each_mode`.
- **Possible upstream fix:** the Gram-Schmidt step keeps `mode.neff` (the mode that it orthogonalizes), and
  `Mode.__add__` does not average neff (a sum of two modes has no neff).

## meow 0.15: a Prism along z is never present inside its z range

- **Where:** `meow/geometries.py`, `Prism._project_axis_z`: `if z < self.h_min or z < self.h_max: return []`. The
  second test should be `z > self.h_max`.
- **Effect:** a prism with `axis="z"` (a cross-section polygon that is constant along the propagation) projects
  only at `z >= h_max`, so cells inside its range do not contain it.
- **Fix in this repository:** `photonsuite.eme._z_prism` uses a subclass with the corrected test. Test:
  `suite/tests/test_eme.py::test_a_polygon_segment_continues_the_guide` (with the upstream prism the polygon half
  is missing and T = 2.7e-29).

## fdtdx 0.6.3: fails when JAX float64 is enabled

- **Effect:** with `jax_enable_x64` on, `run_fdtd` mixes float64 values into its float32 time loop, and the
  equinox while loop raises "`body_fun` must have the same input and output structure".
- **Workaround in this repository:** `photonsuite.fdtd.transmission` runs fdtdx inside `jax.enable_x64(False)`.
  This may be a limit (fdtdx uses float32), not an error.
