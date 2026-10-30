# meow-gpu

A GPU finite-difference eigenmode (FDE) solver for [meow](https://github.com/gdsfactory/meow) (EME), in JAX.
First target: AMD GPUs with ROCm (tested on a Radeon RX 7900 XT, gfx1100, ROCm 10.1.0, JAX 0.11.1).

```python
import meow as mw
from meow_gpu.meow_backend import GPUModeSolver, compute_modes_gpu

solver = GPUModeSolver(precision="single")          # keeps the last modes: the next cell starts from them
modes = [compute_modes_gpu(cs, num_modes=10, solver=solver) for cs in cross_sections]
S, port_map = mw.compute_s_matrix(modes, cells=cells)   # the rest of meow does not change
```

## How it works

- **The same discrete problem as meow's default solver.** The operator is the full-vector Yee-grid operator of the
  tidy3d local mode solver (diagonal media, PEC/PMC walls, polynomial PML), applied matrix-free with 1-D stencils.
  `tests/test_operator.py` checks it against the tidy3d matrix to machine precision (5e-16).
- **Eigensolver.** Block generalized Davidson with a fast-diagonalization preconditioner (the separable operator
  with a reference permittivity, inverted with dense 1-D eigenbases: two small matrix products per side, good for
  a GPU, valid on non-uniform grids and with PML). Fixed-shape buffers, so JAX compiles once per grid size.
- **Warm start.** The solver keeps the modes of its last solve. The cells of a taper change slowly, so the next
  cell needs about 15 iterations instead of 50 to 90.
- **Precision.** `single` (default): complex64 iteration; the tolerance follows the complex64 floor. `mixed`:
  complex64, then a complex128 polish (tidy3d-level agreement, slow on cards with slow FP64). `double`.
- **Device.** klujax (imported by meow through sax) sets the JAX default platform to `cpu`. meow-gpu therefore puts
  its arrays on the GPU explicitly (`meow_gpu/device.py`). Set `MEOW_GPU_DEVICE=cpu` to force the CPU.

Not supported yet (falls back to tidy3d with a warning): bend radius, angled propagation. meow passes diagonal
permittivity only. Modes below cutoff follow the tidy3d direction rule (power flow when it is reliable, else
keff >= 0).

## Results (2026-10-30, RX 7900 XT against one core of a Ryzen 7 7800X3D)

Synthetic structures with textbook indices (Si 3.476, SiO2 1.444, 1.55 um). The reference is tidy3d's local
mode solver (as called by meow) on the same grid.

**One cross-section** (`bench/bench_fde.py`). "Cold" and "warm" time the eigen solve only, after the JIT compile
and the preconditioner setup (first call: about 1 to 5 s more); the tidy3d time includes its setup. Commands:
`bench_fde.py --dx 0.02 --modes 6 --precision single --extra 8`, the same with `--dx 0.01`, and
`bench_fde.py --width 3.0 --w-dom 5.0 --h-dom 2.0 --dx 0.01 --modes 20 --precision single --tol 1e-4 --extra 8`;
`OMP_NUM_THREADS=1`.

| Case | Unknowns | Modes | tidy3d | GPU single, cold | GPU single, warm | max \|dneff\| |
|---|---|---|---|---|---|---|
| 0.5 um strip, 20 nm grid | 30k | 6 | 2.54 s | 0.44 s | 0.15 s | 1.7e-6 |
| 0.5 um strip, 10 nm grid | 120k | 6 | 5.73 s | 1.00 s | 0.52 s | 3.3e-6 |
| 3.0 um strip, 10 nm grid | 200k | 20 | 8.34 s | 2.14 s | 0.73 s | 2.5e-6 |

`mixed` agrees with tidy3d to 2e-11 in neff, but it is 1.5x to 5x slower than `single` on this card.

**EME of a taper** (`bench/bench_eme_taper.py`: Si strip 0.5 -> 2.0 um over 10 um, 30 cells, 10 modes,
20 nm grid, 36k unknowns per cell). The times include the compile, the setup and meow's post-processing of all
cells. The 8-process tidy3d run scales only 1.55x over 1 process (process start and imports included), so the
fairer CPU baseline is between the two rows:

| Mode solver | Mode solves | \|S21\| (fundamental) |
|---|---|---|
| tidy3d, 1 process | 20.6 s | 0.97570552 |
| tidy3d, 8 processes | 13.3 s | 0.97570552 |
| meow-gpu single (1 process) | 6.3 s | 0.97568 (2.3e-5 lower) |
| meow-gpu mixed | 20.9 s | 0.97570552 (4e-12) |

The single-precision error of the S-parameters (about 2e-5, measured on this one case) is far below the
discretization error of a 20 nm grid (neff changes by about 4e-3 between a 20 nm and a 10 nm grid). The mode
fields are less accurate than neff in single precision (field errors up to about 7e-4 at a 10 nm grid, found by
the review): use `mixed` when field accuracy matters. On fine grids the complex64 floor rises as 1/dx^2; the
tolerance is capped at 1e-3 with a warning.

## Known limits and next steps

- At small cross-sections (about 36k unknowns) the fixed cost per iteration (host eigenproblem, kernel launches)
  dominates: the GPU is about 3x faster than one core. Next: solve several cells at once (batched Davidson).
- The complex128 polish is slow on consumer GPUs. Next: mixed-precision refinement of each eigenpair (complex128
  residuals, complex64 corrections).
- With PML, the solver can miss PML eigenvalues that are nearer the target than the kept modes (meow filters PML
  modes, so the kept modes were still correct in the review test). No test solves PML eigenpairs yet.
- meow's own post-processing (normalization, Gram-Schmidt, on the CPU) costs about 35 ms per cell.

## Tests

`uv run pytest` (CPU) checks the operator against the tidy3d matrix, the eigenvalues against tidy3d in each
precision, the warm start, the meow backend (neff and field overlap against `compute_modes_tidy3d`), and the device
placement. On the GPU computer: `bash tools/remote.sh setup-env`, then run pytest with `.venv-rocm`.
