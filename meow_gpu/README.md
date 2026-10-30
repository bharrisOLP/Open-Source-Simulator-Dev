# meow-gpu

A GPU finite-difference eigenmode (FDE) solver for [meow](https://github.com/gdsfactory/meow) (EME), in JAX.
First target: AMD GPUs with ROCm (tested on a Radeon RX 7900 XT, gfx1100, ROCm 10.1.0, JAX 0.11.1).

```python
import meow as mw
from meow_gpu.meow_backend import GPUModeSolver, compute_modes_gpu, compute_modes_gpu_batch

solver = GPUModeSolver(precision="single")          # keeps the last modes: the next cell starts from them
modes = [compute_modes_gpu(cs, num_modes=10, solver=solver) for cs in cross_sections]
# or: 8 cross-sections of the same grid shape in each solve (the same results)
modes = compute_modes_gpu_batch(cross_sections, num_modes=10, solver=solver, batch=8)
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
- **Batched solves.** `compute_modes_gpu_batch(css, num_modes, batch=B)` solves B cross-sections with the same grid
  shape at the same time (`GPUModeSolver.solve_batch`, `fde.eigensolver.davidson_batch`). The device kernels
  (operator, preconditioner, Gram products, Ritz vectors, inserts) run on a leading batch axis (`jax.vmap`); the
  small projected eigenproblems and the SVQB stay on the host, one for each member, in a thread pool. Each member
  has its own subspace size, restarts, stall rule and stop, so each member does the steps of the per-cell solver
  (on the CPU, from the same initial block, the test cases give the same iterations and the same eigenvalues). A
  member that stops keeps its result. The precision modes are the same. Cell order: the cells are divided into B
  runs of neighbouring cells, and batch j solves cell j of each run; member i starts from the last vectors of
  member i of the previous batch (the previous cell of its run). The modes go through the same post-processing and
  direction rule as `compute_modes_gpu`.
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
fairer CPU baseline is between the two rows. These runs used meow's own post-processing on both sides (before the
neff correction below), so the |S21| values compare the two mode solvers, not the physics:

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

**Batched EME of the same taper** (`bench/bench_eme_batch.py`: per-cell, then B = 4, 8 and 16; one JSON line per
case with the mode-solve time, |S21|, max |dneff| against the per-cell run and the iterations). On the RX 7900 XT
(`--repeat 2`, the second run, after compilation; rows in `benchmarks/2026-10-30/eme-batch.jsonl`):

| Case | single: mode solves | single: max \|dneff\| | mixed: mode solves | mixed: max \|dneff\| |
|---|---|---|---|---|
| per-cell | 5.53 s | - | 19.45 s | - |
| B = 4 | 4.75 s | 4.6e-6 | 18.27 s | 5.9e-12 |
| B = 8 | 4.49 s | 7.3e-6 | 20.75 s | 6.3e-12 |
| B = 16 (15 used) | 4.57 s | 4.8e-6 | 18.75 s | 4.9e-12 |

On this card the batched path is only about 1.2x faster in single precision, and it is not faster in mixed
precision, for this taper (30 cells, 36k unknowns per cell). It stays an option; the per-cell path is the default.
The iterations in the same rows show why (time = `modes_s`, which also includes meow's post-processing, about 0.6 s
on a CPU):

- Single precision: a per-cell iteration takes about 11.5 ms (5.53 s for 480 iterations). A batch iteration takes
  about 27 / 43 / 63 ms with 4 / 8 / 15 members (174 / 105 / 73 iterations). So one batch iteration costs 2.4x /
  3.7x / 5.4x a per-cell iteration for 4x / 8x / 15x more members: the work that grows with each member is the
  largest cost, not the fixed cost per iteration.
- For each member iteration the batch costs less (about 6.8 / 5.3 / 4.2 ms against 11.5 ms), but it does more
  member iterations (batch iterations x members: 696 / 840 / 1095 against 480). The causes: the cold first batch
  (about 46 iterations in each member), the members that stopped (they still go through the kernels), and the
  copies of the last cell in a short last run.
- Mixed precision: about 12.0 ms for each per-cell iteration (1625 iterations), and about 31 / 58 / 97 ms for each
  batch iteration (597 / 361 / 194 iterations): 2388 / 2888 / 2910 member iterations against 1625, so no gain.

The cause of the cost for each member (device kernels or host eigenproblems) is not profiled yet.

On a CPU the batched path is not faster (the CPU has no fixed cost to share). Reference run on an Apple M5 Pro CPU
(`--precision single --repeat 1`; times include the JIT compile). These rows are not logged in `benchmarks/`. The
review ran the same command again: the same iterations, |S21| and max |dneff|, and times within about 10 %
(7.8 / 8.8 / 10.5 / 13.9 s).

| Case | Mode solves | Iterations (sequential steps) | Mean iterations of each member | \|S21\| | max \|dneff\| |
|---|---|---|---|---|---|
| per-cell | 7.9 s | 477 | 15.9 | 0.9933414 | - |
| B = 4 | 9.1 s | 170 | 17.7 | 0.9933439 | 3.4e-6 |
| B = 8 | 11.6 s | 101 | 21.1 | 0.9933439 | 4.8e-6 |
| B = 16 (15 used) | 15.1 s | 68 | 29.3 | 0.9933448 | 4.8e-6 |

The per-cell mean 15.9 is 477 / 30: it includes the cold first cell (the JSON field `mean_iterations` leaves that
cell out and gives 14.9; `mean_iterations_all` includes it). The batch means are `mean_member_iterations`: all
members, with the cold first batch and the copies of the last cell. The batch needs 2.8x to 7x fewer sequential
iterations (a batch runs to the stop of its slowest member). Each member does more iterations on average: the first
batch starts cold in all B members (about 46 iterations, against about 15 for a warm start), and a short last run
adds copies of the last cell (B = 4 and 8: 32 members for 30 cells). The differences against the per-cell run are
at the single-precision tolerance.

## Correction of meow's post-processing (each mode keeps its neff)

In meow 0.15, `Mode.__add__` averages the neff of the two modes, and the Gram-Schmidt step of
`post_process_modes` uses it. So after meow's post-processing, every mode after the first has an average of the
neff of the earlier modes (a 0.525 um strip, the first cell of the taper: TM0 2.137 instead of 1.794). The EME then propagates the higher modes
almost in phase with the fundamental mode, and the mode conversion of a taper adds up coherently.
`compute_modes_gpu` therefore uses `meow_gpu.meow_backend.post_process_modes`: the same filter, normalization and
Gram-Schmidt, but each mode keeps its own neff (test `test_post_processing_keeps_the_neff_of_each_mode`; details in
`docs/upstream-issues.md`). For the taper above (mixed precision, 10 modes): |S21|^2 = 0.987 / 0.990 at 30 / 60
cells instead of 0.952 / 0.955, and the 3-D FDTD of the same taper gives 0.9961 on a 20 nm grid
(`benchmarks/2026-10-30/fdtd-v3.jsonl`, row 4: T00 = 0.99614; `suite/README.md`).
`bench_eme_taper.py --post-process meow` gives the old numbers.

## Known limits and next steps

- At small cross-sections (about 36k unknowns) the GPU is about 3x faster than one core. The batched solve (above)
  gains only about 1.2x in single precision and nothing in mixed precision on the RX 7900 XT: the work that grows
  with each member costs more than the fixed cost per iteration, and the batch does more member iterations. Next:
  profile a batch iteration (device kernels against host eigenproblems).
- Limits of the batched solve: a batch costs the iterations of its slowest member (a member that stopped still
  goes through the kernels, as zero rows). The first batch starts cold in all members (no warm start). The host
  eigenproblems grow with B (one for each member, in a thread pool of up to 8 threads). All members need the same
  grid shape (`compute_modes_gpu_batch` groups the cells by grid shape). The device memory grows with B (the
  subspace buffers are B x max_subspace vectors).
- The complex128 polish of the `mixed` mode is slow from Ritz vectors that are already converged in complex64 (a
  small subspace of 3 blocks: about 40 iterations for the 0.45 um strip, 40 nm grid, of `tests/test_batch.py`).
- The complex128 polish is slow on consumer GPUs. Next: mixed-precision refinement of each eigenpair (complex128
  residuals, complex64 corrections).
- With PML, the solver can miss PML eigenvalues that are nearer the target than the kept modes (meow filters PML
  modes, so the kept modes were still correct in the review test). No test solves PML eigenpairs yet.
- meow's own post-processing (normalization, Gram-Schmidt, on the CPU) costs about 35 ms per cell.

## Tests

`uv run pytest` (CPU) checks the operator against the tidy3d matrix, the eigenvalues against tidy3d in each
precision, the warm start, the meow backend (neff and field overlap against `compute_modes_tidy3d`), and the device
placement. `tests/test_batch.py` checks that the batched solve gives the per-cell results (neff, E and H, in each
precision; a batch of 1; the warm start between batches; `compute_modes_gpu_batch` with batch 1, 2 and 8 on a
5-cell taper, with a short last run; two grid shapes and a bend cell (tidy3d fallback) in one list), ragged
convergence (members that stop at very different iterations give the results of `davidson` from the same initial
block), the stop by the stall rule and the iteration limit of each member, and the tidy3d direction rule for modes
below cutoff. On the CPU the iterations and the matvecs must be the same as those of the per-cell solve (all three
precisions); on a GPU the iteration check has a slack and is not done in single precision. On the GPU computer:
`bash tools/remote.sh setup-env`, then run pytest with `.venv-rocm`.
