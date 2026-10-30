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
  complex64, then a complex128 polish (tidy3d-level agreement). Two polishes: `polish="refine"` (default, from
  2026-10-30: mixed-precision refinement of the eigenpairs, see below) and `polish="davidson"` (the earlier
  complex128 Davidson, slow on cards with slow FP64). `double`.
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

`mixed` agrees with tidy3d to 2e-11 in neff, but it is 1.5x to 5x slower than `single` on this card (measured with
the earlier polish, now `polish="davidson"`).

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
| meow-gpu mixed (earlier polish, `polish="davidson"`) | 20.9 s | 0.97570552 (4e-12) |

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

The mixed columns used the earlier polish (now `polish="davidson"`). On this card the batched path is only about
1.2x faster in single precision, and it is not faster in mixed precision, for this taper (30 cells, 36k unknowns
per cell). It stays an option; the per-cell path is the default.
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

## Mixed-precision refinement (the default polish of `mixed`, 2026-10-30)

`fde.eigensolver.refine`. The complex64 Davidson stops at a residual of 1e-3 (`REFINE_SWITCH_TOL`). Then each step:

1. complex128 Rayleigh-Ritz on the subspace [Q, T] (two blocks: an orthonormal basis Q of the last Ritz vectors, and
   the last corrections T) and the complex128 residuals. The stop test is that of the complex128 Davidson (the
   relative residuals of the k wanted pairs below `tol`, 1e-10).
2. complex64 correction solves: for each pair that is not converged, 20 steps (`REFINE_INNER`) of GMRES on the
   projected correction equation of Jacobi-Davidson (Sleijpen and van der Vorst 1996),
   (I - Q Q^H)(A - theta_i)(I - Q Q^H) t_i = -r_i, with the fast-diagonalization preconditioner on the right. The
   projection on the whole block deflates near eigenvalues (a cluster). The GMRES steps run in one compiled device
   loop (`jax.lax.fori_loop`), with no host step.
3. The corrections go back to complex128 (orthogonal to Q, orthonormal), with one complex128 operator apply each.

The complex64 solves change only the rate of convergence. The accuracy comes from the complex128 residuals and the
complex128 Rayleigh-Ritz (iterative refinement: Dongarra, Moler and Wilkinson 1983). The first step selects the b
Ritz pairs nearest the target; the next steps follow each pair by its Ritz value. Reason (a GPU run with 14 GMRES
steps, taper cell 18): with a selection by the distance to the target, a new Ritz value with a residual of 8.6e-2
(it goes to lambda = -1.935, an eigenvalue that the complex64 Davidson did not find) replaced a converged pair, and
the refinement stopped there. The operator is not Hermitian, so a step can also give spurious Ritz values (one at
-1.886 - 8.5e-4i in the same run). At each step the wanted pairs are the k tracked pairs nearest the target
(`_selection`), not the pairs of the first step: the complex64 Ritz values (residual 1e-3) cannot order the two
modes of a near-degenerate pair. With the wanted set of the first step, a mode count that cuts such a pair gave the
other mode of the pair with `converged=True` (review, 2026-10-30: two cores, 1 mode, TE odd instead of TE even). An
identified neighbour (residual below 1e-2) whose Ritz value can still be nearer the target than a wanted value
(within its residual) must also converge before the stop. The result gives the pairs sorted by the distance to the
target. The refinement stops with `converged=False` when the largest residual of the stop test does not halve in 3
steps, when no correction is left, or at the iteration limit. It then gives the step with the smallest residual,
and `GPUModeSolver` warns. `polish="davidson"` gives the earlier method; `GPUModeSolver(switch_tol=...,
refine_inner=...)` changes the two parameters.

**Checks** (`tests/test_refine.py`; the numbers below are from the same cases on the RX 7900 XT, run through
`tools/gpu_run.sh`). 4 modes and 4 extra block vectors. The agreement columns are those of the refinement; the last
column gives the complex128 applies of the refinement / the earlier polish / the double solve.

| Case | Reference | max \|dneff\| | 1 - \|cos\| (fields) | complex128 applies |
|---|---|---|---|---|
| 0.5 um strip, 40 nm grid | double-precision Davidson, same start | 1.6e-12 | <= 1.1e-16 | 24 / 456 / 727 |
| two 0.44 um cores, 2.24 um gap | ARPACK shift-invert, assembled matrix | 4.6e-13 | <= 3.1e-11 | 29 / 398 / 764 |

- The strip: the residuals of the refined pairs, computed again with the assembled SciPy matrix (independent of the
  JAX operator), are 6.3e-13. Iterations: 49 (refine) against 106 (earlier polish).
- The two cores: the even and odd TE supermodes differ by 2.7e-8 in neff (relative eigenvalue splitting 2.4e-8,
  below the complex64 epsilon 1.2e-7). The single-precision solve mixes the two modes (1 - |cos| = 0.32 and 0.27);
  the refinement separates them (the earlier polish: 1.1e-9; double: 8.3e-11).
- A mode count that cuts a near-degenerate pair (the two cores with 1 mode: the TE pair; 3 modes: the TM pair,
  splitting 8.8e-5; start blocks from seeds 0, 1 and 2): the refinement gives the even mode of the pair, as ARPACK
  and the double solve (test bound 1 - |cos| < 1e-5; max |dneff| against the double solve 7.9e-12, in the review
  script of the same cases).
- A tolerance below the complex128 floor (1e-17): the refinement stops after 7 steps, with `converged=False` and
  residuals of 0.9e-14 to 1.4e-14. The step limit also gives `converged=False`, and the solver warns.

**EME of the taper** (`bench/bench_refine.py`: the 30-cell taper of `bench_eme_taper.py`, 10 modes, 20 nm grid; RX
7900 XT, `--repeat 2`, the second run; times as in the tables above, with meow's post-processing; rows in
`benchmarks/2026-10-30/meow-refine.jsonl`):

| Case | Mode solves | complex128 applies | Iterations (sequential) | \|S21\| | max \|dneff\| against double |
|---|---|---|---|---|---|
| double | 28.95 s | 21213 | 1080 | 0.9933381220221 | - |
| mixed, earlier polish (`polish="davidson"`) | 19.48 s | 18642 | 1625 | 0.9933381220224 | 6.2e-12 |
| mixed, refine (default) | 11.77 s | 1729 | 487 | 0.9933381220219 | 3.4e-12 |
| single | 5.59 s | 0 | 480 | 0.9933148474063 | 5.4e-6 |

The refinement makes `mixed` 1.66x faster on this card, with 10.8x fewer complex128 applies. It does the same
number of sequential iterations as `single` (complex64 iterations plus refinement steps), but each refinement step
costs more (20 complex64 GMRES steps on the device, and the complex128 products).

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
- The earlier complex128 polish (`polish="davidson"`) is slow from Ritz vectors that are already converged in
  complex64 (a small subspace of 3 blocks: about 35 iterations for the 0.45 um strip, 40 nm grid, of
  `tests/test_batch.py`). The default polish of `mixed` is now the refinement (see above).
- The refinement is still about 2x slower than `single` on the RX 7900 XT. With fewer GMRES steps
  (`refine_inner`) it can stall (the halving rule stops it, not converged). No fallback to the earlier polish yet.
- Device memory: the GMRES basis holds (`REFINE_INNER` + 1) x b complex64 vectors, 168 b n bytes at 20 steps (b
  block vectors, n unknowns; 0.8 GB for b = 24 and n = 200k). The earlier polish keeps 3 b complex128 vectors in V
  and in A V (96 b n bytes).
- On a CPU the refinement is not faster than the earlier polish: the complex128 rate of a CPU is not low, and the
  refinement does more complex64 work. Not measured with the code of this version.
- All precisions can miss eigenvalues nearer the target than the kept modes. Taper cell 18 (`bench_refine.py`):
  ARPACK on the assembled matrix finds lambda = -1.9350 and -1.9038 (box modes below cutoff, neff 1.391 and 1.380)
  between the 9th (-1.9871) and the 10th kept mode (-1.8877); the double, the mixed and the single solves keep
  -1.8877. The strip of `tests/test_refine.py` (4 modes): ARPACK finds two modes at neff 1.444 and modes at 1.4074
  and 1.3604 nearer the target than the 4th kept mode (1.3413), which tidy3d also gives (`tests/test_solver.py`).
- The batched solve (`davidson_batch`) runs the refinement member after member, after the batched complex64
  Davidson (the same steps as the per-cell solve, but the device kernels of the refinement are not batched).
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
precisions); on a GPU the iteration check has a slack and is not done in single precision.
`tests/test_refine.py` checks the mixed-precision refinement (above): the strip against the double solve and the
SciPy residuals, the two-core cluster against ARPACK, a mode count that cuts a near-degenerate pair (1 and 3 modes
of the two cores, 3 start blocks, against ARPACK and the double solve by neff and field overlap), the complex128
applies against the earlier polish, the tracking and the selection of the pairs, the stop with `converged=False`
(tolerance out of reach, step limit, solver warning), and the earlier polish (per cell and batched). Run all tests
on the GPU computer (`bash tools/remote.sh push`, then `bash tools/gpu_run.sh meow_gpu python -m pytest -q` there;
`tools/gpu_run.sh` uses `.venv-rocm` and the GPU lock; first `bash tools/remote.sh setup-env`).
