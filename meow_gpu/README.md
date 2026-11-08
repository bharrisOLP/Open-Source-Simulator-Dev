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
  direction rule as `compute_modes_gpu`; the post-processing of a batch runs in a second thread while the GPU
  solves the next batch (at most one batch waits for its post-processing). The host thread pool runs with one BLAS
  thread: with threadpoolctl it limits the BLAS while the pool runs; without threadpoolctl it runs only when
  `OMP_NUM_THREADS=1` (or `OPENBLAS_NUM_THREADS=1`) is set, else the host steps run in one thread, with a warning
  (see "Profile of the batched path").
- **Device.** klujax (imported by meow through sax) sets the JAX default platform to `cpu`. meow-gpu therefore puts
  its arrays on the GPU explicitly (`meow_gpu/device.py`). Set `MEOW_GPU_DEVICE=cpu` to force the CPU.

- **Bends.** `cs.mesh.bend_radius`, `bend_axis` and `plane_center` go to the GPU solver (since 2026-10-30; see
  "Bend modes" below).

Not supported yet (falls back to tidy3d with a warning): angled propagation (tidy3d needs its tensorial solver,
from tidy3d-extras, for it). meow passes diagonal permittivity only. Modes below cutoff follow the tidy3d direction
rule (power flow when it is reliable, else keff >= 0).

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
case with the mode-solve time, |S21|, max |dneff| against the per-cell run and the iterations). On the RX 7900 XT,
with the changes of "Profile of the batched path" below (`--precision single --batches 4,8,16 --repeat 2` and the
same with `--precision mixed`; the second run, after compilation; `OMP_NUM_THREADS=1`; the mixed columns use the
refinement, the default polish; rows in `benchmarks/2026-10-30/eme-batch-profiled.jsonl`). These rows are from the
code of commit 88d5158, before the cold-start stop of 2026-11-07 ("Cold starts" below): the single columns do not
depend on it; in the mixed columns the cold first cell and the cold first batch now go to 2e-5 (B = 8 with
`OMP_NUM_THREADS=4`: first batch 72 instead of 46 iterations, 7.17 s against 7.11 s; per cell 8.04 s against
8.13 s; `benchmarks/2026-11-07/meow-cold-switch.jsonl`). B = 4 and 16 were not measured again.

| Case | single: mode solves | single: max \|dneff\| | mixed: mode solves | mixed: max \|dneff\| | Batch iterations |
|---|---|---|---|---|---|
| per-cell | 4.47 s | - | 7.90 s | - | 474 / 480 |
| B = 4 | 2.40 s (1.86x) | 8.9e-6 | 6.60 s (1.20x) | 7.9e-13 | 164 / 165 |
| B = 8 | 2.68 s (1.67x) | 8.9e-6 | 6.90 s (1.15x) | 2.5e-12 | 105 / 101 |
| B = 16 (15 used) | 3.41 s (1.31x) | 8.9e-6 | 7.32 s (1.08x) | 2.5e-12 | 72 / 68 |

The last column gives the sequential iterations (single / mixed; per-cell: the sum over the cells). |S21| of the
fundamental mode: single 0.99331320 (per-cell) and 0.99328601 to 0.99329819 (batches); mixed 0.9933381220219 in
all four rows. All cells keep the same number of modes as the per-cell run. The single-precision max |dneff| is at
the complex64 tolerance (the bound of `tests/test_batch.py` is 2e-5); the mixed-precision max |dneff| is at the
complex128 tolerance. Against the earlier rows below, the mode solves of the taper take 2.40 s instead of 5.53 s in
single precision (B = 4 against the earlier per-cell run), and 6.60 s instead of 11.72 s in mixed precision (B = 4
against the per-cell refinement row of "Mixed-precision refinement"). B = 16 is slower than B = 8: the cold first
batch (52 iterations in single, 49 in mixed precision with the code of 88d5158) is a larger part of 2 batches, and
15 members of a cold batch do the work of 15 cold cells.

Earlier rows (the code of commit 0fabc92, OpenBLAS with its default threads, rows in
`benchmarks/2026-10-30/eme-batch.jsonl`):

| Case | single: mode solves | single: max \|dneff\| | mixed: mode solves | mixed: max \|dneff\| |
|---|---|---|---|---|
| per-cell | 5.53 s | - | 19.45 s | - |
| B = 4 | 4.75 s | 4.6e-6 | 18.27 s | 5.9e-12 |
| B = 8 | 4.49 s | 7.3e-6 | 20.75 s | 6.3e-12 |
| B = 16 (15 used) | 4.57 s | 4.8e-6 | 18.75 s | 4.9e-12 |

The mixed columns used the earlier polish (now `polish="davidson"`). With that code the batched path was only about
1.2x faster in single precision, and not faster in mixed precision, for this taper (30 cells, 36k unknowns per
cell). The batched path stays an option; the per-cell path is the default.
The iterations in the same rows show why (time = `modes_s`, which also includes meow's post-processing, about 1.1 s
for the 30 cells on the CPU: `backend._modes` of the per-cell rows of `benchmarks/2026-10-30/meow-profile-batch.jsonl`,
1.07 s in single and 1.10 s in mixed precision):

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

The cost for each member is profiled in "Profile of the batched path" below.

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

### Profile of the batched path (2026-10-30)

`bench/profile_batch.py` (the 30-cell taper, 10 modes, 8 extra block vectors; RX 7900 XT). Run 1 compiles; run 2
is not instrumented (its time is "run 2" below); run 3 is instrumented: each wrapped function waits for its device
results, and its time is exclusive (the time of the wrapped functions that it calls is not in it). The shares are
those of run 3. "Before": the code of commit b495484 (the profile script then imported numpy before it set
`OMP_NUM_THREADS=1`, so OpenBLAS had its default threads, as in the earlier bench rows; development runs, not
logged). "After": this version (rows in `benchmarks/2026-10-30/meow-profile-batch.jsonl`). Commands:
`python bench/profile_batch.py --precision single --batch 8` (the same with `--precision mixed`, and with
`--batch 0` for the per-cell path). Each category adds the parts of the JSON row, with the mapping
`CATEGORIES` of `bench/profile_batch.py`; `python bench/profile_batch.py --summarize <log>` prints the shares of
each row of a log (for the "after" columns: `benchmarks/2026-10-30/meow-profile-batch.jsonl`):

- batched device kernels: the functions `*_b` (with `_svqb_b`) and `precondition_batch`;
- GMRES: `_correction_krylov` and `_correction_solution`;
- other device kernels of the refinement: `_insert`, `_gram`, `_combine`, `_project_out`, `_ritz`, `_apply_block`,
  `precondition`, `_svqb` and `_take_rows`;
- host eigenproblems: `_host_map: _ritz_coef` and `np.linalg.eig (main thread)`; the other host steps:
  `_host_map: _svqb_coef`, `_svqb_coef`, `_gmres_coef`, `_track` and `_selection`;
- copies: `put` and `backend.put`; Python: the parts that end in "(Python)";
- fields, direction rule, setup: `backend._orient`, `backend.fields`, `backend._fields_batch`,
  `backend.make_problem`, `backend.make_preconditioner`, `backend.stack` and `backend._problem`;
- post-processing: `backend._modes`; "wait, other": the parts in no category and `outside_wrapped_s`.

The exclusive time of `_svqb` and `_svqb_b` is not device time: it is the host code around their kernels and the
copy of their Gram matrix to the host (1.3 % and 0.8 % of run 3 in the logged B = 8 rows, single and mixed). The
"before" columns used the same mapping.

| Part (share of run 3) | single, B = 8: before (not logged) | after | mixed, B = 8: before (not logged) | after |
|---|---|---|---|---|
| batched device kernels (complex64 Davidson) | 41.2 % | 53.5 % | 13.7 % | 17.3 % |
| GMRES of the refinement (device, member after member) | - | - | 48.2 % | 36.7 % |
| other device kernels of the refinement (member after member) | - | - | 9.2 % | 15.3 % |
| host eigenproblems of the projected matrices | 20.4 % | 7.7 % | 6.4 % | 3.1 % |
| host SVQB, GMRES least squares, tracking | 2.0 % | 2.9 % | 1.6 % | 2.4 % |
| host-to-device copies (`put`, `backend.put`) | 5.7 % | 6.3 % | 4.8 % | 6.7 % |
| Python of the solver loops | 10.5 % | 15.7 % | 7.7 % | 13.1 % |
| fields, direction rule, problem setup | 2.7 % | 5.4 % | 1.1 % | 2.2 % |
| meow's post-processing in the main thread | 17.3 % | 0 (second thread) | 7.1 % | 0 (second thread) |
| wait for the last post-processing, other | 0.2 % | 8.5 % | 0.1 % | 3.2 % |
| run 3 (instrumented) | 4.92 s | 3.08 s | 12.77 s | 7.83 s |
| run 2 (not instrumented) | 4.77 s | 2.81 s | 11.82 s | 7.00 s |
| per-cell path, run 2 | 5.77 s | 4.67 s | 12.17 s | 8.12 s |

JIT compiles: none in run 2 or run 3 of any case (no recompile for each member or each batch); the first run costs
1.9 to 3.4 s more in the logged rows. The changes, from the largest share:

1. **GMRES of the refinement** (`_correction_krylov`): the Krylov basis is now (b, steps + 1, n), with the block
   axis first. With the earlier layout (steps + 1, b, n), XLA transposed the basis for the Gram-Schmidt products
   (the batch axis of the products was not the first axis): the optimized HLO of the earlier layout has 5 transposes
   with at least the size of the basis (13,759,200 elements), that of this layout none
   (`benchmarks/2026-11-07/meow-kernel-reduce.txt`, both runs). One call (20 steps, b = 18, 36k unknowns) takes
   35.0 ms instead of 77.9 ms (`bench/kernel_reduce.py`; the two logged runs are in
   `benchmarks/2026-10-30/meow-kernel-reduce.txt`; 2026-11-07: 35.5 / 36.0 ms instead of 76.8 / 77.6 ms). This
   change makes `mixed` faster in the per-cell path too.
2. **Host eigenproblems.** The bench scripts set `OMP_NUM_THREADS=1` after the import of numpy, so OpenBLAS had its
   default threads, and the thread pool of the batched solve oversubscribed the cores. Eight eigenproblems of
   72 x 72 (complex) take 5.4 ms in a pool of 8 threads with `OMP_NUM_THREADS=1` (31 ms in one thread), and
   102 ms in the pool with the default OpenBLAS threads (48 ms in one thread). Two logged runs of
   `bench/kernel_reduce.py`: one with `OMP_NUM_THREADS=1` in the environment, one without. The review measured 45 ms
   in the pool against 50 ms in one thread without the variable: with the default threads, the pool gains nothing,
   and it was up to 2x slower than one thread. The bench scripts now set the variable before the import of numpy,
   and the library now keeps the BLAS at one thread while the pool runs (`_blas_limit` in `fde/eigensolver.py`,
   with threadpoolctl; without threadpoolctl and without `OMP_NUM_THREADS=1`, the host steps run in one thread,
   with a warning). With the limit (`benchmarks/2026-11-07/meow-kernel-reduce.txt`, threadpoolctl 3.8.0), the
   library `_host_map` takes 8.7 ms for the 8 eigenproblems with `OMP_NUM_THREADS=1` and 12.6 ms without thread
   variables, against 102 ms in the pool without the limit (2026-10-30). In the 2026-11-07 run without thread
   variables, one thread took 1024 ms and the pool without the limit 265 ms (another session used the computer).
3. **meow's post-processing** (17.3 % and 7.1 % of run 3 in the "before" columns: about 0.85 to 0.91 s for the
   30 cells, on the CPU) of a batch now runs in a second thread while the GPU solves the next batch
   (`compute_modes_gpu_batch`). After each solve, the results of the previous batch are collected before the new
   batch goes into the queue: at most one batch waits, and an error of the post-processing reaches the caller after
   the next solve. Only the post-processing of the last batch stays on the critical path.

Why the batched path gains less in mixed precision, and what was not changed:

- **A batched refinement (`jax.vmap` over the members) does not make the GMRES faster.** The GMRES kernel under
  `jax.vmap` over 8 members takes 271 ms, and 8 calls take 8 x 35.0 = 280 ms (`bench/kernel_reduce.py`): a gain
  of 1.03x. For the smaller kernels the gain of `jax.vmap` is also small: a Gram matrix of 72 x 18 vectors takes
  0.31 to 0.35 ms for one member and 1.59 to 1.60 ms for 8 members, `_project_out` 0.65 ms and 3.97 to 3.98 ms
  (in each of the two logged runs, 8 members take 4.6 to 6.2x the time of one: a gain of 1.3 to 1.8x for each
  member). So the refinement still runs member after member. A batched refinement in lockstep can only remove the
  other costs of each member (about 15 % small device kernels and part of the 13.1 % Python in the mixed case),
  not the GMRES (36.7 %).
- **The device kernels run far below the peak rate of the card.** A Gram matrix of 72 x 18 vectors of 36k
  complex64 values reads 26 MB and does 0.38 GFLOP in 0.31 to 0.35 ms (about 75 to 84 GB/s and 1.1 to 1.2
  TFLOP/s). The variants that
  were tried (a split contraction, a fused multiply and sum) were not faster (`bench/kernel_reduce.py`). This is
  the largest share of the single-precision batch (53.5 %) and the next item for a faster batch.

## Mixed-precision refinement (the default polish of `mixed`, 2026-10-30)

`fde.eigensolver.refine`. After a warm start the complex64 Davidson stops at a residual of 1e-3
(`REFINE_SWITCH_TOL`); after a cold start at `COLD_SWITCH_TOL` (2e-5; see "Cold starts" below). Then each step:

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
(within its residual) must also converge before the stop. An identified Ritz value that no pair follows (residual
below 1e-2; the refinement computes the residuals of all Ritz pairs) replaces the followed value farthest from the
target when it is nearer the target: an eigenvalue that the complex64 Davidson did not find can come in, a spurious
value (large residual) cannot. The result gives the pairs sorted by the distance to the target. The refinement stops
without convergence when the largest residual of the stop test does not halve in 3 steps, when no correction is
left, or at the iteration limit.

**Fallback.** When the refinement stops without convergence and iterations are left, the fallback runs the steps of
the earlier polish from the same complex64 block: the complex64 Davidson continues to the switch tolerance of the
earlier polish (2e-5), then the complex128 Davidson with 3 blocks. It does not start from the refined block: from
there it cannot find an eigenvalue that the refined block misses (review, 2026-10-30: from the refined block, with
the admission rule turned off, it converged to a wrong 4th mode on the grid of the example). The complex128 stage
stops when its residual does not halve in 30 iterations (only below 1e-3), and the result is the fallback result
when it converged, else the result (refinement or fallback) with the smaller residual. What the code guarantees:
`converged=True` means that the k pairs have complex128 residuals below the tolerance; the mode set is that of a
Davidson run from the complex64 block, so the known limit "all precisions can miss eigenvalues" applies to it as to
the earlier polish. `SolveInfo.polish_used` gives "refine", "refine+davidson" (the fallback ran) or "davidson" (""
for "single", "double", and for "mixed" when no iterations are left after the complex64 stage); the iterations and
the complex128 applies include all parts. Reason (2026-10-30, `examples/chip_to_fiber_coupling.py`): for the weakly
guided mode of a SiN tip (0.3 x 0.3 um, just above a dense band of window modes in a 14 x 14 um window, 4 modes)
the refinement stopped above the tolerance, and the example stopped. Its 4th mode was also not that of the earlier
polish: the second mode of a near-degenerate pair of window modes was not in the block of the complex64 Davidson,
and the tracking did not let it in. With the admission of identified values and the fallback, the variants of this
structure that were tried on the GPU (windows of 8 to 14 um, grids of 50 to 100 nm) converge to the 4 neff of the
earlier polish; the refinement alone did not converge there, and the fallback finished. The test
`test_weakly_guided_mode_converges` (the 14 um window on a 100 nm grid) failed before this change; the test
`test_example_grid_gives_the_modes_of_the_earlier_polish` checks all 4 modes on the 50 nm grid of the example,
with and without the admission rule. `polish="davidson"` gives the earlier method; `GPUModeSolver(switch_tol=...,
refine_inner=...)` changes the two parameters.

**Cold starts (2026-11-07).** After a cold start (no warm-start vectors), the complex64 stage of `refine` stops at
`COLD_SWITCH_TOL` (2e-5, the stop of the earlier polish) instead of 1e-3 (`GPUModeSolver._switch_tol`; also for each
cold member of a batch; a given `switch_tol` is kept). Reason: the protection above was not stable. On the SiN tip
of the tests (`tests/structures.sin_tip` on the 50 nm grid: the structure of the example before 2026-11-07, with a
core one sample off its mirror symmetry; see the docstring), from a cold start, the complex64 stage stopped at 1e-3
before the near-degenerate pair at neff 1.4418809 / 1.4418766 was in its block. Then the refinement either stalled
(the fallback gave the right modes) or converged to the next pair (1.4412281, 1.4411957), and the BLAS decided
which: with `OPENBLAS_NUM_THREADS=1` at the start of the process it converged to the wrong pair also with the
admission rule; with 4 threads only without the admission rule; the code of commit f8c16ec gave the same results
(`bench/diag_tip.py` and the test, RX 7900 XT computer). The BLAS thread count changed while the process ran
(threadpoolctl, 1 to 4 threads) gave the same results as the count at the start. With the stop at 2e-5, the test
passes with 1 and with 4 BLAS threads. A warm start (the next cell of a taper) keeps the stop at 1e-3: on the
30-cell taper the change costs nothing measurable (`bench_refine.py --cases double,mixed-refine --repeat 2`, with
`OMP_NUM_THREADS=4`: 8.04 s, 497 iterations, 1724 complex128 applies, against 8.13 s, 489 iterations, 1736 applies,
with `--switch-tol 1e-3`; max |dneff| against double 2.1e-12 and 1.7e-12; `bench_eme_batch.py --precision mixed
--batches 8 --repeat 2`: 7.17 s against 7.11 s, 126 against 104 batch iterations, the extra iterations in the cold
first batch; rows in `benchmarks/2026-11-07/meow-cold-switch.jsonl`, the same commands without and with
`--switch-tol 1e-3`).

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
  residuals of 0.9e-14 to 1.4e-14 (`refine` alone). In `mixed`, the fallback then runs and stops by its stall rule
  (not at the iteration limit); its result is not worse than that of the refinement. The step limit also gives
  `converged=False`, and the solver warns.
- Precision "mixed" and "double" need `jax_enable_x64`: without it JAX runs complex128 arrays in complex64 without an
  error, so the solver stops with an error.

**EME of the taper** (`bench/bench_refine.py`: the 30-cell taper of `bench_eme_taper.py`, 10 modes, 20 nm grid; RX
7900 XT, `--repeat 2`, the second run; times as in the tables above, with meow's post-processing; rows in
`benchmarks/2026-10-30/meow-refine.jsonl`):

| Case | Mode solves | complex128 applies | Iterations (sequential) | \|S21\| | max \|dneff\| against double |
|---|---|---|---|---|---|
| double | 28.46 s | 21213 | 1080 | 0.9933381220221 | - |
| mixed, earlier polish (`polish="davidson"`) | 19.48 s | 18642 | 1625 | 0.9933381220224 | 6.2e-12 |
| mixed, refine (default) | 11.72 s | 1729 | 487 | 0.9933381220219 | 3.4e-12 |
| single | 5.51 s | 0 | 480 | 0.9933148474063 | 5.4e-6 |

The refinement makes `mixed` 1.66x faster on this card, with 10.8x fewer complex128 applies. It does the same
number of sequential iterations as `single` (complex64 iterations plus refinement steps), but each refinement step
costs more (20 complex64 GMRES steps on the device, and the complex128 products).

After the change of the GMRES basis layout (see "Profile of the batched path"), with `OMP_NUM_THREADS=1`
(`bench_refine.py --cases double,mixed-refine,single --repeat 2`; rows in
`benchmarks/2026-10-30/meow-refine-profiled.jsonl`):

| Case | Mode solves | complex128 applies | Iterations (sequential) | \|S21\| | max \|dneff\| against double |
|---|---|---|---|---|---|
| double | 23.01 s | 21166 | 1081 | 0.9933381220221 | - |
| mixed, refine (default) | 7.91 s | 1722 | 480 | 0.9933381220219 | 2.7e-12 |
| single | 4.37 s | 0 | 474 | 0.9933131973099 | 5.1e-6 |

`mixed` now takes 1.8x the time of `single` (2.1x before), and 2.9x less time than `double`. All cells converged
(largest residual 9.3e-11 in mixed, 9.96e-11 in double).

## Bend modes (2026-10-30)

**Method.** The radial transform of the tidy3d local mode solver
(`tidy3d.components.mode.transforms.radial_transform`, tidy3d 2.12.2), in `fde/operator.py`. The coordinate of the
normal axis (the axis in the bend plane: x for `bend_axis=1`, y for `bend_axis=0`) moves to
u = coordinate + R - plane_center, and w = R phi is the propagation coordinate, so neff is the effective index at
the radius R. The Jacobian is J = diag(1, 1, a), a = R / u, at the E positions (u at the lower cell corner of the
normal axis, for all three E components, as tidy3d) and at the H positions (u at the cell centre). The media become
eps' = J eps J^T / det J and mu' = J J^T / det J:

    eps'_xx = eps_xx / a_e,  eps'_yy = eps_yy / a_e,  eps'_zz = eps_zz a_e,  mu'_xx = mu'_yy = 1 / a_h,  mu'_zz = a_h

The operator gets the diagonal mu of the tidy3d "P Q" form: c = (Dxf Ey - Dyf Ex) / mu_zz,
(A u)_x = mu_yy hy + Dxf d, (A u)_y = -mu_xx hx + Dyf d. The PML speed uses the transformed eps and mu (as
tidy3d). The fields in the cross-section are E = J^T E', H = J^T H' (Ez = a_e Ez', Hz = a_h Hz'). The target of the
eigensolver is the largest physical index (before the transform), as in tidy3d. A straight cross-section has no mu
or Jacobian arrays on the device (the same kernels as before). In a batch with bends, straight members get arrays
of ones. The preconditioner is that of a straight waveguide (with the transformed eps). A version with the bend
factor a^2 and the target frozen in the 1-D operators needed more iterations in a development run (the guided
modes are far from the target, so the frozen term is a bad approximation); it is not kept. The warm-start memory
has one entry for straight cross-sections and one for each bend (radius, bend axis and the shift R - plane_center
of the normal axis; test `test_warm_start_only_from_the_same_kind`). A cross-section starts only from the modes of
its own kind, also in a batch: each member starts from member i of the previous batch if that member was of the
same kind, else from the memory of its kind, else cold (test `test_batch_warm_start_keeps_each_kind`). Reason (the
review of 2026-10-30): with 4 modes (box modes in the mode set), a warm start from the modes of a 5 um bend did
not converge for the straight strip of `tests/test_bend.py` (the batched solve gave the bend's vectors to a straight
member: 500 iterations, residual 0.3, neff off by 0.09). A bend that the transform cannot represent (bend_radius 0,
or the centre of the bend inside the cross-section) falls back to tidy3d with a warning, also in
`compute_modes_gpu_batch` (the other cross-sections of the list stay on the GPU); `make_problem` raises
`ValueError` for it.

**Checks** (`tests/test_bend.py`, `tests/test_operator.py`; on the RX 7900 XT through `tools/gpu_run.sh`):

- The operator (`apply` and `assemble`) equals the matrix that tidy3d's own `compute_modes` gives to its
  eigensolver (captured at the call of `solver_eigs`), relative difference below 1e-14, for 7 cases: radii 1.2, 5,
  10, 50 and -3 um, both bend axes, PML on one or two axes, PMC and PEC symmetries, random complex eps on a random
  non-uniform grid.
- A Si strip 0.5 x 0.22 um in SiO2 (40 nm grid, 5k unknowns), radii 5, 10 and 50 um, both bend axes: TE0 and TM0
  equal `compute_modes_tidy3d` (with the post-processing of meow-gpu) to |dneff| < 1e-8 and 1 - |overlap| < 1e-6 in
  double and mixed precision (single: 2e-5 and 1e-3). All six field components agree to a relative 1e-6 (single:
  5e-3), after one phase for each mode: meow's inner product uses only Ex, Ey, Hx and Hy, and Ez and Hz are the
  components that the back-transform changes. Without the Jacobian (Ez', Hz'), Ez and Hz differ from tidy3d by
  more than 1e-3 (test `test_the_jacobian_changes_ez_and_hz`).
- 4 modes, double precision only: the GPU modes are the 4 eigenvalues of the tidy3d matrix nearest the target
  (ARPACK shift-invert, 8 values, tol 1e-13; |dneff| < 1e-9; residuals with the tidy3d matrix < 1e-9). Mixed and
  single precision can miss a box mode nearer the target, as tidy3d does (see the timing rows below). So the mode
  set beyond the guided modes depends on the solver and the precision. `compute_modes_tidy3d` itself misses
  box modes at the outer wall of the bend. Bench row (`--radii 5 50 --dx 0.04 --repeat 1 --x-min -1.2 --x-max 1.2`,
  mixed, 4 modes), R = 5 um: tidy3d gives neff 2.4468, 1.8002, 1.5969, 1.4822, the GPU solver 2.4468, 1.8002,
  1.6579, 1.5969 (the three common modes agree to 2.2e-14). At R = 50 um all 4 agree (5.0e-11). So the mode list of
  tidy3d is the reference only for the guided modes.
- Radiation loss (PML of 10 cells in x, the domain from -1.0 to 2.4 um, target neff 2.45, 1 mode): Im(neff) of the
  GPU solver equals that of tidy3d (radii 1.5 and 2 um, rel 1e-4, in double and mixed precision), and it falls with
  the radius (1.5, 2, 3 um).
- A large radius tends to the straight modes as 1/R. A negative radius (-5 um, 40 nm grid) gives the TE0 and TM0
  modes of tidy3d. On the 20 nm grid only TE0 is common: the second GPU mode is a box mode (neff 1.789115) that
  tidy3d misses (it gives TM0, 1.782834; bench row below).

**Bench** (`bench/bench_bend.py`, small cases on the RX 7900 XT, double precision, 40 nm grid, the strip in a
2.4 x 1.8 um box, 2 modes). Commands (rows logged by the lead in `benchmarks/2026-10-30/meow-bend.jsonl`; the
review ran them again and got the same numbers):

1. `--radii 5 -5 50 1000 10000 100000 --dx 0.04 --modes 2 --precision double --repeat 1 --x-min -1.2 --x-max 1.2`
2. `--radii 5 -5 --dx 0.02 --modes 2 --precision double --repeat 1 --x-min -1.2 --x-max 1.2`
3. `--radii 1.5 2 3 --pml 10 --x-min -1.0 --x-max 2.4 --target 2.45 --modes 1 --dx 0.04 --precision double
   --repeat 1`
4. `--radii 5 50 --dx 0.04 --repeat 1 --x-min -1.2 --x-max 1.2` (mixed, 4 modes: the 4-mode row above)

| Case | max \|dneff\| against tidy3d | Im(neff) GPU / tidy3d | \|dneff(TE0)\| against straight |
|---|---|---|---|
| R = 5 um | 3.3e-12 | - | 5.0e-3 |
| R = -5 um | 2.0e-12 | - | 7.9e-3 |
| R = 50 um | 3.5e-12 | - | 6.3e-4 |
| R = 1000 / 10^4 / 10^5 um | 2.0e-12 / 2.0e-12 / 3.8e-12 | - | 3.2e-5 / 3.2e-6 / 3.2e-7 |
| R = 1.5 um, PML | 8.6e-13 | 3.96484346e-5 / 3.96484348e-5 | - |
| R = 2 um, PML | 2.7e-13 | 9.820145e-7 / 9.820147e-7 | - |
| R = 3 um, PML | 7.6e-13 | 2.4136e-9 / 2.4130e-9 | - |

The R = 1.5 um row did not converge in 500 iterations (residual 1.02e-10 against the tolerance 1e-10), but its
neff agrees. The R = 2 um row needed 199 iterations (command 3).

GPU timings (`bench/bench_bend.py`, RX 7900 XT, 20 nm grid, 27k unknowns, 4 modes, radii 2 to 100 um, the median
of 3 runs; rows in `benchmarks/2026-10-30/meow-bend.jsonl`): mixed precision 0.22 to 0.30 s per bend, single 0.17
to 0.25 s, double 0.47 to 1.04 s, against 0.30 to 0.43 s for tidy3d on one CPU process. At this size the GPU gains
little (as for straight sections of this size), and double precision on this card is slower than tidy3d.
Agreement with tidy3d in the same rows (a mode is common when a GPU mode is within 1e-6 of it; these rows were made
before the bench used 1e-5 for single):

- TE0: single 0.98e-6 to 1.68e-6; mixed 2.3e-13 or better; double 5.0e-13 or better.
- The other common modes: mixed up to 4.1e-10 (R = 50 um) and 2.9e-10 (R = 100 um; the 4th mode, at the level of
  the residual tolerance), else 2.3e-13 or better; double up to 8.4e-13.
- The mode sets differ beyond the guided modes. At R = 2 to 20 um, tidy3d misses a box mode that all three
  precisions give (for example 2.2575 at R = 2 um, 1.7438 at R = 5 um). At R = 50 and 100 um, the 4th mode of
  mixed precision is 1.38210 and 1.37506 (the same as tidy3d), but double precision gives 1.46067 and 1.44946,
  nearer the target and converged (single: 1.46067 and 1.37506). So mixed and single precision can also miss a box
  mode nearer the target.

The tight bends with PML (one mode, R = 1.5 / 2 / 3 um, mixed): 1.41 / 0.52 / 0.29 s (271 / 90 / 48 iterations)
against 0.39 to 0.42 s.

**Discretization of the transform (a property of tidy3d's discrete problem, shared by both solvers).** tidy3d takes
the Jacobian of all E components at the lower cell corner of the normal axis, half a cell from Ex. This gives an
error of first order in dx / R: the TE0 neff of the +5 um and the -5 um bends (mirror images) differ by 1.29e-2 on
the 40 nm grid and by 6.1e-3 on the 20 nm grid (bench commands 1 and 2). The same offset gives the 1/R difference from
the straight modes at large R (3.2e-5 at R = 1 mm on the 40 nm grid). Use a fine grid for tight bends.

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

## A power-conserving interface (`meow_gpu.interface`, 2026-11-08)

```python
from meow_gpu.interface import compute_interface_s_matrices, post_process_eigenmodes

modes = [compute_modes_gpu(cs, num_modes=40, solver=solver, post_process=post_process_eigenmodes) for cs in css]
S, port_map = mw.compute_s_matrix(modes, cells=cells, interfaces_fn=compute_interface_s_matrices)
```

In photonsuite: `photonsuite.eme.transmission(..., interface="conserving")` (the default stays `"meow"`).

**Why meow's interface does not conserve power.** Without a correction, the kept modes of the 0.5 to 2.0 um taper of
`suite/bench/eme_fdtd_taper.py` carry 1.008 to 1.012 of the input power (`suite/README.md`). Two causes
(`bench/bench_interface.py --diag` puts both into one row, and the tests below check both):

- The formula. Mode matching at a step has two equations: E_t and H_t are continuous. meow (0.15,
  `compute_interface_s_matrix`) tests both equations with the modes of the left guide: (O_LR + O_RL^T) T = 2 I, with
  O_LR[i, j] = <e^L_i, h^R_j>. This is exact for the complete set of modes. For a truncated set the S-matrix is not
  unitary, and O_LR + O_RL^T is almost singular when a kept mode of one side has no partner in the kept modes of the
  other side. Thus meow truncates the solve (TSVD, rcond 1e-3) and then changes the singular values of S (the
  passivity rules "invert" and "clip"). Without the truncation the solve is unstable.
- The product. meow's inner product integrates the field samples with the trapezoidal rule on the nodes. The
  eigenmodes of the discrete operator are not orthogonal in it. meow's Gram-Schmidt step then mixes the modes, and
  each mixed mode propagates with the neff of one eigenmode.

**The interface.** It is written from the field equations (module docstring of `meow_gpu/interface.py`):

- The product is the unconjugated reciprocity product of the Yee grid, <m1, m2> = 1/2 sum (E1x H2y dA_x -
  E1y H2x dA_y), with the area of each Yee sample (the weights of the Poynting flux of the tidy3d solver). Two
  eigenmodes of the discrete operator with different propagation constants are orthogonal in it (discrete
  reciprocity), so the Gram matrices are diagonal to the solver tolerance.
- The E_t equation is tested with the H fields of the right modes, and the H_t equation with the E fields of the
  left modes (the testing of the conservation-of-complex-power technique, Safavi-Naini and MacPhie, IEEE Trans.
  Microw. Theory Tech. 29, 337 (1981)). With P[i, j] = <e^L_i, h^R_j> and the Gram matrices G_L, G_R, each
  excitation gives one square system K [R; T] = rhs, K = [[P^T, -G_R^T], [G_L, P]]. For orthonormal modes with real
  fields (lossless guides, modes above cutoff), the singular basis of P (P = U diag(s) V^T) divides the step into
  independent channels with r = (1 - s^2) / (1 + s^2) and t = 2 s / (1 + s^2). Thus r^2 + t^2 = 1 for every s: S is
  orthogonal (power is conserved) and symmetric (reciprocal) for every truncation of a set of modes above cutoff,
  with no TSVD and no passivity rule. For the complete set the projections are exact. With modes below cutoff in a
  truncated set, P is complex and this proof does not hold.
- The modes (`post_process_eigenmodes`): no Gram-Schmidt (the modes stay eigenmodes, and the interface uses their
  Gram matrices). Each mode goes on the branch that decays in +z (below cutoff: Im(neff) > 0) or that carries power
  in +z (above cutoff, the rule of tidy3d): a mode on the other branch is flipped (neff, E_z, H_x and H_y change
  sign), not removed. Thus both members of a complex pair (neff, -conj(neff)) of the lossless guide stay in the set;
  the member with Re(neff) < 0 also decays in +z and is a forward mode. All modes below cutoff stay in the set (meow
  removes the modes with |Im neff| > 1; `max_imag_neff=1.0` gives that rule). meow's phase rule and the normalization
  <m, m> = 1 in the Yee product. A degenerate cluster of modes above cutoff gets a real basis that is orthonormal in
  the Yee product (the eigensolver gives any basis of the cluster, often with complex coefficients, and then |S|^2 is
  not the power); a degenerate cluster below cutoff is not changed. Removed: modes with fields that are not finite or
  with a self-product of almost zero, and copies of a kept mode (the same neff and parallel E_t fields; a copy makes
  the interface system singular, and a singular system raises `ValueError`).

**Checks** (`tests/test_interface.py`, 15 tests, mixed precision):

- A lossless step (Si strip 0.5 to 1.0 um, 8 modes, all above cutoff): S^H S = I and S = S^T within 1e-7. On the
  same eigenmodes, meow's interface without a passivity rule has a singular value above 1.001, and meow's product
  gives Gram entries off the diagonal above 1e-4; the Yee product gives the identity within 1e-7.
- meow's SAX cascade of narrow, wide and narrow cells (with propagation): the power of each input mode goes into the
  kept modes within 1e-7; S = S^T within 1e-7.
- The star-product cascade (`compute_s_matrix`, narrow, wide and narrow cells) gives the S-matrix and the port map
  of meow's SAX cascade within 1e-10, with the conserving interface and with meow's interface.
- Equal guides: S = [[0, I], [I, 0]] within 1e-12.
- Uniform media (the window filled with n = 1.444, then with n = 2.0): the two first modes of this grid are a
  degenerate pair with neff = n (uniform Ex and uniform Ey). R = r I and T^H T = (1 - r^2) I on the pair within
  1e-8, with the Fresnel value r = (n1 - n2) / (n1 + n2); no coupling into the other modes (1e-7); the fields of
  the pair are real.
- The complete set (a 0.2 and a 0.4 um strip on an 8 x 6 grid; all 96 modes from a dense eigensolve, most of them
  below cutoff): for the incidence of modes 0, 1 and 5, E_t and H_t are continuous at every sample within 1e-8 of
  the field maximum (the exact discrete solution); meow's formula with the same product, no TSVD and no passivity
  rule gives the same column of S within 1e-8; S = S^T within 1e-8.
- The post-processing removes a copy of a mode, a backward copy (flipped back, then a copy), a mode with a NaN
  field and a zero mode; a mode set with a mode twice raises `ValueError`.
- The documented path on the complete set: `post_process_eigenmodes` on all 96 modes of the 8 x 6 grid keeps all
  96 (two strips: with complex pairs and modes on the growing branch; two uniform windows: with degenerate clusters
  below cutoff), all decay in +z, and E_t and H_t are continuous within 1e-8 (review finding: the first version
  removed the members with Re(neff) < 0 of complex pairs and the modes with Im(neff) < 0, 24 and 30 of the 96 modes,
  and the continuity errors were 6.6 % and 17 %).
- `max_imag_neff`; a cluster without a real basis stays as it is; a step with 8 modes on the left and 5 on the right
  (S unitary and symmetric within 1e-7); the star cascade with a non-symmetric S (meow's interface without its
  symmetrization).
- The solver at 60 modes on the first cell of the taper (0.02 um grid): each mode is an eigenpair of the assembled
  operator (relative residual below 1e-9, an ARPACK eigenvalue), no copies, and each complex mode has its conjugate
  eigenvalue in the ARPACK spectrum (a complex pair); `post_process_eigenmodes` keeps all 60.
- Errors: two different grids, a PML window (on either side).

`suite/tests/test_eme.py` (`test_conserving_interface_keeps_the_power`): a 0.5 to 1.2 um step in photonsuite sends
the input power into the kept modes within 1e-6 (meow's interface without a passivity rule: more than 1 + 1e-4); a
straight guide transmits 1 within 1e-9.

**Results** (the files of `benchmarks/2026-11-08/`: `meow-interface.jsonl`, 55 rows, and `meow-interface-2.jsonl`,
12 rows, with meow's SAX cascade; `meow-interface-3.jsonl`, 4 rows, and `meow-interface-4.jsonl`, 25 rows, with the
star cascade, see "Memory of the cascade"; `meow-solver-arpack.jsonl`, 8 rows; `suite-eme-fdtd-taper.jsonl` and
`suite-eme-fdtd-double-step.jsonl` for the FDTD; RX 7900 XT, precision "mixed", 0.02 um grid unless stated, one EME
cell per segment; the commands are at the end of this part). The meow rows give the eighth-wave rows of
photonsuite.eme again (`suite-eme-fdtd-taper.jsonl`), for example T00 = 0.8761 ("invert") and 0.8980 ("none") at
1.55 um and 10 modes. The power columns give the sum of |S|^2 from the input TE0 mode into the left and the right
ports of the modes above cutoff. The conserving rows above 40 modes come from `meow-interface-4.jsonl` (the
post-processing after the review of 2026-11-08: modes on the wrong branch are flipped, not removed). The conserving
rows above 40 modes of `meow-interface.jsonl` and `meow-interface-2.jsonl` came from the first version of
`post_process_eigenmodes`, which removed valid modes (the members with Re(neff) < 0 of complex pairs and the modes
on the growing branch); they are not used. With 40 modes or fewer no mode is below cutoff, and nothing was removed.

**Scope: the mode sets lack modes.** All EME rows here, of both interfaces and at all mode counts, use the mode sets
of the meow_gpu solver. In this window the solver misses about 10 of the 40 eigenvalues nearest the target, all
above cutoff ("The modes that the solver misses", below; the first item of "Known limits and next steps"). So each
"40 modes" set is not the set of the 40 modes nearest the target, and each EME result here is the result for such a
set. It is not known how much the missed modes change T00.

The causes (the 4 `--diag` rows; the taper, 1.55 um, 10 / 20 / 30 / 40 modes):

- meow's interface without the passivity rule: the largest singular value of an interface S-matrix of the taper is
  1.094, 1.165, 1.564, 2.101, and the right-going kept modes carry 1.0115, 1.0107, 1.0103, 1.0089 of the input
  power.
- meow's formula without the TSVD (rcond 1e-12) is unstable with both products (largest singular values 3e5 to
  5e11; T00 up to 4e4).
- The conserving formula with meow's Gram-Schmidt modes and meow's product gives T00 = 0.88708, 0.89439, 0.90394,
  0.91224, within 7.2e-5 of the conserving interface (0.88701, 0.89435, 0.90390, 0.91221), with a largest singular
  value of 1.001 to 1.015. Thus the formula sets T00; the product adds a smaller power error.
- The largest Gram entry off the diagonal: meow's product on the eigenmodes 0.0095, 0.0136, 0.0216, 0.0333; the Yee
  product 1.3e-9 to 5.6e-9.

The taper of the eighth wave (T00 at 1.50, 1.525, 1.55, 1.575 and 1.60 um):

| interface | modes | T00 | power |
|---|---|---|---|
| meow, "invert" (the default) | 40 | 0.8813, 0.8836, 0.8857, 0.8876, 0.8891 | 0.9750 to 0.9784 |
| meow, "none" | 40 | 0.9132, 0.9163, 0.9189, 0.9214, 0.9235 | 1.0085 to 1.0096 |
| conserving | 10 | 0.8813, 0.8842, 0.8870, 0.8900, 0.8926 | 1 within 1e-12 |
| conserving | 20 | 0.8878, 0.8912, 0.8943, 0.8934, 0.8951 | 1 within 1e-12 |
| conserving | 30 | 0.8998, 0.9021, 0.9039, 0.9060, 0.9079 | 1 within 1e-12 |
| conserving | 40 | 0.9061, 0.9094, 0.9122, 0.9148, 0.9173 | 1 within 1e-12 |
| conserving (b) | 60 | 0.9098, 0.9139, 0.9170, 0.9196, 0.9214 | 0.999998 to 1.000000 |
| conserving (b) | 80 | 0.9071, 0.9102, 0.9129, 0.9156, 0.9189 | 0.999952 to 1.000212 |
| 3-D FDTD, 60 cells per um (a) | | 0.9008, 0.9058, 0.9103, 0.9143, 0.9178 | T_flux 0.9996 to 1.0002 |

(a) `benchmarks/2026-11-08/suite-eme-fdtd-taper.jsonl`, the eighth wave. (b) `meow-interface-4.jsonl`; all
requested modes kept; modes below cutoff in all cells (251 to 387 of the stack at 60 modes, 691 to 827 at 80).

- The conserving T00 rises by 0.0025 to 0.0127 for each 10 more modes up to 40 modes, by 0.0036 to 0.0048 from 40 to
  60 modes, and falls by 0.0025 to 0.0041 from 60 to 80 modes. From 40 to 80 modes it stays between 0.9061 and
  0.9214; it has no converged value.
- The conserving interface differs from the FDTD (60 cells per um) by +0.0053, +0.0035, +0.0019, +0.0005, -0.0005
  with 40 modes, +0.0090, +0.0080, +0.0066, +0.0053, +0.0036 with 60 modes and +0.0063, +0.0044, +0.0025, +0.0013,
  +0.0011 with 80 modes (the FDTD limit is known to 2e-3 to 4e-3, `suite/README.md`); meow with "invert" is 0.0196
  to 0.0287 below the FDTD, with "none" 0.0058 to 0.0124 above (40 modes). Over the band, the conserving T00 rises
  by 0.0111, 0.0116 and 0.0118 (40, 60 and 80 modes), the FDTD T00 by 0.0169.
- With modes below cutoff the power on the ports of the modes above cutoff stays within 2.2e-4 of 1 (80 modes).

The double step (0.5 um, then 1.25 um for 0.1 um, then 2.0 um; `--length 0.1 --segments 1`; the FDTD is
photonsuite.fdtd_avg on the same segments):

| method | modes | T00 | power |
|---|---|---|---|
| conserving | 10 | 0.5987, 0.5943, 0.6055, 0.6171, 0.6293 | 1 within 1e-12 |
| conserving | 20 | 0.5839, 0.5946, 0.6058, 0.6174, 0.6294 | 1 within 1e-12 |
| conserving | 40 | 0.5532, 0.5603, 0.5676, 0.5744, 0.5810 | 1 within 1e-12 |
| conserving (e) | 60 | 0.5554, 0.5622, 0.5690, 0.5762, 0.5846 | 1 within 1e-7 |
| conserving (e) | 80 | 0.5561, 0.5605, 0.5645, 0.5713, 0.5807 | 1 within 1e-7 |
| conserving (e) | 120 | 0.5575, 0.5637, 0.5690, 0.5739, 0.5828 | 1 within 1e-7 |
| conserving, 0.01 um grid (b) | 40 | 1.55 um: 0.5677 | 1.00000 |
| conserving, window x -3 to 3, y -1.8 to 2.02 um (c) | 40 | 1.55 um: 0.6101 | 1 (no mode below cutoff) |
| conserving, the same window (c) | 80 | 1.55 um: 0.5825 | 1 (no mode below cutoff) |
| meow, "none" | 40 | 0.5575, 0.5656, 0.5741, 0.5835, 0.5938 | 1.0339 to 1.0355 |
| meow, "invert" | 40 | 0.5312, 0.5366, 0.5424, 0.5487, 0.5550 | 0.9695 to 0.9724 |
| 3-D FDTD, 30 cells per um (d) | | 0.5517, 0.5559, 0.5616, 0.5688, 0.5776 | T_flux 0.9954 to 0.9978 |
| 3-D FDTD, 40 (d) | | 0.5480, 0.5522, 0.5578, 0.5650, 0.5738 | T_flux 0.9954 to 0.9977 |
| 3-D FDTD, 50 (d) | | 0.5501, 0.5543, 0.5599, 0.5670, 0.5758 | T_flux 0.9955 to 0.9976 |
| 3-D FDTD, 60 (d) | | 0.5492, 0.5535, 0.5591, 0.5663, 0.5751 | T_flux 0.9954 to 0.9975 |

(b) `meow-interface-2.jsonl`. A 0.005 um grid did not run: the refinement of the mode solver asked for 4.37 GiB of
GPU memory (its GMRES basis, 168 b n bytes for b = 48 block vectors and n = 582,400 unknowns), above the GPU memory
limit of the shared computer (`XLA_PYTHON_CLIENT_MEM_FRACTION=0.4`). (c) `meow-interface-3.jsonl` (star cascade; no
mode below cutoff, so nothing was removed): the walls 1 um farther away than in the FDTD window; 120 modes did not
run (an error row: a 2.29 GiB allocation of the same GMRES basis, b = 128 and n = 114,600, under the same limit).
(d) `suite-eme-fdtd-double-step.jsonl`. (e) `meow-interface-4.jsonl`; all requested modes kept; 36 to 230 modes
of the stack below cutoff.

The EME and the FDTD do not agree on the double step, and the cause is not known:

- With 40 modes (all above cutoff, power conserved) the EME is above the FDTD (60 cells per um) by +0.0040,
  +0.0068, +0.0085, +0.0082, +0.0058. The FDTD changes by up to 0.0038 between 30 and 60 cells per um, and not
  monotonically (0.5616, 0.5578, 0.5599, 0.5591 at 1.55 um).
- The EME is not converged in the mode count: from 20 to 40 modes T00 falls by 0.031 to 0.048; from 40 to 120
  modes it moves by 0.0034 to 0.0049 at each wavelength and not monotonically (1.55 um: 0.5676, 0.5690, 0.5645 and
  0.5690 at 40, 60, 80 and 120 modes). Over 40 to 120 modes the EME is above the FDTD by 0.0040 to 0.0103.
- Where the power goes at 1.55 um: the EME (40 modes) puts 0.5676 into TE0, 0.4255 into the other transmitted modes
  and 0.0070 into the reflected modes. The FDTD (60 cells per um) puts 0.5591 into TE0 and 0.4377 into the rest of
  the transmitted flux (T_flux 0.9968); its input plane gives a mode power of 1.0037 of the net flux, which is about
  1 + R for a pure TE0 source (R about 0.4 %). Thus the EME puts 0.0122 less power into the other transmitted modes,
  0.0085 more into TE0, and reflects about 0.0032 more. T_flux + 0.0037 = 1.0005: the FDTD loses almost no power
  through the sides of the window.
- The EME grid is not the cause: the 0.01 um grid gives 0.56771 at 1.55 um and 40 modes, 0.00016 above the 0.02 um
  grid (0.56756), against a gap of 0.0085. (The edges of the 1.25 um section, x = +-0.625 um, are not on a grid line
  of the 0.02 um grid; the FDTD averages the permittivity.)
- The walls: the EME window has PEC walls at x = +-2 um and y = -0.8 and 1.02 um (0.8 um from the core), the FDTD a
  0.4 um PML outside the same window, so a field that reaches the walls is reflected in the EME and absorbed in the
  FDTD. In the larger window the EME gives 0.6101 at 40 modes and 0.5825 at 80 modes (1.55 um), against 0.5676 at 40
  modes in the default window. The larger window has more modes above cutoff, so at 40 and 80 modes it is farther
  from convergence in the mode count (the default window gave 0.6055 and 0.6058 at 10 and 20 modes). These rows
  cannot show the effect of the walls; it is not known.
- Not tested: the walls, the modes that the solver misses (below), the mode truncation and the FDTD error. The gap
  of 0.0040 to 0.0085 at 40 modes (0.0040 to 0.0103 over 40 to 120 modes) is not explained.

**The modes that the solver misses** (`meow-solver-arpack.jsonl`, `bench/solver_vs_arpack.py`: the first and the
last cell of the taper, 0.5 and 2.0 um strips, 1.55 um, the 0.02 um grid; ARPACK shift-invert on the assembled
operator at the target of the solver):

- Each mode of the solver is an eigenpair: relative residual 2.4e-11 or less, eigenvalue within 9.2e-13 (relative)
  of an ARPACK eigenvalue; no copies. After `post_process_eigenmodes` each complex mode has the other member of its
  pair in the set, except one mode in the 60-mode and one in the 120-mode set of the 0.5 um cell (the other member is
  outside the set).
- The solver does not give the k eigenvalues nearest the target. Of the nearest 40, 60, 80 and 120 it misses 10, 13,
  15 and 18 (0.5 um cell) and 9, 12, 14 and 18 (2.0 um cell).
- The missed modes above cutoff are the same in both cells: neff = 1.444 (twice), 1.43094, 1.39104, 1.37979,
  1.32187, 1.21849, 1.16624, 1.07105, 0.85716, 0.67428 and 0.49757 (all twelve from 60 modes on; the first ten and
  nine at 40 modes); the others are below cutoff. These values are, within 1.9e-3, the modes of the window without
  the core that have one transverse wavenumber of zero: sqrt(1.444^2 - (m lambda / 2a)^2) with a = 4 um, m = 1 to 7,
  and sqrt(1.444^2 - (n lambda / 2b)^2) with b = 1.82 um, n = 1 to 3, and the two uniform-field modes at 1.444 (an
  interpretation of the logged values; why the solver misses them is open).

**Memory of the cascade.** The rows of `meow-interface.jsonl` and `meow-interface-2.jsonl` used meow's SAX cascade
(`meow.compute_s_matrix`, klu backend). Its analysis step (`sax.backends.klu.analyze_circuit_klu`) builds a boolean
matrix of (all S-matrix entries of the interfaces and the propagations) x (the connected port pairs, in both
directions), and index arrays of the same shape: about 16 n^2 k^3 entries for n interfaces with k modes each. All
memory values here are in GiB (the bench and the probe divide by 2^30). `bench/mem_cascade.py` (synthetic stacks of
22 cells with random interface S-matrices; CPU; `benchmarks/2026-11-08/meow-cascade-memory.jsonl`) gives a peak RSS
increase of 0.56, 0.99 and 2.14 GiB for 10, 20 and 30 modes (7.4e6, 5.8e7 and 1.9e8 entries; 5.5 to 6.3 s): about
0.5 GiB + 8.5 bytes per entry. For the 22 cells of the taper this is about 12 GiB at 57 kept modes and 24 GiB at 73
kept modes; the process peaks of `meow-interface-2.jsonl` were 16.3 GiB (60 modes) and 28.3 GiB (80 modes), on a
computer with 29 GiB. The star product (`meow_gpu.interface.compute_s_matrix`: cell after cell, (I - R R)^-1 of k x
k blocks) increased the peak RSS in `bench/mem_cascade.py` by 0.001 GiB at most (0.01 s) and gave the same T00
within 2e-15; `tests/test_interface.py` checks it against meow's cascade on real modes (1e-10, both interfaces, also
a non-symmetric S). On the taper at 80 modes and 1.55 um the bench with the star cascade (`meow-interface-3.jsonl`)
gives T00 = 0.91223, the value of the SAX row of `meow-interface-2.jsonl` (both with the first version of the
post-processing, so the two cascades had the same modes), with a process peak of 5.45 GiB against 28.3 GiB. The
bench (default `--cascade star`) and `photonsuite.eme` with `interface="conserving"` use the star product. The bench
refuses a row whose estimate (the mode fields: 1.67 MiB per mode on the 200 x 91 grid, 2.86 GiB for each list of 22
cells x 80 modes; or the SAX cascade) is above `--max-gb` (GiB, default 8).

Commands (from the repository folder on the GPU computer). The rows of `meow-interface.jsonl` and
`meow-interface-2.jsonl` came from the bench before `--cascade` existed, that is with the SAX cascade, and from the
first version of `post_process_eigenmodes`. With the bench of today, `--cascade sax` repeats the cascade; the rows
of 60 and 80 modes then also need `--max-gb` above their estimates (about 12 and 24 GiB), which do not fit on the
shared computer. The rows of `meow-interface-3.jsonl` and `meow-interface-4.jsonl` used the star cascade (peak RSS
of the `-4` rows 6.34 GiB):

```bash
B="python bench/bench_interface.py"; W="--wavelengths 1.50,1.525,1.55,1.575,1.60"
bash tools/gpu_run.sh meow_gpu $B --modes 10,20,30,40 $W                    # meow-interface.jsonl: 20 rows
bash tools/gpu_run.sh meow_gpu $B --modes 10,20,30,40 --diag                # 4 rows, 1.55 um
bash tools/gpu_run.sh meow_gpu $B --length 0.1 --segments 1 --modes 10,20,40,60,80,120 $W   # 30 rows
GPU_RUN_TIMEOUT=3600 bash tools/gpu_run.sh meow_gpu $B --modes 60,80 --interfaces conserving $W   # -2.jsonl: 10
bash tools/gpu_run.sh meow_gpu $B --length 0.1 --segments 1 --modes 40 --step 0.01 --interfaces conserving
bash tools/gpu_run.sh meow_gpu $B --length 0.1 --segments 1 --modes 40,80,120 --window=-3,3,-1.8,2.02 \
  --interfaces conserving                                                   # -3.jsonl: 3 rows (1 error row)
bash tools/gpu_run.sh meow_gpu $B --modes 80 --interfaces conserving --wavelengths 1.55   # -3.jsonl: 1 row
GPU_RUN_TIMEOUT=3600 bash tools/gpu_run.sh meow_gpu $B --modes 60,80 --interfaces conserving $W   # -4.jsonl: 10
bash tools/gpu_run.sh meow_gpu $B --length 0.1 --segments 1 --modes 60,80,120 --interfaces conserving $W   # -4: 15
bash tools/gpu_run.sh meow_gpu python bench/solver_vs_arpack.py --widths 0.5,2.0 --modes 40,60,80,120   # 8 rows
for r in 30 40 50 60; do                                                    # suite-eme-fdtd-double-step.jsonl
  bash tools/gpu_run.sh suite python bench/eme_fdtd_taper.py fdtd --resolution $r --length 0.1 --segments 1
done
cd meow_gpu; for k in 10 20 30; do for m in star sax; do                    # meow-cascade-memory.jsonl (CPU)
  JAX_PLATFORMS=cpu ../.venv-rocm/bin/python bench/mem_cascade.py --method $m --cells 22 --modes $k; done; done
```

Limits:

- Straight guides in a closed window only (no PML, no bend, no angled propagation: `ValueError`). The product of a
  PML window or a bend needs the coordinate stretch of the solver.
- Modes below cutoff carry no power. For them S holds the amplitudes of the unconjugated normalization. With such
  modes in a truncated set, P is complex and power conservation is not proven: check the power sums
  (`P_right_prop`, `P_left_prop`; the rows of `meow-interface-4.jsonl` stay within 2.2e-4 of 1). A degenerate
  cluster below cutoff is not changed. In the window of the bench, the sets of 40 modes are all above cutoff.
- The interface is exact for the complete set and conserves power for every truncation of a set above cutoff, but
  T00 still changes with the mode count: the box modes of the closed window stand for the radiation.
- With modes below cutoff in the set, |S|^2 summed over all ports is not the power: the bench gives the sums over
  the ports of the modes above cutoff (`P_right_prop`, `P_left_prop`).
- The eigensolver can miss modes nearer the target than the kept ones ("Known limits"); a missed mode is a
  truncation, not an error of the interface. In the window of the taper the solver misses about 10 modes above
  cutoff ("The modes that the solver misses"; `test_solver_modes_near_cutoff_are_eigenpairs` checks that its modes
  are true eigenpairs without copies).

## Known limits and next steps

- **Open, first: the solver misses modes above cutoff.** In the window of the taper and the double step (x -2 to 2
  um, y -0.8 to 1.02 um, 0.02 um grid) the Davidson solver of meow_gpu gives correct eigenpairs, but not the set
  nearest the target: it misses 10 and 9 of the 40 nearest eigenvalues of the 0.5 and the 2.0 um cell, and 12 modes
  above cutoff from 60 modes on (`meow-solver-arpack.jsonl`; "The modes that the solver misses"). The results that
  use such sets: every EME row of "A power-conserving interface" (meow's interface and the conserving interface, all
  mode counts; the larger window of the double step probably also, not checked), the 10 um taper of "Correction of
  meow's post-processing", and the EME rows of photonsuite in `suite/README.md`. The effect on T00 is not known.
  Next: find why the solver misses these modes (its start block, its preconditioner or its selection; not checked),
  fix it, and repeat these rows.

- At small cross-sections (about 36k unknowns) the GPU is about 3x faster than one core. The batched solve gains
  less in mixed precision than in single precision (see "Profile of the batched path"): the device kernels of a
  batch take a time that grows almost linearly with B, and the refinement runs member after member. Next: a
  batched refinement in lockstep (it removes the small kernels, copies and Python steps of each member, not the
  GMRES time), and faster inner products (the Gram matrices with a long contraction run far below the peak rate of
  the card).
- The host thread pool of the batched solve needs one BLAS thread. With threadpoolctl, the library sets it while
  the pool runs: for a BLAS with pthreads (the PyPI wheels) the calling thread sets the process-wide count, one
  caller at a time (a lock), and other threads of the process also run their BLAS with one thread meanwhile; for an
  OpenMP BLAS (a count for each thread) each pool worker sets its own limit (`_blas_limit`; the OpenMP path is
  tested with a stand-in controller only). Without threadpoolctl, set `OMP_NUM_THREADS=1` before the import of
  numpy; else the host steps run in one thread (with a warning). With the default OpenBLAS threads, the pool gained
  nothing in one run and was up to 2x slower than one thread in another run.
- Limits of the batched solve: a batch costs the iterations of its slowest member (a member that stopped still
  goes through the kernels, as zero rows). The first batch starts cold in all members (no warm start). The host
  eigenproblems grow with B (one for each member, in a thread pool of up to 8 threads). All members need the same
  grid shape (`compute_modes_gpu_batch` groups the cells by grid shape). The device memory grows with B (the
  subspace buffers are B x max_subspace vectors).
- The earlier complex128 polish (`polish="davidson"`) is slow from Ritz vectors that are already converged in
  complex64 (a small subspace of 3 blocks: about 35 iterations for the 0.45 um strip, 40 nm grid, of
  `tests/test_batch.py`). The default polish of `mixed` is now the refinement (see above).
- The refinement is still about 1.8x slower than `single` on the RX 7900 XT (taper bench above). It can stall:
  with fewer GMRES steps (`refine_inner`), and for a weakly guided mode above a dense band of window modes (see
  "Fallback"). Then the
  fallback (the steps of the earlier polish from the complex64 block) finishes the solve, and the solve costs more
  than the earlier polish alone (more complex128 applies too).
- Device memory: the GMRES basis holds (`REFINE_INNER` + 1) x b complex64 vectors, 168 b n bytes at 20 steps (b
  block vectors, n unknowns; 0.8 GB for b = 24 and n = 200k). The earlier polish keeps 3 b complex128 vectors in V
  and in A V (96 b n bytes).
- On a CPU the refinement is not faster than the earlier polish: the complex128 rate of a CPU is not low, and the
  refinement does more complex64 work. Not measured with the code of this version.
- The mode set of `mixed` after a warm start comes from a complex64 stage that stops at 1e-3: a near-degenerate
  pair nearer the target that is not in its block can be missed, and then the result depends on the rounding (see
  "Cold starts"). The warm start makes this less probable (the block holds the modes of a similar problem), but it
  is not checked. Use `switch_tol=2e-5` (or `polish="davidson"`) when such a pair is probable.
- All precisions can miss eigenvalues nearer the target than the kept modes. Taper cell 18 (`bench_refine.py`):
  ARPACK on the assembled matrix finds lambda = -1.9350 and -1.9038 (box modes below cutoff, neff 1.391 and 1.380)
  between the 9th (-1.9871) and the 10th kept mode (-1.8877); the double, the mixed and the single solves keep
  -1.8877. The strip of `tests/test_refine.py` (4 modes): ARPACK finds two modes at neff 1.444 and modes at 1.4074
  and 1.3604 nearer the target than the 4th kept mode (1.3413), which tidy3d also gives (`tests/test_solver.py`).
- The batched solve (`davidson_batch`) runs the refinement member after member, after the batched complex64
  Davidson (the same steps as the per-cell solve, but the device kernels of the refinement are not batched). A
  `jax.vmap` of the GMRES kernel over 8 members takes the time of 8 calls (see "Profile of the batched path"), so
  a batched refinement can only remove the smaller costs of each member.
- With PML, the solver can miss PML eigenvalues that are nearer the target than the kept modes (meow filters PML
  modes, so the kept modes were still correct in the review test). `tests/test_bend.py` solves one PML eigenpair
  (a radiating bend) near a given target.
- Bends: tight bends with PML converge slowly (double precision, 1 mode: 199 iterations at R = 2 um, more than 500
  at R = 1.5 um in the bench domain). The preconditioner does not include the bend; a better one is open work.
  The bend transform of tidy3d has an error of first order in dx / R (see "Bend modes"). The modes beyond the
  guided ones are box modes at the outer wall; tidy3d and the mixed and single solves can miss some that are
  nearer the target than the kept modes (only the double solve is checked against ARPACK), so compare bend mode
  sets of two solvers or precisions only for the guided modes.
- meow's own post-processing (normalization, Gram-Schmidt, on the CPU) costs about 35 ms per cell.

## Tests

`uv run pytest` (CPU) checks the operator against the tidy3d matrix, the eigenvalues against tidy3d in each
precision, the warm start, the meow backend (neff and field overlap against `compute_modes_tidy3d`), and the device
placement. `tests/test_batch.py` checks that the batched solve gives the per-cell results (neff, E and H, in each
precision; a batch of 1; the warm start between batches; `compute_modes_gpu_batch` with batch 1, 2 and 8 on a
5-cell taper, with a short last run; two grid shapes and a cell that falls back to tidy3d in one list; the
post-processing in a second thread, and its errors; the BLAS limit of the host thread pool, by threading layer,
and without threadpoolctl), ragged convergence (members that stop at very different
iterations give the results of `davidson` from the same initial block), the stop by the stall rule and the iteration
limit of each member, and the tidy3d direction rule for modes below cutoff. On the CPU the iterations and the
matvecs must be the same as those of the per-cell solve (all three precisions); on a GPU the iteration check has a
slack and is not done in single precision.
`tests/test_refine.py` checks the mixed-precision refinement (above): the stop of the complex64 stage after a cold
and a warm start (`COLD_SWITCH_TOL`, per cell and batched), a warm start on the SiN tip of the example grid (with
and without the admission rule), the documented core of `structures.sin_tip`, the strip against the double solve and
the SciPy residuals, the two-core cluster against ARPACK, a mode count that cuts a near-degenerate pair (1 and 3
modes of the two cores, 3 start blocks, against ARPACK and the double solve by neff and field overlap), the
complex128 applies against the earlier polish, the tracking and the selection of the pairs, the weakly guided SiN
tip (the fallback; against the earlier polish, the double solve, ARPACK and the SciPy residuals; on the 50 nm grid
of the example all 4 modes, with and without the admission rule; the batched fallback), the stop with
`converged=False` (tolerance out of reach: the stall rule of the fallback and the better result; step limit; solver
warning), the error without `jax_enable_x64`, and the earlier polish (per cell and batched). `tests/test_bend.py`
checks the bend modes (see "Bend modes"), and a batch of bends and a straight cell. `tests/test_interface.py`
checks the power-conserving interface (see "A power-conserving interface"). Run all tests on the GPU
computer (`bash tools/remote.sh push`, then `bash tools/gpu_run.sh meow_gpu python -m pytest -q` there;
`tools/gpu_run.sh` uses `.venv-rocm` and the GPU lock; first `bash tools/remote.sh setup-env`).
