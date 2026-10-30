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
- The refinement is still about 2x slower than `single` on the RX 7900 XT. It can stall: with fewer GMRES steps
  (`refine_inner`), and for a weakly guided mode above a dense band of window modes (see "Fallback"). Then the
  fallback (the steps of the earlier polish from the complex64 block) finishes the solve, and the solve costs more
  than the earlier polish alone (more complex128 applies too).
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
5-cell taper, with a short last run; two grid shapes and a cell that falls back to tidy3d in one list), ragged
convergence (members that stop at very different iterations give the results of `davidson` from the same initial
block), the stop by the stall rule and the iteration limit of each member, and the tidy3d direction rule for modes
below cutoff. On the CPU the iterations and the matvecs must be the same as those of the per-cell solve (all three
precisions); on a GPU the iteration check has a slack and is not done in single precision.
`tests/test_refine.py` checks the mixed-precision refinement (above): the strip against the double solve and the
SciPy residuals, the two-core cluster against ARPACK, a mode count that cuts a near-degenerate pair (1 and 3 modes
of the two cores, 3 start blocks, against ARPACK and the double solve by neff and field overlap), the complex128
applies against the earlier polish, the tracking and the selection of the pairs, the weakly guided SiN tip (the
fallback; against the earlier polish, the double solve, ARPACK and the SciPy residuals; on the 50 nm grid of the
example all 4 modes, with and without the admission rule; the batched fallback), the stop with `converged=False`
(tolerance out of reach: the stall rule of the fallback and the better result; step limit; solver warning), the
error without `jax_enable_x64`, and the earlier polish (per cell and batched).
`tests/test_bend.py` checks the bend modes (see "Bend modes"), and a batch of bends and a straight cell. Run all tests
on the GPU computer (`bash tools/remote.sh push`, then `bash tools/gpu_run.sh meow_gpu python -m pytest -q` there;
`tools/gpu_run.sh` uses `.venv-rocm` and the GPU lock; first `bash tools/remote.sh setup-env`).
