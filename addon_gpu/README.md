# addon_gpu: the add-on route of FDTD on the AMD GPU

This folder runs the GPU time stepping of the Erised add-on (`meep_gpu` 1.0.0, GPL-2.0-or-later,
https://github.com/Erised-AI-Inc/meep-gpu, pinned at commit `8993426`) on an AMD Radeon RX 7900 XT (gfx1100), and
measures it. The MIT FDTD package (version 1.35.0 from conda-forge, double precision, serial build) builds each
simulation, and the add-on time-steps it on the GPU in float32. The add-on supports NVIDIA GPUs (CuPy, with Triton
kernels or hand-written CUDA kernels) and Apple GPUs (Metal). This folder adds the AMD GPU: CuPy for ROCm (hipRTC
compiles the hand-written kernels) and the AMD backend of Triton.

**Status (2026-11-07): version 0.1 works.** Both kernel tables of the add-on run on the AMD GPU, with three small
patches (`tools/env/addon_patches/`). On three cases the kernels give the same float32 words as the array path of
the add-on, and they agree with the CPU solver to 8.3e-6 or better (relative L2). On a 3-D case of 2,097,152 cells
the time loop is 16 times (Triton) to 27 times (hand-written kernels) as fast as the serial CPU solver (float32 on
the GPU, float64 on the CPU). Patch 0003 lets the lift read the permittivity in bulk on this solver build, so a whole
run (setup and time loop) is now faster on the GPU route in both bench cases: 2.19 to 2.34 times in 2-D (1,536,000
cells, 2,400 steps) and 3.12 to 3.37 times in 3-D (2,097,152 cells, 1,280 steps), with the kernel tables. Before
patch 0003 the setup dominated, and the whole 2-D run was 3.4 to 3.7 times slower than on the CPU solver. Small runs
are still slower on the GPU ("Timings").

## What runs on the AMD GPU

Measurements of 2026-10-30 and 2026-11-07, on one RX 7900 XT: ROCm 7.2 runtime from the PyTorch wheel,
PyTorch 2.14.1+rocm7.2, Triton 3.8.0 (AMD), CuPy 14.2.0 (cupy-rocm-7-0), the solver 1.35.0 (double precision,
serial), the add-on 1.0.0.

- **Array path** (CuPy arrays on the GPU): runs without a patch. It is the default run. Without patch 0002 it does
  not keep subnormal values under `keep` (below).
- **Triton kernel table**: runs with `MEEP_GPU_HARDWARE=any`, without a patch.
- **Hand-written kernel table** (`cuda`, RawKernel source compiled by hipRTC): refused without a patch (hipRTC does
  not accept `--fmad=false`); runs with `MEEP_GPU_HARDWARE=any` and patch 0001.
- **Float32 subnormal policy `keep`** (the policy of both NVIDIA tables): without a patch, CuPy does not attain it
  and gives no message; attained with patch 0002.

The add-on reads the RX 7900 XT as compute capability 11.0 and Triton 3.8.0. Both are outside the range that the
add-on supports (compute capability 7.0 to 9.0, Triton 3.1). By default it therefore refuses the kernel tables by
name, and the run takes the array path. `MEEP_GPU_HARDWARE=any` is the switch of the add-on that admits such a
device. Each run then prints a NOTE line, and `driver.fast_path_report()` gives `certified: False`. On every case
below, both tables serve 4 of the 7 sub-steps of a time step (`step_B`, `update_H`, `step_D`, `update_E`). The
sources, the monitors and the other sub-steps run on the array path, as on NVIDIA.

What does not run, or is not attained, and why:

- **Certification.** No AMD identity is certified. The certification of the add-on is a set of device gates on
  named NVIDIA and Apple hardware; those gates have not run on this GPU. The checks below take their place here.
- **The `flush` subnormal policy.** The NVIDIA tables run under `keep` by default. Under `flush` (which a caller can
  install), two executors do not follow it on this GPU: Triton keeps subnormal values in all 6 op classes of
  `triton_subnormal_probe.py` (the flush mechanism of the add-on is the NVIDIA one), and a negation of a subnormal
  value keeps it in CuPy (3 cells of the gate; NVIDIA flushes it). Do not install `flush` on this GPU.
- **The Triton leg of the gate of the add-on** (`parity/meep_gpu/gate_subnormal_policy.py`) does not run with
  Triton 3.8: it calls `JITFunction.cache`, which Triton 3.8 does not have. `triton_subnormal_probe.py` repeats its
  method for the Triton op classes.
- **A parallel CPU reference.** The solver of this environment is a serial build: 1 and 4 OpenMP threads gave the
  same time (bench). No MPI build was compared.

## The checks

All with both patches, unless the item says otherwise. "Words" are float32 words of the final Ez field, compared bit
for bit; "relative" is the relative L2 difference.

1. **Kernels against the array path** (`tests/test_parity.py`, `compare_legs.py`, 2026-11-07). Both tables, 0
   differing words on each case: the 2-D slab waveguide (24,000 words, 800 steps, two flux spectra), the 2-D ring
   resonator of the tutorial of the MIT FDTD package (25,600 words, 8,000 steps, and the 8,001 values of the field
   at a point), the 3-D dielectric sphere (110,592 words, 240 steps, a flux spectrum). The two bench cases also
   gave 0 differing words (1,536,000 and 2,097,152 words).
2. **Against the CPU solver** (its own time stepping, double precision). Field: 2.0e-7 (slab), 1.0e-6 (ring),
   8.0e-7 (sphere), 5.5e-7 (2-D bench case), 8.3e-6 (3-D bench case). Flux: 2.0e-8 to 4.8e-7. Point series of the
   ring: 5.7e-7. The add-on accepts 1e-4. The NumPy reference of the add-on (float32 on the host) is at the same
   level: field 2.0e-7, 1.1e-6 and 7.9e-7 on the three cases.
3. **Resonances of the ring** (Harminv on the point series, three modes). The two tables give the same modes. Against
   the CPU solver: frequency 0.118102, 0.147163 and 0.175247, different by 5.9e-8, 1.1e-6 and 2.2e-7; Q 80.6, 316.5
   and 1686 against 80.7, 316.2 and 1675 (up to 0.67 %). The NumPy reference of the add-on differs from the CPU
   solver by up to 4.4e-7 in frequency and 0.14 % in Q. `tests/test_parity.py` checks the complex frequency
   f + i decay of each mode to 2e-5 of f, and Q to 1 %.
4. **The CuPy array path against the NumPy array path, word for word after every sub-step** (the module
   `meep_gpu/test_cupy_array_path.py` of the add-on, 20 cases, with planted subnormal and signed-zero words). Pinned
   clone: 20 of 20 fail (for example 984 words differ after the first `step_B`). With both patches: 20 of 20 pass.
5. **The subnormal gate of the add-on**, `keep`, host and CuPy legs, each cell against its exact keep word. Without
   patch 0002: FAIL, 23 of the 29 CuPy cells flushed. With patch 0002: PASS, 37 of 37 cells (8 host, 29 CuPy).
6. `triton_subnormal_probe.py` under no policy, `keep` and `flush`: keep in 6 of 6 op classes under each of them.
6a. Two of the tests in `tests/test_patches.py` check the effect of each patch. A RawKernel computes a * b + c with
   a = b = 1 + 2^-12 and c = -(1 + 2^-11): 2^-24 with no option and with `--fmad=true` (fused), 0 with `--fmad=false`
   (patch 0001 turns the fusion off). The keep token of patch 0002 is in the cache key of each compile after the
   keep install, and in no key before it (`keep_key_probe.py`).
7. **Install check of the add-on** (`tools/check_install.py --require-gpu`, a sphere of 24 x 24 x 24 cells, 240
   steps, 2026-10-30, no patch): the NumPy reference against the CPU solver 7.1e-7 (field) and 8.6e-8 (flux); the
   Triton table with `MEEP_GPU_HARDWARE=any` against the NumPy reference 0.000e+00; the check ends "OK: ... work
   together on this host's GPU".
8. **The example check of the add-on** (`examples/run_examples.py --check`, 4 examples, `MEEP_GPU_HARDWARE=any`,
   2026-11-07), once with each table: PASSED. Each table: 0 mismatches against the recorded output of the add-on in
   4 of 4 examples (recorded on another GPU and a single-precision solver; numbers within 1e-4 of the largest number
   on the line); the GPU leg equals the array leg byte for byte in 3 of 3 compared examples, the dispersive slab
   included; the NumPy reference against the CPU solver 2.2e-6 (sphere), 2.7e-6 (slab) and 3.0e-4 (dispersive slab,
   band 2e-3; reported, not judged, with a double-precision solver).
9. The folder of the hand-written kernels (`meep_gpu/cuda_kernels/` of the pinned clone) holds 145 Python files: 66
   test modules, `__init__.py` and 78 other modules, of which 59 hold `__global__` kernel source. No file contains a
   warp intrinsic (`__shfl`, `warpSize`, `__ballot`, `__syncwarp`, `__activemask`, `__ldg`), and no non-test module
   uses `__shared__`: the kernels are elementwise and do not assume a wave size of 32 or 64.

10. **The bulk read of the permittivity (patch 0003)**, 2026-11-07. The module
    `meep_gpu/test_from_meep_bulk_chi1inv.py` of the add-on: 10 of 10 pass with the three patches. Its 6 bulk-path
    cells (a 3-D PML cell with an off-diagonal tensor, metallic walls, a mirror fold on a periodic axis, two
    cylindrical cells, a dispersive background) run on this build only with patch 0003; in each, the read words equal
    `get_chi1inv` of the solver as uint64 at every point, and the installed arrays equal those of the per-point read.
    The final fields after a run, bulk read against per-point read, both tables: 0 of 1,536,000 differing words (2-D
    bench case, 2,400 steps; fluxes equal), 0 of 2,097,152 (3-D bench case, 320 steps; flux equal), and 0 on the slab
    and the sphere of `tests/test_parity.py`.

The tests of this folder: 23 of 23 pass (`tests/test_patches.py`, `tests/test_parity.py`; 78 s, 2026-11-07, on
the pinned clone after `tools/env/addon_gpu.sh` applied the three patches, without `ADDON_GPU_CLONE`).

The test modules of the add-on, run one at a time (2026-11-07, pinned clone, then the work copy with both patches):

| Module | Pinned clone | With both patches |
|---|---|---|
| `meep_gpu/test_backends.py` | 16 passed | 16 passed |
| `meep_gpu/test_subnormal_policy.py` | 118 passed, 2 failed (keep on a ROCm CuPy), 6 skipped | 120 passed, 6 skipped |
| `meep_gpu/test_cupy_array_path.py` | 20 failed | 20 passed |
| `meep_gpu/test_metal_weld_contract.py` | 26 passed | 22 passed, 4 failed (drift) |
| `meep_gpu/test_triton_weld_contract.py` | 27 passed, 5 failed (drift) | 27 passed, 5 failed (drift) |
| `meep_gpu/test_cuda_weld_contract.py` | 36 passed | 36 passed |

The weld contracts are rule 2 of the add-on: they count the entries of the certification ledgers whose recorded
digests no longer match the files. The 5 Triton failures of the pinned clone are in the list of the add-on of
tests that wait for a certification (`tools/ci/pending_certification.txt`). The patches move the counts: Metal 0 to
7 of 410 pinned paths, Triton raw digests 60 to 91 of 577, Triton code digests 8 to 37 of 432, Triton device digests
1 to 1 of 80, CUDA no change. With patch 0003 too, the counts are the same (2026-11-07). On the AMD GPU this
changes no dispatch decision (no AMD identity is certified, and
the kernels dispatched with both patches on every case). On an Apple or NVIDIA host the patched files would put
those certified entries in drift; the patches are for the AMD route only.

The rest of the suite of the add-on has not run on this GPU: its pytest paths (`testpaths` in its
`pyproject.toml`: `meep_gpu` and `parity/meep_gpu`) hold 409 test modules (`test_*.py`: 225 in `meep_gpu/`, 66 in
`meep_gpu/cuda_kernels/`, 113 in `parity/meep_gpu/`, 5 in `parity/meep_gpu/rounds/1.0/`), and the 6 above are the
only ones that ran.

## Timings

`bench/bench_addon.py`, 2026-11-07, one RX 7900 XT, the computer shared with CPU jobs of another session
(`OMP_NUM_THREADS=4`). Two logged runs of the bench (the lead agent logged the rows):

- with the three patches, on the pinned clone (22:39 to 22:43 EST; the GPU legs read the permittivity in bulk,
  `chi1inv_read: "bulk"`): `benchmarks/2026-11-07/addon-bench-full-0003.jsonl`;
- before patch 0003, with patches 0001 and 0002 on a work copy (commit a6938b6 of this repository; the GPU legs
  read the permittivity one point at a time): `benchmarks/2026-11-07/addon-bench-full.jsonl`.

The add-on steps in float32 on the GPU; the CPU solver steps in float64, serially (1 and 4 OpenMP threads gave the
same time). "Step loop" is `time_s` of a row: the time loop, after a warm-up run in the same process (so kernel
compilation is not timed). "Whole run" is `setup_s + time_s` of a row: it adds the setup before the first step (the
lift of the add-on, or `init_sim` of the CPU solver) and the work after the loop. For the GPU legs it is the run
after the warm-up run of the same process: it does not include the start of the process, the start-up of CuPy and
Triton, or the kernel compilation of a first run (with warm disk caches, a review measured 0.4 s to 0.8 s more for
a fresh process). The factor is the time of the CPU solver leg with 1 thread divided by the time of the leg.

With the three patches (`addon-bench-full-0003.jsonl`):

| Case | Time | CPU solver | Array path | Triton table | Hand-written table |
|---|---|---|---|---|---|
| 2-D slab, 1,536,000 cells, 2,400 steps | step loop | 5.10 s | 2.81 s (1.8x) | 1.00 s (5.1x) | 0.84 s (6.1x) |
| | whole run | 6.35 s | 4.67 s (1.36x) | 2.91 s (2.19x) | 2.72 s (2.34x) |
| 3-D sphere, 2,097,152 cells, 1,280 steps | step loop | 36.6 s | 5.32 s (6.9x) | 2.31 s (16x) | 1.34 s (27x) |
| | whole run | 48.3 s | 18.5 s (2.61x) | 15.5 s (3.12x) | 14.3 s (3.37x) |

Before patch 0003 (`addon-bench-full.jsonl`):

| Case | Time | CPU solver | Array path | Triton table | Hand-written table |
|---|---|---|---|---|---|
| 2-D slab, 1,536,000 cells, 2,400 steps | step loop | 5.11 s | 2.78 s (1.8x) | 1.03 s (5.0x) | 0.82 s (6.2x) |
| | whole run | 6.43 s | 23.9 s (0.27x) | 23.6 s (0.27x) | 22.0 s (0.29x) |
| 3-D sphere, 2,097,152 cells, 1,280 steps | step loop | 37.7 s | 5.34 s (7.1x) | 2.35 s (16x) | 1.34 s (28x) |
| | whole run | 49.2 s | 45.3 s (1.09x) | 41.6 s (1.18x) | 43.0 s (1.15x) |

- Rate (with the three patches): 2-D 723 (CPU) against 3,680 (Triton) and 4,408 (hand-written) million cell
  updates per second; 3-D 73 against 1,161 and 1,998. Between progress callbacks (from step 100 on), the tables step
  the 3-D case at 2,761 (Triton) and 2,652 (hand-written).
- Setup before the first step: with the three patches, 2-D 1.87 s to 1.90 s on the GPU route against 1.25 s for the
  CPU solver, 3-D 12.99 s to 13.17 s against 11.63 s; before patch 0003, 2-D 21 to 23 s and 3-D 39 to 42 s.
- The step loops and the fields did not change with patch 0003: the same agreement with the CPU solver (5.5e-7 and
  8.3e-6), 0 words different from the array path, and loop times within 3 % of the earlier rows.
- Small cases are slower on the GPU: the 2-D slab of 24,000 cells steps at 54 to 65 million cell updates per second
  on the GPU against 211 on the CPU solver (bench `--small`, `benchmarks/2026-11-07/addon-bench-small.jsonl`, before
  patch 0003).

Where the setup went before patch 0003 (`profile_setup.py`, the lift under the Python profiler, development runs of
2026-11-07, patches 0001 and 0002). The add-on read the permittivity of a solver build that it had not compared
("1.35.0 double precision ... not a build the bulk read was compared on") one point at a time: 13,824,009 calls of
`get_chi1inv` (9 for each cell) took 27.7 s of a 29.6 s lift of the 2-D bench case, and 18,874,368 calls took 37.3 s
of a 49.4 s lift of the 3-D bench case. The rest was `init_sim` of the solver (1.3 s and 11.6 s), which the CPU solver
also does. The first step (the configuration freeze) took 0.43 s and 0.74 s. With the bulk read forced on, the read
took 0.05 s to 0.06 s per component in 2-D and 0.24 s to 0.30 s in 3-D (32,640 points per component still one at a
time), and the lift 2.4 s and 14.0 s, of which `init_sim` is 1.3 s and 11.7 s.

Patch 0003 lists this build for the bulk read ("The checks", item 10). Development runs before the second bench
(`setup_s` after a warm-up run in the same process; `run_case.py --warmup 2`, 2026-11-07) gave setup times within
0.1 s of the logged rows:

| Case | Table | Per-point read | Bulk read (patch 0003) |
|---|---|---|---|
| 2-D slab, 1,536,000 cells | hand-written | 21.39 s | 1.89 s |
| 2-D slab, 1,536,000 cells | Triton | 21.33 s | 1.89 s |
| 3-D sphere, 2,097,152 cells | hand-written | 39.73 s | 13.24 s |
| 3-D sphere, 2,097,152 cells | Triton | 39.38 s | 13.06 s |

With patch 0003, the setup on the GPU route is the `init_sim` of the solver and less than 2 s more.

## History

- 2026-11-07, about 16:49:53 EST: the GPU computer stopped (kernel 7.0.0-38). The kernel dump shows an Oops in the
  amdgpu/TTM memory manager (list corruption, then a NULL pointer in `ttm_resource_move_to_lru_tail`) during a GPU
  memory allocation (a KFD ioctl) of a Python process. By the kernel time stamps, the Oops came at about 16:49:52.8,
  81 s after the GPU resumed from runtime suspend (16:48:31.65). The three Triton subnormal probes of this folder
  ended at 16:48:33. The last journal entries (16:49:44.58 to 16:49:44.67) are one short login session, which made
  the empty log file of a run of 6 test modules of the add-on (pinned clone) at 16:49:44.64. The Python process of
  the Oops started after that session (its process number is higher), so the module run had started about 8 s
  before the Oops. The other session on the computer used no GPU at that time, so the trigger was very probably
  that module run; the cause is not proven. The computer now runs kernel 7.0.0-42.
- 2026-11-07, 19:24 to 19:38: the same 6 modules ran one at a time, with a short time limit, without a problem of
  the computer. They gave the known failures of the table in "The checks" (pinned clone: 20 in
  `test_cupy_array_path.py`, 2 in `test_subnormal_policy.py`, 5 in the Triton weld contract; with both patches: 4 in
  the Metal weld contract, 5 in the Triton weld contract). The tests, the benches and the example checks of this
  folder followed.

## Open items

- Most checks and the first logged bench used a work copy with patches 0001 and 0002 (`ADDON_GPU_CLONE`). Since
  2026-11-07 22:26 the pinned clone on the GPU computer carries the three patches (`tools/env/addon_gpu.sh`); the
  tests of this folder and the second logged bench ran on it, the example checks have not run again on it.
- 403 of the 409 test modules of the add-on have not run on this GPU ("The checks"). Run them one module at a
  time (see "History").
- Most of the setup that remains is the `init_sim` of the solver (11.6 s in 3-D), which the CPU solver also does.
- Patch 0003 lists one build of the stamp (`1.35.0`, double, Linux x86_64), compared on that one build only; the
  add-on asks for more than one build of a stamp before it lists it.
- The timings are for one 2-D and one 3-D case, against a serial CPU solver.

## Install and run

On the GPU computer, from the repository folder:

1. `bash tools/env/addon_gpu.sh` makes `.local/envs/addon-gpu`, clones the add-on at the pinned commit into
   `.local/upstream/erised-fork`, and applies each patch of `tools/env/addon_patches/` that applies cleanly.
2. Run every command through the GPU lock: `bash tools/gpu_run.sh addon_gpu <command>`. It uses the environment and
   the run-time variables of `tools/env/addon_gpu_activate.sh`.
3. To use a work copy with the patches instead of the pinned clone, set `ADDON_GPU_CLONE=<absolute path of the work
   copy>` before the command.

```bash
bash tools/remote.sh push
bash tools/remote.sh sh <<'EOF'
bash tools/gpu_run.sh addon_gpu python -m pytest -q
bash tools/gpu_run.sh addon_gpu python run_case.py --case slab2d --leg gpu --table cuda --until 20
bash tools/gpu_run.sh addon_gpu python bench/bench_addon.py --small
EOF
```

In a script, a GPU run is `run_on_gpu(sim, until=T)` of the add-on, with `MEEP_GPU_HARDWARE=any` in the
environment (and `MEEP_GPU_KERNEL_TABLE=cuda` or `triton` to select one table). Read `result.driver.gpu`,
`result.driver.active_step_path` and `result.driver.fast_path_report()` after the run.

## Files

- `addon_cases.py`: the three cases (slab waveguide, ring resonator, sphere), with textbook values.
- `run_case.py`: one case on one leg in one process (`gpu` with `--table triton`, `cuda` or `auto`; `array`;
  `reference`; `cpu`, the CPU solver). It prints one `ADDON_JSON` line and saves the field, the fluxes and the
  point series.
- `compare_legs.py`: the relative L2 difference and the differing float32 words of two saved legs.
- `subnormal_gate.py`: runs the subnormal gate of the add-on with the keep cache folder and the hipRTC options.
- `triton_subnormal_probe.py`: keep or flush, bit for bit, for the Triton op classes of the kernels.
- `keep_key_probe.py`: is the keep token of patch 0002 in the CuPy cache key, before and after the keep install.
- `profile_setup.py`: where the setup before the first step goes (the lift under the Python profiler).
- `tests/`: the pytest tests (GPU computer only).
- `bench/bench_addon.py`: the bench, one JSON line per leg (the CPU solver with 1 and 4 OpenMP threads, both
  kernel tables, the array path).

## License

GPL-2.0-or-later (`LICENSE`, the text copied from the add-on). The code of this folder drives GPL code of the
add-on, so it has the same license. The patches in `tools/env/addon_patches/` are also GPL-2.0-or-later (see the
README there).
