# addon_gpu: the add-on route of FDTD on the AMD GPU

This folder runs the GPU time stepping of the Erised add-on (`meep_gpu` 1.0.0, GPL-2.0-or-later,
https://github.com/Erised-AI-Inc/meep-gpu, pinned at commit `8993426`) on an AMD Radeon RX 7900 XT (gfx1100), and
measures it. The MIT FDTD package (version 1.35.0 from conda-forge, double precision, serial build) builds each
simulation, and the add-on time-steps it on the GPU in float32. The add-on supports NVIDIA GPUs (CuPy, with Triton
kernels or hand-written CUDA kernels) and Apple GPUs (Metal). This folder adds the AMD GPU: CuPy for ROCm (hipRTC
compiles the hand-written kernels) and the AMD backend of Triton.

**Status (2026-11-08): version 0.1 works.** Both kernel tables of the add-on run on the AMD GPU, with four small
patches (`tools/env/addon_patches/`). All 409 test modules of the add-on ran on this GPU, one at a time: with the
four patches 379 are clean, and each failure of the other 30 is sorted ("The whole test suite of the add-on"); none
needs a change of a kernel (two of them pass since the runner sets the CuPy cache folder per module). On three cases
the kernels give the same float32 words as the array path of the add-on, and they agree with the CPU solver to
8.3e-6 or better (relative L2). On a 3-D case of 2,097,152 cells the time loop is 16 times (Triton) to 27 times
(hand-written kernels) as fast as the serial CPU solver (float32 on the GPU, float64 on the CPU). Patch 0003 lets
the lift read the permittivity in bulk on this solver build, so a whole run (setup and time loop) is now faster on
the GPU route in both bench cases: 2.19 to 2.34 times in 2-D (1,536,000 cells, 2,400 steps) and 3.12 to 3.37 times
in 3-D (2,097,152 cells, 1,280 steps), with the kernel tables. Before patch 0003 the setup dominated, and the whole
2-D run was 3.4 to 3.7 times slower than on the CPU solver. Small runs are still slower on the GPU ("Timings";
measured with patches 0001 to 0003).

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

The tests of this folder: 53 of 53 pass (`tests/test_patches.py`, `tests/test_parity.py`,
`tests/test_suite_runner.py`; 84 s, ended 2026-11-08 15:25 EST, on the pinned clone with the four patches, without
`ADDON_GPU_CLONE`).

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
1 to 1 of 80, CUDA no change. With patch 0003 too, the counts are the same (2026-11-07). With patch 0004 too:
Metal 9 of 410 paths and 9 of 318 code digests, Triton raw digests 96 of 577, code digests 38 of 432, device digests 1
of 80, CUDA no change (2026-11-08; "The whole test suite of the add-on"). On the AMD GPU this
changes no dispatch decision (no AMD identity is certified, and
the kernels dispatched with both patches on every case). On an Apple or NVIDIA host the patched files would put
those certified entries in drift; the patches are for the AMD route only.

### The whole test suite of the add-on (2026-11-08)

The pytest paths of the add-on (`testpaths` in its `pyproject.toml`: `meep_gpu` and `parity/meep_gpu`) hold 409
test modules (`test_*.py`: 225 in `meep_gpu/`, 66 in `meep_gpu/cuda_kernels/`, 113 in `parity/meep_gpu/`, 5 in
`parity/meep_gpu/rounds/1.0/`; `tests/test_suite_runner.py` checks the count). `suite_runner.py` runs them one at a
time on the pinned clone, with `MEEP_GPU_HARDWARE=any` (the kernel tables of the add-on admit this GPU, uncertified)
and a time limit of 300 s for each module. It sorts the modules into three tiers, in this order: `host` (148
modules that do not name CuPy, Triton, PyTorch or kernel dispatch), `gpu` (234 modules that do) and `certification`
(27 modules: the ledger, weld and record contracts of the add-on, and the modules with a test in its
`tools/ci/pending_certification.txt`). All 409 modules ran twice, one row per module in each file:

- `benchmarks/2026-11-08/addon-suite-any.jsonl` (run `any-2026-11-08`), patches 0001 to 0003. The rows started
  between 02:54:28 and 08:00:36 EST, with no row between 03:35:49 and 07:58:39 (a pause for another session). The
  rows do not record the CuPy settings. `CUPY_CACHE_DIR` was `.local/cupy-cache` for every row (it is set by
  `tools/env/addon_gpu_activate.sh`, which `tools/gpu_run.sh` sources). The 33 rows from 07:58:39 on ran from the
  script `.local/addon-runs/suite/final-2026-11-08.sh` on the GPU computer, which exports
  `CUPY_GPU_MEMORY_LIMIT=8589934592`; for the earlier rows the memory limit is not logged.
- `benchmarks/2026-11-08/addon-suite-any-0004m.jsonl` (run `any-0004m-2026-11-08`), the four patches. The rows
  started between 09:05:53 and 09:29:32 EST, and each row records `CUPY_GPU_MEMORY_LIMIT=8589934592` and
  `CUPY_CACHE_DIR=.local/cupy-cache`.

The computer had no problem: in each file the uptime of the rows never decreases (32,048 s to 50,416 s, and
54,333 s to 55,752 s), and no module reached the time limit. The counts of the rows:

| Patches | Tier | Modules | Clean | Not clean | Passed | Failed | Skipped | Errors |
|---|---|---|---|---|---|---|---|---|
| 0001 to 0003 | `host` | 148 | 148 | 0 | 4,479 | 0 | 268 | 0 |
| 0001 to 0003 | `gpu` | 234 | 227 | 7 | 11,081 | 232 | 817 | 97 |
| 0001 to 0003 | `certification` | 27 | 4 | 23 | 2,269 | 41 | 28 | 0 |
| 0001 to 0003 | all | 409 | 379 | 30 | 17,829 | 273 | 1,113 | 97 |
| four | `host` | 148 | 147 | 1 | 4,478 | 1 | 268 | 0 |
| four | `gpu` | 234 | 228 | 6 | 11,082 | 231 | 817 | 97 |
| four | `certification` | 27 | 4 | 23 | 2,269 | 41 | 28 | 0 |
| four | all | 409 | 379 | 30 | 17,829 | 273 | 1,113 | 97 |

407 of the 409 modules give the same counts and exit code in both files. Patch 0004 makes
`meep_gpu/test_source_deposit.py` pass (68 passed and 1 failed, now 69 passed), and it puts one weld of
`meep_gpu/test_triton_cylindrical_composition.py` in drift (6 passed, now 5 passed and 1 failed: "sources.py drifted
from the bytes this gate executed").

A module is clean when no test failed and no test had an error. Three Metal modules skip at module level by the
declared resource `mps_device` of the add-on (no Apple GPU), so pytest ran no test and exited with code 5; they
count as clean. The counts come from the JUnit file of each module, one outcome for each test case; pytest's own
summary can count one more error where a test also has an error in its teardown (`suite_runner.py`). Whether a
module is clean does not change.

The failures with the four patches (`addon-suite-any-0004m.jsonl`), sorted (the numbers refer to the notes below):

| Class | Module | Failed | Errors |
|---|---|---|---|
| 1. Certification records, listed by the add-on | 22 modules of the `certification` tier | 33 | 0 |
| 2. Certification records, patch 0002 | `meep_gpu/test_metal_weld_contract.py` | 4 | 0 |
| 2. Certification records, patch 0002 | `meep_gpu/test_dispatch_contract.py` | 1 | 0 |
| 2. Certification records, patch 0002 | `meep_gpu/test_triton_complex_conductive_fused_pair.py` | 1 | 0 |
| 3. Certification records, patch 0004 | `meep_gpu/test_triton_cylindrical_composition.py` | 1 | 0 |
| 4. Environment: the solver version | `meep_gpu/test_from_meep.py` | 228 | 0 |
| 5. Environment: the CuPy cache folder | `meep_gpu/test_triton_complex_no_pml_curl.py` | 0 | 83 |
| 5. Environment: the CuPy cache folder | `meep_gpu/test_triton_complex_offdiag_update_e.py` | 0 | 14 |
| 6. Environment: `MEEP_GPU_HARDWARE=any` | `meep_gpu/test_dispatch_contract.py` | 2 | 0 |
| 7. AMD route: Triton 3.8 | `meep_gpu/test_triton_complex_no_pml_conductive.py` | 1 | 0 |
| 7. AMD route: Triton 3.8 | `meep_gpu/test_triton_kernels.py` | 1 | 0 |
| 8. Defect of an add-on test: the order | `meep_gpu/test_dispatch_chunk_division.py` | 1 | 0 |
| 9. AMD GPU: the NaN sign (fixed by patch 0004) | `meep_gpu/test_source_deposit.py` | 0 | 0 |

By class: certification records 40 failed tests (expected); environment 230 failed and 97 errors; AMD route 2
failed; a defect of a test 1 failed. With patches 0001 to 0003 (`addon-suite-any.jsonl`) class 3 has no failure and
class 9 has 1.

1. Each failing test is in `tools/ci/pending_certification.txt` of the add-on.
2. The welds and the dispatch records pin `meep_gpu/subnormal_policy.py`, which patch 0002 changes. The weld of the
   conductive fused pair records the sha256 `1b1e772c...` of the unpatched file.
3. The entry `cylindrical_composition_gate` pins `meep_gpu/sources.py`, which patch 0004 changes.
4. Its case table builds a medium with `D_conductivity_offdiag`, which the solver of this environment (1.35.0)
   refuses (`NotImplementedError` in its `geom.py`); the table is built for every case. The identity reference of the
   add-on is 1.33.0.
5. A module fixture installs `keep` by hand. The add-on then asks for a `CUPY_CACHE_DIR` whose path carries
   `ftz_stripped` (its `docs/guides/kernel-dispatch.md`), and `tools/env/addon_gpu_activate.sh` sets
   `.local/cupy-cache`. With `suite_runner.py --cupy-cache-dir <a folder named ...ftz_stripped>` both modules pass
   (83 and 14 passed, patches 0001 to 0003; `benchmarks/2026-11-08/addon-suite-keepcache.jsonl`). Since these rows,
   the runner gives these two modules the keep folder by default ("The CuPy cache folder of each module"), and both
   pass in the normal run.
6. The two tests expect the refusal of an uncertified Triton version and of an unsupported device; the switch of
   this run admits both. Without the switch both pass (`benchmarks/2026-11-08/addon-suite-default.jsonl`, 08:16 EST,
   patches 0001 to 0003 and the first form of patch 0004: 204 passed; the 2 failures are those of notes 1 and 2).
7. The add-on supports Triton 3.1; the ROCm wheels bring 3.8. One test checks the version ("Triton 3.8.0 is outside
   the validated set ['3.1.0']"). The other blocks every import of `triton` and imports a kernel module again; that
   module takes the real Triton from a sibling module, and `@triton.jit` of Triton 3.8 imports
   `triton.language.core` at that time (`_normalize_ty`), which the block refuses.
8. The test replaces `triton` with a double without `jit` and plans with the real composers; a kernel module that
   is imported for the first time under the double fails, and the planner refuses. The same failure occurs without
   `MEEP_GPU_HARDWARE=any` (`addon-suite-default.jsonl`, 03:28 EST, patches 0001 to 0003). With
   `preimport_triton_kernels.py` the module passed in a development run (not logged; the command is in the docstring
   of that file). It is not specific to this GPU.
9. On this GPU `a - b` with a NaN `b` gives `b` with its sign bit inverted, where NumPy on the x86 host gives `b`
   (`nan_words_probe.py`, `benchmarks/2026-11-08/addon-nan-probe.out`: 4 NaN words of both signs; `+`, `*`,
   `b - 1.5`, the casts and `inf * 0` give equal words). The source deposit subtracts twice: `real(A)` is formed
   as `amp_re * c_re - amp_im * c_im`, and the door subtracts `real(A)` from the stored word. With patches 0001 to
   0003 the amplitude `inf + 0j` of the add-on test stored `0x7fc00000` on the GPU and `0xffc00000` on the host:
   400 words in 76 of 84 cases, all NaN (`addon-nan-probe.out`). The first form of patch 0004 (the door only) left
   `1 + inf j` at 280 words in 70 of 84 cases and `1 + nan j` at 336 words in 84 of 84 cases
   (`addon-nan-probe-0004v1.out`; the review of the change found these counts). The four patches: 0 words in all
   three tables (`addon-nan-probe-0004m.out`), and 69 of 69 tests of the module pass.

No failure needs a change of a kernel. Of the failures that are specific to the AMD route, two come from Triton 3.8
(the only Triton of the ROCm wheels), and one from the NaN sign of the GPU, which patch 0004 fixes in the source
deposit (`tools/env/addon_patches/README.md`). The drift counts of rule 2 (the failure messages of the three weld
contracts): with patches 0001 to 0003, Metal 7 of 410 pinned paths and 7 of 318 code digests, Triton raw digests 91
of 577, code digests 37 of 432 and device digests 1 of 80, CUDA none (`addon-suite-any.jsonl`); with the four
patches, Metal 9 of 410 and 9 of 318, Triton 96 of 577, 38 of 432 and 1 of 80, CUDA none (36 of 36 pass;
`addon-suite-any-0004m.jsonl`). Patch 0004 changes `meep_gpu/host_writes.py` and `meep_gpu/sources.py`. Of the pins
of `meep_gpu/host_writes.py`, only `bit_identity_gate` of the Triton ledger is a raw-tier digest; the
`driver_dispatch` entries pin it in a provenance map (`array_path_oracle.provenance_code_sha256`) that the add-on
reports and does not judge. `meep_gpu/sources.py` is pinned by more entries of the Triton and Metal ledgers.

### The CuPy cache folder of each module (2026-11-08)

The add-on keeps one CuPy cache folder for each float32 subnormal policy (`meep_gpu/subnormal_policy.py`,
`cupy_cache_reasons`): an install of `keep` refuses a `CUPY_CACHE_DIR` whose path does not carry `ftz_stripped`, and
an install of `flush` refuses one that does. Its dispatch installs `keep` itself, derives the keep folder
`<CUPY_CACHE_DIR>-ftz_stripped` (`keep_policy_cache_dir`), and mints it with a marker file beside it; it refuses a
keep folder that holds files and has no marker. The two modules of note 5 install `keep` by hand and set no folder.

`suite_runner.py` now sets the folder per module (`--cache-policy module`, the default): the keep folder that the
dispatch derives (`.local/cupy-cache-ftz_stripped`) for a module whose source installs `keep` by hand, names no
`CUPY_CACHE_DIR` and installs no `flush` (the two modules of note 5, and no other module at the pinned commit), and
the folder of `tools/env/addon_gpu_activate.sh` (`.local/cupy-cache`) for all other modules. If the marker is
missing, the add-on mints the folder first; if the add-on refuses the folder, the module runs with
`.local/cupy-cache`, and its row gives the reason. The alternative, one keep folder for every module (set by the
activate script), was measured too. Both runs cover the 60 modules whose source names the cache folder or the
policy (`--cache-or-policy`), with the four patches, one at a time. Each row is compared with the row of the same
module in `addon-suite-any-0004m.jsonl` ("same": the same counts, exit code and failed tests):

| Cache folders | File (`benchmarks/2026-11-08/`) | Same rows | The two modules of note 5 | Other changes |
|---|---|---|---|---|
| per module (the default) | `addon-suite-cache-module.jsonl` | 58 of 60 | 83 and 14 passed | none |
| the keep folder for all | `addon-suite-cache-keep-all.jsonl` | 57 of 60 | 83 and 14 passed | 3 failed |

With the keep folder for all modules, `meep_gpu/test_triton_no_pml_constitutive.py` gives 169 passed and 3 failed
(was 172 passed): in one test the add-on refuses an install of `flush` ("CUPY_CACHE_DIR ... carries the keep-policy
token 'ftz_stripped' but 'flush' was requested"), and two more tests of the module fail an assertion. So the folder is
set per module, and the activate script keeps `.local/cupy-cache`. Commands (on the GPU computer, from the
repository folder; the rows started 14:36:44 to 14:38:50 and 14:41:35 to 14:43:39 EST, uptime increasing in each
file, no time-out):

```bash
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUPY_GPU_MEMORY_LIMIT=8589934592
uv run --no-project --python 3.12 addon_gpu/suite_runner.py --run cache-module-w10b --cache-or-policy \
  --chunk 60 --chunk-seconds 280
uv run --no-project --python 3.12 addon_gpu/suite_runner.py --run cache-keep-all-w10b --cache-or-policy \
  --chunk 60 --chunk-seconds 280 --cupy-cache-dir $PWD/.local/cupy-cache-ftz_stripped
```

The time limit of each module now uses GNU timeout (`gnutimeout` on the GPU computer, as `tools/gpu_run.sh`): at
the limit it sends SIGTERM, and 30 s later SIGKILL, to the process group of the module, so a process that the module
started also ends. GNU timeout is in that group, so the SIGKILL also ends it (return code -9). A SIGKILL of the
module process from another source before the limit (the OOM killer) also gives -9 (or 137), so a row is a time-out
only when the return code is 124, 137 or -9 and the call took at least the limit. `tests/test_suite_runner.py` checks
this with a command and a second process that both ignore SIGTERM (both end, return code -9, a time-out), and with a
command that ends by SIGKILL before the limit (return code -9, not a time-out). Each row of the two files names
`gnutimeout`; no row of the two files is a time-out.

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
- 2026-11-08: all 409 test modules of the add-on ran one at a time (`suite_runner.py`; "The whole test suite of
  the add-on"; `benchmarks/2026-11-08/addon-suite-any.jsonl`). The rows started between 02:54:28 and 08:00:36 EST,
  with no row between 03:35:49 and 07:58:39 (a pause for another session). The computer had no problem: the uptime
  of the rows increases from 32,048 s to 50,416 s.
- 2026-11-08, 09:05:53 to 09:29:32 EST: all 409 modules again, with the four patches
  (`benchmarks/2026-11-08/addon-suite-any-0004m.jsonl`); the uptime of the rows increases from 54,333 s to 55,752 s.
- 2026-11-08, 14:36:44 to 14:43:39 EST: the 60 modules that name the CuPy cache folder or the subnormal policy, twice
  ("The CuPy cache folder of each module"), one at a time, GNU timeout; the uptime of the rows increases from
  74,183 s to 74,598 s. The whole suite did not run again.

## Open items

- Most checks and the first logged bench used a work copy with patches 0001 and 0002 (`ADDON_GPU_CLONE`). Since
  2026-11-07 22:26 the pinned clone on the GPU computer carries the three patches (`tools/env/addon_gpu.sh`), and
  since 2026-11-08 08:17 also patch 0004; the tests of this folder and the second logged bench ran on it, the example
  checks have not run again on it.
- Patch 0004 makes the words of the source deposit equal those of the host for the three amplitude tables of its
  tests. The other subtractions on the GPU (the array path, the kernels, the `D - P` update of `deposit_repair.py` at
  its repair cells) keep the NaN sign of the GPU. There is no patch 0005 for `deposit_repair.py`: the repair must
  give the words of `update_E` on the same device, and it does, NaN words included (`nan_words_probe.py` part 3;
  `tools/env/addon_patches/README.md`, "No patch 0005").
- The CuPy cache folder is set per module by `suite_runner.py` ("The CuPy cache folder of each module"). A script
  that installs `keep` by hand outside the runner must set a `CUPY_CACHE_DIR` that carries `ftz_stripped` itself.
- `subnormal_policy._clear_triton_jit_caches` of the add-on reads `JITFunction.cache`, which Triton 3.8 does not
  have, so on this GPU a policy install drops no compiled Triton kernel of the process. No test of the suite
  failed by it.
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

The test modules of the add-on run outside the lock, through `suite_runner.py` (each chunk of modules takes the
lock; `touch .local/addon-runs/suite/STOP` stops a run after the current module):

```bash
bash tools/remote.sh sh <<'EOF'
uv run --no-project --python 3.12 addon_gpu/suite_runner.py --run <name> --resume --chunk 200 --chunk-seconds 300
uv run --no-project --python 3.12 addon_gpu/suite_runner.py --run <name> --summary
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
- `suite_runner.py`: runs the test modules of the add-on one at a time, each with its own time limit (GNU timeout)
  and its own CuPy cache folder, in chunks of one hold of the GPU lock, and logs one JSON line per module
  (`--summary`: the totals by tier; a stop file stops a run after the current module).
- `nan_words_probe.py`: the NaN words of the GPU against those of the host (the failure of
  `meep_gpu/test_source_deposit.py`; part 3: the `D - P` of the deposit repair).
- `preimport_triton_kernels.py`: a pytest plugin that imports the Triton kernel modules of the add-on before the
  tests (the order dependence of `meep_gpu/test_dispatch_chunk_division.py`).
- `tests/`: the pytest tests (GPU computer only).
- `bench/bench_addon.py`: the bench, one JSON line per leg (the CPU solver with 1 and 4 OpenMP threads, both
  kernel tables, the array path).

## License

GPL-2.0-or-later (`LICENSE`, the text copied from the add-on). The code of this folder drives GPL code of the
add-on, so it has the same license. The patches in `tools/env/addon_patches/` are also GPL-2.0-or-later (see the
README there).
