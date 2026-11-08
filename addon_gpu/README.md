# addon_gpu: the add-on route of FDTD on the AMD GPU

This folder runs the GPU time stepping of the Erised add-on (`meep_gpu` 1.0.0, GPL-2.0-or-later,
https://github.com/Erised-AI-Inc/meep-gpu, pinned at commit `8993426`) on an AMD Radeon RX 7900 XT (gfx1100), and
measures it. The MIT FDTD package (version 1.35.0 from conda-forge, double precision, serial build) builds each
simulation, and the add-on time-steps it on the GPU in float32. The add-on supports NVIDIA GPUs (CuPy, with Triton
kernels or hand-written CUDA kernels) and Apple GPUs (Metal). This folder adds the AMD GPU: CuPy for ROCm (hipRTC
compiles the hand-written kernels) and the AMD backend of Triton.

**Status (2026-11-07): version 0.1 works.** Both kernel tables of the add-on run on the AMD GPU, with two small
patches (`tools/env/addon_patches/`). On three cases the kernels give the same float32 words as the array path of
the add-on, and they agree with the CPU solver to 8.3e-6 or better (relative L2). On a 3-D case of 2,097,152 cells
the time stepping is 16 times (Triton) to 28 times (hand-written kernels) as fast as the serial CPU solver; the
setup before the first step is longer than on the CPU solver ("Timings").

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
   f + i decay of each mode to 1e-4 of f.
4. **The CuPy array path against the NumPy array path, word for word after every sub-step** (the module
   `meep_gpu/test_cupy_array_path.py` of the add-on, 20 cases, with planted subnormal and signed-zero words). Pinned
   clone: 20 of 20 fail (for example 984 words differ after the first `step_B`). With both patches: 20 of 20 pass.
5. **The subnormal gate of the add-on**, `keep`, host and CuPy legs, each cell against its exact keep word. Without
   patch 0002: FAIL, 23 of the 29 CuPy cells flushed. With patch 0002: PASS, 37 of 37 cells (8 host, 29 CuPy).
6. `triton_subnormal_probe.py` under no policy, `keep` and `flush`: keep in 6 of 6 op classes under each of them.
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
9. The source of the 145 hand-written kernel modules contains no warp intrinsic (`__shfl`, `warpSize`, `__ballot`,
   `__syncwarp`, `__activemask`, `__ldg`), so the kernels do not assume a wave size of 32 or 64.

The tests of this folder: 19 of 19 pass (`tests/test_patches.py`, `tests/test_parity.py`; 72 s, 2026-11-07, with
`ADDON_GPU_CLONE` set to a work copy with both patches).

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
1 to 1 of 80, CUDA no change. On the AMD GPU this changes no dispatch decision (no AMD identity is certified, and
the kernels dispatched with both patches on every case). On an Apple or NVIDIA host the patched files would put
those certified entries in drift; the patches are for the AMD route only.

The rest of the suite of the add-on (about 250 test modules) has not run on this GPU.

## Timings

`bench/bench_addon.py`, 2026-11-07, one RX 7900 XT, both patches, the computer shared with CPU jobs of another
session (`OMP_NUM_THREADS=4`). Time of the step loop (after a warm-up run in the same process, so kernel
compilation is not timed); CPU solver serial (1 and 4 OpenMP threads gave the same time):

| Case | Cells, steps | CPU solver | Array path | Triton table | Hand-written table |
|---|---|---|---|---|---|
| 2-D slab waveguide | 1,536,000, 2,400 | 5.11 s | 2.78 s (1.8x) | 1.03 s (5.0x) | 0.82 s (6.2x) |
| 3-D dielectric sphere | 2,097,152, 1,280 | 37.7 s | 5.34 s (7.1x) | 2.35 s (16x) | 1.34 s (28x) |

- Rate: 2-D 721 (CPU) against 3,580 (Triton) and 4,489 (hand-written) million cell updates per second; 3-D 71 against
  1,141 and 1,999. Between progress callbacks (from step 100 on), both tables step the 3-D case at about 2,650.
- Setup before the first step: 2-D 21 to 23 s on the GPU route against 1.3 s for the CPU solver; 3-D 39 to 42 s
  against 11.5 s. The lift of the add-on reads the permittivity of a solver build that it has not compared ("1.35.0
  double precision ... not a build the bulk read was compared on") one point at a time: its log gives about 0.1 s
  for 24,000 points of each component. For the 3-D case, the whole run (setup and steps) takes 41.6 s (Triton) and
  43.0 s (hand-written) on the GPU route against 49.2 s on the CPU solver; the gain grows with the step count.
- Small cases are slower on the GPU: the 2-D slab of 24,000 cells steps at 54 to 65 million cell updates per second
  on the GPU against 211 on the CPU solver (bench `--small`).

## History

- 2026-10-30, 16:49: the GPU computer stopped about one minute after the start of a run of 6 test modules of the
  add-on (pinned clone; kernel 7.0.0-38). Its kernel dump shows an Oops in the amdgpu/TTM memory manager (list
  corruption, then a NULL pointer in `ttm_resource_move_to_lru_tail`) during a GPU memory allocation of a Python
  process, about 80 s after the GPU resumed from runtime suspend. The cause is not proven. On kernel 7.0.0-42, from
  2026-11-07 19:24, each of the 6 modules ran alone without a problem, followed by the tests, the benches and the
  example check of this folder.

## Open items

- The pinned clone on the GPU computer has no patch yet. The runs above used a work copy with both patches
  (`ADDON_GPU_CLONE`). After `tools/env/addon_gpu.sh` applies the patches, run the tests without that variable.
- The rest of the suite of the add-on (about 250 test modules) has not run on this GPU. Run it one module at a
  time (see "History").
- The setup time: the lift reads the permittivity one point at a time for this solver build. A bulk read that the
  add-on has compared for this build would shorten it.
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
- `tests/`: the pytest tests (GPU computer only).
- `bench/bench_addon.py`: the bench, one JSON line per leg (the CPU solver with 1 and 4 OpenMP threads, both
  kernel tables, the array path).

## License

GPL-2.0-or-later (`LICENSE`, the text copied from the add-on). The code of this folder drives GPL code of the
add-on, so it has the same license. The patches in `tools/env/addon_patches/` are also GPL-2.0-or-later (see the
README there).
