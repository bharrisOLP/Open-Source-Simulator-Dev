# ROCm patches of the Erised add-on (GPL-2.0-or-later)

These patches change the GPU time-stepping add-on of Erised (`meep_gpu`, version 1.0.0, pinned commit `8993426`,
https://github.com/Erised-AI-Inc/meep-gpu) so that its GPU path runs correctly on an AMD GPU through CuPy for ROCm.
Each file is a `git diff` of the pinned clone.

**License.** The add-on is GPL-2.0-or-later, and so are these patches (SPDX-License-Identifier:
GPL-2.0-or-later). A patch file cannot carry a license line, so this file and the `LICENSE` file beside it (the GPL
version 2 text, copied from the clone) state it. The add-on is derived from the MIT FDTD package; its `NOTICE` file
gives the copyright holders.

**How they are applied.** `tools/env/addon_gpu.sh` applies each `*.patch` of this folder, in name order, when
`git apply --check` passes on the pinned clone (`.local/upstream/erised-fork` on the GPU computer). A patch that is
already applied fails that check and is not applied again. To try a patch without a change to the pinned clone,
make a work copy (`git clone -q .local/upstream/erised-fork .local/upstream/addon-work-<n>`), apply the patches
there, and put it first on the Python path (`PYTHONPATH=<work copy>`, or `ADDON_GPU_CLONE=<work copy>` for the
tests of `addon_gpu/`).

The patches:

1. `0001-hiprtc-compile-options.patch` (`meep_gpu/backends.py`). The hand-written kernel table (`cuda`) compiles
   each `cupy.RawKernel` with the NVRTC option `--fmad=false`. On a ROCm build, CuPy gives the options to hipRTC
   (clang), which refuses that spelling ("unknown argument"). Thus the warm pass of every hand-written kernel failed,
   and the run took the array path. The patch translates `--fmad=false` to `-ffp-contract=off`, the clang option
   with the same meaning (no fused multiply-add contraction). The translation wraps `_compile_with_cache_hip`, where
   a ROCm CuPy forms the cache key, preprocesses an empty file with the same options, and compiles. It runs only on a
   ROCm build of CuPy.
2. `0002-hip-keep-subnormals.patch` (`meep_gpu/subnormal_policy.py`). Both NVIDIA kernel tables run under the
   float32 subnormal policy `keep`. The add-on attains it in CuPy when it removes `-ftz=true` from each compile. A
   ROCm CuPy adds `-fcuda-flush-denormals-to-zero` instead, so under `keep` every CuPy executor still flushed
   subnormal values, without a message (the gate of the add-on: 23 of the 29 CuPy cells flushed; the other 6 are a
   negation and a select, which keep subnormal values under both policies on this GPU). The patch removes the clang
   spellings too. After the patch, the gate passes under `keep` on the host and CuPy legs (37 of 37 cells).

   The patch also folds the keep token into the cache key of a ROCm CuPy (`_compile_with_cache_hip`); the add-on
   wrapped only `_compile_with_cache_cuda`, which a ROCm CuPy does not call. On CuPy 14 this token is the only
   barrier against a stale binary: CuPy 14 reads `CUPY_CACHE_DIR` once, when `cupy.cuda.compiler` is imported, so
   the keep cache folder that the add-on names later is not the folder that CuPy uses. Without the token, a keep
   compile reads back a binary that a flush compile wrote under the same key. The wrapper finds the parameter
   `extra_source` by name (`inspect.signature`), because its position has changed between CuPy versions; when the
   parameter is not there, the install of `keep` reports that the token could not be folded in.
   `tools/env/addon_gpu.sh` pins `cupy-rocm-7-0==14.2.0`, and `addon_gpu/tests/test_patches.py` checks that the
   token reaches the key of a keep compile and not the key of a compile before the install.

3. `0003-bulk-chi1inv-read-double.patch` (`meep_gpu/from_meep.py`). The lift reads the inverse permittivity
   (`chi1inv`) of the solver. On a build listed in `BULK_CHI1INV_BUILDS` it copies the stored arrays of the solver in
   bulk (and reads only the points that no chunk owns one at a time); on any other build it calls `get_chi1inv` once
   for each point and each tensor entry. Only single-precision 1.33.0 was listed, so the conda-forge build here
   (1.35.0, double precision, Linux x86_64) read every point one at a time: 27.7 s of a 29.6 s lift of the 2-D bench
   case (13,824,009 calls) and 37.3 s of a 49.4 s lift of the 3-D bench case, under the profiler
   (`addon_gpu/profile_setup.py`). The bulk read already handles a double-precision build (it reads `c_double` words
   when the build is not single precision). The patch adds the stamp `("1.35.0", "double", "Linux", "x86_64")` to the
   list, after the comparisons that the add-on asks for, on this one build: the bulk-path tests of
   `meep_gpu/test_from_meep_bulk_chi1inv.py` (6 cells: the read words equal `get_chi1inv` as uint64 at every point,
   and the installed arrays equal those of the per-point read), and the final fields of the bench cases and the test
   cases after a run, bulk read against per-point read (0 differing words; `addon_gpu/README.md`). The self-check of
   the bulk read stays active: a disagreement restarts the read one point at a time, with a warning.

4. `0004-hip-nan-subtrahend-deposit.patch` (`meep_gpu/host_writes.py`, `meep_gpu/sources.py`). The source deposit
   forms `real(A) = amp_re * c_re - amp_im * c_im` in float64 (`sources._deposit`), and its door
   (`host_writes._rounded_difference`) widens each stored float32 word to float64, subtracts `real(A)` once and
   rounds once: on the host with NumPy, on the GPU with CuPy. IEEE 754 leaves the sign of a NaN result open, and the
   AMD GPU forms `a - b` as `a + (-b)`: a NaN `b` comes back with its sign bit inverted, where NumPy on the x86 host
   returns `b` itself (`addon_gpu/nan_words_probe.py`, `benchmarks/2026-11-08/addon-nan-probe.out`: `1.5 - b` for
   four NaN words of both signs; `+`, `*`, `b - 1.5`, the casts and `inf * 0` give equal words). So both
   subtractions of the deposit can store a NaN word of the other sign. Without the patch, the amplitude `inf + 0j`
   (the non-finite table of `meep_gpu/test_source_deposit.py`) stored `0x7fc00000` on the GPU and `0xffc00000` on
   the host: 400 differing words in 76 of 84 cases, all NaN words (`addon-nan-probe.out`). With the first form of
   this patch (the door only): `inf + 0j` 0 words, `1 + inf j` 280 words in 70 of 84 cases, `1 + nan j` 336 words in
   84 of 84 cases (`benchmarks/2026-11-08/addon-nan-probe-0004v1.out`; the review of the change found these counts).
   With this patch: 0 words in all three tables (`benchmarks/2026-11-08/addon-nan-probe-0004m.out`).

   On a ROCm build of CuPy, `host_writes._subtract_once` subtracts with one elementwise kernel that takes a NaN
   operand as it is, in the host's order (a NaN minuend first, then a NaN subtrahend), and forms every other word by
   the one subtraction. The door uses it, and `sources._deposit` uses it for `real(A)` through
   `host_writes.nan_ordered_subtract`. A subtrahend array of another dtype is widened to float64 first, as the
   in-place subtraction does. A signalling NaN comes back as it is (x86 sets its quiet bit); the rounding to float32
   in the door quiets it on both sides. Every other build, and a minuend that is not a float64 array, keeps the
   in-place subtraction. The patch changes only the source deposit: the other subtractions on the GPU (the array
   path, the kernels, the `D - P` update of `deposit_repair.py` at its repair cells) keep the NaN sign of the GPU,
   and no test of the add-on compares NaN words there. `addon_gpu/tests/test_patches.py` checks the door on 81 pairs
   of a stored word and a subtrahend, a float32 and an int64 subtrahend, and the whole deposit on the 84 cases for
   each of the three amplitude tables; a control shows the sign inversion of the plain device subtraction.

**No patch 0005 (the `D - P` of the deposit repair; decision of 2026-11-08).** `deposit_repair._fresh_at` forms the
float32 `D - P` at the repair cells only, with the same in-place subtraction as the whole-volume form of `update_E`
(`Fields.displacement_minus_polarization`): the repair recomputes the field at the deposit cells as `update_E` of the
same device would, and the add-on states that the subset is bit for bit the subset of the whole. On the GPU it keeps
the NaN sign of the GPU, like the array path and the kernels. A patch that gave the repair the host's words would
break that equality on the GPU, and a patch for every float32 subtraction of the array path and the kernels is a
change of the kernels. No test of the add-on compares NaN words there: of the 409 test modules, 79 name
`deposit_repair` or `_fresh_at`, and 31 write a NaN value (`np.nan`, `float("nan")`, a NaN word, ...); the one
module in both names the repair in a docstring and gives a NaN coefficient to a check that must refuse it. In the
two runs of the whole suite no failure came from the repair (`addon_gpu/README.md`). A NaN in D or P is a field
that has already diverged; the source deposit of patch 0004
makes a NaN from a non-finite amplitude in a finite field, and its test compares those words with the host's.
`addon_gpu/nan_words_probe.py` part 3 measures it on the GPU (`benchmarks/2026-11-08/addon-nan-probe-repair.out`):
for 49 pairs of planted words in D and P (finite, NaN of both signs, a NaN payload, infinities; 35 NaN results), the
repair gives the words of the whole-volume form on the host and on the GPU (0 differing words each), and the GPU
differs from the host in 12 words: where P is a NaN and D is not, in the sign bit only.
`addon_gpu/tests/test_patches.py` checks these facts, so a later patch of the repair alone fails it.

**Rule 2 of the add-on (certification ledgers).** The five files are pinned by the certification ledgers of the add-on,
so the patches put entries of those ledgers in drift. On the AMD GPU no identity is certified in any case: the
kernels run only with `MEEP_GPU_HARDWARE=any`, and each run reports `certified: False`. `addon_gpu/README.md` gives
the checks that replace the certification on this GPU, and the drift counts of the three weld contracts of the
add-on, before and after the patches.
