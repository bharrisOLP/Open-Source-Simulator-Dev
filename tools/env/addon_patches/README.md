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

**Rule 2 of the add-on (certification ledgers).** Both files are pinned by the certification ledgers of the add-on,
so the patches put entries of those ledgers in drift. On the AMD GPU no identity is certified in any case: the
kernels run only with `MEEP_GPU_HARDWARE=any`, and each run reports `certified: False`. `addon_gpu/README.md` gives
the checks that replace the certification on this GPU, and the drift counts of the three weld contracts of the
add-on, before and after the patches.
