#!/usr/bin/env bash
# Make the environments of this repository on the AMD GPU computer. Run: bash tools/remote.sh setup-env
# Each script under tools/env/ makes one environment, uses no sudo, deletes nothing and can run again.
#   tools/env/rocm_jax.sh     .venv-rocm: JAX 0.11.1 on ROCm 10.1.0 (gfx1100), meow, the JAX packages   (default)
#   tools/env/optics_rocm.sh  .venv-optics: PyTorch for ROCm 7.2, Optiland 0.6.3, optics_gpu
#   tools/env/addon_gpu.sh    NOT run here (run it by itself): it clones and runs a third-party GitHub add-on (the
#                             GPU time stepping of the add-on route of FDTD, GPL; approved by the user on
#                             2026-10-30) and makes .local/envs/addon-gpu.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "== tools/env/rocm_jax.sh"
bash tools/env/rocm_jax.sh
echo "== tools/env/optics_rocm.sh"
bash tools/env/optics_rocm.sh
