#!/usr/bin/env bash
# Make the environments of this repository on the AMD GPU computer. Run: bash tools/remote.sh setup-env
# Each script under tools/env/ makes one environment, uses no sudo, deletes nothing and can run again.
#   tools/env/rocm_jax.sh   .venv-rocm: JAX 0.11.1 on ROCm 10.1.0 (gfx1100), meow, meow_gpu   (default)
#   tools/env/fdtd_gpu.sh   NOT run here: it clones and runs a third-party GitHub add-on, which waits for the
#                           approval of the user (STATUS.md, "Decisions for the user").
set -euo pipefail
cd "$(dirname "$0")/.."
echo "== tools/env/rocm_jax.sh"
bash tools/env/rocm_jax.sh
