#!/usr/bin/env bash
# Make the GPU environment of this repository on the AMD GPU computer: a uv virtual environment .venv-rocm with
# the ROCm 10.1.0 pip wheels (gfx1100), JAX 0.11.1 with the AMD ROCm plugin, meow and the meow_gpu package.
# Run: bash tools/remote.sh setup-env (tools/setup_env.sh runs this file).
# No sudo. It deletes nothing. It is safe to run again: uv installs only what is missing or different.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PATH=$HOME/.local/bin:$PATH PYTHONDONTWRITEBYTECODE=1

ROCM_VERSION=10.1.0
JAX_VERSION=0.11.1
GFX=gfx1100
AMD_INDEX=https://stable.repo.amd.com/rocm/whl-next/
VENV=$PWD/.venv-rocm
PY=$VENV/bin/python

[[ -x $PY ]] || uv venv --quiet --python 3.12 "$VENV"
uv pip install --quiet --python "$PY" --index-url "$AMD_INDEX" \
  "rocm[libraries,device-$GFX]==$ROCM_VERSION" \
  "jax_rocm10_plugin==$JAX_VERSION+rocm$ROCM_VERSION" "jax_rocm10_pjrt==$JAX_VERSION+rocm$ROCM_VERSION"
# The jax pins stay in each install, so that the resolver keeps the version that AMD validated for gfx1100.
uv pip install --quiet --python "$PY" "jax==$JAX_VERSION" "jaxlib==$JAX_VERSION" \
  "meow-sim==0.15.0" tidy3d pytest
uv pip install --quiet --python "$PY" --no-deps -e ./meow_gpu -e ./bands_gpu -e ./thermal_gpu -e ./charge_gpu
mkdir -p .local
uv pip freeze --python "$PY" > .local/env-rocm-freeze.txt

ROCR_VISIBLE_DEVICES=0 JAX_ENABLE_X64=1 "$PY" - <<'PY'
import jax, jax.numpy as jnp
devs = jax.devices()
print("jax", jax.__version__, devs)
assert devs[0].platform == "gpu", "JAX did not find the GPU"
x = jnp.ones((2048, 2048), jnp.complex128)
print("complex128 matmul ok:", float(jnp.abs((x @ x)[0, 0])))
PY
