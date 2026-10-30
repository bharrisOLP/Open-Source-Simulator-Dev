#!/usr/bin/env bash
# WAITS FOR THE APPROVAL OF THE USER: this script clones and runs third-party code from GitHub (the Erised
# add-on). The auto-mode classifier blocked it on 2026-10-30. Do not run it before the user approves.
# Environment .local/envs/fdtd-gpu on the AMD GPU computer (Radeon RX 7900 XT, gfx1100), without a system ROCm:
#   - Meep (conda-forge pymeep, serial "nompi" build, double precision) and Python 3.12, with micromamba;
#   - PyTorch for ROCm 7.2 (its wheel brings the HIP runtime and the AMD build of Triton);
#   - CuPy for ROCm 7 (cupy-rocm-7-0); it needs versioned ROCm 7 library names (links to the PyTorch libraries in
#     .local/rocm7-compat) and hipcc for its built-in kernels (from the ROCm 10.1 pip SDK, .venv-hipcc);
#   - the GPU time-stepping add-on of Erised (GPL-2.0-or-later), cloned at a fixed commit into .local/upstream,
#     installed editable without dependencies, with the ROCm patches of fdtd_gpu/patches/ applied.
# tools/env/fdtd_gpu_activate.sh sets the run-time variables. No sudo; it deletes nothing; it can run again.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PATH=$HOME/.local/bin:$PATH PYTHONDONTWRITEBYTECODE=1
ENV=$PWD/.local/envs/fdtd-gpu
PY=$ENV/bin/python
ADDON_REPO=https://github.com/Erised-AI-Inc/meep-gpu.git
ADDON_COMMIT=8993426
TORCH_INDEX=https://download.pytorch.org/whl/rocm7.2
AMD_INDEX=https://stable.repo.amd.com/rocm/whl-next/
export MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX:-$HOME/micromamba}

mkdir -p .local/envs .local/upstream
if [[ ! -x $PY ]]; then
  micromamba create -y -q -p "$ENV" -c conda-forge "python=3.12" "pymeep=1.35.0=nompi_*" "numpy>=2" >/dev/null
fi
uv pip install --quiet --python "$PY" --index-url "$TORCH_INDEX" torch
uv pip install --quiet --python "$PY" cupy-rocm-7-0 pytest

# hipcc from the ROCm pip SDK (only the compiler is used; the runtime is the one of the PyTorch wheel)
[[ -x .venv-hipcc/bin/python ]] || uv venv --quiet --python 3.12 .venv-hipcc
uv pip install --quiet --python .venv-hipcc/bin/python --index-url "$AMD_INDEX" "rocm[devel,device-gfx1100]==10.1.0"

# versioned ROCm 7 names for CuPy
SP=$("$PY" -c "import site; print(site.getsitepackages()[0])")
TLIB=$SP/torch/lib
COMPAT=$PWD/.local/rocm7-compat
mkdir -p "$COMPAT"
for i in 1 2 3 4; do
  missing=$(find "$SP/cupy" "$SP/cupy_backends" -name "*.so" -print0 | LD_LIBRARY_PATH=$COMPAT:$TLIB xargs -0 ldd 2>/dev/null \
    | awk '/not found/ {print $1}' | sort -u)
  [[ -z $missing ]] && break
  for so in $missing; do
    base=${so%%.so*}.so
    [[ -e $TLIB/$base && ! -e $COMPAT/$so ]] && ln -s "$TLIB/$base" "$COMPAT/$so"
  done
done

# the add-on, at a fixed commit, with our patches
ADDON=.local/upstream/erised-fork
[[ -d $ADDON/.git ]] || git clone -q "$ADDON_REPO" "$ADDON"
git -C "$ADDON" fetch -q --depth 1 origin "$ADDON_COMMIT" 2>/dev/null || true
if [[ $(git -C "$ADDON" rev-parse --short=7 HEAD) != "$ADDON_COMMIT" ]]; then
  git -C "$ADDON" checkout -q "$ADDON_COMMIT"
fi
for p in fdtd_gpu/patches/*.patch; do
  [[ -e $p ]] || continue
  if git -C "$ADDON" apply --check "$PWD/$p" 2>/dev/null; then git -C "$ADDON" apply "$PWD/$p" && echo "applied $p"; fi
done
uv pip install --quiet --python "$PY" --no-deps -e "$ADDON"
uv pip freeze --python "$PY" > .local/env-fdtd-gpu-freeze.txt

source tools/env/fdtd_gpu_activate.sh
"$PY" - <<'PY'
import importlib
for m in ("torch", "triton", "cupy"):
    mod = importlib.import_module(m)
    print(m, getattr(mod, "__version__", "?"))
import torch, cupy
print("torch device:", torch.cuda.get_device_name(0), "| cupy devices:", cupy.cuda.runtime.getDeviceCount())
x = cupy.arange(1 << 20, dtype=cupy.float32)
print("cupy elementwise ok:", float((x * 2).sum()) == float(2 * x.sum()))
PY
"$PY" -c "import meep, meep_gpu; print('solver', meep.__version__, '| add-on', getattr(meep_gpu, '__version__', '?'))"
