#!/usr/bin/env bash
# The environment of optics-gpu on the AMD GPU computer: PyTorch for ROCm 7.2 (it brings its own ROCm runtime),
# Optiland 0.6.3 and optics_gpu (editable). Makes .venv-optics in the repository folder; deletes nothing.
# Run on the GPU computer: bash tools/env/optics_rocm.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
export PATH=$HOME/.local/bin:$PATH PYTHONDONTWRITEBYTECODE=1 ROCR_VISIBLE_DEVICES=0
VENV=$PWD/.venv-optics
PY=$VENV/bin/python
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/rocm7.2}
[[ -x $PY ]] || uv venv --quiet --python 3.12 "$VENV"
# the versions of the bench rows in benchmarks/2026-10-30/optics-trace.jsonl (numpy, scipy, pytest: optics_gpu/uv.lock)
uv pip install --quiet --python "$PY" --index-url "$TORCH_INDEX" "torch==2.14.1"
uv pip install --quiet --python "$PY" "optiland==0.6.3" "numpy==2.5.4" "scipy==1.18.1" "pytest==9.2.0"
uv pip install --quiet --python "$PY" --no-deps -e ./optics_gpu
"$PY" -c "import torch, optiland; print('torch', torch.__version__, 'hip', torch.version.hip,
           'gpu', torch.cuda.is_available())"
