#!/usr/bin/env bash
# Run one test or bench command on the GPU computer, with the GPU lock: one gpu_run.sh process at a time in this
# repository (on 2026-10-30, test runs at the same time failed with HIP errors and gave wrong timings; no log was
# kept). The lock does not cover jobs of other sessions or of `remote.sh run`: check `remote.sh usage` before a
# large run. All tests run on the GPU computer, not on the Mac (user, 2026-10-30). Run it on the GPU computer, from
# the repository folder:
#   bash tools/gpu_run.sh <folder> <command...>     for example: bash tools/gpu_run.sh meow_gpu python -m pytest -q
# The command runs in <folder> with .venv-rocm (.venv-optics for optics_gpu) first on PATH. Variables for the
# command go before the script (THERMAL_GPU_DEVICE=cpu bash tools/gpu_run.sh ...). GPU_RUN_TIMEOUT (seconds,
# default 1800) limits the run time, GPU_LOCK_WAIT (seconds, default 7200) the wait for the lock.
set -euo pipefail
usage="usage: bash tools/gpu_run.sh <folder> <command...>"
[[ $# -ge 2 ]] || { echo "$usage" >&2; exit 2; }
cd "$(dirname "$0")/.."
root=$PWD
folder=${1%/}
shift
[[ -d $root/$folder ]] || { echo "gpu_run: no folder $folder" >&2; exit 2; }
venv=$root/.venv-rocm
[[ $folder == optics_gpu ]] && venv=$root/.venv-optics
mkdir -p "$root/.local"
export ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-0} PYTHONDONTWRITEBYTECODE=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.8} PATH=$venv/bin:$PATH
export PYTEST_ADDOPTS="${PYTEST_ADDOPTS:-} -p no:cacheprovider"
cd "$root/$folder"
exec flock -w "${GPU_LOCK_WAIT:-7200}" "$root/.local/gpu.lock" timeout "${GPU_RUN_TIMEOUT:-1800}" "$@"
