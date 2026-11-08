#!/usr/bin/env bash
# Run one test or bench command on the GPU computer, with the GPU lock: one gpu_run.sh process at a time in this
# repository (on 2026-10-30, test runs at the same time failed with HIP errors and gave wrong timings; no log was
# kept). The lock does not cover jobs of other sessions or of `remote.sh run`: check `remote.sh usage` before a
# large run. All tests run on the GPU computer, not on the Mac (user, 2026-10-30). Run it on the GPU computer, from
# the repository folder:
#   bash tools/gpu_run.sh <folder> <command...>     for example: bash tools/gpu_run.sh meow_gpu python -m pytest -q
# The command runs in <folder> with .venv-rocm (.venv-optics for optics_gpu) first on PATH. For addon_gpu and for
# the add-on clone (.local/upstream/erised-fork) it uses .local/envs/addon-gpu with the run-time variables of
# tools/env/addon_gpu_activate.sh (tools/env/addon_gpu.sh makes that environment). Variables for the
# command go before the script (THERMAL_GPU_DEVICE=cpu bash tools/gpu_run.sh ...). GPU_RUN_TIMEOUT (seconds,
# default 1800) limits the run time, GPU_LOCK_WAIT (seconds, default 7200) the wait for the lock. At the time limit
# the process group of the command gets SIGTERM (exit 124); if the command still runs GPU_RUN_KILL_AFTER seconds
# later (default 30), the group gets SIGKILL (exit 137). This needs GNU timeout: the uutils timeout of the GPU
# computer (/usr/bin/timeout) sends SIGKILL only to the direct child, so the script uses gnutimeout when it exists.
# A descendant still keeps the lock (flock runs without -o) when it is in its own process group (setsid, a nested
# timeout), when it ignores SIGTERM and the direct child ends at SIGTERM (no SIGKILL then), or when it is stuck in
# the GPU driver.
set -euo pipefail
usage="usage: bash tools/gpu_run.sh <folder> <command...>"
[[ $# -ge 2 ]] || { echo "$usage" >&2; exit 2; }
cd "$(dirname "$0")/.."
root=$PWD
folder=${1%/}
folder=${folder#./}
shift
[[ -d $root/$folder ]] || { echo "gpu_run: no folder $folder" >&2; exit 2; }
venv=$root/.venv-rocm
case $folder in
  optics_gpu) venv=$root/.venv-optics ;;
  addon_gpu | addon_gpu/* | .local/upstream/erised-fork | .local/upstream/erised-fork/*)
    venv=$root/.local/envs/addon-gpu
    [[ -x $venv/bin/python ]] || { echo "gpu_run: no $venv (run tools/env/addon_gpu.sh)" >&2; exit 2; }
    source tools/env/addon_gpu_activate.sh ;;
esac
mkdir -p "$root/.local"
export ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-0} PYTHONDONTWRITEBYTECODE=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.8} PATH=$venv/bin:$PATH
export PYTEST_ADDOPTS="${PYTEST_ADDOPTS:-} -p no:cacheprovider"
cd "$root/$folder"
tmo=$(command -v gnutimeout || command -v timeout)
exec flock -w "${GPU_LOCK_WAIT:-7200}" "$root/.local/gpu.lock" \
  "$tmo" -k "${GPU_RUN_KILL_AFTER:-30}" "${GPU_RUN_TIMEOUT:-1800}" "$@"
