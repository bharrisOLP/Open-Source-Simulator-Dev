# Source this file (from the repository folder on the AMD GPU computer) before a run in .local/envs/addon-gpu.
# It sets the run-time variables that CuPy and Triton need without a system ROCm (see tools/env/addon_gpu.sh).
_root=$(pwd)
_env=$_root/.local/envs/addon-gpu
_sp=$("$_env/bin/python" -c "import site; print(site.getsitepackages()[0])")
_devel=$("$_root/.venv-hipcc/bin/rocm-sdk" path --root)
export ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-0}   # the integrated GPU (gfx1036) is not supported by ROCm
export ROCM_HOME=$_devel HIP_PATH=$_devel
export PATH=$_env/bin:$_devel/bin:$PATH
export LD_LIBRARY_PATH=$_root/.local/rocm7-compat:$_sp/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
# The CuPy cache folder of the default (native) compiles and of the float32 subnormal policy `flush`. The add-on
# refuses an install of `keep` with a folder whose path does not carry `ftz_stripped`, and of `flush` with one that
# does. Its dispatch installs `keep` itself and mints .local/cupy-cache-ftz_stripped from this folder; for the test
# modules that install `keep` by hand, addon_gpu/suite_runner.py sets that keep folder per module. One keep folder for
# every run would make the add-on refuse each `flush` install (3 failed tests in 1 of 60 modules, addon_gpu/README.md).
export CUPY_CACHE_DIR=$_root/.local/cupy-cache
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
unset _root _env _sp _devel
