# Source this file (from the repository folder on the AMD GPU computer) before a run in .local/envs/fdtd-gpu.
# It sets the run-time variables that CuPy and Triton need without a system ROCm (see tools/env/fdtd_gpu.sh).
_root=$(pwd)
_env=$_root/.local/envs/fdtd-gpu
_sp=$("$_env/bin/python" -c "import site; print(site.getsitepackages()[0])")
_devel=$("$_root/.venv-hipcc/bin/rocm-sdk" path --root)
export ROCR_VISIBLE_DEVICES=0                     # the integrated GPU (gfx1036) is not supported by ROCm
export ROCM_HOME=$_devel HIP_PATH=$_devel
export PATH=$_env/bin:$_devel/bin:$PATH
export LD_LIBRARY_PATH=$_root/.local/rocm7-compat:$_sp/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export CUPY_CACHE_DIR=$_root/.local/cupy-cache
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
unset _root _env _sp _devel
