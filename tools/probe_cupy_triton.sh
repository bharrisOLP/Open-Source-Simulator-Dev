#!/usr/bin/env bash
# Probe on the AMD GPU computer: can CuPy (ROCm build) and Triton (AMD backend, from the PyTorch ROCm wheel) run
# kernels on gfx1100 without a system ROCm? Both are needed for the GPU time stepping of the Erised add-on.
# Makes .venv-probe in the repository folder; deletes nothing. Run: bash tools/probe_cupy_triton.sh
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH=$HOME/.local/bin:$PATH PYTHONDONTWRITEBYTECODE=1 ROCR_VISIBLE_DEVICES=0
VENV=$PWD/.venv-probe
PY=$VENV/bin/python
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/rocm7.2}

[[ -x $PY ]] || uv venv --quiet --python 3.12 "$VENV"
echo "== torch (ROCm wheel, brings its own ROCm runtime and pytorch-triton-rocm)"
uv pip install --quiet --python "$PY" --index-url "$TORCH_INDEX" torch 2>&1 | tail -3
"$PY" - <<'PY'
import torch, sys
print("torch", torch.__version__, "hip", torch.version.hip, "available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).gcnArchName)
    x = torch.ones(1 << 20, device="cuda"); print("sum ok", float(x.sum()))
try:
    import triton, triton.language as tl
    print("triton", triton.__version__)

    @triton.jit
    def axpy(x_ptr, y_ptr, n, a, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = i < n
        tl.store(y_ptr + i, a * tl.load(x_ptr + i, mask=m) + tl.load(y_ptr + i, mask=m), mask=m)

    n = 1 << 20
    x = torch.arange(n, device="cuda", dtype=torch.float32); y = torch.ones(n, device="cuda")
    axpy[(n // 1024,)](x, y, n, 2.0, BLOCK=1024)
    torch.cuda.synchronize()
    print("triton kernel ok:", bool(torch.allclose(y, 2 * torch.arange(n, device="cuda", dtype=torch.float32) + 1)))
except Exception as e:  # noqa: BLE001
    print("triton FAILED:", type(e).__name__, str(e)[:400])
PY
echo "== cupy (ROCm build)"
for pkg in cupy-rocm-7-0 cupy-rocm-6-0; do
  if uv pip install --quiet --python "$PY" "$pkg" 2>/dev/null; then echo "installed $pkg"; break; fi
  echo "no wheel: $pkg"
done
# CuPy-ROCm looks for the ROCm libraries; point it at the libraries in the torch wheel (no system ROCm here)
TLIB=$("$PY" -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
LD_LIBRARY_PATH=$TLIB:${LD_LIBRARY_PATH:-} "$PY" - <<'PY'
try:
    import cupy as cp
    print("cupy", cp.__version__, "devices", cp.cuda.runtime.getDeviceCount(), "hip build", cp.cuda.runtime.is_hip)
    k = cp.RawKernel(r'''
    extern "C" __global__ void axpy(const float* x, float* y, float a, int n) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i < n) y[i] += a * x[i];
    }''', "axpy")
    n = 1 << 20
    x = cp.arange(n, dtype=cp.float32); y = cp.ones(n, dtype=cp.float32)
    k(((n + 255) // 256,), (256,), (x, y, cp.float32(2.0), cp.int32(n)))
    print("cupy RawKernel ok:", bool(cp.allclose(y, 2 * cp.arange(n, dtype=cp.float32) + 1)))
except Exception as e:  # noqa: BLE001
    print("cupy FAILED:", type(e).__name__, str(e)[:400])
PY
