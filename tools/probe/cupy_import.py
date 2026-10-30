"""Probe: import CuPy (ROCm build) and print the root cause of a failure."""
import traceback

try:
    import cupy as cp
    print("cupy", cp.__version__, "devices", cp.cuda.runtime.getDeviceCount(), "hip", cp.cuda.runtime.is_hip)
    k = cp.RawKernel(r'''
    extern "C" __global__ void axpy(const float* x, float* y, float a, int n) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i < n) y[i] += a * x[i];
    }''', "axpy")
    n = 1 << 20
    x = cp.arange(n, dtype=cp.float32)
    y = cp.ones(n, dtype=cp.float32)
    k(((n + 255) // 256,), (256,), (x, y, cp.float32(2.0), cp.int32(n)))
    print("cupy RawKernel ok:", bool(cp.allclose(y, 2 * x + 1)))
except Exception as e:  # noqa: BLE001
    c = e
    while c.__cause__ or c.__context__:
        c = c.__cause__ or c.__context__
    print("cupy FAILED; root cause:", type(c).__name__, str(c)[:600])
    traceback.print_exc(limit=2)
