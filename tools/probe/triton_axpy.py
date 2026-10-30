"""Probe: one Triton kernel on the AMD GPU (through the PyTorch ROCm wheel)."""
import torch
import triton
import triton.language as tl


@triton.jit
def axpy(x_ptr, y_ptr, n, a, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    tl.store(y_ptr + i, a * tl.load(x_ptr + i, mask=m) + tl.load(y_ptr + i, mask=m), mask=m)


n = 1 << 20
x = torch.arange(n, device="cuda", dtype=torch.float32)
y = torch.ones(n, device="cuda")
axpy[(n // 1024,)](x, y, n, 2.0, BLOCK=1024)
torch.cuda.synchronize()
print("triton", triton.__version__, "kernel ok:", bool(torch.allclose(y, 2 * x + 1)))
