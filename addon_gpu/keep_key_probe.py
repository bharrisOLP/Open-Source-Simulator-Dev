# SPDX-License-Identifier: GPL-2.0-or-later
"""Does the keep token of patch 0002 reach the cache key of a CuPy compile on this ROCm build?

    python keep_key_probe.py

CuPy 14 reads ``CUPY_CACHE_DIR`` only when ``cupy.cuda.compiler`` is imported, so the keep token in the cache key is
the only thing that separates a keep binary from a flush binary in the kernel cache (tools/env/addon_patches/README.md).
This probe records each key that CuPy hashes (``cupy.cuda.compiler._hash_hexdigest``), compiles one new RawKernel
before and one after ``meep_gpu.set_subnormal_policy("keep")``, and counts the keys that carry the token. It runs in
its own process, because the policy is installed for the whole process. Prints one line ``ADDON_JSON {...}``.
"""

from __future__ import annotations

import json
import uuid


def main() -> int:
    import cupy

    import meep_gpu
    from meep_gpu import backends
    from meep_gpu import subnormal_policy as sp

    compiler = cupy.cuda.compiler
    keys: list[bytes] = []
    original = compiler._hash_hexdigest

    def spy(value):
        keys.append(value if isinstance(value, bytes) else str(value).encode())
        return original(value)

    compiler._hash_hexdigest = spy
    backends.guard_kernel_compilation(cupy)

    def compile_new_kernel(tag: str) -> list[bytes]:
        first = len(keys)
        name = f"keep_key_probe_{tag}"
        code = (f'extern "C" __global__ void {name}(float* x) {{ x[threadIdx.x] *= 2.0f; }}\n'
                f"// {uuid.uuid4().hex}\n")                 # new source: CuPy must form a new key
        kernel = cupy.RawKernel(code, name, options=("--fmad=false",))
        x = cupy.ones(32, dtype=cupy.float32)
        kernel((1,), (32,), (x,))
        cupy.cuda.Device().synchronize()
        return keys[first:]

    before = compile_new_kernel("before")
    sp.point_cupy_cache_at_keep_policy()
    stamp = meep_gpu.set_subnormal_policy("keep")
    after = compile_new_kernel("after")
    token = sp.CUPY_KEEP_CACHE_KEY_TOKEN.encode()
    rec = {
        "cupy": cupy.__version__,
        "is_hip": backends.cupy_is_hip(cupy),
        "policy": (stamp or {}).get("resolved") if isinstance(stamp, dict) else None,
        "before_keys": len(before),
        "before_with_token": sum(token in key for key in before),
        "after_keys": len(after),
        "after_with_token": sum(token in key for key in after),
    }
    print("ADDON_JSON " + json.dumps(rec), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
