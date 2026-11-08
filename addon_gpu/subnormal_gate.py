# SPDX-License-Identifier: GPL-2.0-or-later
"""Run the subnormal-policy gate of the add-on (``parity/meep_gpu/gate_subnormal_policy.py``) on this host.

    python subnormal_gate.py <clone folder> --policy keep --legs host,cupy_ufunc --out FILE.json

The gate measures, bit for bit, whether each executor (host NumPy, CuPy ufunc, ElementwiseKernel, RawKernel,
reductions, Triton) obeys a float32 subnormal policy. Under ``keep`` the add-on refuses to install the policy unless
``CUPY_CACHE_DIR`` names a folder that carries ``ftz_stripped``, so this wrapper first calls
``point_cupy_cache_at_keep_policy()``, as kernel dispatch does. That only satisfies the check of the add-on: CuPy 14
reads ``CUPY_CACHE_DIR`` once, when ``cupy.cuda.compiler`` is imported, and keeps the folder that it read then. What
separates the keep binaries from the flush binaries in the cache is the keep token that patch 0002 puts into the
cache key (tests/test_patches.py, ``test_the_keep_token_reaches_the_cache_key``). The CuPy compiler front ends are
also guarded (``backends.guard_kernel_compilation``), as ``resolve_backend`` does before a run, so that the
RawKernel leg compiles with the options of the hipRTC patch (tools/env/addon_patches/0001).
"""

from __future__ import annotations

import os
import runpy
import sys


def main(argv: list[str]) -> int:
    clone = os.path.abspath(argv[0])
    rest = argv[1:]
    policy = rest[rest.index("--policy") + 1] if "--policy" in rest else None
    import cupy

    from meep_gpu import backends

    if policy == "keep":
        from meep_gpu.subnormal_policy import point_cupy_cache_at_keep_policy

        point_cupy_cache_at_keep_policy()
    backends.guard_kernel_compilation(cupy)
    gate = os.path.join(clone, "parity", "meep_gpu", "gate_subnormal_policy.py")
    sys.argv = [gate] + rest
    try:
        runpy.run_path(gate, run_name="__main__")
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
