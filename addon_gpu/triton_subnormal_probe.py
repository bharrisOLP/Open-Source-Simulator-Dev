# SPDX-License-Identifier: GPL-2.0-or-later
"""Does a Triton kernel on this GPU keep or flush float32 subnormal values, under a subnormal policy of the add-on?

    python triton_subnormal_probe.py --policy keep|flush|none

The gate of the add-on (``parity/meep_gpu/gate_subnormal_policy.py``) cannot run its Triton leg with Triton 3.8
(it calls ``JITFunction.cache``, which Triton 3.8 does not have), so this probe repeats the method for the op classes
of the shipped kernels: operands are built as uint32 bit patterns, moved as integers, and read back as uint32, with
no float literal on the host. ``--policy`` installs the policy with ``meep_gpu.set_subnormal_policy`` before the
first compile (``none`` installs nothing: the default of Triton on this GPU). Prints one line ``ADDON_JSON {...}``.
"""

from __future__ import annotations

import argparse
import json

# (name, x word, y word, keep word, flush word). Exact in float32: 2^-70 * 2^-65 = 2^-135 (subnormal 0x00004000);
# 1.25 * 2^-126 - 2^-126 = 2^-128 (0x00200000); 2^-100 / 2^35 = 2^-135; a subnormal input times 1.0.
CASES = (
    ("mul_result_subnormal", 0x1C800000, 0x1F000000, 0x00004000, 0x00000000),
    ("sub_cancellation_subnormal", 0x00A00000, 0x00800000, 0x00200000, 0x00000000),
    ("fma_result_subnormal", 0x1C800000, 0x1F000000, 0x00004000, 0x00000000),
    ("div_rn_result_subnormal", 0x0D800000, 0x51000000, 0x00004000, 0x00000000),
    ("mul_subnormal_input", 0x00004000, 0x3F800000, 0x00004000, 0x00000000),
    ("select_transport", 0x00004000, 0x3F800000, 0x00004000, 0x00004000),
)
OPS = {name: index for index, (name, *_rest) in enumerate(CASES)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--policy", choices=("keep", "flush", "none"), default="none")
    args = p.parse_args(argv)

    import numpy as np
    import torch
    import triton
    import triton.language as tl

    import meep_gpu

    stamp = None
    if args.policy != "none":
        if args.policy == "keep":
            from meep_gpu.subnormal_policy import point_cupy_cache_at_keep_policy

            point_cupy_cache_at_keep_policy()
        stamp = meep_gpu.set_subnormal_policy(args.policy, strict=False)

    @triton.jit
    def probe(x_ptr, y_ptr, out_ptr, OP: tl.constexpr, N: tl.constexpr):
        i = tl.arange(0, N)
        x = tl.load(x_ptr + i).to(tl.float32, bitcast=True)
        y = tl.load(y_ptr + i).to(tl.float32, bitcast=True)
        if OP == 0:
            r = x * y
        elif OP == 1:
            r = x - y
        elif OP == 2:
            r = tl.math.fma(x, y, tl.zeros_like(x))
        elif OP == 3:
            r = tl.math.div_rn(x, y)
        elif OP == 4:
            r = x * y
        else:
            r = tl.where(y != 0, x, y)
        tl.store(out_ptr + i, r.to(tl.int32, bitcast=True))

    n = 16
    rows = []
    for name, xw, yw, keep, flush in CASES:
        x = torch.tensor(np.full(n, xw, dtype=np.uint32).view(np.int32), device="cuda")
        y = torch.tensor(np.full(n, yw, dtype=np.uint32).view(np.int32), device="cuda")
        out = torch.zeros(n, dtype=torch.int32, device="cuda")
        probe[(1,)](x, y, out, OP=OPS[name], N=n)
        words = out.cpu().numpy().view(np.uint32)
        word = int(words[0])
        same = bool((words == words[0]).all())
        verdict = "keep" if word == keep else ("flush" if word == flush else "other")
        rows.append({"op": name, "word": f"0x{word:08X}", "keep_word": f"0x{keep:08X}",
                     "flush_word": f"0x{flush:08X}", "verdict": verdict, "all_lanes_equal": same})
    rec = {"policy": args.policy, "triton": triton.__version__, "torch": torch.__version__,
           "device": torch.cuda.get_device_name(0), "rows": rows,
           "policy_stamp": None if stamp is None else {k: stamp.get(k) for k in ("resolved", "installed")
                                                         if isinstance(stamp, dict)}}
    print("ADDON_JSON " + json.dumps(rec, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
