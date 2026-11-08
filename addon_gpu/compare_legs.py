# SPDX-License-Identifier: GPL-2.0-or-later
"""Compare two legs saved by run_case.py --save, and print one line ``ADDON_JSON {...}``.

    python compare_legs.py A.npz B.npz

For each array that both files hold (the final Ez field, the point series, each flux spectrum): the relative L2
difference ||A - B|| / ||B|| and, for arrays of the same shape, the number of float32 words that differ (B is the
reference).
"""

from __future__ import annotations

import json
import sys

import numpy as np


def compare(a_path: str, b_path: str) -> dict:
    out = {"a": a_path, "b": b_path}
    with np.load(a_path) as a, np.load(b_path) as b:
        for name in sorted(set(a.files) & set(b.files)):
            x, y = np.asarray(a[name]), np.asarray(b[name])
            row = {"shape": list(x.shape)}
            if x.shape == y.shape and x.size:
                xd, yd = x.astype(np.float64).ravel(), y.astype(np.float64).ravel()
                norm = float(np.linalg.norm(yd))
                row["rel_l2"] = float(np.linalg.norm(xd - yd) / norm) if norm > 0 else None
                xw = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
                yw = np.ascontiguousarray(y, dtype=np.float32).view(np.uint32)
                row["differing_float32_words"] = int(np.count_nonzero(xw != yw))
                row["words"] = int(x.size)
            out[name] = row
    return out


if __name__ == "__main__":
    print("ADDON_JSON " + json.dumps(compare(sys.argv[1], sys.argv[2])), flush=True)
