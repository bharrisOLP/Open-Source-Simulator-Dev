# SPDX-License-Identifier: GPL-2.0-or-later
"""Where does the setup before the first step go? Profile the lift of one case, and time its first step.

    python profile_setup.py --case slab2d --resolution 40 --scale 4                     the 2-D bench case
    python profile_setup.py --case sphere3d --resolution 32                              the 3-D bench case
    python profile_setup.py --case slab2d --bulk on|off|auto --table cuda|triton|none    the chi1inv read path

The lift is ``meep_gpu.from_meep.lift_simulation`` (the solver's ``init_sim``, the read of the permittivity
(``chi1inv``), the conductivity and source passes, the transfer to the GPU). ``--bulk`` forces the bulk read of the
stored ``chi1inv`` arrays on (``on``) or off (``off``), as the add-on's own test of that read does; ``auto`` keeps
the choice of the add-on (the bulk read only on a build in ``BULK_CHI1INV_BUILDS``). ``--table`` selects the kernel
table for the first step (``none``: the NumPy reference, ``prefer_gpu=False``). Prints the lift lines of the add-on on
stderr, the top functions of the profile, and one line ``ADDON_JSON {...}``.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import os
import pstats
import sys
import time

TABLE_ENV = {"triton": {"MEEP_GPU_HARDWARE": "any", "MEEP_GPU_KERNEL_TABLE": "triton"},
             "cuda": {"MEEP_GPU_HARDWARE": "any", "MEEP_GPU_KERNEL_TABLE": "cuda"}, "none": {}}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--case", default="slab2d")
    p.add_argument("--resolution", type=int, default=None)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--bulk", choices=("auto", "on", "off"), default="auto")
    p.add_argument("--table", choices=tuple(TABLE_ENV), default="cuda")
    p.add_argument("--top", type=int, default=25)
    args = p.parse_args(argv)
    os.environ.update(TABLE_ENV[args.table])
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from meep_gpu import from_meep

    from addon_cases import build

    if args.bulk != "auto":
        forced = args.bulk == "on"
        real_lift = from_meep._lift_epsilon_structured

        def lift_spy(*a, **kw):  # as the add-on's test_from_meep_bulk_chi1inv.py forces the read path
            if not kw.get("_restarted_by"):
                kw["_bulk_read"] = forced
            return real_lift(*a, **kw)

        from_meep._lift_epsilon_structured = lift_spy

    case = build(args.case, args.resolution, None, args.scale)
    prefer_gpu = args.table != "none"
    profile = cProfile.Profile()
    t0 = time.perf_counter()
    profile.enable()
    driver = from_meep.lift_simulation(case.sim, prefer_gpu=prefer_gpu)
    profile.disable()
    lift_s = time.perf_counter() - t0
    try:
        t1 = time.perf_counter()
        driver.step()                       # the first step: the configuration freeze (and the kernel warm pass)
        if prefer_gpu:
            import cupy

            cupy.cuda.Device().synchronize()
        first_step_s = time.perf_counter() - t1
        read = getattr(driver, "chi1inv_read", None) or {}
        components = {name: {k: rep.get(k) for k in ("path", "seconds", "bulk_points", "residual_points")}
                      for name, rep in (read.get("components") or {}).items()}
        active = driver.active_step_path
    finally:
        driver.close()

    stats = pstats.Stats(profile)
    top = []
    for (filename, line, name), (cc, nc, tt, ct, _callers) in sorted(
            stats.stats.items(), key=lambda item: item[1][3], reverse=True)[:args.top]:
        top.append({"function": f"{os.path.basename(filename)}:{line}:{name}", "calls": nc,
                    "tottime_s": round(tt, 3), "cumtime_s": round(ct, 3)})
    text = io.StringIO()
    stats.stream = text
    stats.sort_stats("tottime").print_stats(12)
    print(text.getvalue(), file=sys.stderr)
    rec = {"case": args.case, "resolution": args.resolution, "scale": args.scale, "cells": case.cells,
           "bulk": args.bulk, "table": args.table, "lift_s": round(lift_s, 3), "first_step_s": round(first_step_s, 3),
           "chi1inv_path": read.get("path"), "chi1inv_refusal": read.get("refusal"),
           "chi1inv_stamp": read.get("stamp"), "chi1inv_components": components, "active_step_path": active,
           "top_cumulative": top}
    print("ADDON_JSON " + json.dumps(rec, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
