"""Where the fixed cost of optics_gpu.analysis.pupil_wavefront goes. The steps of one call of the earlier code
(66da5d3: a new PyTorch copy of the optic for each call and three traces through it), made here with Optiland's
functions, each timed alone (torch.cuda.synchronize after each step); then the present call. The median of
--repeat runs after one untimed run. One JSON line per pupil sampling N, with the times in ms:
- xpp: the paraxial exit-pupil point of the original optic (NumPy, host);
- to_dict, check_ref: the dictionary of the optic and the trace of the 6 check rays through the original (NumPy);
- from_dict: the PyTorch copy of the optic (Optic.from_dict under Optiland's PyTorch backend);
- check_copy: the trace of the 6 check rays through the copy and the comparison (device to host);
- five: the trace of the chief ray and the four marginal rays through the copy (trace_generic);
- aim: Optiland's paraxial ray aiming alone for 5 rays on the copy (EPD, entrance pupil: small device traces);
- surfaces_5: Optiland's surface trace alone for 5 rays on the copy (the rays of `aim`);
- main: the trace of all pupil rays through the copy (one trace_generic call);
- call: one whole pupil_wavefront call of the present code (the kept copy: one trace, item 7 of the module
  docstring of analysis.py); call_cold: the same after clear_copy_cache (a new copy).
With --profile, also the counts of device-to-host reads (aten::_local_scalar_dense, aten::item) and of all
operators in one whole call (the kept copy; torch.profiler), and a cProfile of `aim` on a new copy (where the
time of Optiland's paraxial ray aiming goes).

    bash tools/gpu_run.sh optics_gpu python bench/profile_wavefront.py --pupil 64 256 --profile
"""
import argparse
import json
import statistics
import time
import warnings

import numpy as np
import torch

from optics_gpu.analysis import CHECK_P, clear_copy_cache, pupil_wavefront
from optics_gpu.pop import _max_difference, optiland_torch
from optics_gpu.systems import aberrated_singlet, ideal_relay

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
WL = 1.55

ap = argparse.ArgumentParser()
ap.add_argument("--system", choices=("singlet", "relay"), default="singlet")
ap.add_argument("--pupil", type=int, nargs="+", default=[64, 256])
ap.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--repeat", type=int, default=5)
ap.add_argument("--profile", action="store_true")
a = ap.parse_args()
optic = aberrated_singlet() if a.system == "singlet" else ideal_relay()


def sync():
    if a.device == "cuda":
        torch.cuda.synchronize()


def timed(times: dict, name: str, fn):
    sync()
    t0 = time.perf_counter()
    out = fn()
    sync()
    times.setdefault(name, []).append(1e3 * (time.perf_counter() - t0))
    return out


def steps(N: int, times: dict) -> None:
    import optiland.backend as be
    from optiland.optic import Optic

    timed(times, "xpp", lambda: optic.paraxial.exit_pupil_point_gcs())
    data = timed(times, "to_dict", optic.to_dict)
    ref = timed(times, "check_ref", lambda: optic.ray_tracer.trace_generic(0.0, 0.0, CHECK_P[0].copy(),
                                                                           CHECK_P[1].copy(), WL, record=False))
    with optiland_torch(a.device), torch.no_grad():
        copy = timed(times, "from_dict", lambda: Optic.from_dict(data))

        def check():
            got = copy.ray_tracer.trace_generic(0.0, 0.0, be.array(CHECK_P[0].copy()), be.array(CHECK_P[1].copy()),
                                                WL, record=False)
            return _max_difference(ref, got)

        timed(times, "check_copy", check)
        f64 = {"dtype": torch.float64, "device": torch.device(a.device)}
        px = torch.tensor([0.0, 0.0, 0.0, 1.0, -1.0], **f64)
        py = torch.tensor([0.0, 1.0, -1.0, 0.0, 0.0], **f64)
        timed(times, "five", lambda: copy.ray_tracer.trace_generic(0.0, 0.0, px.clone(), py.clone(), WL,
                                                                   record=False))
        aimer = copy.ray_tracer.ray_generator.aimer
        x0, y0, z0, L, M, Nn = timed(times, "aim", lambda: aimer.aim_rays((0.0, 0.0), WL, (px.clone(), py.clone())))

        def surf():
            from optiland.rays import RealRays

            rays = RealRays(x0, y0, z0, L.clone(), M.clone(), Nn.clone(), torch.ones_like(x0),
                            torch.full_like(x0, WL))
            copy.surfaces.trace(rays, record=False)
            return rays

        timed(times, "surfaces_5", surf)
        p = np.linspace(-1.0, 1.0, N)
        PX, PY = np.meshgrid(p, p, indexing="ij")
        ins = PX**2 + PY**2 <= 1
        Px, Py = torch.as_tensor(PX[ins], **f64), torch.as_tensor(PY[ins], **f64)
        timed(times, "main", lambda: copy.ray_tracer.trace_generic(0.0, 0.0, Px.clone(), Py.clone(), WL,
                                                                   record=False))
    timed(times, "call", lambda: pupil_wavefront(optic, (0.0, 0.0), WL, n_pupil=N, device=a.device))

    def cold():
        clear_copy_cache()
        return pupil_wavefront(optic, (0.0, 0.0), WL, n_pupil=N, device=a.device)

    timed(times, "call_cold", cold)


for N in a.pupil:
    steps(N, {})                                   # untimed run
    times: dict = {}
    for _ in range(a.repeat):
        steps(N, times)
    row = {"system": a.system, "device": a.device, "pupil": N, "repeat": a.repeat,
           **{f"{k}_ms": statistics.median(v) for k, v in times.items()},
           "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "torch": torch.__version__}
    if a.profile:
        from torch.profiler import ProfilerActivity, profile

        acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if a.device == "cuda" else [])
        with profile(activities=acts) as prof:
            pupil_wavefront(optic, (0.0, 0.0), WL, n_pupil=N, device=a.device)
            sync()
        ev = prof.key_averages()
        counts = {e.key: e.count for e in ev}
        row["host_reads"] = counts.get("aten::_local_scalar_dense", 0)
        row["item_calls"] = counts.get("aten::item", 0)
        row["aten_ops"] = sum(c for k, c in counts.items() if k.startswith("aten::"))
        row["top_ops"] = sorted(((k, c) for k, c in counts.items() if k.startswith("aten::")),
                                key=lambda kc: -kc[1])[:8]
    print(json.dumps(row), flush=True)

if a.profile:
    # where the time of Optiland's paraxial ray aiming goes (5 rays on the PyTorch copy): cProfile, the functions
    # with the largest cumulative time
    import cProfile
    import io
    import pstats

    from optiland.optic import Optic

    data = optic.to_dict()
    with optiland_torch(a.device), torch.no_grad():
        copy = Optic.from_dict(data)
        f64 = {"dtype": torch.float64, "device": torch.device(a.device)}
        px = torch.tensor([0.0, 0.0, 0.0, 1.0, -1.0], **f64)
        py = torch.tensor([0.0, 1.0, -1.0, 0.0, 0.0], **f64)
        aimer = copy.ray_tracer.ray_generator.aimer
        aimer.aim_rays((0.0, 0.0), WL, (px.clone(), py.clone()))
        pr = cProfile.Profile()
        pr.enable()
        for _ in range(10):
            aimer.aim_rays((0.0, 0.0), WL, (px.clone(), py.clone()))
        sync()
        pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(25)
    print("\n".join(line[:150] for line in s.getvalue().splitlines()[:60]))
