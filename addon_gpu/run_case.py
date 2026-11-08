# SPDX-License-Identifier: GPL-2.0-or-later
"""Run one case on one leg in this process, and print one JSON line that starts with ``ADDON_JSON``.

    python run_case.py --case slab2d --leg gpu --table triton      the Triton kernels of the add-on
    python run_case.py --case slab2d --leg gpu --table cuda        the hand-written kernels (CuPy RawKernel, hipRTC)
    python run_case.py --case slab2d --leg gpu --table auto        both tables, in the order of the add-on
    python run_case.py --case slab2d --leg array                   the CuPy array path on the GPU (dispatch off)
    python run_case.py --case slab2d --leg reference               the NumPy reference of the add-on (host CPU)
    python run_case.py --case slab2d --leg cpu                     the CPU solver itself (its own time stepping)

One leg runs in each process, because a run that dispatches kernels installs the float32 subnormal policy of its
kernel table for the whole process (README of the add-on, "Bit identity"). The kernel tables of the add-on admit
NVIDIA devices of compute capability 7.0 to 9.0 only; on the AMD GPU, ``--table triton|cuda|auto`` therefore sets
``MEEP_GPU_HARDWARE=any`` (the switch of the add-on that admits a device outside that range, uncertified).

``--save FILE.npz`` writes the final field, the flux spectra and the recorded point series for a comparison in
another process. ``--warmup T`` first runs the same case to time T in this process, so that the timed run does not
include the kernel compilation. The step rate is measured between progress callbacks (each one waits for the
device), from the second callback to the end of the run.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

TABLE_ENV = {
    "triton": {"MEEP_GPU_HARDWARE": "any", "MEEP_GPU_KERNEL_TABLE": "triton"},
    "cuda": {"MEEP_GPU_HARDWARE": "any", "MEEP_GPU_KERNEL_TABLE": "cuda"},
    "auto": {"MEEP_GPU_HARDWARE": "any"},
    "none": {},
}
LEGS = ("gpu", "array", "reference", "cpu", "info")


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--case", default="slab2d")
    p.add_argument("--leg", choices=LEGS, default="gpu")
    p.add_argument("--table", choices=tuple(TABLE_ENV), default="triton")
    p.add_argument("--policy", choices=("default", "keep", "flush"), default="default",
                   help="float32 subnormal policy installed before an array or reference leg")
    p.add_argument("--resolution", type=int, default=None)
    p.add_argument("--until", type=float, default=None)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--warmup", type=float, default=0.0, help="run the case to this time first (GPU legs)")
    p.add_argument("--no-series", action="store_true", help="do not record the point series (benches)")
    p.add_argument("--save", default=None)
    p.add_argument("--full-report", action="store_true")
    p.add_argument("--progress-interval", type=int, default=100)
    return p.parse_args(argv)


def _device_sync():
    try:
        import cupy

        cupy.cuda.Device().synchronize()
    except Exception:  # noqa: BLE001 - no CuPy: nothing to wait for
        pass


def versions() -> dict:
    out = {"python": platform.python_version()}
    for name in ("numpy", "meep", "meep_gpu", "cupy", "triton", "torch"):
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "?")
        except Exception:  # noqa: BLE001
            out[name] = None
    try:
        import meep as mp

        out["solver_single_precision"] = bool(mp.is_single_precision())
    except Exception:  # noqa: BLE001
        pass
    try:
        import cupy

        props = cupy.cuda.runtime.getDeviceProperties(0)
        out["device"] = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
        out["gcn_arch"] = (props.get("gcnArchName") or b"").decode() if isinstance(props.get("gcnArchName"), bytes) \
            else props.get("gcnArchName")
        out["cupy_is_hip"] = bool(cupy.cuda.runtime.is_hip)
    except Exception:  # noqa: BLE001
        pass
    out["omp_num_threads"] = os.environ.get("OMP_NUM_THREADS")
    out["solver_links_openmp"] = _solver_links_openmp()
    return out


def _solver_links_openmp():
    """Is an OpenMP runtime mapped into this process after the import of the solver (Linux only)?"""
    try:
        import meep  # noqa: F401

        with open("/proc/self/maps") as fh:
            maps = fh.read()
        return any(name in maps for name in ("libgomp", "libomp.so", "libiomp"))
    except Exception:  # noqa: BLE001
        return None


def _summary(report) -> dict:
    if report is None:
        return {}
    slots = {}
    for name, slot in (report.get("slots") or {}).items():
        if isinstance(slot, dict):
            slots[name] = {k: slot.get(k) for k in ("path", "arm", "table", "reason") if k in slot}
        else:
            slots[name] = str(slot)
    comp = report.get("composition") or {}
    return {
        "step_path": report.get("step_path"),
        "decision": report.get("decision"),
        "refused_because": report.get("refused_because"),
        "table": report.get("table"),
        "certified": report.get("certified"),
        "dispatched_slots": comp.get("dispatched_slots"),
        "array_slots": comp.get("array_slots"),
        "slots": slots,
        "launch_counters": report.get("launch_counters"),
    }


def _install_policy(args, meep_gpu):
    if args.policy == "default" or args.leg not in ("array", "reference"):
        return
    if args.policy == "keep" and meep_gpu.available_gpu() == "cuda":
        from meep_gpu.subnormal_policy import point_cupy_cache_at_keep_policy

        point_cupy_cache_at_keep_policy()
    meep_gpu.set_subnormal_policy(args.policy)


def _gpu_run(case, args, meep_gpu, mp, *, prefer_gpu: bool, record: bool):
    from meep_gpu import run_on_gpu

    series: list[float] = []
    step_functions = []
    harminv = None
    if record and case.point is not None and not args.no_series:
        comp = case.harminv[0] if case.harminv else mp.Ez

        def _rec(sim):
            series.append(complex(sim.get_field_point(comp, case.point)).real)

        step_functions.append(_rec)
    if record and case.harminv is not None:
        harminv = mp.Harminv(case.harminv[0], case.point, case.harminv[1], case.harminv[2])
        step_functions.append(mp.after_sources(harminv))
    marks: list[tuple[int, float]] = []

    def _progress(done, total):
        _device_sync()
        marks.append((int(done), time.perf_counter()))

    kwargs = dict(prefer_gpu=prefer_gpu, step_functions=tuple(step_functions), progress_cb=_progress,
                  progress_interval=args.progress_interval)
    if case.until is not None:
        kwargs["until"] = case.until
    else:
        kwargs["until_after_sources"] = case.until_after_sources
    t0 = time.perf_counter()
    result = run_on_gpu(case.sim, **kwargs)
    _device_sync()
    total = time.perf_counter() - t0
    return result, total, marks, series, harminv


def _steady_rate(marks, cells):
    if len(marks) < 3:
        return None, None
    (s0, t0), (s1, t1) = marks[1], marks[-1]
    if s1 <= s0 or t1 <= t0:
        return None, None
    step_s = (t1 - t0) / (s1 - s0)
    return step_s, cells / step_s / 1e6


def run(args) -> dict:
    for key, value in (TABLE_ENV[args.table].items() if args.leg == "gpu" else ()):
        os.environ[key] = value
    if args.leg == "array":
        os.environ["MEEP_GPU_DISPATCH"] = "0"
    import numpy as np
    import meep as mp
    import meep_gpu

    from addon_cases import build

    rec: dict = {"case": args.case, "leg": args.leg, "table": args.table if args.leg == "gpu" else None,
                 "resolution": args.resolution, "scale": args.scale, "versions": versions(),
                 "meep_gpu_file": meep_gpu.__file__,
                 "env": {k: v for k, v in os.environ.items() if k.startswith("MEEP_GPU_")}}
    if args.leg == "info":
        return rec
    _install_policy(args, meep_gpu)

    if args.warmup > 0 and args.leg in ("gpu", "array"):
        warm = build(args.case, args.resolution, args.warmup, args.scale)
        if warm.until is None:
            warm.until, warm.until_after_sources = args.warmup, None
        result, warm_total, _, _, _ = _gpu_run(warm, args, meep_gpu, mp, prefer_gpu=True, record=False)
        rec["warmup"] = {"until": args.warmup, "steps": result.steps, "total_s": warm_total}
        result.close()

    case = build(args.case, args.resolution, args.until, args.scale)
    rec["cells"] = case.cells
    rec["shape"] = list(case.shape)
    flux_names = [f"flux_{i}" for i in range(len(case.fluxes))]
    harminv = None
    series: list[float] = []

    if args.leg == "cpu":
        t0 = time.perf_counter()
        case.sim.init_sim()
        rec["setup_s"] = time.perf_counter() - t0
        step_functions = []
        if case.point is not None and not args.no_series:
            comp = case.harminv[0] if case.harminv else mp.Ez

            def _rec(sim):
                series.append(complex(sim.get_field_point(comp, case.point)).real)

            step_functions.append(_rec)
        if case.harminv is not None:
            harminv = mp.Harminv(case.harminv[0], case.point, case.harminv[1], case.harminv[2])
            step_functions.append(mp.after_sources(harminv))
        t0 = time.perf_counter()
        if case.until is not None:
            case.sim.run(*step_functions, until=case.until)
        else:
            case.sim.run(*step_functions, until_after_sources=case.until_after_sources)
        run_s = time.perf_counter() - t0
        steps = int(case.sim.timestep())
        rec.update(steps=steps, meep_time=float(case.sim.meep_time()), run_s=run_s,
                   step_s=run_s / max(steps, 1), mcells_per_s=case.cells * steps / run_s / 1e6)
        field = np.asarray(case.sim.get_array(component=mp.Ez))
        spectra = [np.asarray(mp.get_fluxes(f), dtype=np.float64) for f in case.fluxes]
        rec["gpu"] = None
        rec["active_step_path"] = "solver"
    else:
        result, total, marks, series, harminv = _gpu_run(case, args, meep_gpu, mp,
                                                         prefer_gpu=args.leg != "reference", record=True)
        try:
            step_s, rate = _steady_rate(marks, case.cells)
            driver = result.driver
            report = driver.fast_path_report()
            rec.update(steps=int(result.steps), meep_time=float(result.meep_time), total_s=total,
                       loop_s=float(result.wall_time_s), step_s=step_s, mcells_per_s=rate,
                       loop_mcells_per_s=case.cells * result.steps / result.wall_time_s / 1e6,
                       gpu=driver.gpu, active_step_path=driver.active_step_path,
                       subnormal_policy=meep_gpu.get_subnormal_policy())
            rec["dispatch"] = _summary(report)
            rec["certified"] = (report or {}).get("certified")
            if args.full_report:
                rec["report"] = report
            field = np.asarray(result.get_array("Ez"))
            spectra = [np.asarray(result.get_flux_spectrum(f), dtype=np.float64) for f in case.fluxes]
        finally:
            result.close()

    field = np.ascontiguousarray(field)
    rec["field_shape"] = list(field.shape)
    rec["field_dtype"] = str(field.dtype)
    rec["field_max_abs"] = float(np.abs(field).max())
    rec["field_l2"] = float(np.linalg.norm(field.ravel().astype(np.float64)))
    rec["fluxes"] = [s.tolist() for s in spectra]
    rec["series_len"] = len(series)
    if harminv is not None:
        # The error of a mode comes back as a complex number from the run-loop facade of the add-on (measured: a
        # TypeError in float()), so the frequency, decay and Q are read as real parts, the amplitude and the error
        # as magnitudes.
        rec["harminv"] = [{"freq": float(np.real(m.freq)), "decay": float(np.real(m.decay)),
                           "Q": float(np.real(m.Q)), "amp": abs(complex(m.amp)), "err": abs(complex(m.err))}
                          for m in harminv.modes]
    if args.save:
        arrays = {"field": field, "series": np.asarray(series, dtype=np.float64)}
        arrays.update({n: s for n, s in zip(flux_names, spectra)})
        np.savez(args.save, **arrays)
    return rec


def main(argv=None) -> int:
    args = parse(argv)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    rec = run(args)
    sys.stdout.flush()
    print("ADDON_JSON " + json.dumps(rec, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
