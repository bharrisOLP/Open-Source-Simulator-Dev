# SPDX-License-Identifier: GPL-2.0-or-later
"""Run the test modules of the add-on one at a time on the GPU computer, and log one JSON line per module.

Run it on the GPU computer, from the repository folder, OUTSIDE the GPU lock (each chunk of modules takes the lock
itself through tools/gpu_run.sh, and each module has its own time limit):

    uv run --no-project --python 3.12 addon_gpu/suite_runner.py --list
    uv run --no-project --python 3.12 addon_gpu/suite_runner.py --run suite-any --start 0 --count 20
    uv run --no-project --python 3.12 addon_gpu/suite_runner.py --run suite-any --resume --max-seconds 3000
    uv run --no-project --python 3.12 addon_gpu/suite_runner.py --run suite-any --not-clean   (run them again)
    uv run --no-project --python 3.12 addon_gpu/suite_runner.py --run suite-any --summary     (totals by tier)

The modules are the ``test_*.py`` files under the pytest paths of the add-on (``meep_gpu`` and
``parity/meep_gpu``, without the folders that its ``norecursedirs`` excludes), in the clone
``.local/upstream/erised-fork`` (pinned commit, with the patches of tools/env/addon_patches/). They run in this
order:

1. ``host``: modules that do not name CuPy, Triton, PyTorch or kernel dispatch, and that hold no test of the
   certification suite of the add-on;
2. ``gpu``: modules that name one of them;
3. ``certification``: the ledger, weld and record contracts (``CERTIFICATION_FILES`` of the root conftest of the
   add-on) and the modules with a test in ``tools/ci/pending_certification.txt``. A module named on the command
   line runs its certification tests too (AGENTS.md of the add-on), so these modules come last.

Each module runs alone, as ``timeout -k 30 <--timeout> python -m pytest -q <module> --junitxml=<file>`` in the clone
(default 300 s; SIGTERM at the limit, SIGKILL 30 s later). The modules run in chunks: one call of
``bash tools/gpu_run.sh .local/upstream/erised-fork`` (one hold of the GPU lock) runs at most ``--chunk`` modules, one
after the other, and starts no new module after ``--chunk-seconds`` (default 240 s); then the next chunk waits for
the lock again, so other runs of this repository get the GPU between chunks. The runner prints the uptime and the
load before each chunk. With ``--hardware any`` (the default) the module runs with ``MEEP_GPU_HARDWARE=any``, the
switch of the add-on that admits this GPU for its kernel tables (uncertified); ``--hardware default`` leaves the
variable unset. The JSON line goes to ``.local/addon-runs/suite/<run>.jsonl`` and to the standard output; the pytest
output and the JUnit file of the module go to ``.local/addon-runs/suite/<run>/``. Each line holds the module, the
counts (passed, failed, skipped, errors; ``null`` after a time-out), the seconds of the test session and of the
whole call, the exit code, the failed tests with the first line of their messages, how many of them the add-on lists
as pending certification, the uptime and load of the computer before the module (a low uptime shows a restart), and
the ``CUPY_GPU_MEMORY_LIMIT`` and ``CUPY_CACHE_DIR`` of the pytest call (a folder in this repository relative to
it).

The counts come from the JUnit file, one outcome for each test case: a case with a failure counts as failed, else a
case with an error counts as an error, else skipped or passed. pytest's own summary can differ when a test also has
an error in its teardown (a failed test with a teardown error: pytest says "1 failed, 1 error"; a passed test with a
teardown error: "1 passed, 1 error", here 1 error). Whether a module is clean does not change.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parent
CLONE_REL = ".local/upstream/erised-fork"
CLONE = REPO / CLONE_REL
OUT = REPO / ".local" / "addon-runs" / "suite"
TEST_ROOTS = ("meep_gpu", "parity/meep_gpu")
# norecursedirs of the add-on (pyproject.toml) that can occur under its pytest paths
SKIP_DIRS = {"results", "_scratch", "examples", "benchmarks", "docs", "tools", "site", "build", "dist", "venv",
             "node_modules", "CVS", "_darcs", "{arch}"}
# the ledger, weld and record contracts: CERTIFICATION_FILES of the root conftest.py of the add-on
CERTIFICATION_FILES = (
    "meep_gpu/cuda_kernels/test_certification_metadata.py",
    "meep_gpu/cuda_kernels/test_certification_record.py",
    "meep_gpu/test_cuda_weld_contract.py",
    "meep_gpu/test_dispatch_contract.py",
    "meep_gpu/test_metal_weld_contract.py",
    "meep_gpu/test_regate_writers.py",
    "meep_gpu/test_triton_weld_contract.py",
)
GPU_WORDS = re.compile(r"\b(cupy|triton|torch|RawKernel|MEEP_GPU_DISPATCH|MEEP_GPU_HARDWARE|run_on_gpu)\b")
TIERS = ("host", "gpu", "certification")
MESSAGE_CHARS = 300
KILL_AFTER_S = 30            # timeout -k: SIGKILL this many seconds after the SIGTERM at the time limit
TIMEOUT_EXITS = (124, 137)   # the exit codes of timeout(1) at the limit: SIGTERM ended the call, or SIGKILL did
MAX_FAILURES = 40


def modules(clone: Path = CLONE) -> list[str]:
    found = []
    for root in TEST_ROOTS:
        for path in (clone / root).rglob("test_*.py"):
            rel = path.relative_to(clone)
            if any(part in SKIP_DIRS or part.startswith(".") or part.endswith(".egg") for part in rel.parts[:-1]):
                continue
            found.append(rel.as_posix())
    return sorted(found)


def pending_ids(clone: Path = CLONE) -> set[str]:
    path = clone / "tools" / "ci" / "pending_certification.txt"
    ids = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            ids.add(line)
    return ids


def module_id(module: str) -> str:
    """``meep_gpu/test_x.py`` -> ``meep_gpu.test_x``, the CLASSNAME prefix of the JUnit identifiers."""
    return module[:-3].replace("/", ".")


def tier(module: str, pending: set[str], clone: Path = CLONE) -> str:
    prefix = module_id(module) + "::"
    if module in CERTIFICATION_FILES or any(p.startswith(prefix) for p in pending):
        return "certification"
    text = (clone / module).read_text(errors="replace")
    return "gpu" if GPU_WORDS.search(text) else "host"


def ordered(clone: Path = CLONE) -> list[tuple[str, str]]:
    pending = pending_ids(clone)
    rows = [(tier(m, pending, clone), m) for m in modules(clone)]
    return sorted(rows, key=lambda r: (TIERS.index(r[0]), r[1]))


def is_pending(test_id: str, pending: set[str]) -> bool:
    return test_id in pending or test_id.split("[", 1)[0] in pending


def junit_counts(xml_path: Path, pending: set[str]) -> dict:
    root = ET.parse(xml_path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    counts = {"passed": 0, "failed": 0, "skipped": 0, "errors": 0}
    failures = []
    session_s = 0.0
    for suite in suites:
        session_s += float(suite.get("time", 0.0))
        for case in suite.iter("testcase"):
            cls, name = case.get("classname", ""), case.get("name", "")
            test_id = f"{cls}::{name}" if cls else name
            kinds = [child.tag for child in case if child.tag in ("failure", "error", "skipped")]
            if "failure" in kinds:
                kind = "failed"
            elif "error" in kinds:
                kind = "errors"
            elif "skipped" in kinds:
                kind = "skipped"
            else:
                kind = "passed"
            counts[kind] += 1
            if kind in ("failed", "errors"):
                node = case.find("failure") if kind == "failed" else case.find("error")
                message = (node.get("message") or node.text or "").strip().splitlines()
                failures.append({"test": test_id, "kind": kind, "pending": is_pending(test_id, pending),
                                 "message": (message[0] if message else "")[:MESSAGE_CHARS]})
    counts["session_s"] = round(session_s, 2)
    counts["failures"] = failures[:MAX_FAILURES]
    counts["pending_failed"] = sum(1 for f in failures if f["pending"])
    return counts


def row_environment(env: dict) -> dict:
    """The CuPy settings of a pytest call, for its row: a folder in this repository is written relative to it."""
    cache = env.get("CUPY_CACHE_DIR")
    if cache:
        path = Path(cache).resolve()
        if path == REPO or REPO in path.parents:
            cache = path.relative_to(REPO).as_posix()
    return {"cupy_gpu_memory_limit": env.get("CUPY_GPU_MEMORY_LIMIT"), "cupy_cache_dir": cache}


def machine_state() -> dict:
    with open("/proc/uptime") as fh:
        uptime_s = float(fh.read().split()[0])
    return {"uptime_s": round(uptime_s), "load1": round(os.getloadavg()[0], 2)}


def run_module(module: str, tier_name: str, run: str, args, pending: set[str]) -> dict:
    """Run one module with its own time limit. Called inside the GPU lock (``--inner``), in the clone folder."""
    # New names for each call: the runner deletes and overwrites no file on the GPU computer, and pytest gets a new
    # --basetemp folder, so it does not clean up the temporary folders of earlier sessions either. The --basetemp
    # folder is in the system temporary folder, not in this repository: a pytest run that a test starts in its
    # tmp_path would otherwise take the pyproject.toml of this repository as its rootdir, and its test identifiers
    # would change (parity/meep_gpu/test_conftest_evidence_marker.py failed so).
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = f"{module_id(module)}-{args.hardware}-{stamp}"
    folder = OUT / run
    folder.mkdir(parents=True, exist_ok=True)
    xml_path = folder / f"{slug}.xml"
    log_path = folder / f"{slug}.log"
    basetemp = Path(tempfile.gettempdir()) / "addon-suite" / run / slug
    basetemp.parent.mkdir(parents=True, exist_ok=True)   # pytest makes --basetemp, but not its parent folders
    env = hardware_env(args.hardware, args.cupy_cache_dir)
    cmd = ["timeout", "-k", str(KILL_AFTER_S), str(args.timeout), sys.executable, "-m", "pytest", "-q", module,
           f"--junitxml={xml_path}", f"--basetemp={basetemp}"]
    state = machine_state()
    started = dt.datetime.now().isoformat(timespec="seconds")
    t0 = time.perf_counter()
    with open(log_path, "x") as log:
        proc = subprocess.run(cmd, cwd=CLONE, env=env, stdout=log, stderr=subprocess.STDOUT)
    wall_s = time.perf_counter() - t0
    row = {"module": module, "tier": tier_name, "hardware": args.hardware, "started": started, **state,
           **row_environment(env),
           "exit": proc.returncode, "timed_out": proc.returncode in TIMEOUT_EXITS}
    if xml_path.exists():
        counts = junit_counts(xml_path, pending)
        row.update({k: counts[k] for k in ("passed", "failed", "skipped", "errors")})
        row["seconds"] = counts["session_s"]
        row["wall_s"] = round(wall_s, 2)
        row["pending_failed"] = counts["pending_failed"]
        row["failures"] = counts["failures"]
    else:
        row.update({"passed": None, "failed": None, "skipped": None, "errors": None, "seconds": None,
                    "wall_s": round(wall_s, 2), "pending_failed": None, "failures": []})
        tail = log_path.read_text(errors="replace").strip().splitlines()[-3:]
        row["tail"] = [line[:MESSAGE_CHARS] for line in tail]
    deselected = re.search(r"(\d+) deselected", log_path.read_text(errors="replace"))
    row["deselected"] = int(deselected.group(1)) if deselected else 0
    return row


def hardware_env(hardware: str, cupy_cache_dir: str | None = None) -> dict:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.pop("MEEP_GPU_HARDWARE", None)
    if hardware == "any":
        env["MEEP_GPU_HARDWARE"] = "any"
    if cupy_cache_dir:
        env["CUPY_CACHE_DIR"] = cupy_cache_dir
    return env


def stop_requested(stop_file: Path, since: float) -> bool:
    """True when the stop file was written after ``since`` (time.time()); the runner never deletes it."""
    try:
        return stop_file.stat().st_mtime > since
    except OSError:
        return False


def last_rows(jsonl: Path, hardware: str) -> dict[str, dict]:
    """The last logged row of each module of the run, for one ``--hardware`` setting."""
    rows: dict[str, dict] = {}
    if not jsonl.exists():
        return rows
    for line in jsonl.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("hardware") == hardware:
            rows[row["module"]] = row
    return rows


#: pytest's exit code when it ran no test: a module that skips at module level by a declared resource of the add-on
#: (``requires_resource_skip(..., allow_module_level=True)``, for example the Metal modules without an Apple GPU)
NO_TESTS_RAN = 5


def clean(row: dict) -> bool:
    """No failure and no error: exit 0, or exit 5 for a module that only skipped (a declared module-level skip)."""
    if row.get("failed") or row.get("errors"):
        return False
    return row["exit"] == 0 or (row["exit"] == NO_TESTS_RAN and bool(row.get("skipped"))
                                and not row.get("passed"))


def summary(rows: dict[str, dict], order: list[tuple[str, str]]) -> None:
    """Totals by tier from the last row of each module, then one line per module that is not clean."""
    print(f"{'tier':13s} {'modules':>7s} {'run':>4s} {'clean':>5s} {'not':>4s} {'passed':>7s} {'failed':>6s} "
          f"{'skipped':>7s} {'errors':>6s} {'pending':>7s} {'timeout':>7s}")
    for name in (*TIERS, "all"):
        mods = [m for t, m in order if name in (t, "all")]
        ran = [rows[m] for m in mods if m in rows]
        total = {k: sum(r.get(k) or 0 for r in ran) for k in ("passed", "failed", "skipped", "errors",
                                                                "pending_failed")}
        print(f"{name:13s} {len(mods):7d} {len(ran):4d} {sum(clean(r) for r in ran):5d} "
              f"{sum(not clean(r) for r in ran):4d} {total['passed']:7d} {total['failed']:6d} "
              f"{total['skipped']:7d} {total['errors']:6d} {total['pending_failed']:7d} "
              f"{sum(bool(r['timed_out']) for r in ran):7d}")
    for t, m in order:
        r = rows.get(m)
        if r is not None and not clean(r):
            print(f"NOT CLEAN {m} ({t}): passed {r['passed']}, failed {r['failed']}, skipped {r['skipped']}, "
                  f"errors {r['errors']}, pending {r['pending_failed']}, exit {r['exit']}, {r['seconds']} s")


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--list", action="store_true", help="print the ordered modules with their tier, and stop")
    p.add_argument("--run", default="suite", help="name of the run: .local/addon-runs/suite/<run>.jsonl")
    p.add_argument("--tier", choices=TIERS, action="append", help="only modules of this tier (repeatable)")
    p.add_argument("--modules", nargs="*", help="these modules (paths in the clone), in this order")
    p.add_argument("--start", type=int, default=0, help="index of the first module in the selected order")
    p.add_argument("--count", type=int, default=None, help="at most this many modules")
    p.add_argument("--resume", action="store_true", help="skip the modules that the run already logged")
    p.add_argument("--not-clean", action="store_true",
                   help="only the modules whose last row in the run has a failure, an error or a non-zero exit")
    p.add_argument("--summary", action="store_true", help="print the totals of the run by tier, and stop")
    p.add_argument("--timeout", type=int, default=300,
                   help="time limit of the pytest call of each module, in seconds (timeout -k 30 inside the chunk)")
    p.add_argument("--hardware", choices=("any", "default"), default="any",
                   help="any: MEEP_GPU_HARDWARE=any (the kernel tables admit this GPU); default: unset")
    p.add_argument("--max-seconds", type=float, default=None,
                   help="start no new chunk after this many seconds of this call")
    p.add_argument("--chunk", type=int, default=20, help="at most this many modules in one hold of the GPU lock")
    p.add_argument("--chunk-seconds", type=float, default=240,
                   help="in one hold of the GPU lock, start no new module after this many seconds")
    p.add_argument("--inner", action="store_true",
                   help="(used by the runner itself) run --modules here, inside the GPU lock, in this process")
    p.add_argument("--stop-after-timeouts", type=int, default=2,
                   help="stop after this many modules in a row that reach the time limit")
    p.add_argument("--cupy-cache-dir", default=None,
                   help="CUPY_CACHE_DIR for the modules (the add-on refuses a 'keep' install by hand unless the "
                        "folder name carries 'ftz_stripped', and a 'flush' install if it does)")
    p.add_argument("--stop-file", default=str(OUT / "STOP"),
                   help="touch this file to stop the run after the current module (only a write after the start "
                        "of the run counts)")
    p.add_argument("--started", type=float, default=None, help=argparse.SUPPRESS)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse(argv)
    if not CLONE.is_dir():
        print(f"suite_runner: no clone {CLONE_REL} (run tools/env/addon_gpu.sh)", file=sys.stderr)
        return 2
    rows = ordered()
    if args.list:
        for i, (t, m) in enumerate(rows):
            print(f"{i:4d} {t:13s} {m}")
        counts = {t: sum(1 for r in rows if r[0] == t) for t in TIERS}
        print("modules:", len(rows), counts)
        return 0
    tiers = dict((m, t) for t, m in rows)
    if args.modules:
        unknown = [m for m in args.modules if m not in tiers]
        if unknown:
            print(f"suite_runner: not a test module of the pytest paths: {unknown}", file=sys.stderr)
            return 2
        selected = list(args.modules)
    else:
        selected = [m for t, m in rows if not args.tier or t in args.tier]
    selected = selected[args.start:]
    jsonl = OUT / f"{args.run}.jsonl"
    logged = last_rows(jsonl, args.hardware)
    if args.summary:
        summary(logged, rows)
        return 0
    if args.resume:
        selected = [m for m in selected if m not in logged]
    if args.not_clean:
        selected = [m for m in selected if m in logged and not clean(logged[m])]
    if args.count is not None:
        selected = selected[:args.count]
    OUT.mkdir(parents=True, exist_ok=True)
    if args.inner:
        return inner(selected, tiers, jsonl, args)
    return outer(selected, jsonl, args)


def inner(selected: list[str], tiers: dict[str, str], jsonl: Path, args) -> int:
    """Run the modules one at a time in this process (inside one hold of the GPU lock); log a row for each."""
    pending = pending_ids()
    t0 = time.perf_counter()
    timeouts_in_a_row = 0
    since = args.started if args.started is not None else time.time()
    for i, module in enumerate(selected, 1):
        if stop_requested(Path(args.stop_file), since):
            print(f"suite_runner: stop file {args.stop_file} written; {len(selected) - i + 1} modules not started",
                  flush=True)
            return 4
        if time.perf_counter() - t0 > args.chunk_seconds:
            print(f"suite_runner: chunk budget used after {time.perf_counter() - t0:.0f} s; "
                  f"{len(selected) - i + 1} modules go to the next chunk", flush=True)
            break
        print(f"[{i}/{len(selected)}] {module} ({tiers[module]}) ...", flush=True)
        row = run_module(module, tiers[module], args.run, args, pending)
        with open(jsonl, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        print("SUITE_JSON " + json.dumps({k: row[k] for k in ("module", "passed", "failed", "skipped", "errors",
                                                               "seconds", "exit", "pending_failed", "uptime_s",
                                                               "load1")}), flush=True)
        timeouts_in_a_row = timeouts_in_a_row + 1 if row["timed_out"] else 0
        if timeouts_in_a_row >= args.stop_after_timeouts:
            print(f"suite_runner: {timeouts_in_a_row} modules in a row reached the time limit; stopped", flush=True)
            return 3
    return 0


def outer(selected: list[str], jsonl: Path, args) -> int:
    """Give the modules to ``inner`` in chunks, one hold of the GPU lock (tools/gpu_run.sh) for each chunk."""
    t0 = time.perf_counter()
    started = time.time()
    remaining = list(selected)
    while remaining:
        if stop_requested(Path(args.stop_file), started):
            print(f"suite_runner: stop file {args.stop_file} written; {len(remaining)} modules not started",
                  flush=True)
            return 4
        if args.max_seconds is not None and time.perf_counter() - t0 > args.max_seconds:
            print(f"suite_runner: stopped after {time.perf_counter() - t0:.0f} s; {len(remaining)} modules not "
                  f"started", flush=True)
            break
        chunk = remaining[:args.chunk]
        state = machine_state()
        print(f"suite_runner: chunk of {len(chunk)} modules ({len(remaining)} left); uptime {state['uptime_s']} s, "
              f"load {state['load1']}", flush=True)
        before = len(jsonl.read_text().splitlines()) if jsonl.exists() else 0
        env = hardware_env(args.hardware, args.cupy_cache_dir)
        # the time limit of the whole hold: the budget, then at most one module that started just before its end
        env["GPU_RUN_TIMEOUT"] = str(int(args.chunk_seconds + args.timeout + 120))
        cmd = ["bash", "tools/gpu_run.sh", CLONE_REL, "python", str(PACKAGE / "suite_runner.py"), "--inner",
               "--run", args.run, "--hardware", args.hardware, "--timeout", str(args.timeout),
               "--chunk-seconds", str(args.chunk_seconds), "--stop-after-timeouts", str(args.stop_after_timeouts),
               "--stop-file", args.stop_file, "--started", repr(started),
               *(["--cupy-cache-dir", args.cupy_cache_dir] if args.cupy_cache_dir else []), "--modules", *chunk]
        proc = subprocess.run(cmd, cwd=REPO, env=env)
        lines = jsonl.read_text().splitlines()[before:] if jsonl.exists() else []
        logged = {json.loads(line)["module"] for line in lines}
        if not logged:
            print(f"suite_runner: a chunk logged no module (exit {proc.returncode}); stopped", flush=True)
            return 1
        remaining = [m for m in remaining if m not in logged]
        if proc.returncode in (3, 4):
            return proc.returncode           # time limits in a row, or the stop file (the inner call says which)
        if proc.returncode != 0:
            print(f"suite_runner: a chunk ended with exit {proc.returncode}; stopped", flush=True)
            return proc.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
