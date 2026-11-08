# SPDX-License-Identifier: GPL-2.0-or-later
"""suite_runner.py: it selects the test modules that the add-on collects, sorts them by the certification suite of
the add-on, reads the counts of a JUnit report, gives each module its CuPy cache folder by the rule of the add-on, and
limits the time of a module with GNU timeout. These tests use no GPU (the GPU computer only: /proc, GNU timeout)."""

from __future__ import annotations

import ast
import json
import os
import signal
import subprocess
import sys
import time
import tomllib
import types

import pytest

import suite_runner
from conftest import PINNED_CLONE

# The test modules in the pytest paths of the add-on at the pinned commit 8993426 (counted on 2026-11-08)
MODULES_BY_FOLDER = {"meep_gpu": 225, "meep_gpu/cuda_kernels": 66, "parity/meep_gpu": 113,
                     "parity/meep_gpu/rounds/1.0": 5}


def test_the_modules_are_those_of_the_pytest_paths_of_the_addon():
    mods = suite_runner.modules(PINNED_CLONE)
    assert len(mods) == len(set(mods)) == sum(MODULES_BY_FOLDER.values()) == 409
    by_folder: dict[str, int] = {}
    for m in mods:
        folder = m.rsplit("/", 1)[0]
        by_folder[folder] = by_folder.get(folder, 0) + 1
    assert by_folder == MODULES_BY_FOLDER
    assert not [m for m in mods if "/results/" in m or "/_scratch/" in m]


def test_the_pytest_paths_and_excluded_folders_are_those_of_the_addon():
    config = tomllib.loads((PINNED_CLONE / "pyproject.toml").read_text())["tool"]["pytest"]["ini_options"]
    assert tuple(config["testpaths"]) == suite_runner.TEST_ROOTS
    for pattern in config["norecursedirs"]:
        name = pattern.rsplit("/", 1)[-1]
        if name in (".*", "*.egg"):
            continue                       # dot folders and eggs: suite_runner.modules tests them by name
        assert name in suite_runner.SKIP_DIRS, pattern


def test_the_certification_files_are_those_of_the_conftest_of_the_addon():
    tree = ast.parse((PINNED_CLONE / "conftest.py").read_text())
    values = [node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == "CERTIFICATION_FILES" for t in node.targets)]
    assert len(values) == 1
    assert ast.literal_eval(values[0]) == suite_runner.CERTIFICATION_FILES


def test_every_module_with_a_test_of_the_certification_suite_comes_last():
    order = suite_runner.ordered(PINNED_CLONE)
    pending = suite_runner.pending_ids(PINNED_CLONE)
    assert pending, "the add-on lists tests that wait on its certification round"
    tiers = [t for t, _ in order]
    assert tiers == sorted(tiers, key=suite_runner.TIERS.index)
    certification = {m for t, m in order if t == "certification"}
    assert set(suite_runner.CERTIFICATION_FILES) <= certification
    for test_id in pending:
        module = test_id.split("::", 1)[0].replace(".", "/") + ".py"
        if (PINNED_CLONE / module).exists():
            assert module in certification, test_id


JUNIT = """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest" errors="1" failures="2" skipped="1" tests="6" time="3.25">
<testcase classname="meep_gpu.test_x" name="test_ok" time="0.1"/>
<testcase classname="meep_gpu.test_x" name="test_ok2[a]" time="0.1"/>
<testcase classname="meep_gpu.test_x" name="test_bad[b]" time="0.1">
<failure message="AssertionError: 1 != 2&#10;second line">trace</failure></testcase>
<testcase classname="meep_gpu.test_x" name="test_listed[c]" time="0.1"><failure message="drift">t</failure></testcase>
<testcase classname="meep_gpu.test_x" name="test_setup" time="0.1"><error message="failed on setup">t</error>
</testcase>
<testcase classname="meep_gpu.test_x" name="test_skip" time="0.0"><skipped message="[requires_resource]"/>
</testcase>
</testsuite></testsuites>
"""


def test_the_junit_counts_and_the_pending_tests(tmp_path):
    xml = tmp_path / "report.xml"
    xml.write_text(JUNIT)
    counts = suite_runner.junit_counts(xml, {"meep_gpu.test_x::test_listed"})
    assert {k: counts[k] for k in ("passed", "failed", "skipped", "errors")} == {"passed": 2, "failed": 2,
                                                                                 "skipped": 1, "errors": 1}
    assert counts["session_s"] == pytest.approx(3.25)
    assert counts["pending_failed"] == 1
    assert [(f["test"], f["kind"], f["pending"], f["message"]) for f in counts["failures"]] == [
        ("meep_gpu.test_x::test_bad[b]", "failed", False, "AssertionError: 1 != 2"),
        ("meep_gpu.test_x::test_listed[c]", "failed", True, "drift"),
        ("meep_gpu.test_x::test_setup", "errors", False, "failed on setup"),
    ]


def test_the_last_row_of_a_module_counts_and_a_clean_row_has_no_failure(tmp_path):
    jsonl = tmp_path / "run.jsonl"
    rows = [{"module": "a.py", "hardware": "any", "exit": 1, "failed": 0, "errors": 3},
            {"module": "a.py", "hardware": "any", "exit": 0, "failed": 0, "errors": 0},
            {"module": "b.py", "hardware": "default", "exit": 1, "failed": 1, "errors": 0}]
    jsonl.write_text("\n".join(json.dumps(r) for r in rows) + "\nnot json\n")
    last = suite_runner.last_rows(jsonl, "any")
    assert list(last) == ["a.py"] and suite_runner.clean(last["a.py"])
    assert not suite_runner.clean(suite_runner.last_rows(jsonl, "default")["b.py"])
    assert not suite_runner.clean({"exit": 124, "failed": None, "errors": None})


def test_a_declared_module_skip_is_clean_and_an_empty_run_is_not():
    assert suite_runner.clean({"exit": 5, "passed": 0, "failed": 0, "skipped": 1, "errors": 0})
    assert not suite_runner.clean({"exit": 5, "passed": 0, "failed": 0, "skipped": 0, "errors": 0})
    assert not suite_runner.clean({"exit": 5, "passed": 0, "failed": 0, "skipped": 1, "errors": 1})


def test_only_a_stop_file_written_after_the_start_stops_the_run(tmp_path):
    stop = tmp_path / "STOP"
    assert not suite_runner.stop_requested(stop, since=0.0)
    stop.write_text("")
    written = stop.stat().st_mtime
    assert suite_runner.stop_requested(stop, since=written - 1.0)
    assert not suite_runner.stop_requested(stop, since=written + 1.0)


def test_a_row_records_the_cupy_settings_with_a_repository_folder_relative():
    repo = suite_runner.REPO
    env = {"CUPY_GPU_MEMORY_LIMIT": "8589934592", "CUPY_CACHE_DIR": str(repo / ".local" / "cupy-cache")}
    assert suite_runner.row_environment(env) == {"cupy_gpu_memory_limit": "8589934592",
                                                 "cupy_cache_dir": ".local/cupy-cache"}
    assert suite_runner.row_environment({}) == {"cupy_gpu_memory_limit": None, "cupy_cache_dir": None}
    outside = "/var/cache/cupy-elsewhere"
    assert suite_runner.row_environment({"CUPY_CACHE_DIR": outside})["cupy_cache_dir"] == outside


def test_a_kill_after_the_time_limit_counts_as_a_time_out():
    assert suite_runner.KILL_AFTER_S > 0
    assert set(suite_runner.TIMEOUT_EXITS) == {124, 137, -9}


# The two modules of the pinned commit that install the policy 'keep' in a module fixture and set no cache folder: with
# the cache folder of the environment, each test of them had an error ("CUPY_CACHE_DIR ... does not carry the policy
# token"; benchmarks/2026-11-08/addon-suite-any-0004m.jsonl: 83 and 14 errors)
KEEP_BY_HAND_MODULES = ["meep_gpu/test_triton_complex_no_pml_curl.py",
                        "meep_gpu/test_triton_complex_offdiag_update_e.py"]


def test_only_the_modules_that_install_keep_by_hand_get_the_keep_cache_folder():
    keep = [m for m in suite_runner.modules(PINNED_CLONE) if suite_runner.cache_policy(m, PINNED_CLONE) == "keep"]
    assert keep == KEEP_BY_HAND_MODULES
    assert suite_runner.keeps_by_hand('install_subnormal_policy("keep")')
    assert not suite_runner.keeps_by_hand('install_subnormal_policy("keep")\nmonkeypatch.setenv("CUPY_CACHE_DIR", x)')
    assert not suite_runner.keeps_by_hand("install_subnormal_policy('keep')\ninstall_subnormal_policy('flush')")
    assert not suite_runner.keeps_by_hand('install_subnormal_policy("keep")\n'
                                          'os.environ.setdefault("MEEP_GPU_SUBNORMAL_POLICY", "flush")')
    assert not suite_runner.keeps_by_hand("install_subnormal_policy()")


def test_sixty_modules_name_the_cache_folder_or_the_policy():
    named = [m for m in suite_runner.modules(PINNED_CLONE) if suite_runner.names_cache_or_policy(m, PINNED_CLONE)]
    assert len(named) == 60
    assert set(KEEP_BY_HAND_MODULES) <= set(named)
    args = suite_runner.parse(["--cache-or-policy", "--modules", "meep_gpu/test_backends.py"])
    assert args.cache_or_policy and args.cache_policy == "module"


def test_the_keep_cache_folder_follows_the_rule_of_the_addon():
    from meep_gpu import subnormal_policy as sp

    assert suite_runner.KEEP_TOKEN == sp.CUPY_CACHE_POLICY_TOKEN
    assert suite_runner.KEEP_MARKER_SUFFIX == sp.KEEP_POLICY_MARKER_SUFFIX
    for base in ("/w/.local/cupy-cache", "/w/.local/cupy-cache/", "/w/c-ftz_stripped", "/w/ftz_stripped/c"):
        folder = suite_runner.keep_cache_dir(base)
        assert folder == sp.keep_policy_cache_dir(base), base
        assert sp.cupy_cache_reasons(sp.KEEP, folder) == []
        assert sp.keep_policy_cache_marker(folder) == folder + suite_runner.KEEP_MARKER_SUFFIX


def _cache_args(**kw):
    import argparse

    return argparse.Namespace(**{"cupy_cache_dir": None, "cache_policy": "module", **kw})


def test_a_keep_module_gets_a_keep_folder_that_the_addon_minted(tmp_path):
    base = str(tmp_path / "cupy-cache")
    keep = base + "-ftz_stripped"
    assert suite_runner.module_cache_dir("plain", _cache_args(), base) == (None, "plain", None)
    assert suite_runner.module_cache_dir("keep", _cache_args(cache_policy="plain"), base) == (None, "plain", None)
    assert suite_runner.module_cache_dir("keep", _cache_args(cupy_cache_dir="/x"), base) == ("/x", "given", None)
    assert suite_runner.module_cache_dir("plain", _cache_args(cupy_cache_dir="/x"), base) == ("/x", "given", None)
    folder, label, note = suite_runner.module_cache_dir("keep", _cache_args(), base)
    assert (folder, label) == (keep, "keep") and "minted" in note
    marker = json.loads((tmp_path / "cupy-cache-ftz_stripped.meep_gpu_keep_policy").read_text())
    assert marker["written_by"] == "meep_gpu.subnormal_policy" and marker["directory"] == keep
    assert suite_runner.module_cache_dir("keep", _cache_args(), base) == (keep, "keep", None)   # marked now


def test_a_keep_folder_that_the_addon_refuses_is_not_used(tmp_path):
    base = str(tmp_path / "cupy-cache")
    unmarked = tmp_path / "cupy-cache-ftz_stripped"
    unmarked.mkdir()
    (unmarked / "an-entry").write_text("")              # files of unknown provenance, no marker
    folder, label, note = suite_runner.module_cache_dir("keep", _cache_args(), base)
    assert (folder, label) == (None, "plain")
    assert note.startswith("the add-on refused the keep folder") and "marker" in note
    assert not suite_runner.keep_folder_minted(str(unmarked))
    folder, label, note = suite_runner.module_cache_dir("keep", _cache_args(), None)
    assert (folder, label) == (None, "plain") and "no CUPY_CACHE_DIR" in note


def test_the_module_time_limit_is_gnu_timeout():
    program = suite_runner.timeout_program()
    version = subprocess.run([program, "--version"], capture_output=True, text=True, timeout=30).stdout
    assert "GNU coreutils" in version, (program, version.splitlines()[:1])


GRANDCHILD = """
import signal, subprocess, sys, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
child = subprocess.Popen([sys.executable, "-c",
                          "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(20)"])
open(sys.argv[1], "w").write(str(child.pid))
time.sleep(20)
"""


def _gone(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/status") as fh:
            return any(line.startswith("State:") and "Z" in line.split()[1] for line in fh)   # a zombie: ended
    except FileNotFoundError:
        return True


def test_the_time_limit_ends_a_process_that_the_module_started(tmp_path):
    """The command of a module ignores SIGTERM and starts a process that ignores SIGTERM too: GNU timeout sends
    SIGKILL to the process group KILL_AFTER seconds after the limit, so both end, and GNU timeout too (it is in the
    group: return code -9, which counts as a time-out). The uutils timeout sends SIGKILL only to its direct child and
    exits 137, and the second process would run on (here for 20 s)."""
    pid_file = tmp_path / "pid"
    t0 = time.monotonic()
    proc = subprocess.run([suite_runner.timeout_program(), "-k", "1", "2", sys.executable, "-c", GRANDCHILD,
                           str(pid_file)], timeout=60)
    assert proc.returncode == -signal.SIGKILL and proc.returncode in suite_runner.TIMEOUT_EXITS
    assert time.monotonic() - t0 < 10
    pid = int(pid_file.read_text())
    deadline = time.monotonic() + 5
    while not _gone(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert _gone(pid), f"process {pid} that the command started still runs after the time limit"


def test_a_row_names_the_cache_policy_and_the_timeout_program(tmp_path, monkeypatch):
    """run_module (with a stand-in for the pytest call): the row records the cache policy, the folder of the call and
    the timeout program; a keep module gets the keep folder in the environment of its call."""
    calls = []

    def fake_run(cmd, cwd, env, stdout, stderr):
        calls.append((cmd, env))
        return subprocess.CompletedProcess(cmd, 0)

    base = str(tmp_path / "cupy-cache")
    (tmp_path / "cupy-cache-ftz_stripped").mkdir()
    (tmp_path / "cupy-cache-ftz_stripped.meep_gpu_keep_policy").write_text("{}")
    monkeypatch.setenv("CUPY_CACHE_DIR", base)
    monkeypatch.setattr(suite_runner, "OUT", tmp_path / "out")
    monkeypatch.setattr(suite_runner, "subprocess", types.SimpleNamespace(run=fake_run, STDOUT=subprocess.STDOUT))
    monkeypatch.setattr(suite_runner, "tempfile", types.SimpleNamespace(gettempdir=lambda: str(tmp_path / "tmp")))
    args = _cache_args(hardware="any", timeout=300)
    keep_row = suite_runner.run_module(KEEP_BY_HAND_MODULES[0], "gpu", "t", args, set())
    plain_row = suite_runner.run_module("meep_gpu/test_backends.py", "gpu", "t", args, set())
    assert calls[0][1]["CUPY_CACHE_DIR"] == base + "-ftz_stripped" and calls[1][1]["CUPY_CACHE_DIR"] == base
    assert (keep_row["cache_policy"], keep_row["cupy_cache_dir"]) == ("keep", base + "-ftz_stripped")
    assert (plain_row["cache_policy"], plain_row["cupy_cache_dir"]) == ("plain", base)
    assert calls[0][0][0] == suite_runner.timeout_program() and calls[0][0][1:4] == ["-k", "30", "300"]
    assert keep_row["timeout_program"] == os.path.basename(suite_runner.timeout_program())
