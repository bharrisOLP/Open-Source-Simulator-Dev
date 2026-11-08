# SPDX-License-Identifier: GPL-2.0-or-later
"""suite_runner.py: it selects the test modules that the add-on collects, sorts them by the certification suite of
the add-on, and reads the counts of a JUnit report. These tests use no GPU."""

from __future__ import annotations

import ast
import json
import tomllib

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
    assert set(suite_runner.TIMEOUT_EXITS) == {124, 137}
