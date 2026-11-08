# SPDX-License-Identifier: GPL-2.0-or-later
"""Shared helpers of the addon_gpu tests. Run them on the GPU computer only:

    bash tools/gpu_run.sh addon_gpu python -m pytest -q

The tests use the add-on that the environment imports (the pinned clone, with the patches of
tools/env/addon_patches/ applied by tools/env/addon_gpu.sh). ``ADDON_GPU_CLONE=<folder>`` puts another clone of the
add-on first on the Python path, in this process and in each leg process (a work copy with the patches applied).
Each leg runs in its own process (run_case.py), because a run that dispatches kernels installs the float32
subnormal policy of its kernel table for the whole process.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
REPO = PACKAGE.parent
PINNED_CLONE = REPO / ".local" / "upstream" / "erised-fork"
CLONE = os.environ.get("ADDON_GPU_CLONE")
if CLONE:
    sys.path.insert(0, str(Path(CLONE).resolve()))
sys.path.insert(0, str(PACKAGE))


def leg_env(extra: dict | None = None) -> dict:
    env = dict(os.environ)
    paths = [str(PACKAGE)]
    if CLONE:
        paths.insert(0, str(Path(CLONE).resolve()))
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    for key in [k for k in env if k.startswith("MEEP_GPU_")]:
        env.pop(key)                     # each leg sets its own switches (run_case.py --table)
    env.update(extra or {})
    return env


def parse_json(stdout: str) -> dict:
    for line in reversed(stdout.splitlines()):
        if line.startswith("ADDON_JSON "):
            return json.loads(line[len("ADDON_JSON "):])
    raise AssertionError("no ADDON_JSON line in the output:\n" + stdout[-4000:])


def run_script(args: list[str], timeout: float = 900, env_extra: dict | None = None) -> dict:
    proc = subprocess.run([sys.executable, *args], cwd=PACKAGE, env=leg_env(env_extra), capture_output=True,
                          text=True, timeout=timeout)
    if proc.returncode != 0:
        raise AssertionError(f"{args} exited {proc.returncode}:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}")
    return parse_json(proc.stdout)


def run_leg(tmp_path: Path, case: str, leg: str, *, table: str | None = None, policy: str | None = None,
            resolution: int | None = None, until: float | None = None, extra: list[str] | None = None,
            env_extra: dict | None = None, timeout: float = 900) -> tuple[dict, dict]:
    """Run one leg in its own process; return (the JSON record, the saved arrays)."""
    name = f"{case}-{leg}-{table or 'x'}-{policy or 'x'}-{resolution}-{until}"
    save = tmp_path / f"{name}.npz"
    args = ["run_case.py", "--case", case, "--leg", leg, "--save", str(save)]
    if table:
        args += ["--table", table]
    if policy:
        args += ["--policy", policy]
    if resolution:
        args += ["--resolution", str(resolution)]
    if until:
        args += ["--until", str(until)]
    args += extra or []
    rec = run_script(args, timeout=timeout, env_extra=env_extra)
    with np.load(save) as data:
        arrays = {k: np.array(data[k]) for k in data.files}
    return rec, arrays


def rel_l2(a, b) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    norm = np.linalg.norm(b)
    return float(np.linalg.norm(a - b) / norm) if norm > 0 else float(np.linalg.norm(a - b))


def differing_words(a, b) -> int:
    """How many float32 words differ, bit for bit (the arrays must have the same shape)."""
    a = np.ascontiguousarray(np.asarray(a, dtype=np.float32))
    b = np.ascontiguousarray(np.asarray(b, dtype=np.float32))
    assert a.shape == b.shape, (a.shape, b.shape)
    return int(np.count_nonzero(a.view(np.uint32) != b.view(np.uint32)))


@pytest.fixture(scope="session")
def legs_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("legs")
