# SPDX-License-Identifier: GPL-2.0-or-later
"""The ROCm patches of tools/env/addon_patches/: they apply to the pinned clone, and they do what they say."""

from __future__ import annotations

import subprocess

import pytest

from conftest import PINNED_CLONE, REPO, run_script

PATCHES = sorted((REPO / "tools" / "env" / "addon_patches").glob("*.patch"))


def test_there_are_patches_and_a_licence():
    assert [p.name for p in PATCHES][:2] == ["0001-hiprtc-compile-options.patch", "0002-hip-keep-subnormals.patch"]
    assert (REPO / "tools" / "env" / "addon_patches" / "LICENSE").read_text().startswith(
        (PINNED_CLONE / "LICENSE").read_text()[:200])


@pytest.mark.parametrize("patch", PATCHES, ids=lambda p: p.name)
def test_each_patch_applies_to_the_pinned_clone_or_is_applied(patch):
    """tools/env/addon_gpu.sh applies a patch when ``git apply --check`` passes; once applied, the reverse passes."""
    forward = subprocess.run(["git", "-C", str(PINNED_CLONE), "apply", "--check", str(patch)], capture_output=True)
    reverse = subprocess.run(["git", "-C", str(PINNED_CLONE), "apply", "--reverse", "--check", str(patch)],
                             capture_output=True)
    assert forward.returncode == 0 or reverse.returncode == 0, forward.stderr.decode() + reverse.stderr.decode()


def test_hip_options_translates_only_the_nvrtc_fmad_spellings():
    from meep_gpu import backends

    assert backends.hip_options(("--fmad=false",)) == ("-ffp-contract=off",)
    assert backends.hip_options(("-fmad=true", "-I/include", "-DX=1")) == ("-ffp-contract=fast", "-I/include",
                                                                          "-DX=1")
    assert backends.hip_options(()) == ()


def test_guard_wraps_the_rocm_compile_path_once():
    import cupy

    from meep_gpu import backends

    assert backends.cupy_is_hip(cupy), "these tests are for a ROCm build of CuPy"
    compiler = cupy.cuda.compiler
    backends.guard_kernel_compilation(cupy)
    first = compiler._compile_with_cache_hip
    backends.guard_kernel_compilation(cupy)
    assert compiler._compile_with_cache_hip is first
    assert getattr(first, backends._HIP_OPTIONS_MARK, False)


def test_rawkernel_with_the_fmad_option_compiles_and_runs_on_the_amd_gpu():
    import cupy
    import numpy as np

    from meep_gpu import backends

    backends.guard_kernel_compilation(cupy)
    code = r'''extern "C" __global__ void axpy(const float* x, const float* y, float* out, float a, int n) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i < n) out[i] = a * x[i] + y[i];
    }'''
    kernel = cupy.RawKernel(code, "axpy", options=("--fmad=false",))
    x = cupy.arange(1000, dtype=cupy.float32)
    y = cupy.ones(1000, dtype=cupy.float32)
    out = cupy.empty_like(x)
    kernel((4,), (256,), (x, y, out, np.float32(0.5), np.int32(1000)))
    np.testing.assert_array_equal(cupy.asnumpy(out), 0.5 * np.arange(1000, dtype=np.float32) + 1)


def test_keep_policy_holds_on_every_cupy_executor(tmp_path):
    """The gate of the add-on, keep policy, host and CuPy legs: every cell must agree (verdict PASS)."""
    import json
    import os
    import sys

    clone = os.environ.get("ADDON_GPU_CLONE") or str(PINNED_CLONE)
    out = tmp_path / "gate.json"
    legs = "host,cupy_ufunc,cupy_elementwise,cupy_rawkernel,cupy_reduction"
    proc = subprocess.run([sys.executable, "subnormal_gate.py", clone, "--policy", "keep", "--legs", legs,
                           "--out", str(out)], cwd=str(REPO / "addon_gpu"), capture_output=True, text=True,
                          timeout=600, env={**os.environ, "PYTHONPATH": clone})
    record = json.loads(out.read_text())
    assert record["verdict"] == "PASS", record["failures"]
    cells = sum(len(row) for row in record["matrix"].values())
    assert cells >= 37, cells
    assert proc.returncode == 0


@pytest.mark.parametrize("policy", ["none", "keep"])
def test_triton_keeps_subnormals(policy):
    rec = run_script(["triton_subnormal_probe.py", "--policy", policy], timeout=600)
    assert [row["verdict"] for row in rec["rows"]] == ["keep"] * 6, rec["rows"]
    assert all(row["all_lanes_equal"] for row in rec["rows"])


def test_triton_does_not_flush_under_a_requested_flush():
    """A known limit on this GPU, kept as a test so that a change is seen: Triton for AMD keeps subnormal values
    under the add-on's flush policy (its flush mechanism is the NVIDIA one). The NVIDIA tables run under keep."""
    rec = run_script(["triton_subnormal_probe.py", "--policy", "flush"], timeout=600)
    assert rec["policy_stamp"]["resolved"] == "flush"
    assert [row["verdict"] for row in rec["rows"]] == ["keep"] * 6, rec["rows"]
