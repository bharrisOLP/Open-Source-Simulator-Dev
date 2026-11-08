# SPDX-License-Identifier: GPL-2.0-or-later
"""The ROCm patches of tools/env/addon_patches/: they apply to the pinned clone, and they do what they say."""

from __future__ import annotations

import subprocess

import pytest

from conftest import PINNED_CLONE, REPO, run_script

PATCHES = sorted((REPO / "tools" / "env" / "addon_patches").glob("*.patch"))


def test_there_are_patches_and_a_licence():
    assert [p.name for p in PATCHES] == ["0001-hiprtc-compile-options.patch", "0002-hip-keep-subnormals.patch",
                                         "0003-bulk-chi1inv-read-double.patch",
                                         "0004-hip-nan-subtrahend-deposit.patch"]
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


def test_fmad_false_turns_off_the_fused_multiply_add_on_the_amd_gpu():
    """a * b + c with a = b = 1 + 2^-12 and c = -(1 + 2^-11), in float32.

    The exact a * b is 1 + 2^-11 + 2^-24. Rounded to float32 first (no fused multiply-add) it is 1 + 2^-11 (a tie,
    to even), and the sum is 0. Fused, the sum is 2^-24. hipRTC fuses by default, so the operands show whether the
    translated option (-ffp-contract=off) reaches the compiler.
    """
    import cupy
    import numpy as np

    from meep_gpu import backends

    backends.guard_kernel_compilation(cupy)
    code = r'''extern "C" __global__ void muladd(const float* a, const float* b, const float* c, float* out, int n) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i < n) out[i] = a[i] * b[i] + c[i];
    }'''
    n = 256
    a = cupy.full(n, 1 + 2.0 ** -12, dtype=cupy.float32)
    c = cupy.full(n, -(1 + 2.0 ** -11), dtype=cupy.float32)
    results = {}
    for options in ((), ("--fmad=false",), ("--fmad=true",)):
        out = cupy.empty(n, dtype=cupy.float32)
        cupy.RawKernel(code, "muladd", options=options)((1,), (n,), (a, a, c, out, np.int32(n)))
        results[options] = cupy.asnumpy(out)
    np.testing.assert_array_equal(results[("--fmad=false",)], np.zeros(n, dtype=np.float32))
    np.testing.assert_array_equal(results[()], np.full(n, 2.0 ** -24, dtype=np.float32))
    np.testing.assert_array_equal(results[("--fmad=true",)], np.full(n, 2.0 ** -24, dtype=np.float32))


def test_the_keep_token_reaches_the_cache_key():
    """Patch 0002: on CuPy 14 the keep token in the cache key is the only barrier against a stale flush binary."""
    rec = run_script(["keep_key_probe.py"], timeout=600)
    assert rec["is_hip"] and rec["policy"] == "keep", rec
    assert rec["before_keys"] >= 1 and rec["before_with_token"] == 0, rec
    assert rec["after_keys"] >= 1 and rec["after_with_token"] == rec["after_keys"], rec


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


# Patch 0004: the stored float32 words and the float64 subtrahends of the deposit door, as bit patterns. Normals,
# both zeros, both infinities, quiet NaN words of both signs and one with a payload. No subnormal stored word: this
# process installs no subnormal policy, so a ROCm CuPy flushes it (patch 0002 and its tests are about that).
_STORED_WORDS = (0x3FC00000, 0xC0100000, 0x00000000, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000, 0xFFC00000,
                 0x7FC00123)
_SUBTRAHEND_WORDS = (0x3FF0000000000000, 0x8000000000000000, 0x0000000000000000, 0x7FF8000000000000,
                     0xFFF8000000000000, 0x7FF8000000000123, 0x7FF0000000000000, 0xFFF0000000000000,
                     0x3E45798EE2308C3A)


def _door_operands():
    import numpy as np

    stored = np.array(_STORED_WORDS, dtype=np.uint32).view(np.float32)
    subtrahends = np.array(_SUBTRAHEND_WORDS, dtype=np.uint64).view(np.float64)
    return np.repeat(stored, subtrahends.size), np.tile(subtrahends, stored.size)


def test_the_deposit_door_stores_the_host_words_on_the_amd_gpu():
    """Patch 0004: ``host_writes._rounded_difference`` on the GPU equals NumPy on the host word for word, for every
    pair of a stored word and a subtrahend (81 pairs, real and complex storage), NaN operands included."""
    import cupy
    import numpy as np

    from meep_gpu import host_writes

    stored, subtrahends = _door_operands()
    with np.errstate(all="ignore"):
        host = host_writes._rounded_difference(stored.copy(), subtrahends, None)  # noqa: SLF001
    door = host_writes._rounded_difference  # noqa: SLF001
    device = cupy.asnumpy(door(cupy.asarray(stored), cupy.asarray(subtrahends), None))
    np.testing.assert_array_equal(device.view(np.uint32), host.view(np.uint32))

    # complex storage: the real words against real(A), the imaginary words (the stored words reversed) against imag(A)
    parts = np.stack([stored, stored[::-1]], axis=1).copy()
    pairs = parts.reshape(-1).view(np.complex64)
    with np.errstate(all="ignore"):
        host_c = host_writes._rounded_difference(pairs.copy(), subtrahends, subtrahends[::-1].copy())  # noqa: SLF001
    device_c = cupy.asnumpy(host_writes._rounded_difference(  # noqa: SLF001
        cupy.asarray(pairs), cupy.asarray(subtrahends), cupy.asarray(subtrahends[::-1].copy())))
    np.testing.assert_array_equal(device_c.view(np.uint32), host_c.view(np.uint32))


def test_without_the_patch_the_gpu_inverts_the_sign_of_a_nan_subtrahend():
    """The control of patch 0004: the plain device subtraction, widened and rounded as the door does it, differs from
    the host only where the subtrahend is a NaN and the stored word is not, and there only in the sign bit."""
    import cupy
    import numpy as np

    stored, subtrahends = _door_operands()
    with np.errstate(all="ignore"):
        host = (stored.astype(np.float64) - subtrahends).astype(np.float32).view(np.uint32)
    device = cupy.asnumpy((cupy.asarray(stored).astype(cupy.float64) - cupy.asarray(subtrahends))
                          .astype(cupy.float32)).view(np.uint32)
    differ = host != device
    assert np.array_equal(differ, np.isnan(subtrahends) & ~np.isnan(stored)), np.flatnonzero(differ)
    assert np.all((host ^ device)[differ] == 0x80000000)


@pytest.mark.parametrize("amplitude", [complex(float("inf"), 0.0), complex(1.0, float("inf")),
                                       complex(1.0, float("nan"))], ids=["inf+0j", "1+inf_j", "1+nan_j"])
def test_the_source_deposit_stores_the_host_words_for_non_finite_amplitudes(amplitude):
    """Patch 0004, the whole deposit: ``sources._deposit`` on the GPU stores the host's words in every case of the
    add-on test (each field configuration, Ez and Hz, each current). ``1 + inf j`` and ``1 + nan j`` reach the
    subtraction that forms ``real(A)`` (``amp_re * c_re - amp_im * c_im``) with a NaN subtrahend, before the door."""
    import cupy
    import numpy as np

    from meep_gpu import sources as src
    from meep_gpu import test_source_deposit as t

    cases = differ = nan_words = 0
    for configuration in sorted(t.FIELD_CONFIGURATIONS):
        for component in ("Ez", "Hz"):
            for current in t.FIELD_CURRENTS.values():
                got, _, _ = t._device_deposit_words(cupy, src._deposit, configuration, component,  # noqa: SLF001
                                                    [amplitude], current, t._NORMAL_PLANTS)
                with np.errstate(all="ignore"):
                    host, _, _ = t._field_deposit_words(src._deposit, configuration, component,  # noqa: SLF001
                                                        [amplitude], current, t._NORMAL_PLANTS)
                cases += 1
                differ += int(np.count_nonzero(host != got))
                nan_words += int(np.count_nonzero(np.isnan(host.view(np.float32))))
    assert cases == 84
    assert nan_words > 0, "the table must store NaN words, or it does not test the NaN order"
    assert differ == 0, f"{differ} words differ from the host"


@pytest.mark.parametrize("dtype", ["float32", "int64"])
def test_the_door_widens_a_subtrahend_of_another_dtype_on_the_amd_gpu(dtype):
    """Patch 0004: the kernel takes float64 operands, so the door widens a float32 or int64 subtrahend first, as the
    in-place subtraction of the other builds does; the words equal the host's."""
    import cupy
    import numpy as np

    from meep_gpu import host_writes

    stored = np.array([1.5, -2.25, 0.0, -0.0, 3.0e5, 7.0], dtype=np.float32)
    subtrahends = np.array([1, -3, 0, 2, 70000, -1], dtype=dtype)
    door = host_writes._rounded_difference  # noqa: SLF001
    host = door(stored.copy(), subtrahends, None)
    device = cupy.asnumpy(door(cupy.asarray(stored), cupy.asarray(subtrahends), None))
    np.testing.assert_array_equal(device.view(np.uint32), host.view(np.uint32))
