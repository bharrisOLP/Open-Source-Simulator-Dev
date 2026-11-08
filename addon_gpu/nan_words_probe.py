# SPDX-License-Identifier: GPL-2.0-or-later
"""NaN words on the AMD GPU against the host: the probe of the one failure of meep_gpu/test_source_deposit.py.

The failing test of the add-on (``test_on_a_cuda_device_the_silent_skip_stores_the_always_deposit_and_host_words``)
deposits an amplitude ``inf + 0j`` with a current ``(+0, +0)``. The host forms ``real(A) = inf * 0 - 0 * 0`` in
float64, which is a NaN, and the door of the deposit (``host_writes.subtract_double``) widens each stored float32
word to float64, subtracts ``real(A)`` once and rounds once to float32: on the host with NumPy, on the GPU with CuPy.
The stored words are therefore NaN words, and their bits (sign and payload) come from the arithmetic unit.

Part 1 (``elementary``): for float64 NaN operands of both signs and two payloads, the words of ``a - b``, ``a + b``,
``a * b``, ``b - a`` and of the float64 to float32 cast, on the host (NumPy) and on the GPU (CuPy elementwise
kernels), and the NaN that ``inf * 0`` forms on each. Part 2 (``deposit``): the cases of the failing test (every
field configuration of the test, Ez and Hz, each current) for three amplitude tables: ``inf + 0j`` (the
``non-finite`` table of the test), ``1 + inf j`` and ``1 + nan j``, with the host and device words of each differing
word in hex. The last two reach the subtraction that forms ``real(A)`` (``amp_re * c_re - amp_im * c_im``) with a
NaN subtrahend, before the door of the deposit.

Part 3 (``repair``): the float32 ``D - P`` of the deposit repair (``deposit_repair._fresh_at``, which patch 0004 does
not change) at its cells, against the whole-volume ``D - P`` that ``update_E`` forms
(``Fields.displacement_minus_polarization``), on the host and on the GPU, for every pair of planted words (finite,
NaN of both signs, a NaN payload, infinities). The repair must give the words of the whole-volume form on the same
device: it recomputes the field at the deposit cells as ``update_E`` of that device would.

Run it on the GPU computer, through the GPU lock:

    bash tools/gpu_run.sh addon_gpu python nan_words_probe.py

It prints one line per finding and one ``NAN_JSON`` line at the end.
"""

from __future__ import annotations

import json
import sys

import numpy as np

QNAN_POS = 0x7FF8000000000000
QNAN_NEG = 0xFFF8000000000000
QNAN_POS_PAYLOAD = 0x7FF8000000000123
QNAN_NEG_PAYLOAD = 0xFFF8000000000123


def f64(word: int) -> np.ndarray:
    return np.array([word], dtype=np.uint64).view(np.float64)


def hex64(x) -> str:
    return f"{int(np.asarray(x, dtype=np.float64).reshape(-1).view(np.uint64)[0]):016x}"


def hex32(x) -> str:
    return f"{int(np.asarray(x, dtype=np.float32).reshape(-1).view(np.uint32)[0]):08x}"


def elementary(cupy) -> list[dict]:
    rows = []
    finite = np.array([1.5], dtype=np.float64)
    for name, word in (("+qnan", QNAN_POS), ("-qnan", QNAN_NEG), ("+qnan payload", QNAN_POS_PAYLOAD),
                       ("-qnan payload", QNAN_NEG_PAYLOAD)):
        b = f64(word)
        ops = {
            "1.5 - b": (lambda xp, a, b: a - b),
            "1.5 + b": (lambda xp, a, b: a + b),
            "1.5 * b": (lambda xp, a, b: a * b),
            "b - 1.5": (lambda xp, a, b: b - a),
            "float32(1.5 - b)": (lambda xp, a, b: (a - b).astype(xp.float32)),
            "float32(b)": (lambda xp, a, b: b.astype(xp.float32)),
        }
        for op, fn in ops.items():
            with np.errstate(all="ignore"):
                host = fn(np, finite, b)
            dev = cupy.asnumpy(fn(cupy, cupy.asarray(finite), cupy.asarray(b)))
            to_hex = hex32 if host.dtype == np.float32 else hex64
            rows.append({"operand": name, "op": op, "host": to_hex(host), "device": to_hex(dev),
                         "equal": to_hex(host) == to_hex(dev)})
    with np.errstate(all="ignore"):
        host = np.array([np.inf]) * np.array([0.0])
    dev = cupy.asnumpy(cupy.array([np.inf]) * cupy.array([0.0]))
    rows.append({"operand": "inf, 0", "op": "inf * 0", "host": hex64(host), "device": hex64(dev),
                 "equal": hex64(host) == hex64(dev)})
    return rows


#: The amplitude tables of part 2: the ``non-finite`` table of the add-on test, and two with a non-finite imaginary
#: part.
AMPLITUDE_TABLES = {"inf + 0j": [complex(np.inf, 0.0)], "1 + inf j": [complex(1.0, np.inf)],
                    "1 + nan j": [complex(1.0, np.nan)]}


def deposit(cupy, label: str) -> list[dict]:
    from meep_gpu import sources as src
    from meep_gpu import test_source_deposit as t

    rows = []
    amplitudes = AMPLITUDE_TABLES[label]
    for configuration in sorted(t.FIELD_CONFIGURATIONS):
        for component in ("Ez", "Hz"):
            for current_label, current in t.FIELD_CURRENTS.items():
                got, _, _ = t._device_deposit_words(cupy, src._deposit, configuration, component,  # noqa: SLF001
                                                    amplitudes, current, t._NORMAL_PLANTS)
                with np.errstate(all="ignore"):
                    host, _, _ = t._field_deposit_words(src._deposit, configuration, component,  # noqa: SLF001
                                                        amplitudes, current, t._NORMAL_PLANTS)
                differ = np.flatnonzero(host != got)
                pairs = sorted({(f"{int(host[i]):08x}", f"{int(got[i]):08x}") for i in differ})
                both_nan = all(((int(host[i]) & 0x7F800000) == 0x7F800000 and (int(host[i]) & 0x7FFFFF))
                               and ((int(got[i]) & 0x7F800000) == 0x7F800000 and (int(got[i]) & 0x7FFFFF))
                               for i in differ)
                rows.append({"amplitude": label, "configuration": configuration, "component": component,
                             "current": current_label,
                             "words": int(host.size), "differ": int(differ.size), "both_nan": bool(both_nan),
                             "pairs_host_device": pairs[:6]})
    return rows


#: float32 words of part 3, planted in D and in P (every pair of them at one repair cell each): finite words, NaN words
#: of both signs and with a payload, and infinities of both signs.
REPAIR_WORDS = (0x3FC00000, 0xC0100000, 0x7FC00000, 0xFFC00000, 0x7FC00123, 0x7F800000, 0xFF800000)


def _repair_fields(xp, d_words: np.ndarray, p_words: np.ndarray):
    """A 2-D grid with a Lorentzian on Ez (as ``meep_gpu/test_deposit_repair.py`` builds one), D and P set to the
    given float32 words; ``xp`` is NumPy (the host) or CuPy (the GPU)."""
    from meep_gpu.dispersion import PolarizationState, Susceptibility
    from meep_gpu.fields import Fields
    from meep_gpu.grid import Grid

    grid = Grid(resolution=10.0, cell_size=(1.2, 1.2, 0.0), dimensions=2, xp=xp)
    fields = Fields(grid=grid)
    fields.polarizations.append(PolarizationState(Susceptibility(1.0, 0.1, "lorentzian"),
                                                  {"Ex": 0.0, "Ey": 0.0, "Ez": 0.25}, grid, xp.float32))
    fields.enable_field_storage()
    shape = fields.Dz.shape
    fields.Dz[...] = xp.asarray(d_words.view(np.float32).reshape(shape))
    fields.polarizations[0].P["Ez"][...] = xp.asarray(p_words.view(np.float32).reshape(shape))
    return fields


def repair(cupy) -> dict:
    """Part 3: ``D - P`` of the deposit repair (``deposit_repair._fresh_at``, at the repair cells only) against the
    whole-volume ``D - P`` of ``update_E`` (``Fields.displacement_minus_polarization``), on the host and on the GPU,
    with every pair of ``REPAIR_WORDS`` (D, P) at one cell each (cells 0, 2, 4, ...; noise elsewhere)."""
    from meep_gpu import deposit_repair

    pairs = np.array([(d, p) for d in REPAIR_WORDS for p in REPAIR_WORDS], dtype=np.uint32)
    rng = np.random.default_rng(20261108)
    size = 12 * 12
    d_words = rng.standard_normal(size).astype(np.float32).view(np.uint32)
    p_words = rng.standard_normal(size).astype(np.float32).view(np.uint32)
    linear = np.arange(len(pairs), dtype=np.int64) * 2
    d_words[linear], p_words[linear] = pairs[:, 0], pairs[:, 1]
    words = {}
    for side, xp in (("host", np), ("device", cupy)):
        fields = _repair_fields(xp, d_words, p_words)
        assert fields.Dz.size == size, fields.Dz.shape
        index = xp.asarray(linear)
        with np.errstate(all="ignore"):
            subset = deposit_repair._fresh_at(fields, "Ez", index)  # noqa: SLF001
            whole = fields.displacement_minus_polarization("Ez").reshape(-1)[index]
        to_host = (lambda a: a) if xp is np else cupy.asnumpy
        words[side + " repair"] = np.ascontiguousarray(to_host(subset), dtype=np.float32).view(np.uint32)
        words[side + " whole"] = np.ascontiguousarray(to_host(whole), dtype=np.float32).view(np.uint32)
    host, device = words["host repair"], words["device repair"]
    differ = np.flatnonzero(host != device)
    d_nan, p_nan = np.isnan(pairs.view(np.float32)[:, 0]), np.isnan(pairs.view(np.float32)[:, 1])
    return {"cells": int(len(pairs)),
            "nan_results": int(np.count_nonzero(np.isnan(host.view(np.float32)))),
            "device_repair_vs_device_whole": int(np.count_nonzero(words["device repair"] != words["device whole"])),
            "host_repair_vs_host_whole": int(np.count_nonzero(words["host repair"] != words["host whole"])),
            "device_vs_host": int(differ.size),
            "device_vs_host_sign_only": bool(np.all((host ^ device)[differ] == 0x80000000)),
            "differ_where_p_nan_d_not": bool(np.array_equal(differ, np.flatnonzero(p_nan & ~d_nan))),
            "pairs_host_device": sorted({(f"{int(host[i]):08x}", f"{int(device[i]):08x}") for i in differ})}


def main() -> int:
    import cupy

    out = {"device": cupy.cuda.runtime.getDeviceProperties(0)["name"].decode(errors="replace"),
           "elementary": elementary(cupy)}
    for row in out["elementary"]:
        print(f"{row['op']:18s} {row['operand']:15s} host {row['host']} device {row['device']} "
              f"{'equal' if row['equal'] else 'DIFFERENT'}", flush=True)
    summary = {}
    for label in AMPLITUDE_TABLES:
        rows = deposit(cupy, label)
        for row in rows:
            if row["differ"]:
                print(f"deposit {label}: {row['configuration']}, {row['component']}, current {row['current']}: "
                      f"{row['differ']} of {row['words']} words differ (all NaN: {row['both_nan']}); host/device "
                      f"{row['pairs_host_device']}", flush=True)
        summary[label] = {"cases": len(rows), "cases_differ": sum(1 for r in rows if r["differ"]),
                          "words_differ": sum(r["differ"] for r in rows),
                          "differ_all_nan": all(r["both_nan"] for r in rows if r["differ"])}
    rep = repair(cupy)
    print(f"repair D - P: {rep['cells']} cells, {rep['nan_results']} NaN results; repair against the whole-volume "
          f"form: host {rep['host_repair_vs_host_whole']}, device {rep['device_repair_vs_device_whole']} differing "
          f"words; device against host {rep['device_vs_host']} (sign only: {rep['device_vs_host_sign_only']}; where P "
          f"is a NaN and D is not: {rep['differ_where_p_nan_d_not']}); host/device {rep['pairs_host_device']}",
          flush=True)
    print("NAN_JSON " + json.dumps({"device": out["device"], "deposit": summary, "repair": rep}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
