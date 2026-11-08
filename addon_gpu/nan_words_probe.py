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
    print("NAN_JSON " + json.dumps({"device": out["device"], "deposit": summary}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
