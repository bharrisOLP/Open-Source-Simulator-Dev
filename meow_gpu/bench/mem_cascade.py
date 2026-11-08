"""Peak memory of meow's SAX cascade (klu backend) against the star product of meow_gpu.interface, on synthetic
stacks without a mode solve: n cells with k modes each, random interface S-matrices (unitary, near a port swap)
and random propagation phases. One stack in each process (the peak RSS of a process only grows):

    JAX_PLATFORMS=cpu python bench/mem_cascade.py --method sax --cells 22 --modes 20
    JAX_PLATFORMS=cpu python bench/mem_cascade.py --method star --cells 22 --modes 20

Prints one JSON line: the peak RSS before and after the cascade (GiB), the time, the number of entries of the boolean
matrix that sax.backends.klu.analyze_circuit_klu builds (len(Si) x len(Cj): all interface and propagation entries
times the connected port pairs), and T00 and sum |S|^2 of the first input (the same seed gives the same stack, so
the two methods must give the same values).
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np  # noqa: E402


def peak_gb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 3)     # Linux: kB


def stack(n: int, k: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    phases = [np.exp(2j * np.pi * rng.random(k)) for _ in range(n)]
    swap = np.block([[np.zeros((k, k)), np.eye(k)], [np.eye(k), np.zeros((k, k))]])
    ifaces = []
    for _ in range(n - 1):
        noise = rng.normal(size=(2 * k, 2 * k)) + 1j * rng.normal(size=(2 * k, 2 * k))
        q = np.linalg.qr(np.eye(2 * k) + 0.3 * noise)[0]
        ifaces.append(swap @ q)
    return phases, ifaces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["sax", "star"], required=True)
    ap.add_argument("--cells", type=int, default=22)
    ap.add_argument("--modes", type=int, default=20)
    a = ap.parse_args()
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from meow_gpu.interface import _blocks, _star

    n, k = a.cells, a.modes
    phases, ifaces = stack(n, k)
    pm = {**{f"left@{j}": j for j in range(k)}, **{f"right@{j}": k + j for j in range(k)}}
    entries = ((n - 1) * (2 * k) ** 2 + n * 2 * k) * ((n - 1) * 4 * k)
    before = peak_gb()
    t0 = time.perf_counter()
    if a.method == "sax":
        from meow.eme.cascade import compute_s_matrix_sax

        def props(modes, cells, cell_lengths=None):
            return {f"p_{i}": {**{(f"left@{j}", f"right@{j}"): jnp.asarray(ph[j]) for j in range(k)},
                               **{(f"right@{j}", f"left@{j}"): jnp.asarray(ph[j]) for j in range(k)}}
                    for i, ph in enumerate(phases)}

        def ifs(modes):
            return {f"i_{i}_{i + 1}": (jnp.asarray(s), pm) for i, s in enumerate(ifaces)}

        with jax.default_device(jax.devices("cpu")[0]):
            s, port_map = compute_s_matrix_sax([[None] * k] * n, cell_lengths=[0.0] * n, interfaces_fn=ifs,
                                               propagations_fn=props)
        s = np.asarray(s)
    else:
        def prop(ph):
            d = np.diag(ph)
            return np.zeros_like(d), d, d, np.zeros_like(d)

        total = prop(phases[0])
        for i, sm in enumerate(ifaces):
            total = _star(total, _blocks(sm, pm, k, k))
            total = _star(total, prop(phases[i + 1]))
        r_ll, t_lr, t_rl, r_rr = total
        s = np.block([[r_ll, t_rl], [t_lr, r_rr]]).T
        port_map = pm
    seconds = time.perf_counter() - t0
    i0 = port_map["left@0"]
    t00 = s[i0, port_map["right@0"]]
    print(json.dumps({"method": a.method, "cells": n, "modes": k, "klu_bool_entries": entries,
                      "peak_gb_before": before, "peak_gb_after": peak_gb(), "seconds": round(seconds, 2),
                      "T00": [float(t00.real), float(t00.imag)], "sum_abs2_row0": float(np.sum(np.abs(s[i0]) ** 2))}),
          flush=True)


if __name__ == "__main__":
    main()
