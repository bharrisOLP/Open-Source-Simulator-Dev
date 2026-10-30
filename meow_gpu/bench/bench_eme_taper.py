"""EME benchmark: a linear Si strip taper with meow, mode solver = tidy3d (CPU) or meow_gpu (GPU).

    python bench/bench_eme_taper.py --backend gpu --precision single
    python bench/bench_eme_taper.py --backend tidy3d --procs 8

Synthetic structure (textbook indices): Si core 220 nm thick, width 0.5 -> 2.0 um over 10 um, SiO2 cladding,
wavelength 1.55 um. Prints one JSON line: the mode-solve time, the S-matrix time, |S| of the fundamental mode
from the left port to the right port, and the neff of the first and the last cell.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np


def build(n_cells: int, dx: float, w0: float = 0.5, w1: float = 2.0, length: float = 10.0):
    import meow as mw
    oxide = mw.IndexMaterial(name="sio2", n=1.444)
    si = mw.IndexMaterial(name="si", n=3.476)
    clad = mw.Structure(material=oxide,
                        geometry=mw.Box(x_min=-3, x_max=3, y_min=-2, y_max=2, z_min=-1, z_max=length + 1))
    poly = np.array([[0.0, -w0 / 2], [length, -w1 / 2], [length, w1 / 2], [0.0, w0 / 2]])
    core = mw.Structure(material=si, mesh_order=1,
                        geometry=mw.Prism(poly=poly, h_min=0.0, h_max=0.22, axis="y"))
    mesh = mw.Mesh2D(x=np.arange(-2.0, 2.0 + dx / 2, dx), y=np.arange(-0.8, 1.02 + dx / 2, dx))
    lengths = np.full(n_cells, length / n_cells)
    cells = mw.create_cells([clad, core], mesh, lengths)
    env = mw.Environment(wl=1.55, T=25.0)
    return cells, [mw.CrossSection.from_cell(cell=c, env=env) for c in cells]


def _tidy3d_one(args):
    n_cells, dx, i, num_modes = args
    import meow as mw
    _, css = build(n_cells, dx)
    return mw.compute_modes_tidy3d(css[i], num_modes=num_modes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["tidy3d", "gpu"], required=True)
    ap.add_argument("--precision", default="single")
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--cells", type=int, default=30)
    ap.add_argument("--modes", type=int, default=10)
    ap.add_argument("--dx", type=float, default=0.02)
    ap.add_argument("--tol", type=float, default=None)
    ap.add_argument("--extra", type=int, default=8)
    a = ap.parse_args()

    import jax
    import meow as mw
    cells, css = build(a.cells, a.dx)
    row = {"backend": a.backend, "precision": a.precision if a.backend == "gpu" else "double", "procs": a.procs,
           "tol": a.tol, "extra": a.extra, "cells": a.cells, "modes": a.modes, "grid": list(css[0].nx.shape), "device": str(jax.devices()[0])}
    t0 = time.perf_counter()
    if a.backend == "gpu":
        jax.config.update("jax_enable_x64", True)
        from meow_gpu.meow_backend import GPUModeSolver, compute_modes_gpu
        solver = GPUModeSolver(precision=a.precision, tol=a.tol, extra=a.extra)
        modes, its = [], []
        for cs in css:
            modes.append(compute_modes_gpu(cs, num_modes=a.modes, solver=solver))
            its.append(solver.last_info.iterations)
        row["iterations"] = its
        row["mean_iterations"] = float(np.mean(its[1:]))
        from meow_gpu.meow_backend import device_summary
        row["solver_device"] = device_summary()
    elif a.procs == 1:
        modes = [mw.compute_modes_tidy3d(cs, num_modes=a.modes) for cs in css]
    else:
        import multiprocessing as mp
        with ProcessPoolExecutor(a.procs, mp_context=mp.get_context("spawn")) as ex:
            modes = list(ex.map(_tidy3d_one, [(a.cells, a.dx, i, a.modes) for i in range(len(css))]))
    row["modes_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    with jax.default_device(jax.devices("cpu")[0]):
        S, port_map = mw.compute_s_matrix(modes, cells=cells)
    row["smatrix_s"] = time.perf_counter() - t0
    S = np.asarray(S)
    row["S21_00"] = float(abs(S[port_map["left@0"], port_map["right@0"]]))
    row["S11_00"] = float(abs(S[port_map["left@0"], port_map["left@0"]]))
    row["neff_first"] = [round(float(np.real(m.neff)), 8) for m in modes[0][:4]]
    row["neff_last"] = [round(float(np.real(m.neff)), 8) for m in modes[-1][:4]]
    row["n_modes_kept"] = [len(m) for m in modes]
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
