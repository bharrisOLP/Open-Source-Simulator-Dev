"""The mode sets of meow_gpu at 40 to 120 modes against ARPACK, on the cells of bench/bench_interface.py.

    python bench/solver_vs_arpack.py [--widths 0.5,2.0] [--modes 40,60,80,120] [--wavelength 1.55] [--step 0.02]

For each strip width (a 0.22 um thick Si strip in SiO2, window x -2 to 2 um, y -0.8 to 1.02 um; textbook indices Si
3.476, SiO2 1.444) and mode count: the modes of the solver (precision "mixed", a new solver, as in the bench) and the
eigenvalues of ARPACK on the assembled operator (scipy eigs, shift-invert at the target of the solver, k = modes +
40). Prints one JSON line:
- residual_max: the largest relative residual |A u - lambda u| / (|lambda_target| |u|) of the solver modes;
- arpack_distance_max: the largest distance of a solver eigenvalue from the nearest ARPACK eigenvalue (relative);
- missed: how many of the `modes` ARPACK eigenvalues nearest the target the solver does not give, and their neff
  (rounded; the first 20);
- above_cutoff, below_cutoff, complex (Re and Im of neff not zero), re_negative, growing (Im(neff) < 0): counts of
  the raw solver modes;
- kept and pairs_complete: the modes after post_process_eigenmodes, and how many of its complex modes have the other
  member of their pair (-conj(neff)) in the set;
- copies: pairs of solver modes with the same neff and parallel E_t fields.
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np  # noqa: E402

N_SI, N_SIO2 = 3.476, 1.444


def cross_section(width, step, wavelength):
    import meow as mw
    structs = [mw.Structure(material=mw.IndexMaterial(name="SiO2", n=N_SIO2),
                            geometry=mw.Box(x_min=-3, x_max=3, y_min=-2, y_max=2, z_min=-1, z_max=2)),
               mw.Structure(material=mw.IndexMaterial(name="Si", n=N_SI), mesh_order=1,
                            geometry=mw.Box(x_min=-width / 2, x_max=width / 2, y_min=0, y_max=0.22, z_min=-1,
                                            z_max=2))]
    mesh = mw.Mesh2D(x=np.arange(-2.0, 2.0 + step / 2, step), y=np.arange(-0.8, 1.02 + step / 2, step))
    cell = mw.Cell(structures=structs, mesh=mesh, z_min=0.0, z_max=1.0)
    return mw.CrossSection.from_cell(cell=cell, env=mw.Environment(wl=wavelength, T=25.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", default="0.5,2.0")
    ap.add_argument("--modes", default="40,60,80,120")
    ap.add_argument("--wavelength", type=float, default=1.55)
    ap.add_argument("--step", type=float, default=0.02)
    a = ap.parse_args()
    import jax
    jax.config.update("jax_enable_x64", True)
    import scipy.sparse.linalg as sla

    from meow_gpu.fde.operator import assemble
    from meow_gpu.interface import _is_copy, post_process_eigenmodes
    from meow_gpu.meow_backend import GPUModeSolver, _problem, _target, compute_modes_gpu, device_summary

    tol = 1e-9
    for w in [float(v) for v in a.widths.split(",")]:
        cs = cross_section(w, a.step, a.wavelength)
        p = _problem(cs)
        a_op = assemble(p).astype(complex)
        target = _target(p, None)
        scale = abs(target)
        for nm in [int(v) for v in a.modes.split(",")]:
            t0 = time.perf_counter()
            raw = compute_modes_gpu(cs, num_modes=nm, solver=GPUModeSolver(precision="mixed"),
                                    post_process=lambda ms: ms)
            t_solve = time.perf_counter() - t0
            lam_ref = sla.eigs(a_op, k=nm + 40, sigma=target, which="LM", return_eigenvectors=False)
            lam_ref = lam_ref[np.argsort(np.abs(lam_ref - target))]
            lam = np.array([-complex(m.neff) ** 2 for m in raw])
            res = [np.linalg.norm(a_op @ u - v * u) / (scale * np.linalg.norm(u)) for v, u in
                   ((v, np.concatenate([np.asarray(m.Ex).ravel(), np.asarray(m.Ey).ravel()])) for v, m in
                    zip(lam, raw))]
            dist = [np.min(np.abs(lam_ref - v)) / scale for v in lam]
            missed = [v for v in lam_ref[:nm] if np.min(np.abs(lam - v)) / scale > 1e-7]
            n = [complex(m.neff) for m in raw]
            small = [abs(x.imag) <= tol * abs(x) for x in n]
            got = post_process_eigenmodes(raw)
            ng = [complex(m.neff) for m in got]
            cplx = [x for x in ng if abs(x.imag) > tol * abs(x) and abs(x.real) > tol * abs(x)]
            row = {"width_um": w, "step_um": a.step, "wavelength": a.wavelength, "modes": nm,
                   "grid": list(np.shape(cs.nx)), "solve_s": round(t_solve, 1),
                   "residual_max": float(max(res)), "arpack_distance_max": float(max(dist)),
                   "missed": len(missed),
                   "missed_neff": [[round(float(np.sqrt(-v).real), 5), round(float(np.sqrt(-v).imag), 5)]
                                   for v in missed[:20]],
                   "above_cutoff": int(sum(small)), "below_cutoff": int(len(n) - sum(small)),
                   "complex": int(sum(abs(x.imag) > tol * abs(x) and abs(x.real) > tol * abs(x) for x in n)),
                   "re_negative": int(sum(x.real < -tol * abs(x) for x in n)),
                   "growing": int(sum(x.imag < -tol * abs(x) for x in n)),
                   "kept": len(got), "complex_kept": len(cplx),
                   "pairs_complete": int(sum(min(abs(y + np.conj(x)) for y in ng) < 1e-6 for x in cplx)),
                   "copies": int(sum(_is_copy(raw[i], raw[j], tol) for i in range(len(raw)) for j in range(i))),
                   "device": device_summary()}
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
