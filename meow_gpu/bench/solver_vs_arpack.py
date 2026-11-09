"""The mode sets of meow_gpu at 40 to 120 modes against ARPACK, on the cells of bench/bench_interface.py.

    python bench/solver_vs_arpack.py [--widths 0.5,2.0] [--modes 40,60,80,120] [--wavelength 1.55] [--step 0.02]
                                     [--search deflated,full]

For each strip width (a 0.22 um thick Si strip in SiO2, window x -2 to 2 um, y -0.8 to 1.02 um; textbook indices Si
3.476, SiO2 1.444): the assembled operator A, its unknowns of the PEC walls W (``operator.wall_unknowns``: Ex on the
near y wall, Ey on the near x wall) and the other unknowns R. One "wall" line per width:
- coupling_wr, coupling_rw: max |A[W, R]| and max |A[R, W]| (zero: A is block-diagonal);
- wall_formula_max: the largest relative distance of the eigenvalues of the wall block A[W, W] (dense eig) from the
  1-D Neumann formula -(n^2 - (2 sin(m pi / 2N) / (k0 dl))^2), m = 0 .. N - 1, of each wall (N cells of size dl);
- arpack_s: the times of the two ARPACK runs (scipy eigs, shift-invert at the target of the solver): the full
  matrix A (k = max(modes) + 40) and the restricted matrix A[R, R] (k = max(modes) + 20).
One line per width, mode count and search space ("deflated": the solver of today; "full": the start block and the
preconditioner keep the wall unknowns, the solver up to 505c4c5, ``make_preconditioner(deflate_walls=False)``);
precision "mixed", a new solver for each line (cold start), after one solve of the same size that compiles:
- residual_max: the largest relative residual |A u - lambda u| / (|lambda_target| |u|) of the solver modes;
- wall_max: the largest |u_W| / |u| of the solver modes;
- missed_full: how many of the ``modes`` eigenvalues of A nearest the target the solver does not give; missed_wall:
  how many of these are eigenvalues of the wall block (within 1e-9, relative); missed_full_neff: their neff
  (rounded; the first 20);
- missed: the same against A[R, R] (the eigenvalues without the wall modes; a cut through a complex pair can count
  one: then distance_max stays at the tolerance); missed_neff;
- distance_max: the largest difference of the sorted distances |lambda - target| of the solver set and of the
  ``modes`` nearest eigenvalues of A[R, R] (relative; zero within the tolerance when the sets are equal);
- arpack_distance_max: the largest distance of a solver eigenvalue from the nearest eigenvalue of A[R, R];
- iterations, matvecs, matvecs_hi, solve_s (the eigen solve and the fields, no post-processing), polish;
- above_cutoff, below_cutoff, complex, re_negative, growing: counts of the raw solver modes; kept and pairs_complete:
  the modes after post_process_eigenmodes, and how many of its complex modes have the other member of their pair
  (-conj(neff)) in the set; copies: pairs of solver modes with the same neff and parallel E_t fields;
- same_as_deflated (the "full" lines, when both ran): the largest difference of the sorted distances |lambda -
  target| of the two search spaces (relative); conjugate_swaps: eigenvalues of this line whose conjugate (not the
  value) is in the "deflated" set (a cut through a complex pair: both members have the same distance);
  differ_otherwise: the other eigenvalues that are not in the "deflated" set;
- max_rss_gb: the peak resident memory of the process so far.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np  # noqa: E402

N_SI, N_SIO2 = 3.476, 1.444
SAME = 1e-9          # two eigenvalues are the same when they differ by less than this (relative to |target|)


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


def neumann_wall_values(p, n_wall):
    """The eigenvalues of the 1-D Neumann formula for the two walls (uniform grid and wall index n_wall)."""
    vals = []
    for dl in (p.dl_f[0], p.dl_f[1]):
        n = len(dl)
        kt = 2 * np.sin(np.arange(n) * np.pi / (2 * n)) / (p.k0 * float(np.mean(dl)))
        vals.append(-(n_wall**2 - kt**2))
    return np.concatenate(vals)


def nearest(values, target, k):
    return values[np.argsort(np.abs(values - target), kind="stable")[:k]]


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20      # Linux: kB


def neff_list(lams):
    return [[round(float(np.sqrt(-v).real), 5), round(float(np.sqrt(-v).imag), 5)] for v in lams[:20]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", default="0.5,2.0")
    ap.add_argument("--modes", default="40,60,80,120")
    ap.add_argument("--wavelength", type=float, default=1.55)
    ap.add_argument("--step", type=float, default=0.02)
    ap.add_argument("--search", default="deflated,full", help="deflated (today), full (up to 505c4c5), or both")
    a = ap.parse_args()
    import jax
    jax.config.update("jax_enable_x64", True)
    import scipy.sparse.linalg as sla

    import meow_gpu.meow_backend as backend
    from meow_gpu.fde.eigensolver import make_preconditioner
    from meow_gpu.fde.operator import assemble, wall_unknowns
    from meow_gpu.interface import _is_copy, post_process_eigenmodes
    from meow_gpu.meow_backend import GPUModeSolver, _problem, _target, compute_modes_gpu, device_summary

    searches = [s for s in a.search.split(",") if s]
    if not set(searches) <= {"deflated", "full"}:
        raise SystemExit("--search: deflated and/or full")
    modes = [int(v) for v in a.modes.split(",")]
    tol = 1e-9
    for w in [float(v) for v in a.widths.split(",")]:
        cs = cross_section(w, a.step, a.wavelength)
        p = _problem(cs)
        a_op = assemble(p).astype(complex).tocsr()
        walls = wall_unknowns(p).ravel()
        iw, ir = np.flatnonzero(walls), np.flatnonzero(~walls)
        a_rr = a_op[ir][:, ir].tocsc()
        a_ww = a_op[iw][:, iw].toarray()
        target = _target(p, None)
        scale = abs(target)
        lam_wall = np.linalg.eigvals(a_ww)
        formula = neumann_wall_values(p, N_SIO2)
        wall_formula = max(np.min(np.abs(formula - v)) for v in lam_wall) / scale
        wall_formula = max(wall_formula, max(np.min(np.abs(lam_wall - v)) for v in formula) / scale)
        t0 = time.perf_counter()
        lam_full = sla.eigs(a_op, k=max(modes) + 40, sigma=target, which="LM", return_eigenvectors=False)
        t_full = time.perf_counter() - t0
        t0 = time.perf_counter()
        lam_rest = sla.eigs(a_rr, k=max(modes) + 20, sigma=target, which="LM", return_eigenvectors=False)
        t_rest = time.perf_counter() - t0
        print(json.dumps({"line": "wall", "width_um": w, "step_um": a.step, "wavelength": a.wavelength,
                          "grid": list(p.shape), "unknowns": int(walls.size), "wall_unknowns": int(walls.sum()),
                          "coupling_wr": float(abs(a_op[iw][:, ir]).max()),
                          "coupling_rw": float(abs(a_op[ir][:, iw]).max()),
                          "wall_formula_max": float(wall_formula),
                          "wall_above_cutoff": int(np.sum(np.real(lam_wall) < 0)),
                          "arpack_s": {"full": round(t_full, 1), "rest": round(t_rest, 1)},
                          "max_rss_gb": round(rss_gb(), 2)}), flush=True)
        for nm in modes:
            ref_full = nearest(lam_full, target, nm)
            ref_rest = nearest(lam_rest, target, nm)
            got_values = {}
            # one solve of this block size first: the JIT compile is not in the times below
            compute_modes_gpu(cs, num_modes=nm, solver=GPUModeSolver(precision="mixed"), post_process=lambda ms: ms)
            for search in searches:
                saved = backend.make_preconditioner
                if search == "full":
                    backend.make_preconditioner = lambda q: make_preconditioner(q, deflate_walls=False)
                try:
                    solver = GPUModeSolver(precision="mixed")
                    t0 = time.perf_counter()
                    raw = compute_modes_gpu(cs, num_modes=nm, solver=solver, post_process=lambda ms: ms)
                    t_solve = time.perf_counter() - t0
                finally:
                    backend.make_preconditioner = saved
                info = solver.last_info
                lam = np.array([-complex(m.neff) ** 2 for m in raw])
                got_values[search] = lam
                vecs = [np.concatenate([np.asarray(m.Ex).ravel(), np.asarray(m.Ey).ravel()]) for m in raw]
                res = [np.linalg.norm(a_op @ u - v * u) / (scale * np.linalg.norm(u)) for v, u in zip(lam, vecs)]
                wall_part = [np.linalg.norm(u[iw]) / np.linalg.norm(u) for u in vecs]
                missed_full = [v for v in ref_full if np.min(np.abs(lam - v)) / scale > SAME]
                missed_wall = [v for v in missed_full if np.min(np.abs(lam_wall - v)) / scale < SAME]
                missed = [v for v in ref_rest if np.min(np.abs(lam - v)) / scale > SAME]
                dist = np.sort(np.abs(lam - target)) - np.sort(np.abs(ref_rest - target))
                n = [complex(m.neff) for m in raw]
                small = [abs(x.imag) <= tol * abs(x) for x in n]
                got = post_process_eigenmodes(raw)
                ng = [complex(m.neff) for m in got]
                cplx = [x for x in ng if abs(x.imag) > tol * abs(x) and abs(x.real) > tol * abs(x)]
                row = {"line": "solver", "search": search, "width_um": w, "step_um": a.step,
                       "wavelength": a.wavelength, "modes": nm, "grid": list(p.shape),
                       "solve_s": round(t_solve, 2), "iterations": info.iterations, "matvecs": info.matvecs,
                       "matvecs_hi": info.matvecs_hi, "polish": info.polish_used, "converged": info.converged,
                       "residual_max": float(max(res)), "wall_max": float(max(wall_part)),
                       "missed_full": len(missed_full), "missed_wall": len(missed_wall),
                       "missed_full_neff": neff_list(missed_full),
                       "missed": len(missed), "missed_neff": neff_list(missed),
                       "distance_max": float(np.max(np.abs(dist)) / scale),
                       "arpack_distance_max": float(max(np.min(np.abs(lam_rest - v)) for v in lam) / scale),
                       "above_cutoff": int(sum(small)), "below_cutoff": int(len(n) - sum(small)),
                       "complex": int(sum(abs(x.imag) > tol * abs(x) and abs(x.real) > tol * abs(x) for x in n)),
                       "re_negative": int(sum(x.real < -tol * abs(x) for x in n)),
                       "growing": int(sum(x.imag < -tol * abs(x) for x in n)),
                       "kept": len(got), "complex_kept": len(cplx),
                       "pairs_complete": int(sum(min(abs(y + np.conj(x)) for y in ng) < 1e-6 for x in cplx)),
                       "copies": int(sum(_is_copy(raw[i], raw[j], tol) for i in range(len(raw))
                                         for j in range(i))),
                       "device": device_summary()}
                if search == "full" and "deflated" in got_values:
                    other = got_values["deflated"]
                    d = np.sort(np.abs(other - target)) - np.sort(np.abs(lam - target))
                    row["same_as_deflated"] = float(np.max(np.abs(d)) / scale)
                    diff = [v for v in lam if np.min(np.abs(other - v)) / scale > SAME]
                    swaps = [v for v in diff if np.min(np.abs(other - np.conj(v))) / scale < SAME]
                    row["conjugate_swaps"] = len(swaps)
                    row["differ_otherwise"] = len(diff) - len(swaps)
                row["max_rss_gb"] = round(rss_gb(), 2)
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
