"""EME interfaces on the SOI taper of photonsuite (suite/bench/eme_fdtd_taper.py): meow's interface against the
power-conserving interface of meow_gpu.interface, with the same modes.

    python bench/bench_interface.py [--modes 10,20,30,40] [--wavelengths 1.55] [--diag]
        [--interfaces meow_invert,meow_none,conserving] [--max-imag-neff 1.0] [--window=-2,2,-0.8,1.02]
        [--cascade star] [--max-gb 8] [--pairs both] [--pair-variants as_solved,neither]
    common: [--length 2] [--segments 20] [--w0 0.5] [--w1 2.0] [--step 0.02]

The device (the same segments as suite/bench/eme_fdtd_taper.py; textbook indices Si 3.476, SiO2 1.444): a Si strip
220 nm thick in SiO2, 1.5 um straight at w0, a linear taper to w1 over `length` um in `segments` steps (each segment
has the width at its centre), 1 um straight at w1; window x -2 to 2 um, y -0.8 to 1.02 um; one EME cell per segment.
`--length 0.1 --segments 1`: a double step 0.5 -> 1.25 -> 2.0 um (a 0.1 um long middle section).

For each wavelength and mode count the script solves the eigenmodes once (meow_gpu, precision "mixed", a new solver
for each wavelength, as photonsuite.eme does) and then cascades them with each interface:
- meow_invert, meow_none: meow_gpu's post-processing (meow's Gram-Schmidt, each mode keeps its neff) and meow's
  interface (TSVD rcond 1e-3, passivity "invert" or "none"): the rows of photonsuite.eme of the eighth wave.
- conserving: post_process_eigenmodes (no Gram-Schmidt; modes below cutoff kept unless --max-imag-neff; each mode on
  the branch that decays in +z, or with power flow in +z above cutoff) and meow_gpu.interface (the Yee product, the
  test fields of the conservation-of-complex-power technique).
P_right and P_left sum |S|^2 from left@0 over all ports of a side; P_right_prop and P_left_prop only over the ports of
the modes above cutoff (|S|^2 of a mode below cutoff is not a power). Counts over all cells: complex_neff (raw modes
below cutoff), flipped_raw (raw modes that post_process_eigenmodes puts on the other branch: Im(neff) < 0),
negative_re_neff (kept modes with Re(neff) < 0: members of complex pairs (neff, -conj(neff)), forward modes).
--pairs: the rule of the variant "conserving" for a complex pair that the mode count cuts
(meow_gpu.interface.complete_complex_pairs: "both", the default since 2026-11-08 night, adds the missing member;
"neither" removes the member without its partner; "as_solved" keeps the set of the solver, the rule of the rows up
to that day). --pair-variants adds one variant "conserving_<rule>" for each rule, on the same modes (no second
solve). pair_modes_added and pair_modes_removed count over all cells the modes that "both" adds and that "neither"
removes (0 when no pair is cut). --search-space full: the search space of 505c4c5 (the preconditioner keeps the
unknowns of the PEC walls, ``make_preconditioner(deflate_walls=False)``, as in bench/solver_vs_arpack.py), to check
that a pair rule gives the same T00 from the two iteration paths.
The rows of benchmarks/2026-11-08/meow-interface.jsonl, -2 and -3 above 40 modes came from the first version of
post_process_eigenmodes, which removed the modes with Re(neff) < 0 or Im(neff) < 0 (its count: backward_raw).
With --diag, also the two mixed cases (which change of the two causes the difference), and for each interface of the
taper: the largest singular value of S, the Gram matrices of the eigenmodes in the two products, and how real the
fields are. Prints one JSON line for each wavelength and mode count (with the peak memory of the process; since
2026-11-08 evening also "converged" (all cells), "not_converged_cells" and "residual_max" of the mode solves: before,
a solve that did not converge gave only a warning on stderr); a row that raises prints a JSON line with "error" and
the end of the traceback (and the traceback on stderr), and the script goes on with the next row. Since 2026-11-08
night each row also gives gpu_peak_gb and gpu_limit_gb (the device memory pool of JAX; null without statistics).
--window moves the PEC walls of the closed EME window (the FDTD of photonsuite.fdtd_avg has a PML outside the
default window); write it with "=" (the value starts with "-").

Memory: --cascade star (the default since 2026-11-08 afternoon) cascades with the star product of
meow_gpu.interface.compute_s_matrix. The rows of benchmarks/2026-11-08/meow-interface.jsonl and meow-interface-2.jsonl
used meow's SAX cascade (klu backend), which needs about 0.5 GiB + 8.5 bytes x sax_entries() (bench/mem_cascade.py):
for the 22 cells of the taper about 12 GiB at 57 kept modes and 24 GiB at 73 (the logged peaks of the process: 16.3 GiB
and 28.3 GiB). A row whose estimated memory (the mode fields, or the SAX cascade) is above --max-gb is refused with an
error row.
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import resource
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "4")   # before numpy: OpenBLAS reads it when it loads
import time  # noqa: E402

import numpy as np  # noqa: E402

N_SI, N_SIO2 = 3.476, 1.444
WINDOW = ((-2.0, 2.0), (-0.8, 1.02))          # the window of suite/bench/eme_fdtd_taper.py (default of --window)


def segments(w0, w1, length, n_seg):
    """(z0, z1, width) of each segment (the construction of suite/bench/eme_fdtd_taper.py)."""
    segs = [(0.0, 1.5, w0)]
    for k in range(n_seg):
        segs.append((1.5 + length * k / n_seg, 1.5 + length * (k + 1) / n_seg, w0 + (w1 - w0) * (k + 0.5) / n_seg))
    return segs + [(1.5 + length, 2.5 + length, w1)]


def build(segs, step, wavelength, window=WINDOW):
    """meow cells (one per segment) and cross-sections; the structures of photonsuite.eme.structures (the
    background gets the highest mesh order, each later shape a lower one)."""
    import meow as mw
    (x0, x1), (y0, y1) = window
    oxide, si = mw.IndexMaterial(name="SiO2", n=N_SIO2), mw.IndexMaterial(name="Si", n=N_SI)
    z_lo, z_hi = segs[0][0], segs[-1][1]
    structs = [mw.Structure(material=oxide, mesh_order=len(segs) + 1,
                            geometry=mw.Box(x_min=x0 - 1, x_max=x1 + 1, y_min=y0 - 1, y_max=y1 + 1, z_min=z_lo - 1,
                                            z_max=z_hi + 1))]
    for k, (z0, z1, w) in enumerate(segs):
        structs.append(mw.Structure(material=si, mesh_order=len(segs) - k,
                                    geometry=mw.Box(x_min=-w / 2, x_max=w / 2, y_min=0.0, y_max=0.22, z_min=z0,
                                                    z_max=z1)))
    mesh = mw.Mesh2D(x=np.arange(x0, x1 + step / 2, step), y=np.arange(y0, y1 + step / 2, step))
    cells = mw.create_cells(structs, mesh, np.array([s[1] - s[0] for s in segs]), z_min=z_lo)
    env = mw.Environment(wl=wavelength, T=25.0)
    return cells, [mw.CrossSection.from_cell(cell=c, env=env) for c in cells]


SAX_GB_PER_ENTRY, SAX_GB_FIXED = 8.5e-9, 0.5     # bench/mem_cascade.py, 22 cells, 10 to 30 modes (README)


def sax_entries(modes) -> int:
    """The entries of the boolean matrix of sax.backends.klu.analyze_circuit_klu for meow's netlist: (all S-matrix
    entries of the interfaces and the propagations) x (the connected port pairs, both directions)."""
    k = [len(m) for m in modes]
    pairs = list(zip(k[:-1], k[1:]))
    return (sum((a + b) ** 2 for a, b in pairs) + sum(2 * n for n in k)) * sum(2 * (a + b) for a, b in pairs)


def cascade(modes, cells, interfaces_fn, method="star", max_gb=8.0):
    """T00 and the sums of |S|^2 from left@0 into all right and all left ports, and into the ports of the modes above
    cutoff only (P_*_prop; |S|^2 of a mode below cutoff is not a power). method "star": the star product of
    meow_gpu.interface.compute_s_matrix (memory of the order of k^2); "sax": meow.compute_s_matrix (SAX, klu), which
    is refused when its estimated memory is above max_gb."""
    import jax
    import meow as mw

    from meow_gpu.interface import compute_s_matrix
    if method == "star":
        S, pm = compute_s_matrix(modes, [c.length for c in cells], interfaces_fn=interfaces_fn)
    else:
        need = SAX_GB_FIXED + SAX_GB_PER_ENTRY * sax_entries(modes)
        if need > max_gb:
            raise MemoryError(f"the SAX cascade needs about {need:.1f} GiB here (above --max-gb {max_gb}): "
                              "use --cascade star")
        with jax.default_device(jax.devices("cpu")[0]):          # SAX with klujax runs on the CPU
            S, pm = mw.compute_s_matrix(modes, cells=cells, interfaces_fn=interfaces_fn)
    S = np.asarray(S)
    i0 = pm["left@0"]

    def power(side, ms, prop):
        return float(sum(abs(S[i0, pm[f"{side}@{k}"]]) ** 2 for k, m in enumerate(ms)
                         if not prop or abs(np.imag(m.neff)) < 1e-9 * abs(m.neff)))

    return {"T00": float(abs(S[i0, pm["right@0"]]) ** 2), "P_right": power("right", modes[-1], False),
            "P_left": power("left", modes[0], False), "P_right_prop": power("right", modes[-1], True),
            "P_left_prop": power("left", modes[0], True)}


def sigma_max(interfaces: dict) -> float:
    return float(max(np.linalg.svd(np.asarray(s), compute_uv=False)[0] for s, _ in interfaces.values()))


def gram_offdiag(modes, overlap) -> float:
    g = overlap(modes, modes)
    return float(np.max(np.abs(g - np.diag(np.diag(g)))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="10,20,30,40")
    ap.add_argument("--wavelengths", default="1.55")
    ap.add_argument("--length", type=float, default=2.0)
    ap.add_argument("--segments", type=int, default=20)
    ap.add_argument("--w0", type=float, default=0.5)
    ap.add_argument("--w1", type=float, default=2.0)
    ap.add_argument("--step", type=float, default=0.02)
    ap.add_argument("--window", default="-2,2,-0.8,1.02",
                    help="x0,x1,y0,y1 (um) of the closed EME window (PEC walls); the default is the FDTD window")
    ap.add_argument("--interfaces", default="meow_invert,meow_none,conserving",
                    help="a subset of meow_invert, meow_none, conserving (meow's interface is slow above 40 modes)")
    ap.add_argument("--max-imag-neff", type=float, default=None,
                    help="conserving: remove modes with a larger |Im neff| (default: keep all; 1.0 is meow's rule)")
    ap.add_argument("--cascade", choices=["star", "sax"], default="star",
                    help="star: the star product of meow_gpu.interface (default); sax: meow's SAX cascade (klu)")
    ap.add_argument("--max-gb", type=float, default=8.0,
                    help="refuse a row whose estimated memory (mode fields, SAX cascade) is above this (GiB)")
    ap.add_argument("--pairs", choices=["both", "neither", "as_solved"], default="both",
                    help="conserving: the rule for a complex pair that the mode count cuts (default: both)")
    ap.add_argument("--pair-variants", default="",
                    help="more conserving variants on the same modes, one per rule (for example as_solved,neither)")
    ap.add_argument("--search-space", choices=["deflated", "full"], default="deflated",
                    help="full: the preconditioner of 505c4c5 (keeps the PEC-wall unknowns); default: the solver")
    ap.add_argument("--diag", action="store_true")
    a = ap.parse_args()
    if a.search_space == "full":
        import meow_gpu.meow_backend as backend
        from meow_gpu.fde.eigensolver import make_preconditioner
        backend.make_preconditioner = lambda q: make_preconditioner(q, deflate_walls=False)
    bad = [r for r in a.pair_variants.split(",") if r and r not in ("both", "neither", "as_solved")]
    if bad:
        ap.error(f"--pair-variants: unknown rules {bad}")

    import jax
    jax.config.update("jax_enable_x64", True)

    segs = segments(a.w0, a.w1, a.length, a.segments)
    x0, x1, y0, y1 = (float(v) for v in a.window.split(","))
    window = ((x0, x1), (y0, y1))
    for wl in [float(v) for v in a.wavelengths.split(",")]:
        cells, css = build(segs, a.step, wl, window)
        for n_modes in [int(v) for v in a.modes.split(",")]:
            try:                                      # the modes of a row are freed when one_row returns
                one_row(a, wl, n_modes, cells, css, window)
            except Exception as exc:                  # one JSON row with the error, then the next row
                traceback.print_exc()
                print(json.dumps({"case": "SOI taper", "length_um": a.length, "segments": a.segments,
                                  "step_um": a.step, "window": list(window), "wavelength": wl, "modes": n_modes,
                                  "error": f"{type(exc).__name__}: {exc}",
                                  "traceback": traceback.format_exc().splitlines()[-6:],
                                  "max_rss_gb": max_rss_gb(), **gpu_memory_gb()}), flush=True)


def max_rss_gb() -> float:
    """The peak resident memory of this process (Linux: ru_maxrss in kB)."""
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)


def gpu_memory_gb() -> dict:
    """The peak and the limit of the device memory pool of JAX on the first device (GiB; since 2026-11-08 night, after
    the GPU computer stopped answering during a 200-mode row), or None values where the backend gives no statistics."""
    try:
        import jax
        stats = jax.devices()[0].memory_stats() or {}
    except Exception:                                  # a backend without statistics: the row still prints
        stats = {}
    return {k: (round(stats[n] / 2**30, 2) if n in stats else None)
            for k, n in (("gpu_peak_gb", "peak_bytes_in_use"), ("gpu_limit_gb", "bytes_limit"))}


def one_row(a, wl, n_modes, cells, css, window):
    """Solve the modes of all cells and print the row of this wavelength and mode count."""
    import jax
    import meow as mw
    from meow.eme.interface import overlap_matrix as meow_overlap
    from meow.fde.post_process import filter_modes, normalize_modes

    from meow_gpu.interface import (complete_complex_pairs, compute_interface_s_matrices, interface_blocks,
                                    overlap_matrix, post_process_eigenmodes)
    from meow_gpu.meow_backend import GPUModeSolver, compute_modes_gpu, device_summary, post_process_modes

    meow_ip = functools.partial(meow_overlap, inner_product=mw.inner_product)

    def blocks_s(m1, m2, overlap):
        r_ll, t_lr, t_rl, r_rr = interface_blocks(m1, m2, overlap=overlap)
        s = np.block([[r_ll, t_rl], [t_lr, r_rr]]).T
        ports = [f"left@{i}" for i in range(len(m1))] + [f"right@{i}" for i in range(len(m2))]
        return jax.numpy.asarray(s), {p: i for i, p in enumerate(ports)}

    def conserving_with(overlap):
        return lambda modes, **_: {f"i_{i}_{i + 1}": blocks_s(m1, m2, overlap)
                                   for i, (m1, m2) in enumerate(zip(modes[:-1], modes[1:]))}

    solver = GPUModeSolver(precision="mixed")
    need_gs = a.diag or any(k.startswith("meow") for k in a.interfaces.split(","))
    nx, ny = np.shape(css[0].nx)
    lists = 1 + need_gs + (2 if a.diag else 0)        # eig, gs; with --diag also the raw and meow-normalized modes
    fields_gb = len(css) * n_modes * 6 * nx * ny * 16 * lists / 2**30
    if fields_gb > a.max_gb:
        raise MemoryError(f"the mode fields need about {fields_gb:.1f} GiB ({lists} lists of {len(css)} cells x "
                          f"{n_modes} modes on {nx} x {ny} samples; above --max-gb {a.max_gb}): lower --modes")
    # post-process each cell after its solve and keep the raw modes only for --diag: the modes of 22 cells take
    # about 1.67 MiB per mode on the 0.02 um grid (2.86 GiB for 22 cells x 80 modes, for each list)
    rules = [r for r in a.pair_variants.split(",") if r and r != a.pairs]
    raw, gs, eig, counts, t_solve, infos = [], [], [], np.zeros(2, int), 0.0, []
    by_rule = {r: [] for r in rules}         # the other rules share the mode objects of the same base set
    pair_counts = {"added": 0, "removed": 0}
    for cs in css:
        t0 = time.perf_counter()
        r = compute_modes_gpu(cs, num_modes=n_modes, solver=solver, post_process=lambda ms: ms)
        t_solve += time.perf_counter() - t0
        infos.append(solver.last_info)
        counts += [sum(abs(np.imag(m.neff)) > 1e-9 * abs(m.neff) for m in r),
                   sum(np.imag(m.neff) < -1e-9 * abs(m.neff) for m in r)]
        if need_gs:
            gs.append(post_process_modes(r))   # meow's Gram-Schmidt is slow
        base = post_process_eigenmodes(r, max_imag_neff=a.max_imag_neff, pairs="as_solved")
        eig.append(complete_complex_pairs(base, a.pairs))
        for rule in rules:
            by_rule[rule].append(complete_complex_pairs(base, rule))
        pair_counts["added"] += len(complete_complex_pairs(base, "both")) - len(base)
        pair_counts["removed"] += len(base) - len(complete_complex_pairs(base, "neither"))
        if a.diag:
            raw.append(r)
    row = {"case": "SOI taper", "w0": a.w0, "w1": a.w1, "length_um": a.length, "segments": a.segments,
           "step_um": a.step, "window": list(window), "wavelength": wl, "modes": n_modes, "cells": len(cells),
           "max_imag_neff": a.max_imag_neff, "pairs": a.pairs, "search_space": a.search_space,
           "pair_modes_added": pair_counts["added"], "pair_modes_removed": pair_counts["removed"],
           "min_modes_kept": min(len(m) for m in eig), "max_modes_kept": max(len(m) for m in eig),
           "min_modes_kept_gs": min(len(m) for m in gs) if need_gs else None,
           "complex_neff": int(counts[0]),
           "negative_re_neff": int(sum(np.real(m.neff) < -1e-9 * abs(m.neff) for ms in eig for m in ms)),
           "flipped_raw": int(counts[1]),
           "solve_s": round(t_solve, 1), "device": device_summary(), "cascade": a.cascade,
           "fields_gb_estimate": round(fields_gb, 2),
           "converged": all(i.converged for i in infos),
           "not_converged_cells": [k for k, i in enumerate(infos) if not i.converged],
           "residual_max": float(max(i.residual for i in infos))}
    variants = {
        "meow_invert": (gs, mw.compute_interface_s_matrices),
        "meow_none": (gs, functools.partial(mw.compute_interface_s_matrices, passivity_method="none")),
        "conserving": (eig, compute_interface_s_matrices),
    }
    variants = {k: v for k, v in variants.items() if k in a.interfaces.split(",")}
    if "conserving" in a.interfaces.split(","):
        for rule in rules:
            variants[f"conserving_{rule}"] = (by_rule[rule], compute_interface_s_matrices)
    if a.diag:
        # the formula of meow (no passivity rule) with the eigenmodes and the Yee product; the conserving
        # formula with the Gram-Schmidt modes and meow's product
        from meow_gpu.interface import yee_inner_product
        variants["meow_formula_yee"] = (eig, functools.partial(
            mw.compute_interface_s_matrices, inner_product=yee_inner_product, conjugate=False,
            passivity_method="none", tsvd_rcond=1e-12))
        variants["meow_formula_meowproduct_eig"] = (
            [normalize_modes(filter_modes(m), mw.inner_product) for m in raw],
            functools.partial(mw.compute_interface_s_matrices, passivity_method="none", tsvd_rcond=1e-12))
        variants["conserving_meowproduct_gs"] = (gs, conserving_with(meow_ip))
    for name, (modes, fn) in variants.items():
        t0 = time.perf_counter()
        res = cascade(modes, cells, fn, a.cascade, a.max_gb)
        res["s"] = round(time.perf_counter() - t0, 2)
        if a.diag:
            res["sigma_max"] = sigma_max(fn(modes))
        row[name] = res
    if a.diag:
        re_e = [float(np.max(np.abs(np.imag(m.Ex))) / np.max(np.abs(m.Ex))) for ms in eig for m in ms]
        row["gram_offdiag_yee_eig"] = max(gram_offdiag(m, overlap_matrix) for m in eig)
        row["gram_offdiag_meow_eig"] = max(gram_offdiag(m, meow_ip) for m in variants[
            "meow_formula_meowproduct_eig"][0])
        row["gram_offdiag_meow_gs"] = max(gram_offdiag(m, meow_ip) for m in gs)
        row["gram_offdiag_yee_gs"] = max(gram_offdiag(m, overlap_matrix) for m in gs)
        row["max_imag_over_abs_Ex"] = max(re_e)
        row["neff_first_cell"] = [round(float(np.real(m.neff)), 6) for m in raw[0]]
    row["max_rss_gb"] = max_rss_gb()
    row.update(gpu_memory_gb())
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
