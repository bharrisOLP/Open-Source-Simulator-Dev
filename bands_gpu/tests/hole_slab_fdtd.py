"""The hole slab of the MPB example hole-slab (after S. G. Johnson et al., "Guided modes in photonic crystal slabs",
Phys. Rev. B 60, 5751 (1999); air holes r 0.3 a in a slab of eps 12 and thickness 0.5 a) in 3-D FDTD with fdtd_gpu:
an independent method (Yee grid, time stepping, Harminv) for the z-even (TE-like) bands of bands_gpu at K and M.

fdtd_gpu has no Bloch phases (only k_point = 0, periodic faces) and cubic voxels. So:

- The k point folds onto Gamma in a rectangular supercell of the primitive lattice a1 = (1, 0), a2 = (1/2, Ly/2):
  M = b2 / 2 in the cell 1 x Ly (2 holes), K = (b2 - b1) / 3 in the cell 3 x Ly (6 holes). Real dipoles at the same
  point of each primitive cell, with the weights cos(2 pi k . R) (R: the translation of that cell), excite only the
  Bloch modes at k and -k (the same frequencies, time reversal): the weights are orthogonal to the phases of the other
  k points that fold onto Gamma. The probe signals are summed with the same weights before Harminv.
- The y period of the supercell must be a whole number of voxels, and the translation a2 too (then the grid is
  exactly periodic with the primitive lattice and the weights select k exactly). sqrt(3) is irrational, so the FDTD
  crystal has Ly = 7/4 (`LY`) at resolutions that are multiples of 8: a crystal 1.04 % longer along y than the
  triangular one (a1 = (1, 0), a2 = (1/2, 7/8); "the 7/4 crystal"). bands_gpu solves the same crystal in an oblique
  cell, so the comparison is at the same eps; bands_gpu also gives the change to the triangular crystal.
- z-even: dipoles E_x and E_y at z = 0, probes of E_x and E_y at z = 0. With an odd number of cells along z, the plane
  z = 0 holds the E_x and E_y nodes and the Yee grid is mirror symmetric. The averaged eps of fdtd_gpu is not exactly
  mirror symmetric (its sample lattice is centrally symmetric about each voxel center, not mirror symmetric: the
  E_x and E_y components of the voxels at the slab faces differ from their mirror voxels), so the z-odd modes, which
  have E_x = E_y = 0 at z = 0 in a symmetric structure, are suppressed but not exactly excluded. parity="odd" is the
  control: E_z dipoles and probes at z = +-dx/2 (the two E_z nodes next to z = 0, the same sign) for the z-odd modes.
- PML above and below the slab (fdtd_gpu PML, along z only); the in-plane faces are periodic.

eps at the interfaces: fdtd_gpu keeps the diagonal of the inverse tensor of Kottke et al. (Phys. Rev. E 77, 036611,
2008) at each Yee E component (`fdtd_gpu._averaging`, DIAGONAL "inverse"; or "forward", the diagonal of the forward
tensor); bands_gpu uses the full tensor (with the analytic normal of the hole). The off-diagonal terms that fdtd_gpu
drops make its error first order in dx for E in the plane of the curved hole wall (fdtd_gpu/README.md, check 8).
"""

from __future__ import annotations

import math
import time

import numpy as np

EPS_SLAB, RADIUS, THICKNESS = 12.0, 0.3, 0.5
LY = 7 / 4                        # the y period of the rectangular cell (sqrt(3) in the triangular crystal)
SRC_OFFSET = (0.31, 0.17)         # the dipoles, from a hole center (in the slab material: |d| = 0.354 > r)
PROBE_OFFSET = (-0.13, 0.36)      # the probes, from a hole center (|d| = 0.383)
THETA = 0.6                       # the in-plane direction of the dipoles (rad from x): E_x and E_y both excited
POINTS = {"G": (0.0, 0.0), "M": (0.0, 0.5), "K": (-1 / 3, 1 / 3)}      # reduced coordinates (b1, b2)


def reciprocal(ly: float = LY) -> np.ndarray:
    """b1, b2 (rows; units 1/a, without 2 pi) of a1 = (1, 0), a2 = (1/2, ly/2): a_i . b_j = delta_ij."""
    return np.linalg.inv(np.array([[1.0, 0.0], [0.5, ly / 2]])).T


def k_cartesian(point: str, ly: float = LY) -> np.ndarray:
    """The Cartesian k (units 2 pi / a) of a point "G", "M" or "K" (reduced coordinates as bands_gpu)."""
    return np.asarray(POINTS[point]) @ reciprocal(ly)


def supercell(point: str, ly: float = LY) -> tuple[int, list[tuple[float, float]]]:
    """(nx, R): the supercell nx x ly in which k folds onto Gamma, and the translations R of its primitive cells."""
    nx = 3 if point == "K" else 1
    R = [(i + 0.5 * j, j * ly / 2) for j in (0, 1) for i in range(nx)]
    k = k_cartesian(point, ly)
    for v in ((nx, 0.0), (0.0, ly)):                 # k . (supercell vector) must be a whole number: k folds onto 0
        if abs(k @ v - round(k @ v)) > 1e-12:
            raise ValueError(f"{point} does not fold onto Gamma in the cell {nx} x {ly}")
    return nx, R


def bloch_weights(point: str, ly: float = LY) -> np.ndarray:
    """cos(2 pi k . R) for the translations R of supercell(point)."""
    _, R = supercell(point, ly)
    return np.cos(2 * np.pi * (np.asarray(R) @ k_cartesian(point, ly)))


def run_fdtd(point: str, res: int, ly: float = LY, air: float = 1.5, pml: float = 1.0, fcen: float = 0.3,
             df: float = 0.24, fwidth: float = 0.3, t_after: float = 150.0, diagonal: str = "inverse",
             shift: tuple[float, float] = (0.0, 0.0), parity: str = "even") -> dict:
    """One fdtd_gpu run of the hole slab with the Bloch weights of `point`. air: the air above and below the slab (a)
    before the PML; t_after: the time after the source (a / c); shift: a shift of the whole crystal (holes, sources,
    probes) in the plane, in voxels (without a shift the hole centers are on voxel corners at each resolution that is
    a multiple of 8); parity: "even" (E_x and E_y at z = 0) or "odd" (E_z at z = +-dx/2, the control). Returns the
    cell, the Harminv modes of the weighted sum of the probes (freq, Q, amplitude over the largest), the run time and
    the signal."""
    import fdtd_gpu as fg
    import fdtd_gpu._averaging as avg

    nx, R = supercell(point, ly)
    lx, dx = float(nx), 1.0 / res
    for v in [lx, ly] + [c for r in R for c in r]:          # the grid is exactly periodic with a1 and a2
        if abs(v * res - round(v * res)) > 1e-9:
            raise ValueError(f"resolution {res}: {v} is not a whole number of voxels (use a multiple of 8 for 7/4)")
    nz = int(round((THICKNESS + 2 * air + 2 * pml) * res)) | 1         # odd: z = 0 is the center of a voxel
    lz = nz * dx

    def snap(p: float, length: float) -> float:
        """The center of the voxel that holds p, after p is wrapped into [-length/2, length/2)."""
        p = (p + length / 2) % length - length / 2
        return -length / 2 + (math.floor((p + length / 2) / dx) + 0.5) * dx

    h0 = (-lx / 2 + 0.25 + shift[0] * dx, -0.5 + shift[1] * dx)
    holes = [((h0[0] + rx + lx / 2) % lx - lx / 2, (h0[1] + ry + ly / 2) % ly - ly / 2) for rx, ry in R]

    def points(offset):
        base = (snap(h0[0] + offset[0], lx), snap(h0[1] + offset[1], ly))
        return [(snap(base[0] + rx, lx), snap(base[1] + ry, ly)) for rx, ry in R]

    if parity == "even":            # (component, z of the cell, amplitude factor): E_x and E_y in the cell at z = 0
        nodes = [(fg.Ex, 0.0, math.cos(THETA)), (fg.Ey, 0.0, math.sin(THETA))]
    elif parity == "odd":           # E_z of the cells at z = 0 and z = -dx: the nodes z = +dx/2 and -dx/2
        nodes = [(fg.Ez, 0.0, 1.0), (fg.Ez, -dx, 1.0)]
    else:
        raise ValueError("parity: 'even' or 'odd'")
    w = bloch_weights(point, ly)
    src = fg.GaussianSource(fcen, fwidth=fwidth)
    sources = [fg.Source(src, comp, center=fg.Vector3(x, y, z), amplitude=c * f)
               for (x, y), c in zip(points(SRC_OFFSET), w) if abs(c) > 1e-12 for comp, z, f in nodes]
    probes = [fg.Harminv(comp, fg.Vector3(x, y, z), fcen, df)
              for x, y in points(PROBE_OFFSET) for comp, z, _ in nodes]
    geometry = [fg.Block(size=fg.Vector3(fg.inf, fg.inf, THICKNESS), material=fg.Medium(epsilon=EPS_SLAB))]
    geometry += [fg.Cylinder(radius=RADIUS, height=THICKNESS, center=fg.Vector3(x, y, 0), material=fg.air)
                 for x, y in holes]
    old, avg.DIAGONAL = avg.DIAGONAL, diagonal
    try:
        t0 = time.perf_counter()
        sim = fg.Simulation(cell_size=fg.Vector3(lx, ly, lz), resolution=res, geometry=geometry, sources=sources,
                            boundary_layers=[fg.PML(pml, direction=fg.Z)], k_point=fg.Vector3())
        sim.run(*probes, until_after_sources=t_after)
        wall = time.perf_counter() - t0
    finally:
        avg.DIAGONAL = old
    sig = sum(c * (probes[2 * j].data + probes[2 * j + 1].data) for j, c in enumerate(w))
    modes = fg.harminv(sig, sim.dt, fcen - df / 2, fcen + df / 2)
    amax = max((abs(m.amp) for m in modes), default=1.0)
    return {"shape": list(sim.shape), "cells": int(np.prod(sim.shape)), "steps": int(sim.n_steps),
            "dt": sim.dt, "run_s": round(sim.run_seconds, 3), "setup_s": round(sim.setup_seconds, 3),
            "wall_s": round(wall, 3), "n_averaged": int(sim.n_averaged), "device": sim.device,
            "fit_err": modes[0].err if modes else None,
            "modes": [[m.freq, m.Q, abs(m.amp) / amax] for m in modes], "signal": sig}


def strong_modes(modes, q_min: float = 1e3, rel_amp: float = 1e-2) -> list[float]:
    """The frequencies of the modes with |Q| >= q_min and an amplitude of at least rel_amp of the largest, in order.
    (|Q|: at a Q above about 1e6 the fit can give a small growth, Q < 0; fdtd_gpu/README.md, check 3.)"""
    return [f for f, q, a in modes if abs(q) >= q_min and a >= rel_amp]


def bands_cell(n: int, ly: float | None = LY, height: float | None = None, layers: int = 4):
    """The bands_gpu cell of the hole slab: an oblique cell a1 = (1, 0), a2 = (1/2, ly/2) (ly None: the triangular
    lattice), n x n in the plane, layers n + 1 points along z (odd for the z parity); height (a) None: (layers n + 1)
    / n, so that dz = 1 / n and the slab faces (z = +-0.25) are on voxel faces (n a multiple of 4) or voxel centers
    (n = 2 mod 4): the subpixel samples then give the exact fill fraction (README, the slab table). The height 4 of
    bench/bench_hole_slab.py puts the faces on subpixel samples (tests/test_slab3d.py)."""
    from bands_gpu.geometry import TRIANGULAR, Block, Cell, Cylinder

    nz = layers * n + 1
    h = nz / n if height is None else height
    if ly is None:
        basis, size = TRIANGULAR, (1.0, 1.0, h)
    else:
        basis, size = ((1.0, 0.0, 0.0), (0.5, ly / 2, 0.0), (0.0, 0.0, 1.0)), (1.0, math.hypot(0.5, ly / 2), h)
    return Cell(size=size, shape=(n, n, nz), basis=basis,
                objects=(Block(size=(np.inf, np.inf, THICKNESS), eps=EPS_SLAB),
                         Cylinder(radius=RADIUS, height=THICKNESS, eps=1.0)))


def even_bands(cell, points=("K", "M"), num: int = 4, sub: int = 8, tol: float = 1e-8,
               parity: str = "even") -> dict:
    """The z-even bands of bands_gpu (num per point; parity="odd": the z-odd bands) at the points, with the convergence
    flags and the iterations."""
    from bands_gpu.findk import BandEvaluator

    ev = BandEvaluator(cell, num, None, parity=parity, mirror_axis=2, tol=tol, sub=sub)
    out = {}
    for p in points:
        f, _, ok = ev.solve((*POINTS[p], 0.0))
        out[p] = (np.asarray(f, float), bool(ok))
    out["iterations"] = ev.iterations
    return out
