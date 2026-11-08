# fdtd-gpu

A small FDTD front end on [fdtdx](https://github.com/ymahlau/fdtdx) (MIT, JAX) in the script style of the MIT FDTD
package (Python interface): `Simulation(cell_size, resolution, geometry, sources, boundary_layers)`, geometry objects
with materials by index, Gaussian-pulse sources in frequency units, flux and mode monitors, `run(until=...)` or run
until the fields decay. Users of that package can run their usual device scripts on an AMD GPU (tested on a Radeon
RX 7900 XT, gfx1100, ROCm 10.1.0, JAX 0.11.1, fdtdx 0.6.3). No code of the MIT package is used; the interface is
written from its documentation.

```python
import fdtd_gpu as fg                                             # um; frequency in 1/um (c = 1); time in um/c

si, sio2 = fg.Medium(index=3.45), fg.Medium(index=1.45)
core = fg.Block(size=fg.Vector3(fg.inf, 0.4, fg.inf), material=si)
src = fg.EigenModeSource(fg.GaussianSource(1 / 1.55, fwidth=0.15), center=fg.Vector3(-2.5), size=fg.Vector3(0, 2),
                         eig_band=1, eig_parity=fg.ODD_Z)        # 2-D TM (Ez) mode, launched along +x
sim = fg.Simulation(cell_size=fg.Vector3(8, 4, 0), resolution=20, geometry=[core], sources=[src],
                    boundary_layers=[fg.PML(1.0)], default_material=sio2)
mon = sim.add_mode_monitor(1 / 1.55, 0.15, 5, fg.ModeRegion(center=fg.Vector3(2.5), size=fg.Vector3(0, 2)))
flux = sim.add_flux(1 / 1.55, 0.15, 5, fg.FluxRegion(center=fg.Vector3(2.5), size=fg.Vector3(0, 2)))
sim.run(until_after_sources=fg.stop_when_fields_decayed(20, fg.Ez, fg.Vector3(2.5), 1e-6))
alpha = sim.get_eigenmode_coefficients(mon, [1], eig_parity=fg.ODD_Z).alpha  # (bands, freqs, 2): forward, backward
P = sim.get_fluxes(flux)
```

## Supported subset

Supported:
- **Cell:** `cell_size` (a size of 0 makes the cell 2-D or 1-D), `resolution` (pixels per um), centered at the
  origin.
- **Materials:** `Medium(epsilon=...)` or `Medium(index=...)`: lossless, non-dispersive, isotropic.
- **Geometry:** `Block` (axis-aligned), `Cylinder` and `Prism` along x, y or z, `Sphere`. Later objects replace
  earlier ones. `eps_averaging=True` (the default, as in the MIT package): subpixel averaging of the permittivity at
  each Yee E component (see "Differences"); `eps_averaging=False`: the staircase.
- **Boundaries:** `PML(thickness, direction, side)` inside the cell. Other faces are metal (PEC), or periodic with
  `k_point=Vector3()`.
- **Sources:** `Source(GaussianSource or ContinuousSource, component, center, size, amplitude)`: a point, line or
  plane current. `EigenModeSource(src, center, size, eig_band, eig_parity, eig_kpoint)`.
- **Monitors:** `add_flux` (`FluxRegion` with weights; `get_flux_data`, `load_minus_flux_data`), `add_mode_monitor`
  with `get_eigenmode_coefficients(mon, bands, eig_parity)` (alpha and neff), `get_flux_freqs`, `get_dft_array`,
  `get_epsilon`.
- **Run:** `run(until=T)`, `run(until_after_sources=T)`, `stop_when_fields_decayed(dt, c, pt, decay_by)`,
  `Harminv(c, pt, fcen, df)` with `after_sources`.

Not supported (the option raises NotImplementedError, or the name is not in the package): `geometry_center`,
`symmetries`, non-uniform grids; dispersion, conductivity, anisotropy, nonlinearity; rotated blocks (`e1`, `e2`,
`e3`), tilted axes, sloped prism sidewalls, `MaterialGrid`; Bloch phases (`k_point` other than 0); complex
amplitudes, `amp_func`, custom source times, `eig_match_freq=False`; field arrays (`get_array`), near-to-far, force,
energy and LDOS monitors; other step functions; a second `run` on the same `Simulation`.

`eig_parity` in 3-D: `ODD_Z` (TM-like: E mostly along z) and `EVEN_Z` (TE-like: E mostly in the plane normal to z)
select the polarization in the mode solver; a y parity is ignored. In 2-D only the mode with E along the trivial
axis is supported (Ez in an x-y cell: `NO_PARITY` or `ODD_Z`). An in-plane (Hz, `EVEN_Z`) 2-D mode raises
NotImplementedError in `EigenModeSource` and `get_eigenmode_coefficients`: on the three periodic cells of the trivial
axis the fdtdx mode solver gives NaN fields (source) or silently the Ez mode (monitor). 1-D cells have no mode
sources or mode monitors.

## Differences from the MIT package (read before you compare numbers)

- **Subpixel averaging** (`eps_averaging=True`, the default; `fdtd_gpu/_averaging.py`). Each E component of each
  cell gets its own permittivity: the average over the cube of side dx centered at the component. fdtdx keeps the
  Yee layout of Taflove; this package puts E_a at the cell center plus dx / 2 along a. The rule is the diagonal of
  the anisotropic average of Kottke, Farjadpour and Johnson (Phys. Rev. E 77, 036611, 2008):
  1 / eps_aa = n_a^2 <1/eps> + (1 - n_a^2) / <eps>, with n the interface normal. fdtdx gets three permittivity
  components per cell and keeps its elementwise update: the FDTD loop takes the same time (ring at res 80: 8.83 s
  averaged, 8.79 s staircase). Differences from the MIT package:
  - only the diagonal: the off-diagonal terms (the slanted part of the normal) are dropped. The rule is exact for
    an interface normal to an axis, and second order for E parallel to all interfaces (checks 3, 6, 7). For E in
    the plane of a curved interface (a 2-D disk or ring with Hz; in 3-D the sidewalls of rings and bends) the error
    stays first order, of the size of the staircase error, but the Q is no longer limited by the staircase (check
    8). The full tensor needs the 9-component (anisotropic) update of fdtdx (its documentation: about 3 times the
    cost per step; not measured here);
  - <eps> and <1/eps> come from fixed sample points (a rank-1 lattice of 1024, 1974 and 2042 points per voxel in
    1-D, 2-D and 3-D), not from an adaptive integration with a tolerance (`subpixel_tol` is not accepted). The
    fill fraction of a plane normal to an axis is exact to 1 / (2 n) = 5e-4;
  - the normal is the weighted first moment of eps on a regular grid (24 x 24 in 2-D, 12 x 12 x 12 in 3-D) over
    the circumscribed ball of the voxel, not the analytic normal of the object. For a slanted plane the averaged
    components are within 2 % (median 0.5 %) of the exact diagonal (`test_slanted_plane_kottke_diagonal`);
  - only voxels next to a change of the staircase material are sampled: a feature thinner than one cell that
    holds no cell center is not seen (as by the staircase);
  - the JAX path (float32) can put a sample point within about 1e-7 um of a surface on the other side: one of
    the about 2000 points of a voxel, up to 1.3e-3 of eps in 1.3 % of the voxels of the test geometry
    (`test_numpy_and_jax_agree`).
  `eps_averaging=False` gives the staircase (each cell takes the material of its center), which converges at
  about first order in the cell size. Setup time (`--case setup`: a 3-D strip and ring in an 8 x 4 x 4 um cell):
  3.48 s at res 20 (1.0 M cells, 36,688 sampled voxels; with the start of JAX and the compile), then 2.92 s at
  res 50 (16 M cells, 247,880 voxels) in the same process. The 2-D checks take 0.02 to 0.7 s. Small problems are
  sampled with NumPy, larger ones on the GPU. `get_epsilon` returns 3 / (1/eps_xx + 1/eps_yy + 1/eps_zz) of each
  cell (each component at its own Yee point).
- **float32 fields**, and the time step of fdtdx: dt = 0.99 dx / (c sqrt 3) in all dimensions (`Courant` is
  ignored).
- **One-way mode sources.** `EigenModeSource` is the fdtdx TFSF mode source: the mode is solved at the center
  frequency only, and its absolute amplitude differs from the MIT package. Normalize by an input monitor or a
  reference run (as usual). A `Source` with a size emits in both directions, as in the MIT package.
- **Gaussian pulse.** Current exp(-(t - t0)^2 / (2 w^2)) cos(2 pi f t) with w = 1 / fwidth (the spectral width of
  the MIT package), but the peak is at t0 = 6 w (not 5 w) and the source ends at 12 w.
  `GaussianSource.fourier_transform(f)` gives its spectrum in the DFT convention of the monitors.
- **DFT and flux.** Phasors sum f(t) exp(i 2 pi f t) dt / sqrt(2 pi) (the convention of the MIT package). The fdtdx
  detectors co-locate all fields on the E_z Yee point and take H as the mean of H before and after the step, so E
  and H are at the same time (check 5: arg(H / E) of a plane wave below 2e-4 rad). The flux of the Yee grid is
  high by O((k dx)^2) (check 5). The cells of a monitor plane have the trapezoidal weight 1/2 at the plane edges,
  so that six planes close a box.
- **Mode coefficients.** alpha+- = (int E x Hm* +- int Em* x H) / (2 sqrt(int Re Em x Hm*)), so |alpha|^2 is a
  flux. On the Yee grid the co-located fields are averages of neighbor points (for a plane normal to x or y, the
  transverse H is an average of two points along the normal); this gives a small false backward coefficient
  (check 2). `EigCoeffs.neff` is the effective index from the mode solver at each frequency.
- **Harminv** is a matrix pencil fit of the signal after the sources (`fdtd_gpu.harminv`), not filter
  diagonalization. It uses only the band [fcen - df/2, fcen + df/2]. `freq`, `decay` and `Q` have the meaning of
  the MIT package; `amp` and `err` do not: `amp` is the complex amplitude of the frequency-shifted and low-pass
  filtered signal (about half of the real cosine amplitude, times the gain of the filter), and `err` is one relative
  residual of the whole fit (the same number for all modes).
- **Prism center.** As in the MIT package: with `center`, the centroid of all vertices (the mean of the vertices,
  moved by height / 2 along the axis) goes to `center`.
- **2-D and 1-D** cells have 3 periodic cells on each axis of size 0 (fdtdx needs one axis of one cell to find the
  normal of a plane). Fluxes in 2-D are per unit length, in 1-D per unit area, as in the MIT package.
- Runs use the first GPU (`jax.default_device`) inside `jax.enable_x64(False)`.

## Checks (each against an independent reference)

`tests/cases.py` has the cases and `tests/references.py` the references. `tests/test_validation.py` (checks 1 to 5)
and `tests/test_averaging.py` (checks 6 to 8, and the convergence of check 3) run them on small grids;
`bench/bench_fdtd.py` runs them on larger grids (one JSON line per case, with the resolution and the cells). The
numbers of checks 1 to 3 and 6 to 8, of the res-20 rows of checks 4 and 5, and the setup times are from the 39
rows of `benchmarks/2026-11-07/fdtd-gpu-avg.jsonl` (subpixel averaging on, unless the text says "staircase"; RX
7900 XT, 2026-11-07: `--case all`, the two 3-D waveguide commands and the six last commands under "Tests and
bench", in that order; other jobs can share the computer, so times can change between runs). Line 31 of that file
(check 7, staircase, res 160) is cut by a warning message; its value is from a repeat of the same command on the
same day (max abs(R - R_exact) 0.04655, max abs(R + T - 1) 3.4e-5). The other rows of checks 4 and 5 are from
version 0.1 (the staircase only) in `benchmarks/2026-10-30/fdtd-gpu.jsonl`: these checks have no geometry, so the
averaging does not change them (the res-20 rows of the two files are equal). "res" is the resolution in cells per
um; times are the FDTD loops (with compile).

1. **1-D Bragg mirror** (quarter-wave stack at 1 um, n 2.0 / 1.25, 6 pairs, in air; 41 frequencies from 0.6 to
   1.4 / um). R from a flux plane with the incident fields subtracted (`load_minus_flux_data`), T from a flux plane
   behind the stack. Reference: exact transfer matrices (Born and Wolf).
   - res 200: max abs(R - R_exact) 0.0024, max abs(R + T - 1) 5.3e-5 (6.4 s for both runs). The error is the
     numerical dispersion of the Yee grid (the stop band moves to a lower frequency). The layer faces are on cell
     faces, so the averaged permittivity is the staircase (check 7 puts them between grid lines). The test (res 80)
     requires max abs(R - R_exact) < 0.02 and abs(R + T - 1) < 1e-3.
2. **Straight Si strip** (n 3.45 in 1.45, width 0.4 um; 2-D Ez, and 3-D with a 0.22 um core). Mode source, mode
   monitors and flux planes 1 um after the source and 0.5 um before the output PML; 5 frequencies from 0.570 to
   0.720 / um. Reference: a lossless straight guide transmits 1.
   - 2-D, res 20: T_mode 0.99991 to 1.000000, T_flux 0.99916 to 1.000000. False backward coefficient
     abs(alpha-)^2 / abs(alpha+)^2: 4.4e-4 to 1.3e-3 (the co-located fields of the Yee grid). abs(alpha+)^2 / input
     flux: 1.0001 to 1.0009. neff from the mode solver: 3.117 to 3.215, against 3.108 to 3.207 for the exact slab
     (`references.slab_te_neff`; +0.008 to +0.009 from the discretization of the mode solver at 0.05 um). The faces
     are on grid lines, so these numbers are those of the staircase.
   - 3-D, res 20 (1.0 M cells, 2.8 s; the average 0.69 s): T_mode 0.9979 to 1.00003. T_flux 0.99999 at the center
     frequency, 0.9855 and 0.990 at the band edges. At the input, abs(alpha+)^2 is 0.954 to 0.974 of the flux (4.4
     cells across the core). neff from the mode solver 1.948 to 2.412 (staircase 1.900 to 2.373: the staircase core
     has 4 cells, 0.20 um, along z).
   - Cause of the low T_flux at the band edges: the mode source is solved at its center frequency only, so at
     other frequencies it also launches radiation that crosses the input plane and leaves before the output plane.
     Run with the source centered (and its mode solved) at the edge frequency 0.570 / um (`--src-freq 0.5701613`):
     T_flux at 0.570 / um is 1.000001 (0.9855 before), and it falls at the other frequencies (0.9945 at 0.608 / um,
     0.978 at 0.645 / um). Outside the band of the source the transmission is noise: there the source spectrum is
     1.5e-5 (0.683 / um) and 2.7e-9 (0.720 / um) of its peak, both fluxes are float32 noise, and their ratio has no
     meaning (row 7 of the 0.1 rows gave T_mode 32.8 and T_flux 37.8 at 0.720 / um). The case gives `src_rel` (the
     amplitude spectrum of the source over its peak) and reports T_mode and T_flux as null where src_rel < 1e-3
     (`cases.SRC_REL_MIN`).
3. **2-D Si ring** (n 3.4 in air, inner radius 1.5 um, width 0.22 um; Ez), a short dipole in the ring, `Harminv` on
   Ez after the sources. Reference: the exact resonances of the dielectric ring (Bessel and Hankel functions,
   `references.ring_resonance`), azimuthal orders m = 15 to 21. E is parallel to the interfaces, so the averaged
   rule is the arithmetic mean.
   - Frequency error of the strongest mode of each m, averaged: -0.17 to -0.34 % at res 40 (2.6 s), -0.08 to
     -0.15 % at res 60 (4.1 s), -0.04 to -0.09 % at res 80 (8.8 s): second order (res 40 to 80: 1.97 to 2.02 for
     each m). What remains is mostly the dispersion of the Yee grid (the frequencies are low). Staircase: -0.43 to
     -0.69 %, -0.02 to -0.20 %, -0.09 to -0.14 % (no clear order: the staircase error and the dispersion change
     sign against each other).
   - Free spectral range of neighbor orders, averaged: 0.0275 to 0.0280 (res 40), 0.0277 to 0.0281 (res 60),
     0.0277 to 0.0282 (res 80); staircase 0.0266 to 0.0285, 0.0274 to 0.0287, 0.0275 to 0.0283; exact 0.0278 to
     0.0282.
   - Q (exact: above 1e9): averaged 1.2e5 to 1.4e7 (res 40), 1.1e7 to 8.5e7 (res 60), 1.4e7 to 1.2e8 (res 80);
     staircase 5.6e3 to 1.5e4, 4.9e4 to 6.0e5, 4.7e4 to 4.3e5 (the rough staircase circle scatters, and a mode can
     split into a pair; the check uses the strongest mode of each m). At a Q above about 1e6 the fit can give a
     small growth (Q < 0); the cases use abs(Q).
   - The test (res 60) requires errors below 0.5 % for the three strongest modes and the free spectral range within
     2 %; `test_averaging.py` requires the order above 1.7 (res 40 to 80) for m = 17 to 20, errors below 0.12 % at
     res 80 and Q above 5e4 at res 40.
4. **3-D point dipole** (Ez) in a homogeneous medium, flux through a closed box of half width 1 um, 5 frequencies
   from 0.4 to 0.6 / um. Reference: the analytic radiated power n omega^2 abs(I)^2 / (12 pi) of a dipole current I
   (Jackson), with the source spectrum of `GaussianSource.fourier_transform` (Purcell factor 1).
   - n = 1: 1.0022 to 1.0060 at res 10, 1.0006 to 1.0015 at res 20 (1.0 M cells, 1.7 s), 1.0002 to 1.0007 at
     res 30: second order in the cell size. n = 1.5, res 20: 1.0029 to 1.0066.
   - A box of half width 0.5 um, res 10: 0.99996 to 1.0036.
   - The test requires the error below 3e-3 at res 20 and an observed order above 1.7 (res 10 to 20) in vacuum,
     and below 1 % for n = 1.5 at res 20.
5. **1-D current sheet** (Ex) in a homogeneous medium, flux plane 3 um away, 7 frequencies from 0.4 to 1.6 / um.
   Reference: the exact flux abs(I)^2 / (4 n) per side, and E and H in phase.
   - Vacuum: flux / exact 1.0014 to 1.0220 at res 20, 1.0006 to 1.0045 at res 40 (second order at the high
     frequencies: the Yee grid makes abs(E / H) low by up to 2.2 % at res 20, 1.6 / um). n = 1.5, res 20: 1.0037 to
     1.071. arg(Hy / Ex) below 1.8e-4 rad in all rows.

6. **Subpixel averaging: periodic slab** (n 3.45 in 1.45, width 0.37 um, so the faces are between grid lines;
   periodic along x with the period 0.5 um, so the guided mode has beta = 2 pi / 0.5 um; a dipole at the center,
   `Harminv` after the sources). Reference: `references.slab_bloch_frequency`, the exact slab mode with the exact Yee
   dispersion along x and in time, so the remaining error is the discretization across the slab (the geometry).
   Relative frequency error, averaged / staircase:

   | res | Ez (E parallel to the faces) | Hz (E normal to the faces, `--e-normal`) |
   |---|---|---|
   | 20 | -2.2e-4 / +4.3e-3 | +3.7e-3 / +1.24e-2 |
   | 40 | -2.3e-4 / -2.4e-3 | -2.0e-4 / -4.0e-3 |
   | 80 | +1.7e-5 / -1.9e-3 | -8.0e-5 / -3.5e-3 |

   As an effective index (neff = beta / (2 pi f)) at res 40: averaged +7.2e-4 (Ez) and +5.9e-4 (Hz), staircase
   +7.6e-3 and +1.17e-2. The staircase rounds the width to whole cells and stops converging; the averaged error is
   10 to 100 times smaller. It is second order in its envelope only: its coefficient changes with the position of
   the face in the cell (20 to 40 for Ez: no gain; 40 to 80: 14 times). The test (res 40) requires the averaged
   error below 1e-3 and the staircase error at least 4 times larger, for both polarizations.
7. **Subpixel averaging: Bragg mirror with faces between grid lines** (check 1 with lambda0 = 1.03 um, `--lam0
   1.03`: layers 0.12875 and 0.206 um). Max abs(R - R_exact) at res 40 / 80 / 160: averaged 0.114 / 0.0265 /
   0.0062 (order 2.1 and 2.1; what remains is the dispersion of the Yee grid, as in check 1), staircase 0.257 /
   0.049 / 0.047 (no gain from 80 to 160). abs(R + T - 1) below 3.4e-5 in all rows. The test requires the order
   above 1.8 (res 80 to 160), the averaged error below 1e-2 at res 160 and the staircase error 4 times larger.
8. **Subpixel averaging: 2-D disk with in-plane E** (n 3.4 in air, radius 1.0 um, Hz; an Ex dipole 0.1 um inside
   the rim; `--case disk`). Reference: the exact resonances of the dielectric disk (`references.disk_resonance`,
   Hz and (1 / eps) dHz/dr continuous). Here the normal turns through all angles, and the diagonal rule drops the
   off-diagonal terms of the tensor of Kottke et al. Frequency error of m = 8, 9, 10:

   | res | averaged | staircase |
   |---|---|---|
   | 40 | +0.061, 0.000, -0.054 % | +0.088, -0.014, +0.001 % |
   | 60 | +0.096, +0.074, +0.053 % | +0.119, +0.069, +0.069 % |
   | 80 | +0.092, +0.081, +0.067 % | +0.115, +0.055, +0.035 % |

   Neither converges at second order. The averaged errors fit e = G / res - D / res^2 (a first-order geometry error
   and the second-order dispersion of the grid; G and D from res 40 and 60 predict res 80 to 1e-4) with G = 0.12 to
   0.14 um^-1: about +0.16 % at res 80 before the dispersion, of the same size as the staircase error. This is the
   known limit of the diagonal rule (see "Differences"). The averaged circle has no staircase roughness: Q of
   m = 8 to 10 is 7.6e4 to 2.0e5 (res 40), 1.4e5 to 8.9e5 (res 60), 2.7e5 to 1.1e6 (res 80); staircase 4.8e3 to
   4.3e4, 9.3e3 to 1.6e4, 1.2e4 to 4.1e4. The test (res 60) requires errors below 0.2 % and the averaged Q of
   m = 8, 9 at least 10 times the staircase Q.

Tests: 24 (13 without FDTD runs: geometry, Prism center, 2-D parities, Harminv, the averaged permittivity against
exact fill fractions and the formula of Kottke et al., NumPy against JAX, the transfer to fdtdx; 11 FDTD checks,
some with more than one run).

## Tests and bench (on the GPU computer)

```bash
bash tools/gpu_run.sh fdtd_gpu env PYTHONPATH=src python -m pytest -q
B="env PYTHONPATH=src python bench/bench_fdtd.py"
bash tools/gpu_run.sh fdtd_gpu $B --case all          # bragg 200, waveguide 20, ring 40, dipole 20, sheet 20
bash tools/gpu_run.sh fdtd_gpu $B --case waveguide --three-d
bash tools/gpu_run.sh fdtd_gpu $B --case waveguide --three-d --src-freq 0.5701613
bash tools/gpu_run.sh fdtd_gpu $B --case ring --resolution 60
for a in "--resolution 10" "--resolution 30" "--resolution 20 --index 1.5" "--resolution 10 --half 0.5"; do
  bash tools/gpu_run.sh fdtd_gpu $B --case dipole $a; done
for a in "--resolution 40" "--resolution 20 --index 1.5"; do bash tools/gpu_run.sh fdtd_gpu $B --case sheet $a; done
bash tools/gpu_run.sh fdtd_gpu $B --case ring --resolution 40,60,80 --both          # check 3, averaged and staircase
bash tools/gpu_run.sh fdtd_gpu $B --case slab --resolution 20,40,80 --both          # check 6 (Ez)
bash tools/gpu_run.sh fdtd_gpu $B --case slab --resolution 20,40,80 --both --e-normal   # check 6 (Hz)
bash tools/gpu_run.sh fdtd_gpu $B --case bragg --resolution 40,80,160 --both --lam0 1.03   # check 7
bash tools/gpu_run.sh fdtd_gpu $B --case disk --resolution 40,60,80 --both          # check 8
bash tools/gpu_run.sh fdtd_gpu $B --case setup --resolution 20,50                   # time of the average
```

If the package is not installed in `.venv-rocm`, use `PYTHONPATH=src`. It needs fdtdx 0.6.3 (with tidy3d for its mode
solver), JAX, NumPy and SciPy.

## License

The code of this package is MIT-licensed (the `LICENSE` file at the root of the repository). fdtdx is MIT-licensed;
its mode solver uses tidy3d (LGPL-2.1 or later) as a library. The interface follows the published documentation of
the MIT FDTD package (GPL); no code of that package is used.
