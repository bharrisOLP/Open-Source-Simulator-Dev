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
  earlier ones.
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

- **Staircase geometry.** Each cell takes the material of its center (no subpixel averaging; `eps_averaging=True`
  raises). Curved and slanted boundaries converge at about first order in the cell size (the ring check below).
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

`tests/cases.py` has the five cases and `tests/references.py` the references. `tests/test_validation.py` runs them
on small grids; `bench/bench_fdtd.py` runs them on larger grids (one JSON line per case, with the resolution and the
cells). All numbers are from bench rows of this version on the RX 7900 XT (2026-10-30; the commands are under "Tests
and bench"); "res" is the resolution in cells per um; times are the FDTD loops (with compile).
The bench rows are in `benchmarks/2026-10-30/fdtd-gpu.jsonl` (seconds, steps and cells of each case).

1. **1-D Bragg mirror** (quarter-wave stack at 1 um, n 2.0 / 1.25, 6 pairs, in air; 41 frequencies from 0.6 to
   1.4 / um). R from a flux plane with the incident fields subtracted (`load_minus_flux_data`), T from a flux plane
   behind the stack. Reference: exact transfer matrices (Born and Wolf).
   - res 200: max abs(R - R_exact) 0.0024, max abs(R + T - 1) 3.6e-5 (4.2 s for both runs). The error is the
     numerical dispersion of the Yee grid (the stop band moves to a lower frequency). The test (res 80) requires
     max abs(R - R_exact) < 0.02 and abs(R + T - 1) < 1e-3.
2. **Straight Si strip** (n 3.45 in 1.45, width 0.4 um; 2-D Ez, and 3-D with a 0.22 um core). Mode source, mode
   monitors and flux planes 1 um after the source and 0.5 um before the output PML; 5 frequencies from 0.570 to
   0.720 / um. Reference: a lossless straight guide transmits 1.
   - 2-D, res 20: T_mode 0.99991 to 1.000000, T_flux 0.99915 to 1.000000. False backward coefficient
     abs(alpha-)^2 / abs(alpha+)^2: 4.4e-4 to 1.3e-3 (the co-located fields of the Yee grid). abs(alpha+)^2 / input
     flux: 1.0001 to 1.0009. neff from the mode solver: 3.117 to 3.215, against 3.108 to 3.207 for the exact slab
     (`references.slab_te_neff`; +0.008 to +0.009 from the discretization of the mode solver at 0.05 um).
   - 3-D, res 20 (1.0 M cells, 2.0 s): T_mode 0.9980 to 1.0003. T_flux 1.0000 at the center frequency, 0.983 and
     0.989 at the band edges. At the input, abs(alpha+)^2 is 0.95 to 0.97 of the flux (4.4 cells across the core).
   - Cause of the low T_flux at the band edges: the mode source is solved at its center frequency only, so at
     other frequencies it also launches radiation that crosses the input plane and leaves before the output plane.
     Run with the source centered (and its mode solved) at the edge frequency 0.570 / um (`--src-freq 0.5701613`):
     T_flux at 0.570 / um is 0.999995 (0.983 before), and the other frequencies now fall (0.975 at 0.645 / um).
3. **2-D Si ring** (n 3.4 in air, inner radius 1.5 um, width 0.22 um; Ez), a short dipole in the ring, `Harminv` on
   Ez after the sources. Reference: the exact resonances of the dielectric ring (Bessel and Hankel functions,
   `references.ring_resonance`), azimuthal orders m = 15 to 21.
   - Frequency error of the strongest mode of each m: -0.43 to -0.69 % at res 40 (2.0 s), -0.02 to -0.20 % at
     res 60 (3.6 s). The free spectral range of neighbor orders: 0.0266 to 0.0285 at res 40 and 0.0274 to 0.0287 at
     res 60, against 0.0278 to 0.0282 (exact). The staircase circle is rough: the Q is 5.6e3 to 1.5e4 at res 40 and
     5.0e4 to 6.0e5 at res 60 (exact: above 1e9), and a mode can split into a pair; the check uses the strongest
     mode of each m. The test (res 60) requires errors below 0.5 % for the three strongest modes and the free
     spectral range within 2 %.
4. **3-D point dipole** (Ez) in a homogeneous medium, flux through a closed box of half width 1 um, 5 frequencies
   from 0.4 to 0.6 / um. Reference: the analytic radiated power n omega^2 abs(I)^2 / (12 pi) of a dipole current I
   (Jackson), with the source spectrum of `GaussianSource.fourier_transform` (Purcell factor 1).
   - n = 1: 1.0022 to 1.0060 at res 10, 1.0006 to 1.0015 at res 20 (1.0 M cells, 1.2 s), 1.0002 to 1.0007 at
     res 30: second order in the cell size. n = 1.5, res 20: 1.0029 to 1.0066.
   - A box of half width 0.5 um, res 10: 0.99996 to 1.0036.
   - The test requires the error below 3e-3 at res 20 and an observed order above 1.7 (res 10 to 20) in vacuum,
     and below 1 % for n = 1.5 at res 20.
5. **1-D current sheet** (Ex) in a homogeneous medium, flux plane 3 um away, 7 frequencies from 0.4 to 1.6 / um.
   Reference: the exact flux abs(I)^2 / (4 n) per side, and E and H in phase.
   - Vacuum: flux / exact 1.0014 to 1.0220 at res 20, 1.0006 to 1.0045 at res 40 (second order at the high
     frequencies: the Yee grid makes abs(E / H) low by up to 2.2 % at res 20, 1.6 / um). n = 1.5, res 20: 1.0037 to
     1.071. arg(Hy / Ex) below 1.8e-4 rad in all rows.

Tests: 13 (7 without FDTD runs: geometry, Prism center, 2-D parities, Harminv; 6 FDTD checks, some with more
than one run).

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
```

If the package is not installed in `.venv-rocm`, use `PYTHONPATH=src`. It needs fdtdx 0.6.3 (with tidy3d for its mode
solver), JAX, NumPy and SciPy.

## License

MIT FDTD-package names are used only for the interface. fdtdx is MIT-licensed. This package has no license file yet
(see the decisions in `STATUS.md`).
