# photonsuite

One device description for the GPU solvers of this repository (the idea of a Tidy3D-like suite, for
cross-sections): a `CrossSection` of rectangles with materials that carry the properties of every physics, and
solvers that sample it on their own windows and grids.

```python
from photonsuite import CrossSection, Material, Rect            # the import enables float64 in JAX
from photonsuite.physics import Window, mode_index, phase_shifter_ppi, modulator_vpil

si = Material("Si", n=3.476, dn_dT=1.86e-4, kappa=148.0, eps_r=11.7, semiconductor=True)
sio2 = Material("SiO2", n=1.444, dn_dT=0.95e-5, kappa=1.4, eps_r=3.9)
metal = Material("heater", n=1.444, kappa=100.0)
xs = CrossSection(sio2, (Rect(-5, 5, -5, -2, si), Rect(-0.25, 0.25, 0, 0.22, si), Rect(-1, 1, 2.2, 2.3, metal, tag="heater")))

neff = mode_index(xs, Window(-1.5, 1.5, -0.9, 1.1, 0.02), wavelength=1.55, num_modes=2)     # meow-gpu
p_pi, dneff, th = phase_shifter_ppi(xs, Window(-1.5, 1.5, -0.9, 1.1, 0.02), Window(-5, 5, -5, 2.3, 0.05),
                                    "heater", 1.55)                                          # thermal-gpu + meow-gpu
```

| Function | Solvers | Output |
|---|---|---|
| `mode_index` | meow-gpu (FDE) | complex mode indices, with an optional index change dn(x, y) |
| `temperature` | thermal-gpu | temperature rise (interpolator), heater power per length |
| `phase_shifter_ppi` | thermal-gpu + meow-gpu | P_pi = lambda / (2 dneff/dP') |
| `carriers` | charge-gpu | electrons and holes at a reverse bias (contacts: every semiconductor cell on the left and right window sides; each contact takes v_n or v_p from the sign of its mean doping) |
| `modulator_vpil` | charge-gpu + meow-gpu | V_pi L and the loss change per bias (negative: fewer carriers) |

Units: um for geometry, cm^-3 for doping, W/m for heater power per length.

- Every solve is checked: a mode, thermal or charge solve that does not converge raises
  `photonsuite.physics.ConvergenceError`. The thermal check uses the true residual |b - A T| / |b|.
- The solvers need float64. `import photonsuite` enables `jax_enable_x64`; each function raises RuntimeError if it
  is off.
- A `Rect` holds [x0, x1) x [y0, y1) (with a 1e-9 um tolerance): a grid sample on a left or bottom edge is inside, on
  a right or top edge outside, independent of the rounding of the grid. There is no sub-pixel averaging yet.

## Checks (`tests/test_workflows.py`)

The suite reproduces the standalone examples on the same grids, within 1e-3 relative: P_pi 18.997 mW
(`examples/thermo_optic_phase_shifter.py`, which prints 19.0) and V_pi L 1.386, 1.600, 1.955 V cm at 1, 2 and 4 V
(`examples/pn_modulator.py`). Also: the edge rule of `Rect`, the float64 check, and a thermal solve that does not
converge.

## 3-D FDTD (`photonsuite.fdtd`, with fdtdx) and EME (`photonsuite.eme`)

The two adapters take the same device: a list of `Segment(z0, z1, CrossSection)` along z. The segments must be
contiguous and have one background material (`fdtd.check_segments` raises ValueError); for the FDTD, the first
segment must be at least 1 um and the last at least 0.5 um long (the source and the detectors are there).

- `fdtd.transmission(segments, x_range, y_range, wavelength, step)`: a mode source, mode-overlap detectors and
  Poynting-flux detectors at the input and the output. fdtdx (MIT, JAX, float32) runs it on the AMD GPU; the
  adapter turns float64 off for the run. `Rect` shapes become boxes, `Polygon` shapes fdtdx `ExtrudedPolygon`
  objects along z (a binary mask at the cell centers: no sub-pixel smoothing, because fdtdx 0.6.3 smooths only GDS
  layers extruded along its axis 2). fdtdx is not a dependency of this package (the GPU environment installs it:
  `tools/env/rocm_jax.sh`).
- `eme.transmission(segments, x_range, y_range, wavelength, step, cell_length, num_modes)`: meow with the
  meow-gpu mode solver. `Rect` and `Polygon` shapes (a polygon becomes a prism along z; meow 0.15 drops such
  prisms, and the adapter corrects that: `docs/upstream-issues.md`). Each shape gets its own meow mesh order, so a
  later shape replaces an earlier one, as in `CrossSection`.

| Case (`bench/`, JSON rows in `benchmarks/2026-10-30/fdtd-v3.jsonl`) | Result |
|---|---|
| straight Si strip 0.5 x 0.22 um, 5 um, 25 nm grid (3.3 M cells) | T = 0.9999995; 10.8 s on the RX 7900 XT |
| Si taper 0.5 -> 2.0 um over 10 um, calibrated (below): 100 segments, 25 nm grid (8.4 M cells) | T00 = 0.9974 (T_flux 1.0017) |
| the same, 200 segments, 25 nm | T00 = 0.9981 (T_flux 1.0019) |
| the same, 100 segments, 20 nm (15 M cells) | T00 = 0.9961 (T_flux 1.0003); 213 s with the calibration run |
| the same taper with 80 degree sidewalls (`Polygon`), 25 nm / 20 nm (`bench/sloped_taper.py`) | T00 = 0.9985 / 0.9975 (T_flux 1.0022 / 1.0014) |

The rows in `fdtd.jsonl` (same day) came from the adapter before three corrections (diagnostic rows in
`fdtd-diagnostics.jsonl`):
- Each shape now fills exactly the cells whose centers are inside it. Before, fdtdx rounded the position and the
  size of each box or polygon separately: on a 40 nm grid, where the 100 nm segments are 2.5 cells, that left oxide
  gaps (T_flux 0.44 against 0.999 after the fix), and sloped polygons moved by up to half a cell.
- The volume is a whole number of cells on each axis (65.5 cells failed in fdtdx).
- The run is placed on the first GPU: an FDTD run after an EME run in the same process was 9 to 13 times slower
  (149 s against 16 s at 40 nm; 1307 s against 99 s at 25 nm).
The vertical-wall T00 values changed by 0.1 % at most; the sloped values by 0.2 %.

Transmission into the fundamental mode of the wide output guide: the ratio of two mode overlaps is biased when the
input and the output modes differ (it gave 1.004 at 20 nm). `bench/fdtd_taper_calibrated.py` measures T00 =
T_flux x f_out / f_cal: the Poynting-flux transmission, times the fundamental share of the output power, divided by
the same share for a straight output guide (which removes the overlap bias).

**FDTD against EME (the taper).** meow 0.15 gave T00 = 0.952 to 0.955. The cause was an error in meow's mode
post-processing: its Gram-Schmidt step averages the neff of the higher modes, so the EME propagated them almost in
phase with the fundamental mode (`docs/upstream-issues.md`). With the correction in meow-gpu, the EME gives 0.9867,
0.9900 and 0.9912 at 30, 60 and 120 cells (10 modes, 20 nm grid; `benchmarks/2026-10-30/eme-neff-fix.jsonl`), and
0.9899 with 20 modes at 60 cells (before the correction, 20 modes lowered the result by 0.8 %). The FDTD gives
0.9961 on the 20 nm grid and 0.9974 / 0.9981 on the 25 nm grid; its flux transmission is 1.0003 to 1.0019 (above 1),
so the FDTD values carry an error of up to about 0.2 %. The two methods agree within about 0.5 % (0.9912 against
0.9961). The EME itself does not conserve power: at 60 cells the kept modes on both sides hold 0.994 of the input.
This loss (0.6 %) is of the same order as the remaining gap; its cause (for example the interface regularization of
meow, or the truncated mode basis) is not tested yet. With meow's own post-processing the total is the same (0.994),
but 4 % of the power goes into the higher modes.

These EME numbers come from `meow_gpu/bench/bench_eme_taper.py` (the taper as one meow prism, no straight input
and output). `bench/eme_taper.py` runs `photonsuite.eme` on the segment list of the FDTD runs: T00 = 0.9888 with
0.2 um cells (between the bench values at 0.33 and 0.17 um cells). With 80 degree sidewalls the EME gives 0.9878
(0.10 % lower than with vertical walls), and the FDTD 0.9975 at 20 nm (0.14 % higher; 0.9985 at 25 nm, 0.11 %
higher). The two methods do not agree on the sign of this small change; it is below the accuracy of this comparison
(the FDTD flux transmission is 0.03 % to 0.2 % above 1, and the EME loses 0.6 % of the power).

## Next

- Layer stacks (from a GDS cross-section); band structures (bands-gpu) for periodic waveguides.
