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
| `switching.phase_transient` | thermal-gpu (transient) + meow-gpu | the mode index change against time after the heater switches on or off, and its 10-90 % time (below) |

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

## Switching time of a thermo-optic phase shifter (`photonsuite.switching`)

```python
from photonsuite.switching import phase_transient
from thermal_gpu.nonlinear import SI_POWER_LAW, SIO2_LINEAR

optics, thermal = Window(-1.5, 1.5, -0.9, 1.1, 0.02), Window(-5, 5, -5, 2.3, 0.05)
on = phase_transient(xs, optics, thermal, "heater", 1.55, switch="on", core_tag="core",
                     full_times=(1.5e-6, 5e-6, 20e-6))           # the core Rect needs tag="core"
on.times, on.dneff, on.phase(100.0)                             # s; mode index change; rad for a 100 um heater
on.t10, on.t90, on.switch_time                                  # s: 10 % and 90 % of the step, and the difference
off = phase_transient(xs, optics, thermal, "heater", 1.55, switch="off", core_tag="core",
                      kappa_laws={"Si": SI_POWER_LAW, "SiO2": SIO2_LINEAR})      # kappa(T) per material name
```

1. Heat: the transient of the thermal window of `temperature` (thermal-gpu, BDF2; steps from `dt0` = 10 ns that
   double every 20 steps; 200 steps to 204.6 us). A constant kappa uses `solve_transient2d`; laws per material
   name (`kappa_laws`, `rho_cp_laws`) use `solve_transient_kappa2d`. "on" starts at 300 K with the heater on; "off"
   starts at the steady state with the heater on (the steady solve of the same model) with the heater off. Each
   thermal cell around the optics window is a monitor, so its temperature is known at each step.
2. Modes: one mode solve without heat. The first-order change of the mode index for a real index change dn:
   dneff = sum n dn |E|^2 dA / (eta0 Re sum (E x H*)_z dA), with each E component at its Yee point (A. W. Snyder
   and J. D. Love, "Optical Waveguide Theory", 1983, the chapter on perturbation theory). With dn = dn/dT dT and
   the interpolation of `phase_shifter_ppi`, dneff = sum S_j dT_j over the thermal cells: `on.sensitivity` (1/K).
3. Full mode solves at the final time and at `full_times` (the checks).
4. The 10 % and 90 % times of the step of dneff (`thermal_gpu.transient3d.rise_time`).

The heat-diffusion estimate of the stack (`slab_switch_time`): the heat crosses the oxide between the heater and
the Si substrate (L = 2 um buried oxide + 2.2 um = 4.2 um; the substrate conducts 100 times better and is the heat
sink), alpha = kappa / (rho c_p) = 1.4 / (2220 x 745) = 8.47e-7 m^2/s. The slowest mode has tau = 4 L^2 / (pi^2
alpha) = 8.45 us; one exponential gives the 10-90 % time tau ln 9 = 18.56 us. A 1-D slab heated at its top (the
series of Carslaw and Jaeger, "Conduction of Heat in Solids", 1959) gives 18.72 us at the core center (2.11 um
above the substrate; 10 % at 1.87 us, 90 % at 20.59 us).

Results on the cross-section of the example (with the core tag; rho c_p of Incropera, Tables A.1 and A.2: Si
2330 x 712, SiO2 2220 x 745, the heater metal 19300 x 132 J/m^3/K), 100 W/m, thermal grid 50 nm, optics grid 20 nm
(`tests/test_switching.py`; the same values are in the rows of `benchmarks/2026-11-08/suite-switching.jsonl` from
`bash tools/gpu_run.sh suite python bench/phase_switching.py --scale-check`):

| Model | Switch | 10 % / 90 % (us) | 10-90 % time of the phase (us) | of the mean core temperature (us) |
|---|---|---|---|---|
| constant kappa | on | 1.418 / 19.357 | 17.9385 | 17.9387 |
| constant kappa | off | 1.418 / 19.357 | 17.9385 | 17.9387 |
| Si and SiO2 kappa(T) | on | 1.406 / 19.012 | 17.606 | 17.606 |
| Si and SiO2 kappa(T) | off | 1.387 / 19.257 | 17.870 | 17.871 |

With kappa(T) the switch-on is 1.86 % faster and the switch-off 0.38 % faster than with a constant kappa: the oxide
conducts better when it is hot, and the end of the switch-off (which sets its 90 % time) is near 300 K. On the 20 nm
thermal grid (constant kappa, switch-on; the last two rows of the same file, from `bash tools/gpu_run.sh suite python
bench/phase_switching.py --models constant --switches on --thermal-step 0.02`): 17.9205 us (thermal-gpu: 17.920 us),
P_pi 18.922 mW.

Checks (measured; the test bounds are larger):

| Check | Reference | Agreement |
|---|---|---|
| full mode solve at the end of the switch-on (204.6 us) | `phase_shifter_ppi` (steady): dneff 4.0796e-3, P_pi 18.997 mW | 2.8e-10 relative |
| first-order dneff at 1.48, 4.92, 20.3 and 204.6 us | full mode solves | 2.6e-5, 1.1e-4, 2.2e-4, 2.4e-4 relative: a second-order error (0.058 dneff at each time) |
| the same, the steady field times 0.1 and 1 (the two "scale-check" rows of `benchmarks/2026-11-08/suite-switching.jsonl`, `--scale-check`) | full mode solves | 2.39e-5 and 2.38e-4 (ratio 9.96): the formula is the exact derivative of the discrete mode index |
| dneff(t) against the mean core rise dT_core(t) | dneff = s dT_core, s = sum S = 2.0124e-4 /K (the change for a uniform rise of 1 K; a full solve differs by 1.2e-5) | within 0.205 % (on) and 3.9e-5 (off) at each step with dT_core above 1 % of its steady value; steady state 3.9e-5 |
| switch-off, constant kappa | the mirror image of the switch-on (a linear problem) | below 1e-8 of the step (test bound) |
| 10-90 % time of the mean core temperature | thermal-gpu with its own structure (`heater_stack`, `solve_transient2d`) | below 1e-8 (test bound); 17.939 us as in the thermal-gpu README |
| 10-90 % time of the phase | the same of the mean core temperature | 1.2e-5 (the cladding adds little) |
| 10-90 % time of the phase | heat-diffusion estimate: 1-D slab at the core center / one exponential | 0.958 / 0.967 of the estimate |
| `slab_switch_time` | 1-D thermal-gpu transient of the oxide column (10 nm cells) | 3.4e-5 (10 %), 2.9e-6 (90 %), 6.6e-6 (10-90 %) |
| kappa(T), switch-on | thermal-gpu (`bench_transient_kappa_t.py`): steady core rise 20.0780 K, rise time 0.98145 of the constant kappa | 20.07799 K, 0.981446 |

The 2-D rise is 4.2 % shorter than the 1-D estimate: under the 2 um heater the heat also spreads sideways, and
this part of the steady rise probably settles faster than the 1-D slab.

Limits:
- A 2-D cross-section (a long heater; thermal-gpu shows that the 3-D heater of 100 um switches as its cross-section
  at mid-length). A step of the heater power; dn/dT is constant.
- The first-order formula: the fundamental mode, a real dn. Its error is second order (2.4e-4 relative, 9.7e-7 of
  dneff, at 100 W/m); it grows with the power. The 10-90 % times use the first-order curve; check it with `full_times`.
- The 10-90 % time depends on the time steps: thermal-gpu measured 17.905, 17.939 and 17.945 us with 10, 20 and 40
  steps per stage (dt0 20, 10 and 5 ns).
- One monitor per thermal cell around the optics window: 2,816 cells at 50 nm (2.2 s for the switch-on with 5 mode
  solves), about 16,000 at 20 nm (3.9 s; `seconds` of the logged rows, shared limits of the computer).

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
  later shape replaces an earlier one, as in `CrossSection`. `cell_length=None`: one cell per segment.
  `interface="conserving"` uses the power-conserving interface of meow-gpu (`meow_gpu/README.md`, "A
  power-conserving interface") in place of meow's (the default, `interface="meow"`).
- `fdtd_avg.mode_transmission(segments, x_range, y_range, wavelengths, resolution)`: fdtd_gpu with subpixel
  averaging, a band of wavelengths in one run (the last section).

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

## EME against 3-D FDTD with subpixel averaging (`photonsuite.fdtd_avg`)

`fdtd_avg.mode_transmission(segments, x_range, y_range, wavelengths, resolution)` runs the same segment list with
fdtd_gpu (this repository; fdtdx inside): subpixel averaging of the permittivity at each Yee E component (Kottke et
al.; second order in the fdtd_gpu checks), a mode source, and mode monitors whose modes the fdtdx mode solver
computes at each frequency, so one run gives the whole band. The first segment must be at least 1.5 um long,
the last at least 1.0 um and the device at least 2.0 um (the planes stay 0.5 um from the segment ends). The mode
coefficient |alpha|^2 of a pure mode differs from its flux by the factor b (the co-located fields of the Yee grid;
b = 0.992 for the 0.5 um strip at 40 cells per um), and b depends on the mode, so two more runs (straight guides of
the input and the output cross-section) measure it: T00 = (|a_out|^2 / b_out) / (|a_in|^2 / b_in).
`eme.transmission(..., cell_length=None)` gives one EME cell per segment, so the EME samples the piecewise-constant
device exactly along z.
`fdtd_gpu` is not a dependency (as fdtdx); the tests find it through `pythonpath` in `pyproject.toml`.

The case (`bench/eme_fdtd_taper.py`; textbook indices, constant over the band: Si 3.476, SiO2 1.444): a Si strip
220 nm thick in SiO2, 1.5 um straight at 0.5 um width, a linear taper to 2.0 um over 2 um in 20 segments of 0.1 um
(each with the width at its center), 1 um straight at 2.0 um; window x -2 to 2 um, y -0.8 to 1.02 um; TE; five
wavelengths from 1.50 to 1.60 um. The taper is short, so about 9 % of the power goes into higher modes (T00 near 0.91):
the comparison tests the mode coupling, not only a transmission near 1.

The EME rows of this section use the mode sets of the meow-gpu solver, which lack some modes above cutoff
(about 10 of the 40 nearest eigenvalues; `meow_gpu/README.md`, the first open item); how much this changes T00
is not known.

Results: the rows of `benchmarks/2026-11-08/suite-eme-fdtd-taper.jsonl` (the commands below; RX 7900 XT, shared
computer). T00 at 1.50, 1.525, 1.55, 1.575 and 1.60 um.

FDTD (subpixel averaging; the times are the FDTD loops of the three runs):

| cells per um | cells | T00 | T_flux | b_in / b_out at 1.55 um | time |
|---|---|---|---|---|---|
| 30 | 1.8 M | 0.9082, 0.9129, 0.9171, 0.9207, 0.9238 | 1.0017 to 1.0029 | 0.9864 / 0.9901 | 9.6 s |
| 40 | 4.3 M | 0.9032, 0.9081, 0.9125, 0.9163, 0.9197 | 1.0005 to 1.0014 | 0.9923 / 0.9944 | 28.6 s |
| 50 | 8.3 M | 0.9012, 0.9063, 0.9109, 0.9150, 0.9185 | 0.99996 to 1.0006 | 0.9950 / 0.9964 | 86.2 s |
| 60 | 14.4 M | 0.9008, 0.9058, 0.9103, 0.9143, 0.9178 | 0.9996 to 1.0002 | 0.9966 / 0.9975 | 172.7 s |

- Convergence: the change of T00 from 30 to 40, 40 to 50 and 50 to 60 cells per um is -0.0041 to -0.0050,
  -0.0012 to -0.0020 and -0.0004 to -0.0008 (over the band): faster than first order, but the ratios of the steps
  scatter (1.6 to 5.5 for the last two), so no single order fits. A second-order extrapolation from 50 and 60 gives
  0.9000, 0.9047, 0.9090, 0.9128, 0.9161, within 0.0017 of the res-60 values: the FDTD limit is known to about 2e-3.
- Power: T_flux goes to 1 (0.9996 to 1.0002 at res 60), so the taper radiates little; the rest of the power is in the
  higher modes. The straight guides of the calibration transmit 0.99984 and 0.99996 of the flux.
- The mode factor: b goes to 1 at about second order; b_in / b_out changes T00 by 0.37 % (res 30) to 0.09 % (res 60).
- The FDTD and the EME see the same cross-sections: the TE0 index of the fdtdx mode solver at the planes (1.55 um,
  res 30 to 60) is 2.4460, 2.4479, 2.4505, 2.4468 (0.5 um strip) and 2.8207, 2.8243, 2.8260, 2.8232 (2.0 um);
  meow-gpu (`bench/taper_modes.py`, grids 0.04, 0.02, 0.01 um) gives 2.4518, 2.4436, 2.4450 and 2.8252, 2.8194,
  2.8213. The FDTD propagation index (the phase of a_out / a_in in the straight guides, res 60) is 2.4493 and 2.8273:
  above the mode solver by 0.0026 and 0.0041, the dispersion of the Yee grid, (k n dx)^2 (1 - S^2) / 24 relative with
  S = c dt / (n dx), which also gives 0.0026 and 0.0041. This index error changes the beat phase of TE0 and TE2 in
  the taper; it is a probable cause of the change of T00 with the grid (not tested separately).
- Staircase (`--staircase`, res 40): 0.9019, 0.9069, 0.9115, 0.9156, 0.9193, 0.0004 to 0.0013 below the average.

EME (one cell per segment, 22 cells; 0.02 um grid; times for the five wavelengths 44 s at 10 modes, 105 s at 20, 198 s
at 30, 300 s at 40):

| meow interface setting | modes | T00 | power in the right modes |
|---|---|---|---|
| TSVD rcond 1e-3, passivity "invert" (default) | 10 | 0.8705, 0.8731, 0.8761, 0.8784, 0.8809 | 0.9875 to 0.9879 |
| the same | 20 | 0.8727, 0.8759, 0.8791, 0.8784, 0.8808 | 0.9869 to 0.9871 |
| passivity "clip" | 20 | 0.8853, 0.8887, 0.8920, 0.8907, 0.8932 | 0.9984 to 0.9991 |
| passivity "none" | 10 | 0.8922, 0.8952, 0.8980, 0.9010, 0.9035 | 1.0107 to 1.0122 |
| passivity "none" | 20 | 0.8983, 0.9019, 0.9053, 0.9035, 0.9061 | 1.0100 to 1.0113 |
| passivity "none" | 30 | 0.9096, 0.9123, 0.9145, 0.9165, 0.9183 | 1.0095 to 1.0105 |
| passivity "none" | 40 | 0.9132, 0.9163, 0.9189, 0.9214, 0.9235 | 1.0080 to 1.0094 |
| passivity "none", TSVD rcond 1e-8 | 20 | unstable | 1.6 to 132 |

Agreement and its causes:
- At 1.55 um the EME differs from the FDTD (0.9103 at res 60) by -0.034 and -0.031 with the meow defaults (10 and 20
  modes), -0.018 with "clip", and -0.012, -0.005, +0.004, +0.009 without a correction (10, 20, 30, 40 modes). The
  FDTD is converged to about 2e-3; the EME is not converged.
- The EME is limited by its interfaces, not by its grid or its cells. Without a correction the kept modes carry 0.8
  to 1.2 % more power than the input: the overlap solve of meow with a truncated set of modes in the closed window
  (PEC walls; the box modes stand for the radiation) does not conserve power. "invert" replaces each singular value
  1 + d of an interface by 1 / (1 + d), so the gain becomes a loss of the same size and T00 falls by 0.022 (10
  modes) and 0.026 (20 modes); "clip" removes only the gain. With more modes the uncorrected T00 rises (0.898,
  0.905, 0.915, 0.919 at 10 to 40 modes) and the power error stays near 1 %: no convergence up to 40 modes. The
  FDTD value is between the 20- and the 30-mode values. Almost no truncation of the solve (rcond 1e-8) is unstable.
- The EME grid: 0.04 to 0.02 um changes T00 at 1.55 um by +0.0022 (20 modes, "invert"; 0.8769 against 0.8791), 0.02
  to 0.01 um by +0.0007 ("none", 20 modes; 0.9060 against 0.9053).
- The slope over the band: from 1.50 to 1.60 um the FDTD T00 rises by 0.0169 (res 60; 0.0156 at res 30), the EME by
  0.0078 to 0.0113 (all rows above but the unstable one). The slopes differ by about 0.007, more than the FDTD
  slope changes with the grid (0.0013).
- The earlier comparison on the 10 um taper (above: EME 0.9912 with "invert", FDTD 0.9961, and the EME lost 0.6 % of
  the power) has the sign of the same interface loss; it is not repeated here.

Commands (from the repository folder on the GPU computer; the logged runs set `PYTHONPATH`, which is not
necessary when `tools/env/rocm_jax.sh` installed fdtd_gpu into .venv-rocm):

```bash
B="python bench/eme_fdtd_taper.py"
for r in 30 40 50 60; do bash tools/gpu_run.sh suite env PYTHONPATH=../fdtd_gpu/src $B fdtd --resolution $r; done
bash tools/gpu_run.sh suite env PYTHONPATH=../fdtd_gpu/src $B fdtd --resolution 40 --staircase
for a in "--modes 10" "--modes 20" "--modes 20 --step 0.04" "--modes 20 --passivity clip" \
         "--modes 10 --passivity none" "--modes 20 --passivity none" "--modes 30 --passivity none" \
         "--modes 40 --passivity none" "--modes 20 --passivity none --rcond 1e-8" \
         "--modes 20 --passivity none --step 0.01 --wavelengths 1.55"; do bash tools/gpu_run.sh suite $B eme $a; done
bash tools/gpu_run.sh suite python bench/taper_modes.py
```

Tests (`tests/test_fdtd_avg.py`): the staircase of the adapter equals the suite's own sampling at every cell center
(three segments: a Rect, a sloped Polygon, a fill order), the averaged eps along the propagation axis keeps the
area of each cross-section to 2e-3, the planes must stay off the segment ends, and a straight guide at 20 cells per um
gives T00 within 3e-3 of 1, T_flux within 1e-2, 0.9 < b < 1.01 and a phase index within 0.05 of the mode solver.
`tests/test_eme.py`: one cell per segment on a straight guide (T within 1e-4 of 1).

Limits: the EME has no converged value on this taper (its interface treatment, above); the FDTD runs in float32, with
non-dispersive indices, and gives only the fundamental-mode coefficient (TE2 and higher: not compared yet). The
older adapter `photonsuite.fdtd` puts its input detector 1.0 um into the device, on the end of a 1.0 um first segment
in `bench/fdtd_taper*.py`: its overlap ratio `transmission` sees the start of the taper there (its calibrated T00 uses
the input flux, which does not depend on the plane); not changed.

## Next

- Layer stacks (from a GDS cross-section); band structures (bands-gpu) for periodic waveguides.
