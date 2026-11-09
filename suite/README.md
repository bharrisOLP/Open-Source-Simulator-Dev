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
| `pin.pin_phase_shifter` | charge-gpu (drift-diffusion) + thermal-gpu + meow-gpu | a forward-biased PIN with self-heating: the mode index change of the carriers and of the heat, the net phase, L_pi, P_pi, I_pi, the loss, the bias where the two parts cancel (below) |

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

## A PIN phase shifter with self-heating (`photonsuite.pin`)

A forward current changes the index of the waveguide in two ways, with opposite signs: the injected free carriers
lower the index and add loss (plasma dispersion), and the heat of the current raises the index (dn/dT).
`pin_phase_shifter` calculates both parts on one cross-section.

```python
from photonsuite.physics import Window
from photonsuite.pin import pin_rib, pin_phase_shifter

xs = pin_rib()        # the lateral PIN rib of charge_gpu.injection.rib_pin, on 2 um of oxide over a Si substrate
optics, charge = Window(-1.5, 1.5, -0.5, 0.9, 0.02), Window(-1.5, 1.5, -0.5, 0.72, 0.01)
r = pin_phase_shifter(xs, optics, charge, Window(-5, 5, -5, 2.3, 0.05), 1.55, biases=[0.9, 1.0, 1.1, 1.2],
                      tau=1e-9, core_tag="core")
r.dneff_carrier, r.dneff_thermal, r.dneff     # complex changes of the mode index from 0 V: carriers, heat, both
r.current, r.power                            # I' (A/m) and I' V (W/m)
r.phase_per_length, r.l_pi, r.p_pi, r.i_pi    # rad/m, m, W, A (from the net change)
r.loss_db_per_cm, r.crossover()               # the loss that the current adds; the bias where the parts cancel
# coupling="two-way": the carriers at the device temperature that their own heat makes (step 6)
t = pin_phase_shifter(xs, optics, charge, Window(-5, 5, -5, 2.3, 0.05), 1.55, biases=[0.9, 1.0, 1.1, 1.2],
                      tau=1e-9, core_tag="core", coupling="two-way")
t.electrical.loop.temperature, t.electrical.loop.iterations, t.electrical.loop.history   # K, solves, per step
```

1. Carriers (charge-gpu): the cross-section on the charge window (`physics.charge_device`: the cell centers, an
   ohmic contact on the left and on the right window side, the rule of `carriers`), drift-diffusion with Newton and
   SuperLU (`drift_diffusion.sweep`, continuation from 0 V in steps of 50 mV), SRH lifetimes tau_n = tau_p = tau.
   `pin_rib` with the charge window of the snippet gives the Device of `rib_pin` (the same arrays; test).
2. Heat: the heat density of the drift-diffusion solution without the thermoelectric (Peltier and Thomson) terms,
   G. K. Wachutka, IEEE Trans. Computer-Aided Design 9, 1141 (1990):
   H = -J_n . grad(phi_n) - J_p . grad(phi_p) + q R (phi_p - phi_n), with the quasi-Fermi potentials phi_n and
   phi_p. The first two terms are the Joule heat of the two currents (|J|^2 / sigma, not negative); the last term
   is the heat of the recombination. The discrete form uses the Scharfetter-Gummel face currents of charge-gpu
   (each face gives -I (phi_b - phi_a) for each carrier, one half to each cell; a contact face has the contact
   voltage at its node). Summation by parts with the discrete continuity equations gives sum H = I' V exactly, to
   the residual of the solve. The electrostatic J . E is not this heat: its sum is I' (V - V_c), with V_c =
   V_T (asinh(N_A / 2 n_i) + asinh(N_D / 2 n_i)) = 0.952 V at 1e18 cm^-3 (the potentials of the two contact
   nodes), so it is negative below 0.952 V. Option `heat=<tags>`: I' V spread uniformly over the shapes with these
   tags (a simpler model).
3. Temperature (thermal-gpu, steady): the boundary conditions of `physics.temperature` (300 K at the bottom of the
   thermal window, h = 10 W/m^2/K at the top, adiabatic sides). The heat of each charge cell goes to the thermal
   cells in proportion to the area of overlap, so the sum stays I' V on any thermal grid.
4. Modes (meow-gpu, the fundamental mode): full mode solves with the carrier change, with the thermal change, and
   with both. The carrier change: Soref and Bennett at 1.55 um (R. A. Soref and B. R. Bennett, IEEE J. Quantum
   Electron. 23, 123 (1987); `charge_gpu.modulator.plasma_dispersion`, from the absolute concentrations at V and
   at 0 V, as `modulator_vpil`), dk = dalpha lambda / (4 pi), the mean over 8 x 8 points in the mode cell around
   each Yee point (as `examples/pn_modulator_rc.py` of charge-gpu). The thermal change: dn/dT dT at the Yee points
   (as `phase_shifter_ppi`; Si 1.86e-4 /K, G. Cocorullo and I. Rendina, Electron. Lett. 28, 83 (1992); SiO2
   0.95e-5 /K, the value of the other examples). `plasma=` takes other coefficients (for example those of
   M. Nedeljkovic, R. Soref and G. Z. Mashanovich, IEEE Photon. J. 3, 1171 (2011), for another wavelength); the
   default refuses a wavelength other than 1.55 um.
5. Outputs: the net phase per length 2 pi Re(dneff) / lambda, L_pi = lambda / (2 |Re dneff|), P_pi = I' V L_pi,
   I_pi = I' L_pi, the added loss from Im(dneff), and the first bias interval where Re(dneff) changes sign
   (`crossover()`: linear in I' between two biases; or a bias where it is exactly 0). Also the first-order
   estimates of both parts (the weights of `switching.index_weights` times the complex dn), the Joule and
   recombination parts of the heat, the sum of J . E, and the mean temperature rise of the core and of the Si of
   the charge window.
6. Coupling (`coupling`). "one-way" (the default): the carriers are solved at 300 K, so the heat does not change
   the currents. This path is the code of 0753ed0, with the heat from the public face currents of charge-gpu in
   place of its private fields (the same heat, bit for bit: test). "two-way" (`pin.electrothermal`): a fixed-point
   loop on the device temperature T. charge-gpu takes one temperature per device (`Device.T`: V_T, n_i(T), the
   mobility and the SRH lifetimes; `charge_gpu/README.md`, "Temperature"), not a field, so T is the area mean of
   the steady temperature over the Si cells of the charge window:
   - G(T) = 300 K + the mean rise of the heat of the drift-diffusion solution at T (the heat and the thermal model
     of steps 2 and 3);
   - T_1 = G(300 K) (step 1 is the one-pass feedback of `--device-temperature`), then
     T_k+1 = T_k + damping (G(T_k) - T_k) (default damping 1: no damping), each step a Newton solve at
     `Device.T` = T_k from the last solution;
   - stop when |G(T_k) - T_k| <= `loop_tol` (default 1e-4 K); ConvergenceError after `loop_max_iter` (30) solves.

   The returned solutions are those at the converged T; steps 2 to 5 use them as they are. The carrier change is
   from the carriers at 0 V and 300 K (the unbiased device). The loop converges when the gain dG/dT (the rise
   times the relative change of the current per K) is below 1; here it is 0.001 to 0.107 (below).

Results: the rows of `benchmarks/2026-11-08/suite-pin.jsonl` (the commands below; RX 7900 XT, shared computer).
`pin_rib`, the windows of the snippet, the fundamental mode at 1.55 um (neff = 2.58936 at 0 V), dneff from 0 V.
Lifetimes 1 ns (the first run of the file):

| V | I' (A/m) | I' V (W/m) | Joule part of the heat | core rise (K) | dneff_carrier | dneff_thermal | dneff (both) | L_pi (mm) | P_pi (mW) | I_pi (mA) | added loss (dB/cm) | loss in L_pi (dB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.80 | 0.5910 | 0.4728 | 0.062 | 0.114 | -1.9661e-4 | 2.1420e-5 | -1.7519e-4 | 4.42 | 2.09 | 2.61 | 2.50 | 1.11 |
| 0.85 | 1.546 | 1.314 | 0.123 | 0.314 | -3.9713e-4 | 5.9090e-5 | -3.3804e-4 | 2.29 | 3.01 | 3.55 | 5.81 | 1.33 |
| 0.90 | 3.828 | 3.445 | 0.222 | 0.814 | -7.2299e-4 | 1.5314e-4 | -5.6987e-4 | 1.36 | 4.69 | 5.21 | 11.83 | 1.61 |
| 0.95 | 8.389 | 7.970 | 0.339 | 1.86 | -1.1448e-3 | 3.4995e-4 | -7.9488e-4 | 0.975 | 7.77 | 8.18 | 20.35 | 1.98 |
| 1.00 | 15.92 | 15.92 | 0.451 | 3.68 | -1.6079e-3 | 6.9200e-4 | -9.1606e-4 | 0.846 | 13.47 | 13.47 | 30.34 | 2.57 |
| 1.05 | 26.80 | 28.14 | 0.546 | 6.45 | -2.0820e-3 | 1.2137e-3 | -8.6868e-4 | 0.892 | 25.1 | 23.9 | 41.06 | 3.66 |
| 1.10 | 41.23 | 45.36 | 0.623 | 10.33 | -2.5564e-3 | 1.9458e-3 | -6.1135e-4 | 1.27 | 57.5 | 52.3 | 52.2 | 6.62 |
| 1.15 | 59.36 | 68.26 | 0.683 | 15.49 | -3.0275e-3 | 2.9168e-3 | -1.1201e-4 | 6.92 | 472 | 411 | 63.6 | 44.0 |
| 1.20 | 81.23 | 97.48 | 0.731 | 22.05 | -3.4941e-3 | 4.1536e-3 | 6.5727e-4 | 1.18 | 114.9 | 95.8 | 75.1 | 8.86 |

The test prints of `tests/test_pin.py` (0.9, 1.0, 1.1 and 1.2 V) are the same in all digits of this table.

- The two parts have opposite signs: the carriers lower the index (and add loss), the heat raises it. The heat
  grows faster with the bias than the carrier change: dneff_thermal / |dneff_carrier| = 0.109 at 0.8 V, 0.430 at
  1.0 V and 1.189 at 1.2 V. So, below the cross-over (where the carriers win), the net change has its largest size
  at 1.0 V (the shortest L_pi on that side, 0.846 mm, at P_pi 13.5 mW); it changes sign between 1.15 and 1.2 V, and
  above that bias the heat wins and the phase turns. On the thermal side the net change grows again with the bias
  (1.2 V: +6.5727e-4, L_pi 1.18 mm, P_pi 114.9 mW, a core rise of 22.05 K). This holds in this thermal model only:
  with the heat path of `--heat-sink` the carriers win up to 1.3 V (the thermal environment, below).
- The cross-over (`crossover()`, linear in I' between two biases): 1.157 V, 62.5 A/m, 72.4 W/m with the 50 mV
  steps of the table; 1.159 V, 62.8 A/m, 72.8 W/m with 5 mV steps (`--biases 1.15 1.155 1.16 1.165`: the sign
  changes between 1.155 and 1.16 V); 1.148 V with the 0.1 V steps of the test. These values belong to this thermal
  model: the thermal environment changes dneff_thermal much more than the grids do. A wider window moves the
  cross-over by 9.5 to 13 mV to higher bias, and the heat path of `--heat-sink` removes it up to 1.3 V (the
  thermal environment, below).
- At 0.9 V the heat raises the core by 0.81 K and cancels 21 % of the carrier change: L_pi is 1.36 mm, against
  1.07 mm from the carrier part alone (1.072 mm in the charge-gpu example: `benchmarks/2026-10-30/charge-dd.jsonl`).
- The heat source: the recombination heat is most of the power at low bias (0.938 at 0.8 V), the Joule heat at
  high bias (0.731 at 1.2 V). The sum of J . E is -0.0901, -0.2007, 0.7575 and 20.1115 W/m at 0.8, 0.9, 1.0 and
  1.2 V: I' (V - 0.952 V) (checks below).
- The loss over L_pi rises from 1.11 dB (0.8 V) to 2.57 dB at the shortest L_pi (1.0 V).

Lifetimes 10 ns (`--tau 1e-8`, to 1.3 V): at 0.9 V the current is 2.053 A/m (3.828 A/m at 1 ns) and the carrier
change is larger (-1.0379e-3), so dneff_thermal / |dneff_carrier| is smaller: 0.076 at 0.9 V and 0.270 at 1.0 V
(0.212 and 0.430 at 1 ns). Below the cross-over the shortest L_pi is 0.362 mm at 1.05 V (P_pi 12.98 mW, I' 34.17
A/m, a core rise of 7.98 K); the cross-over is at 1.206 V, 128.3 A/m, 154.7 W/m (between 1.2 and 1.25 V; the core
rise is 32.85 K at 1.2 V; with the heat path of `--heat-sink` there is no cross-over up to 1.3 V). Above it the
heat wins: L_pi is 0.549 mm at 1.25 V and 0.218 mm at 1.3 V (net dneff
+3.5577e-3, P_pi 60.8 mW, I_pi 46.8 mA, a core rise of 62.1 K: far outside the 300 K carrier model, below). The
Joule part of the heat is larger than at 1 ns at the same bias (0.760 against 0.222 at 0.9 V).

Checks (`tests/test_pin.py`; the measured values are test prints, of two runs where the text says so; the test
bounds are larger):

| Check | Reference | Agreement |
|---|---|---|
| the Device of `pin_rib` on the charge window | `charge_gpu.injection.rib_pin()` | the same arrays (edges to 1e-15 m), the same contacts |
| I' at 0.9 and 1.0 V | the logged charge-gpu example (`benchmarks/2026-10-30/charge-dd.jsonl`) | 3.7e-13 and 3.5e-14 or less (two runs) |
| dneff_carrier at 0.9 V | the same example with `--modes` (its own sampling of the Si: the Yee points on y = 0 are oxide there) | -7.2299e-4 against -7.2300e-4: 2.3e-5 |
| the sum of the heat source, 0.9 to 1.2 V | I' V: an identity of the discrete scheme (summation by parts; it holds for any cell potentials), so it checks the continuity residual and the flux code, not where the heat goes | 2.1e-13 or better, also after the move to the thermal grid |
| the same, a 1-D diode (`diode_1d`, 1e17 cm^-3) at 0.5 and 0.7 V | I' V | 3.7e-8 (I' = 5.8e-5 A/m, near the current floor of charge-gpu) and 8.2e-12 |
| the Joule and the recombination heat of each cell (the local checks) | not negative | above -1e-12 I' V (test) |
| the recombination heat of each cell | q R A (phi_p - phi_n) from the public methods `DDSolution.recombination` and `quasi_fermi` | the same values (difference 0.0 at 0.9 to 1.2 V) |
| the sum of J . E | I' (V - V_c), V_c = 2 V_T asinh(N / 2 n_i) = 0.9524 V (analytic) | 6.3e-15 of I' V or better |
| the move to the thermal grid | a source cell over 2 x 2 thermal cells: the overlap areas by hand; a random source on a grid that is not aligned | within the test bounds 1e-14 and 1e-12 (not printed) |
| the thermal part with I' V uniform in the core cells (`heat="core"`), 1.0 V | `phase_shifter_ppi` with the same heat in the same cells (the same q, bit for bit) | 1.9e-11 or less (three runs) |
| the carrier part, 0.9 to 1.2 V | the first-order perturbation (`switching.index_weights` times the complex dn) | 7.1e-5, 1.5e-4, 2.4e-4, 3.2e-4: a second-order difference (the error divided by \|dneff\| changes by less than 10 %) |
| the thermal part | the same | 1.0e-5, 4.7e-5, 1.3e-4, 2.8e-4 (the same) |
| both changes in one solve | the sum of the two parts | the cross term is 2.4e-5 to 6.3e-4 of \|dneff_carrier\| |
| dn/dT = 0 (all materials), 1.1 V | no heat: dneff_thermal = 0, dneff = dneff_carrier | 6.0e-14 or less; 4.6e-12 or less; the carrier part of the four-bias run to 7.8e-12 or less (three runs) |
| no plasma dispersion (`plasma` gives 0), 1.1 V | no carriers: dneff_carrier = 0, dneff = dneff_thermal | 3.0e-13 or less; 4.7e-11 or less; the thermal part of the four-bias run to 1.6e-11 or less (three runs) |
| the heat of `heat_source` from the public face currents, 0.9 to 1.2 V, at 300 K and at the loop temperatures | the heat of 0753ed0 (the private `_fluxes` and mesh fields of charge-gpu; its code is in the test) | the same arrays and the same J . E, bit for bit |
| the two-way loop with the temperature rise forced to 0 (the steady temperature gives 300 K everywhere), 0.9 to 1.2 V | the 300 K (one-way) result | one solve per bias at 300 K; I' 4.0e-14, n and p 4.0e-15, psi 9.5e-15 V; dneff_carrier at 1.1 V 2.8e-12 |
| the two-way loop, 0.9 to 1.2 V: G(T) - T per step | convergence | 0.9 V: 8.0e-1, 1.5e-2, 2.7e-4, 5.1e-6 K (3 solves); 1.0 V: 3.6, 1.6e-1, 7.4e-3, 3.3e-4, 1.5e-5 K (4); 1.1 V: to 1.0e-5 K (5); 1.2 V: 22, 1.4, 8.5e-2, 5.1e-3, 3.0e-4, 1.8e-5 K (5) (test: the ratio of two steps below 0.2) |
| its first step (the device at 300 K plus the rise of the 300 K heat) | the one-pass feedback of the logged bench (`I_hot_rel` of `suite-pin.jsonl`, 0.9 to 1.2 V) | within the test bound 1e-8 (not printed) |
| the converged state | its own heat (the mean rise of the Si that `pin_optical` finds again); the heat integral against I' V; `Device.T` of the solution (V_T) | within 1e-4 K (the stop rule); within 1e-9 (test bound, not printed; the identity of the checks above); V_T of T exactly |
| the fixed point at 1.2 V | `scipy.optimize.brentq` on G(T) - T, with G from fresh solves at T (from the 300 K solution) and the mean of `charge_gpu.temperature.mean_temperature` | 323.62433 K against 323.62431 K: 1.9e-5 K (brentq: 6 solves; the loop: 5) |
| damping 0.5 at 0.9 V | the loop without damping | 300.81087 K against 300.81093 K: 5.9e-5 K (the stop rule: the last residual 6.3e-5 K); 14 solves against 3 |

Also tested: the errors for biases that are not positive and increasing, for a charge window without a p and an n
contact, for a wavelength other than 1.55 um with the default plasma dispersion, for a heat source outside the
thermal window, for a heat tag that no shape has, for a slab shorter than the contacts, and for a thermal solve
that does not converge (ConvergenceError); `crossover()` at an exact zero on a bias and without a zero (synthetic
values); the thermal options of `pin_rib` do not change the charge Device; the mode is the largest index of a
4-mode solve (2.589362; the next is 2.199027); the loop: ConvergenceError when one solve is the limit (1.2 V),
ValueError for tol <= 0, a damping outside (0, 1], max_iter < 1, a PinElectrical without the lifetime, a user
mobility (a value at one temperature), and an unknown `coupling`. Time of a run of the 21 tests: 53.5 s (the
drift-diffusion sweep to 1.2 V on the CPU of the GPU computer: about 19 s; peak RSS 1.95 GiB).

The bench `bench/pin_self_heating.py` writes one JSON row per bias (also the Joule and recombination parts, the
first-order values and the rise of the Si) and one row per run with the cross-over. Its options: `--tau`
(lifetimes), `--biases`, `--thermal-step` and `--optics-step` (the grids), `--heat <tags>` (I' V uniform in the
shapes with these tags, in place of the heat density), `--device-temperature` (the current at 300 K plus the mean
rise of the Si: the size of the feedback that the workflow leaves out), and the thermal environment:
`--thermal-half-width W` (the thermal window and the substrate +-W um wide; default 5), `--substrate-depth D` (the
substrate D um deep; default 3) and `--heat-sink` (a crude heat path to the contacts: the p+ and n+ slab continues
to the window sides, and both window sides are at 300 K over their whole height; the carriers do not change), and
`--two-way` (the loop of step 6, with `--loop-tol` and `--damping`; the rows also give the loop record and the
one-way values of the same run). The rows of `benchmarks/2026-11-08/suite-pin.jsonl` come from these commands, in
this order (from the repository folder on the GPU computer):

```bash
B="python bench/pin_self_heating.py"
for a in "--tau 1e-9 --device-temperature" \
         "--tau 1e-8 --device-temperature --biases 0.8 0.85 0.9 0.95 1.0 1.05 1.1 1.15 1.2 1.25 1.3" \
         "--thermal-step 0.02" "--thermal-step 0.01" "--optics-step 0.01" "--heat slab,core" \
         "--heat slab,core,p,n" "--biases 1.15 1.155 1.16 1.165"; do bash tools/gpu_run.sh suite $B $a; done
```

The rows of `benchmarks/2026-11-08/suite-pin-2.jsonl` (the thermal environment) come from these commands:

```bash
B="python bench/pin_self_heating.py"
for a in "--thermal-half-width 20" "--thermal-half-width 20 --substrate-depth 18" \
         "--heat-sink --biases 0.8 0.85 0.9 0.95 1.0 1.05 1.1 1.15 1.2 1.25 1.3" \
         "--heat-sink --tau 1e-8 --biases 0.8 0.85 0.9 0.95 1.0 1.05 1.1 1.15 1.2 1.25 1.3"; do
  bash tools/gpu_run.sh suite $B $a; done
```

The rows of `benchmarks/2026-11-08/suite-pin-3.jsonl` (the two-way coupling) come from this command:

```bash
bash tools/gpu_run.sh suite python bench/pin_self_heating.py --two-way --tau 1e-9 1e-8 \
  --biases 0.8 0.85 0.9 0.95 1.0 1.05 1.1 1.15 1.2 1.25 1.3
```

Each run takes 24.7 to 34.7 s (the drift-diffusion sweep 19.5 to 20.9 s on the CPU; the thermal and mode part
4.2 to 7.2 s, 15.2 s with the 10 nm mode grid); peak RSS 1.77 to 2.03 GiB.

Grids, heat models and thermal environments (lifetimes 1 ns; the two files):

| Run | neff (0 V) | dneff_thermal at 1.0 V | at 1.2 V | dneff_carrier at 1.0 V | L_pi at 1.0 V (mm) | cross-over |
|---|---|---|---|---|---|---|
| default: thermal grid 50 nm, mode grid 20 nm, the heat density | 2.58936 | 6.9200e-4 | 4.1536e-3 | -1.6079e-3 | 0.8460 | 1.157 V, 62.5 A/m |
| thermal grid 20 nm (`--thermal-step 0.02`) | 2.58936 | 6.9385e-4 | 4.1460e-3 | -1.6079e-3 | 0.8477 | 1.158 V, 62.7 A/m |
| thermal grid 10 nm (`--thermal-step 0.01`) | 2.58936 | 6.9230e-4 | 4.1471e-3 | -1.6079e-3 | 0.8463 | 1.158 V, 62.7 A/m |
| mode grid 10 nm (`--optics-step 0.01`) | 2.56908 | 6.9441e-4 | 4.1676e-3 | -1.6428e-3 | 0.8170 | 1.161 V, 64.1 A/m |
| I' V uniform in the intrinsic zone (`--heat slab,core`) | 2.58936 | 7.2551e-4 | 4.4432e-3 | -1.6079e-3 | 0.8781 | 1.143 V, 56.8 A/m |
| I' V uniform in all the Si of the device (`--heat slab,core,p,n`) | 2.58936 | 7.0001e-4 | 4.2870e-3 | -1.6079e-3 | 0.8535 | 1.152 V, 60.1 A/m |
| thermal window and substrate +-20 um (`--thermal-half-width 20`; `suite-pin-2.jsonl`) | 2.58936 | 6.5187e-4 | 3.9072e-3 | -1.6079e-3 | 0.8105 | 1.170 V, 68.3 A/m |
| the same, substrate 18 um deep (`--substrate-depth 18`) | 2.58936 | 6.6235e-4 | 3.9714e-3 | -1.6079e-3 | 0.8195 | 1.167 V, 66.7 A/m |
| a heat path to the contacts (`--heat-sink`, to 1.3 V) | 2.58936 | 2.5135e-4 | 1.4174e-3 | -1.6079e-3 | 0.5713 | none up to 1.3 V |

- Thermal grid: the 50 nm grid differs from the 10 nm grid by -0.44 % (0.8 V) to +0.16 % (1.2 V) in
  dneff_thermal; the 20 nm grid by +0.22 % at 1.0 V (it samples the 90 nm slab as 80 nm). The cross-over moves by
  less than 1 mV.
- Mode grid: 10 nm against 20 nm changes neff by -0.020, dneff_carrier by +2.2 % (1.0 V; +2.1 % at 1.2 V),
  dneff_thermal by +0.35 % and the cross-over by +3.6 mV. The 20 nm grid is that of the charge-gpu example (the
  carrier part agrees with it, above); it is not converged.
- Heat distribution: the same power, uniform in the intrinsic zone, gives 4.8 % (1.0 V) and 7.0 % (1.2 V) more
  dneff_thermal than the heat density of the drift-diffusion solution and moves the cross-over 14.5 mV lower;
  uniform in all the Si, 1.2 % and 3.2 % more. So, in this thermal environment, the place of the heat matters at
  the level of a few percent; the thermal environment itself matters more (below).
- The thermal environment (`suite-pin-2.jsonl`; the carriers are the same, dneff_carrier to 6e-11):
  - The thermal window and the substrate +-20 um wide (the adiabatic sides farther away): dneff_thermal is 5.5 %
    (0.8 V) to 5.9 % (1.2 V) lower than in the default model; the cross-over moves 13 mV higher (1.170 V). With
    the substrate also 18 um deep: 4.1 % to 4.4 % lower, the cross-over 9.5 mV higher (1.167 V).
  - The heat path of `--heat-sink`: dneff_thermal is 0.405 (0.8 V) to 0.341 (1.2 V) of the default (59 % to 66 %
    lower; the core rise at 1.2 V is 7.53 K against 22.05 K), and there is no cross-over up to 1.3 V: the net
    change has its largest size at 1.2 V (-2.0774e-3, L_pi 0.373 mm, P_pi 36.4 mW) and is still -1.8775e-3 at
    1.3 V. At 10 ns: 63 % (0.8 V) to 68 % (1.3 V) lower, no cross-over up to 1.3 V, the shortest L_pi 0.173 mm
    at 1.25 V (P_pi 35.9 mW, a core rise of 14.79 K).
  - The test prints of `tests/test_pin.py` at 1.0 V (6.5187e-4 and 2.5135e-4) are the same values.
  - So the default model (adiabatic sides at +-5 um, no heat path to the contacts) gives more dneff_thermal than
    a device with a heat path. The thermal environment changes dneff_thermal and the cross-over much more than the
    grids and the heat distribution do: the cross-over and the sign change are results of the thermal model.
- Feedback of the device temperature (`--device-temperature`: one more drift-diffusion solve at 300 K plus the
  mean rise of the Si, from the 300 K solution; all converged): the current at the same bias rises by 0.36 %,
  1.86 %, 4.52 % and 6.69 % at 0.8, 0.9, 1.0 and 1.15 V (300.1, 300.8, 303.6 and 315.5 K), at 1 ns; at 10 ns by
  up to 10.66 % (1.15 V, 322.5 K), and by 5.94 % at 1.3 V (362.9 K). The two-way loop (below) gives the
  self-consistent values; its first step is this one pass.

Two-way coupling (`coupling="two-way"`, step 6; the rows of `benchmarks/2026-11-08/suite-pin-3.jsonl`, command
above; each row has the one-way values of the same run). Lifetimes 1 ns, the default grids and thermal model:

| V | T (K) | solves | I' one-way (A/m) | I' two-way: change | one pass (step 1) | core rise (K) | dneff_carrier | dneff_thermal | dneff (both) | L_pi (mm) | P_pi (mW) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.80 | 300.1105 | 2 | 0.5910 | +0.361 % | +0.360 % | 0.1143 | -1.9711e-4 | 2.1497e-5 | -1.7562e-4 | 4.41 | 2.09 |
| 0.90 | 300.8109 | 3 | 3.828 | +1.898 % | +1.862 % | 0.829 | -7.3071e-4 | 1.5600e-4 | -5.7472e-4 | 1.35 | 4.73 |
| 1.00 | 303.8151 | 4 | 15.92 | +4.730 % | +4.516 % | 3.85 | -1.6389e-3 | 7.2426e-4 | -9.1486e-4 | 0.847 | 14.12 |
| 1.05 | 306.8010 | 4 | 26.80 | +5.981 % | +5.645 % | 6.83 | -2.1252e-3 | 1.2854e-3 | -8.4017e-4 | 0.922 | 27.5 |
| 1.10 | 311.0272 | 5 | 41.23 | +6.844 % | +6.411 % | 11.03 | -2.6057e-3 | 2.0778e-3 | -5.2873e-4 | 1.47 | 71.0 |
| 1.15 | 316.6172 | 5 | 59.36 | +7.153 % | +6.689 % | 16.59 | -3.0708e-3 | 3.1244e-3 | +5.2101e-5 | 14.9 | 1088 |
| 1.20 | 323.6243 | 5 | 81.23 | +6.789 % | +6.381 % | 23.55 | -3.5132e-3 | 4.4355e-3 | +9.1995e-4 | 0.842 | 87.7 |
| 1.25 | 332.0251 | 5 | 106.9 | +5.693 % | +5.417 % | 31.88 | -3.9267e-3 | 6.0070e-3 | +2.0768e-3 | 0.373 | 52.7 |
| 1.30 | 341.7252 | 4 | 136.3 | +3.870 % | +3.754 % | 41.51 | -4.3071e-3 | 7.8217e-3 | +3.5096e-3 | 0.221 | 40.6 |

(T: the converged device temperature, 300 K plus the mean rise of the Si; solves: the drift-diffusion solves of
the loop; the one-way values of 0.8 to 1.2 V are those of the first table; one-way at 1.25 and 1.3 V: net dneff
+1.7221e-3 and +3.1065e-3, core rise 30.16 and 39.93 K.)

- Convergence: 2 to 5 solves to |G(T) - T| <= 1e-4 K (the last residual 1.4e-6 to 6.4e-5 K). The ratio of two
  successive residuals (the gain dG/dT) is 0.0036 at 0.8 V, 0.045 at 1.0 V, 0.060 to 0.067 from 1.1 to 1.2 V and
  0.030 to 0.038 at 1.3 V, so no damping is necessary. The loop takes 19.0 s for the 11 biases (42 solves) on the
  CPU of the GPU computer.
- The converged current is 0.361 % (0.8 V) to 7.153 % (1.15 V) above the one-way current; the one pass gives
  0.360 % to 6.689 %, so it misses up to 0.46 percentage points (1.15 V).
- The two parts grow by different amounts: at 1.0 V the carrier change by 1.93 %, the thermal change by 4.66 %, so
  the net change stays within 0.13 % of the one-way value (L_pi 0.847 mm against 0.846 mm), and P_pi rises by
  4.9 % (the current rises by 4.7 %). The thermal part grows more than the carrier part at every bias (dneff_thermal /
  |dneff_carrier| 0.442 against 0.430 at 1.0 V, 1.263 against 1.189 at 1.2 V), so the net change near the
  cross-over changes much: at 1.15 V it has the other sign (+5.2e-5 against -1.1e-4).
- The cross-over moves 11.8 mV lower: 1.1455 V, 61.85 A/m, 70.85 W/m (between 1.1 and 1.15 V) against 1.1573 V,
  62.54 A/m, 72.38 W/m one-way: a shift of the same size as that of the wider thermal windows above (9.5 to 13 mV
  higher), in the other direction.
- Lifetimes 10 ns (the same command; `tau_s` 1e-8): 1 to 6 solves (the last residual 1.6e-6 to 9.1e-5 K); the
  gain is 0.0010 at 0.8 V and 0.075 to 0.107 from 1.05 to 1.25 V. The converged current is 0.096 % (0.8 V) to
  11.892 % (1.15 V) above the one-way current, against 0.096 % to 10.663 % of the one pass (up to 1.23 percentage
  points less, at 1.15 V). T is 325.21 K at 1.15 V and 366.89 K at 1.3 V. Below the cross-over the shortest L_pi is
  0.366 mm at 1.05 V (P_pi 14.37 mW; one-way 0.362 mm, 12.98 mW). The cross-over moves 20.6 mV lower: 1.1854 V,
  125.4 A/m, 148.7 W/m (between 1.15 and 1.2 V) against 1.2060 V, 128.3 A/m, 154.7 W/m one-way; at 1.2 V the net
  change has the other sign (+3.9229e-4 against -1.9330e-4). At 1.3 V: net dneff +4.4708e-3 (one-way
  +3.5577e-3), L_pi 0.173 mm, P_pi 51.4 mW, a core rise of 66.10 K (62.10 K one-way). The loop takes 20.9 s
  (47 solves); each lifetime of the command takes 49.9 to 51.4 s (the 300 K sweep about 20 s), peak RSS
  1.91 and 2.00 GiB.
- The rise is not uniform over the Si of the charge window: max - min is 0.013 K at 0.8 V and 4.54 K at 1.3 V
  (1 ns), 7.36 K at 1.3 V (10 ns), 8 % to 13 % of the mean rise at every bias of both lifetimes. The loop gives
  all the Si the mean temperature.

Limits:
- Coupling: by default one-way (the carriers at 300 K; the bias scale of the first tables is that of a device at
  300 K). The two-way loop gives the device one temperature, the mean of the Si: charge-gpu has no temperature
  field (a field needs the thermoelectric terms of the current equations), and the field varies by 8 % to 13 % of
  the mean rise over the Si (above). The temperature models of charge-gpu (n_i of Misiakos and Tsamakis above
  340 K is an extrapolation; the mobility of Arora et al.; tau ~ T^-1/2 is an assumption: `charge_gpu/README.md`)
  set the size of the feedback; the plasma dispersion keeps the 300 K coefficients of Soref and Bennett.
- The heat source has no thermoelectric terms: the full model releases about E_g per recombination and has
  Peltier heat at the junctions and the contacts (Wachutka 1990). The total stays I' V; only its distribution
  changes. Its effect on dneff_thermal is not measured; the uniform heat models above change dneff_thermal by
  1.2 % to 7.0 %.
- The models of charge-gpu (no Auger recombination, no band-gap narrowing, no velocity saturation, Boltzmann
  statistics; its README) and their range: charge-gpu tests forward bias to 1.0 V; the rows above 1.0 V are
  outside that range (converged, but not checked against another solver), and the mean rise of the Si is above
  20 K at 1.2 V (1 ns) and from 1.15 V on (10 ns).
- The thermal model of `physics.temperature`: a 2-D cross-section of a long device, steady state, constant
  kappa (the bulk value of Si) and dn/dT. The default thermal environment has adiabatic window sides 5 um from the
  center and a slab that ends at the contacts in oxide, with no heat path to metal contacts. Both make the device
  hotter than with a heat path: the sides at +-20 um give 4 % to 6 % less dneff_thermal, and the crude heat path of
  `--heat-sink` 2.5 to 3.1 times less, with no cross-over up to 1.3 V (above). So the size of dneff_thermal, the
  cross-over (about 1.16 V here; the heat distribution and the window width move it by up to 15 mV) and the sign
  change are results of this thermal model, not of a PIN device in general; a device study needs its own thermal
  environment (the contacts, the metal, the substrate). The 50 nm
  thermal cells sample the 90 nm slab as 100 nm and the 220 nm rib as 200 nm (the heat stays in its place: the
  overlap); the grid rows above give the effect.
- `heat_source` uses the public `DDSolution.face_currents()` of charge-gpu (the face currents of the solve, the
  face and contact indices, the contact node potentials, the cell areas), `quasi_fermi()` and `recombination()`.
- The plasma dispersion of Soref and Bennett at 1.55 um only (`plasma=` for other coefficients), a constant
  base index, and the 20 nm mode grid of the default (2.2 % in dneff_carrier against 10 nm, above).
- The fundamental mode only. The workflow asks meow-gpu for one mode at its default target (just above the
  largest index of the cross-section, Si), so the eigenvalue nearest the target is the mode with the largest
  index. The modes that the meow-gpu solver did not give (`meow_gpu/README.md`, "The modes of the PEC walls":
  spurious modes of the PEC walls, now excluded by the solver) have neff = 1.444 or less in the taper window, far
  below neff = 2.589: they do not change this workflow. `test_the_mode_is_the_fundamental_mode` checks that neff at
  0 V is the largest index of a solve with 4 modes.

## Next

- Layer stacks (from a GDS cross-section); band structures (bands-gpu) for periodic waveguides.
