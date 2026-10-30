# thermal-gpu

Steady and transient heat conduction for photonic heaters on the GPU (JAX; AMD ROCm first):
-div(kappa grad T) = q (and rho c_p dT/dt on the left for transients) on a 2-D rectilinear, non-uniform grid (a
device cross-section), with Dirichlet, adiabatic or convective (Robin) boundaries. Steady state and transients
also in 3-D (`solver3d.py`, `transient3d.py`): a finite-length heater with its ends, and its thermal rise time.
Steady state with a temperature-dependent conductivity kappa(T) in 2-D and 3-D (`nonlinear.py`).

```python
from thermal_gpu.solver import solve_steady
from thermal_gpu.structures import heater_stack

x, y, kappa, q, bc, masks = heater_stack(dx=0.02e-6, power_per_length=100.0)   # SI units; 100 W/m = 10 mW / 100 um
res = solve_steady(x, y, kappa, q, bc)                                          # res.T: (nx, ny) in K

from thermal_gpu.solver3d import solve_steady3d
from thermal_gpu.structures import heater_3d

x, y, z, kappa, q, bc, masks = heater_3d(dx=0.05e-6, length=100e-6, power=10e-3)   # z: the waveguide axis
res = solve_steady3d(x, y, z, kappa, q, bc)                                        # res.T: (nx, ny, nz) in K

from thermal_gpu.structures import heat_capacity, material_labels
from thermal_gpu.transient3d import geometric_stages, solve_transient3d

stages = geometric_stages(1e-8, 200e-6, per_stage=20)          # 10 ns steps, doubled every 20 steps, to 200 us
tr = solve_transient3d(x, y, z, kappa, q, bc, heat_capacity(kappa), stages, scheme="bdf2",
                       monitors=[masks["waveguide"]])            # tr.monitors: (n_steps + 1, 1) mean T in K

from thermal_gpu.nonlinear import SI_POWER_LAW, SIO2_CONSTANT, Constant, material_kappa, solve_steady_kappa3d

kappa_t = material_kappa(material_labels(kappa), [SI_POWER_LAW, SIO2_CONSTANT, Constant(100.0)])
nl = solve_steady_kappa3d(x, y, z, kappa_t, q, bc)                                 # Picard iteration; nl.converged
```

## Method

- Cell-centered finite volumes; the face conductance is the series (harmonic) form L / (d1/k1 + d2/k2), exact for
  layered media and robust at contrasts of 100 (Si/SiO2) and more.
- Conjugate gradients as one jitted `lax.while_loop` (no host synchronization per iteration), preconditioned with
  aggregation multigrid (`multigrid.py`, default) or Jacobi. Multigrid: 2 x 2 aggregation with the Galerkin
  coarse operator (again a conductance stencil, so the contrasts stay exact), a symmetric V-cycle with damped
  Jacobi and an over-corrected coarse step, a dense solve below 1024 cells. On the heater stack: 17, 21, 24 and
  27 CG iterations at 50, 20, 10 and 5 nm (Jacobi: 809, 2505, 3982 at 50, 20, 10 nm). Cells with a large aspect
  ratio need more iterations (the limits are in the docstring of `multigrid.py`).
- Default tolerance 1e-12 relative: the system is ill-conditioned (large contrast, weak convective top), and 1e-9
  gave errors of 0.5 % of the temperature rise against the direct solve. `SteadyResult.residual` is the true
  residual |b - A T| / |b| after the loop, and `converged` is False when it is above the tolerance: at max_iter,
  and in float32 (which cannot reach 1e-12; enable `jax_enable_x64`). An unknown `precond` raises ValueError.
- Transients (`solve_transient`): implicit Euler, one CG solve per step, started from the previous step.
  `TransientResult.converged` is False when a step does not reach the tolerance (`max_residual`).

## Method in 3-D (`fv3d.py`, `multigrid3d.py`, `solver3d.py`)

- The same finite volumes on a 3-D rectilinear, non-uniform grid (edges x, y, z): a 7-point conductance operator,
  face conductance S / (d1/k1 + d2/k2) with the face area S, and Dirichlet, adiabatic (default) or Robin boundaries
  on each of the six sides. These raise ValueError: an unknown side or boundary kind, a boundary with the wrong
  number of values, a value that is not finite, h <= 0, a wrong shape, edges that are not finite or do not
  increase, kappa that is not finite and positive, q that is not finite. In the structures, y is vertical and z is
  the waveguide axis.
- The same CG loop (one jitted `lax.while_loop`), the same true-residual `converged` flag and the same `SteadyResult`.
  Preconditioners: 3-D aggregation multigrid (default) or Jacobi (`precond="jacobi"`).
- Multigrid: 2 x 2 x 2 aggregation with the Galerkin coarse operator P^T A P, which is again a 7-point conductance
  operator; the V-cycle, smoother, over-correction and dense coarsest solve of the 2-D module (same constants).
  Long cells (a phase shifter has cells of 50 nm in x and y but up to 2 um in z): an axis is aggregated only when
  its median face conductance is at least 0.25 of the strongest axis (semi-coarsening). With cells 25 times longer
  in z: 20 CG iterations, against 121 with plain 2 x 2 x 2 aggregation. On cubic cells: 13, 17, 20 and 23 iterations
  at 16^3, 32^3, 64^3 and 128^3 cells (2.1 M cells).
- `cycle="W"` (option): two coarse cycles on each level. On the 3-D heater at 50 nm the V-cycle needs 62 iterations
  (8 levels, because semi-coarsening reduces the cells only by 4 per level) and the W-cycle 14, at 0.38 of the
  V-cycle time on the CPU (0.36 at 25 nm; on the GPU 0.60 and 0.35). The W-cycle runs many small kernels on the
  coarse levels: on the GPU its first solve (setup and compilation) takes 8 to 76 s against 2.9 to 3.7 s for the
  V-cycle (heater), so the V-cycle stays the default.
- Both cycles are positive definite preconditioners (the proof is in the docstring of `multigrid3d.py`). The
  eigenvalues of B A are in (0, 1.6] for the W-cycle and in (0, 1.6^(L-1)] for the V-cycle with L levels.

## Method: transients in 3-D and 2-D (`transient3d.py`)

- rho c_p dT/dt = div(kappa grad T) + q with the finite volumes of `fv3d.py`: M dT/dt = b - A T, M = rho_cp V.
  Each step is one symmetric positive definite solve (d M + A) T_new = rhs with the 3-D multigrid CG of
  `solver3d.py`; d M is an extra diagonal (as in the 2-D `solve_transient`), so the coarse levels keep it exactly.
- Schemes: `euler` (implicit Euler, order 1, L-stable), `bdf2` (order 2, L-stable; variable steps with the step
  ratio below 1 + sqrt(2); the first step is an euler step) and `cn` (Crank-Nicolson, order 2, A-stable but not
  L-stable: with steps much larger than the time constant of a mode, that mode changes its sign at each step).
- Time steps: a constant dt, or stages [(dt, n), ...] (`geometric_stages`: the step doubles every n steps, for
  responses with a wide range of time constants). Each stage builds one multigrid hierarchy for its main operator
  and runs as one jitted `lax.scan` on the device (no host synchronization per step). A step with an other
  diagonal (the first BDF2 step, the first step after a change of dt) uses the hierarchy of its stage as the
  preconditioner; CG solves its own operator.
- The unknowns are T - T_ref (T_ref: the first Dirichlet temperature). This shift is exact, and the relative
  residual then refers to the temperature change and the heat input, not to 300 K. `converged` is False when the
  true residual of one step is above tol (default 1e-10; CG stops at tol / 2, because the true residual can drift
  above the recursive one by round-off). Monitors: volume-weighted mean temperatures of masks or cells at each
  step. `solve_transient2d`: the 2-D cross-section as a 3-D problem with one z cell. `rise_time`: the 10 % and 90 %
  times of a monitor curve.

## Method: temperature-dependent conductivity (`nonlinear.py`)

- Picard iteration: kappa at the cell temperatures of the last iterate, the linear finite-volume problem (series
  face conductance of the two cell values), a multigrid CG solve from the last iterate. The operator stays
  symmetric positive definite (a Newton step adds terms that are not symmetric, so CG would not apply). The
  hierarchy of the first iteration is the preconditioner of the next iterations (`reuse_mg=True`, default). A new
  hierarchy is built when one solve needs more than 3 times the CG iterations of the first solve on the current
  one (`rebuild_factor`); `reuse_mg=False` builds one at each iteration.
- Stop when max |T_k+1 - T_k| <= 1e-10 max(1 K, rise). `converged`: the true nonlinear residual
  |b - A(kappa(T)) T| / |b| of the final field (in T - T_ref) is at most tol (1e-10) and the last CG solve
  converged. A power law kappa ~ T^-n with n > 1 has no steady solution above a limit of the heating (thermal
  runaway); then the flag is False.
- Laws (`PowerLaw`, `LinearLaw`, `Constant`, each with its Kirchhoff transform and the inverse; `material_kappa`
  maps material labels to laws). Each law has a valid range (t_min, t_max). When a cell of the final field is
  outside the range of its law, the solver gives a RuntimeWarning and `out_of_range` > 0 (`converged` refers to
  the discrete problem only).
  - Si: kappa = 148 (300 / T)^1.3 W/m/K (`SI_POWER_LAW`, valid from 300 to 800 K). Source: V. Palankovski, PhD
    thesis, TU Wien (2000), and V. Palankovski and R. Quay, "Analysis and Simulation of Heterostructure Devices",
    Springer (2004): kappa_300 = 148, alpha = -1.3, a fit to the data of Glassbrenner and Slack, Phys. Rev. 134,
    A1058 (1964). Within 3 % of Incropera et al., "Fundamentals of Heat and Mass Transfer", Table A.1 (98.9 and
    61.9 W/m/K at 400 and 600 K).
  - SiO2: nearly constant (`SIO2_CONSTANT`, 1.4 W/m/K). Option `SIO2_LINEAR`: 1.4 (1 + 9.42e-4 (T - 300)), valid
    from 300 to 600 K: the slope of fused silica in Incropera Table A.2 (1.38 W/m/K at 300 K, 1.51 at 400 K); at
    600 K the ratio to the 300 K value is within 2 % of the table (1.75 / 1.38).
- Heat capacities (`structures.heat_capacity`), Incropera Tables A.1 (Si, tungsten) and A.2 (fused silica) at
  300 K: Si 2330 x 712, fused silica 2220 x 745, and tungsten 19300 x 132 J/m^3/K for the generic heater metal.

## Checks (`tests/test_steady.py`)

| Case | Reference | Agreement |
|---|---|---|
| uniformly heated slab, Dirichlet both sides | exact parabola | 1e-4 relative (200 cells) |
| two layers (Si 148 / SiO2 1.4 W/m/K) | exact series resistance | 1e-9 relative |
| heater over a waveguide, non-uniform grid, Robin top | scipy sparse direct solve of the same system | 1e-7 of the rise |
| slab with T = 0 on both sides, initial sin(pi y / L) (transient) | decay exp(-t/tau), tau = L^2 / (pi^2 alpha) | 1 % |

3-D (`tests/test_steady3d.py`; measured values, the test bounds are larger):

| Case | Reference | Agreement |
|---|---|---|
| uniformly heated slab along x, y and z (200 cells; 3 x 4 non-uniform cells across) | exact parabola + q h^2 / (8 k) (the discrete solution on a uniform grid) | 1e-13 of the maximum |
| two layers (Si 148 / SiO2 1.4 W/m/K) across x, y and z, equal cells; and random cells (d1 != d2) with a Robin side | exact series resistance 1 / (t1/k1 + t2/k2 (+ 1/h)) | 3e-11 relative |
| heater block and waveguide, non-uniform 26 x 22 x 18 grid, contrast 148 / 1.4, Robin top | scipy sparse direct solve of the same system | 3e-10 (V), 2e-10 (W), 1.4e-9 (Jacobi) of the rise |
| the same, also Dirichlet and Robin on the x and z sides | scipy sparse direct solve of the same system | 4.0e-10 (V), 3.1e-10 (W), 2.6e-9 (Jacobi) of the range of T |
| z-invariant cross-section (non-uniform, Robin top and side), adiabatic z ends | the 2-D solver in each z cell | 9e-11 of the rise |
| heater on 3 of 9 z cells, same cross-section | 2-D solver with the z-averaged source = z average of the 3-D field (z fluxes cancel) | 3e-10 of the rise |
| coarse operator, 5 grids with odd sizes and factors (2,2,2), (1,2,2), (2,1,2), (2,2,1), (1,1,2) | explicit sparse P^T A P | 1e-14 relative |
| V- and W-cycle as matrices on a deep hierarchy (coarsest 8 cells), cells 1:1, 1:20 and 1:0.1 | symmetric; eigenvalues of B A in the proven bounds | V: 0.31 to 5.9 (5.9 on the flat cells), W: 0.44 to 1.40 |
| 3-D heater, 100 um, 10 mW, 100 nm x-y grid | 2-D cross-section at 100 W/m | mid-length rise 7e-6 relative |

Transients (`tests/test_transient3d.py`; measured values, the test bounds are larger):

| Case | Reference | Agreement |
|---|---|---|
| 3-D box mode (2 x 3 x 4 um, 16^3 cells, T = 0 on all sides), dt = tau / 40 to t = tau | exact discrete recurrence of each scheme with the discrete eigenvalue (the sampled sine is an exact eigenvector) | amplitude 3e-13 to 5e-12; the field stays the mode to 1e-12 to 9e-12 |
| the same | continuous time constant tau = 1 / (alpha pi^2 sum 1/L^2) | 1.6 % (euler), 0.35 % (bdf2), 0.32 % (cn); the grid alone: 0.32 % |
| the same at dt = tau / 10, / 20, / 40 | exp(-lambda_h t): order in time | 0.97, 0.99 (euler); 2.07, 2.03 (bdf2); 2.00, 2.00 (cn) |
| heater block, non-uniform 26 x 22 x 18 grid, contrast 148 / 1.4, three heat capacities, Robin and Dirichlet sides, stages with dt ratios 2 and 0.5 | scipy sparse direct solves of the same recurrences (an independent stepper in T) | 4e-12 to 7e-12 of the rise (V and W cycle, each scheme) |
| heater block from 300 K, steps growing to 30 us | `solve_steady3d` | 3.6e-10 of the rise (euler, bdf2) |
| steady field as the initial field, 5 steps | fixed point | 2.1e-10 of the rise (each scheme) |
| 2-D cross-section with one z cell (`solve_transient2d`) | the 2-D `solve_transient` (implicit Euler) | below 1e-8 of the rise |
| 3-D heater (100 um, 10 mW; 6 um wide, coarse z), rise time 10 % to 90 % at mid-length | 2-D cross-section (100 W/m), same steps | below 1e-3 |

kappa(T) (`tests/test_nonlinear.py`):

| Case | Reference | Agreement |
|---|---|---|
| Si slab, 10 um, uniform heating, T = 300 K on both sides, T_max = 450 K | Kirchhoff: theta = 300 + q y (L - y) / (2 kappa0), T = inverse(theta) | 2.86e-4 and 7.15e-5 of the rise at 100 and 200 cells (order 2.0); 11 Picard iterations |
| 100 um Si (power law) + 1 um SiO2 (linear law), 300 K and 500 K, no source | Kirchhoff in each layer, the same flux (interface 414.75 K) | 4.2e-6 and 1.06e-6 of the 200 K step (order 2.0) |
| heater block over Si, non-uniform 26 x 22 x 18 grid, Robin sides, Si at 409 to 410 K | an independent Picard iteration with scipy direct solves | 1.6e-12 (default: one hierarchy, no rebuild needed) and 3.1e-12 (a new hierarchy each iteration) of the rise |
| Si slab heated above the runaway limit | no steady solution | `converged` False |
| Si slab at 765 K, 200 cells: a new hierarchy each iteration, automatic rebuild (default), first hierarchy only | the same field | below 1e-9 of the rise; CG iterations 14, 47 (3 hierarchies) and 143 (at 1186 K: 16, 65 and 245) |
| Si slab at 1186 K (Si law valid to 800 K) | valid range | RuntimeWarning, `out_of_range` > 0 (no warning at 765 K) |
| constant kappa through the Picard path, 2-D heater stack | `solve_steady` | below 1e-8 of the rise |

## Example: thermo-optic phase shifter (`examples/thermo_optic_phase_shifter.py`)

Temperature (this package) -> index change (dn/dT of Si 1.86e-4 /K, SiO2 0.95e-5 /K) -> mode index (meow-gpu) ->
P_pi = lambda / (2 dneff/dP'). For the synthetic SOI stack of `structures.heater_stack` (0.5 x 0.22 um Si
waveguide on 2 um oxide, 2 um heater with its base 2.2 um above the oxide-waveguide interface, so about 2 um above
the waveguide top): waveguide rise 20.3 K at 100 W/m, P_pi = 19.0 mW (50 nm thermal grid: `--dx 0.05e-6`). Published SOI heaters of this type are typically 20 to 30 mW.

## Example: the 3-D heater with its ends (`structures.heater_3d`)

The cross-section of `heater_stack` along z: a 100 um heater with 10 mW (100 W/m), the waveguide and the layers
continue 20 um past each end (adiabatic z ends). The z grid is 0.25 um at the heater ends and grows to 2 um.
At 50 nm in x and y (200 x 146 x 94 = 2.7 M cells):

| Quantity | 3-D | 2-D cross-section (100 W/m) |
|---|---|---|
| waveguide rise at mid-length | 20.2663 K | 20.2664 K (difference 7e-6 relative) |
| waveguide rise at the heater ends | 10.67 K (0.53 of the 2-D rise) | - |
| rise 5 um inside the ends / 5 um outside | 17.3 K / 2.9 K | - |
| z integral of the rise / (2-D rise x 100 um) | 1.0001 | 1 |

So the 2-D model gives the mid-length temperature of a 100 um heater exactly, and also the phase (the z integral
of the rise): the heat that flows past the ends still heats the waveguide. At each end the rise falls from 85 % to
15 % of the 2-D rise between 5 um inside and 5 um outside. With metal leads of 4 um width past the ends
(`lead_w=4e-6`, 100 nm grid), the leads take heat away: 0.47 of the 2-D rise at the ends, and the z integral is
0.997 of the 2-D value.

## Example: the thermal rise time of the 3-D heater (`bench/bench_transient3d.py`)

The heater of `structures.heater_3d` (100 um, 10 mW) switches on at t = 0; the structure starts at 300 K. The
rise time from 10 % to 90 % of the steady waveguide rise sets the switching speed of the thermo-optic phase
shifter (the fall after switch-off is the mirror image of this curve, as the problem is linear). BDF2, 10 ns
steps that double every 20 steps up to 200 us (200 steps). Small runs on the GPU computer (the same values on a
CPU):
`bench_transient3d.py --case heater3d --dx 0.1e-6 --compare2d` (100 x 73 x 94 = 686k cells):

| Quantity | 3-D, waveguide at mid-length | 2-D cross-section (100 W/m) |
|---|---|---|
| steady rise | 20.2868 K | 20.2870 K |
| 10 % / 90 % time | 1.4164 us / 19.355 us | 1.4164 us / 19.356 us |
| rise time 10 % to 90 % | 17.939 us | 17.939 us (difference 3e-5 relative) |

So at mid-length a 100 um heater switches as its 2-D cross-section. The response has a fast start (10 % after
1.4 us: the heater and the oxide near it) and a slow tail (90 % after 19 us: the heat must cross the 2 um buried
oxide to the substrate). Time-step check on the 2-D cross-section at 50 nm (`--case cross2d --dx 0.05e-6
--per-stage 10 --dt0 2e-8 --dt-scan 3`: per-stage 10, 20, 40 with dt0 20, 10, 5 ns): 17.905, 17.939 and 17.945 us
(the differences fall by 5.2).

## Example: kappa(T) on the 2-D heater (`bench/bench_kappa_t.py`)

The waveguide rise of `structures.heater_stack` with a constant kappa, with Si kappa(T), and with Si and SiO2
kappa(T) (`--dx 0.05e-6 0.02e-6 --power 100 200`, on the GPU computer; the same values on a CPU):

| Grid, power | constant kappa | Si kappa(T) | Si and SiO2 kappa(T) |
|---|---|---|---|
| 50 nm, 100 W/m | 20.2664 K | 20.2667 K (+1.3e-5) | 20.0780 K (-0.93 %), 7 Picard iterations |
| 20 nm, 100 W/m | 20.3541 K | 20.3545 K (+1.5e-5) | 20.1641 K (-0.93 %) |
| 50 nm, 200 W/m | 40.5329 K | 40.5340 K (+2.7e-5) | 39.7926 K (-1.83 %) |
| 20 nm, 200 W/m | 40.7083 K | 40.7095 K (+3.0e-5) | 39.9619 K (-1.83 %) |

The 3-D heater at mid-length (`--case heater3d --dx 0.1e-6 --power 100`) gives the same ratios: 20.2868 K, +1.3e-5
with Si kappa(T), 20.0981 K (-0.93 %) with Si and SiO2 kappa(T).

The kappa(T) of Si has almost no effect. The Si waveguide is at about 320 K (its kappa is 8 % lower), but it is an
island in the oxide, not in series with the heat path; the Si substrate stays near 300 K, and the oxide sets the
thermal resistance. The small increase of the oxide conductivity with T lowers the
rise by 0.93 % at 100 W/m and by 1.83 % at 200 W/m: the constant-kappa model overestimates the rise by these amounts.

## GPU timings (RX 7900 XT against the same algorithm on the 16 CPU threads of the same PC)

Heater stack, steady state, CG time after compilation (`bench/bench_steady.py`; the JSON rows are in
`benchmarks/2026-10-30/`):

| Grid | Cells | Multigrid CG, GPU / CPU | Jacobi CG, GPU / CPU |
|---|---|---|---|
| 20 nm | 183k | 0.099 s / 0.228 s (21 iterations) | 0.238 s / 1.061 s (2505 iterations) |
| 10 nm | 730k | 0.120 s / 0.297 s (24) | 0.765 s / 5.212 s (3982) |
| 5 nm | 2.92 M | 0.208 s / 1.303 s (27) | not run |

Multigrid rows: `thermal-mg.jsonl`. Jacobi rows: `run-20261030-0626.jsonl` (Jacobi was the default of the solver
at that time; `--precond jacobi` now gives the same runs). With multigrid the GPU gains 2.3x to 6.3x at these
sizes; on the small grids, most of the time is fixed cost per iteration.

3-D (`bench/bench_steady3d.py`, rows in `benchmarks/2026-10-30/thermal3d.jsonl`; CG time after compilation):

| Case | Cells | V-cycle, GPU / CPU | W-cycle, GPU / CPU |
|---|---|---|---|
| heater, 100 nm | 686k | 0.066 s / 0.97 s (41 iterations) | 0.076 s / 0.63 s (14) |
| heater, 50 nm | 2.7 M | 0.23 s / 5.9 s (62) | 0.14 s / 2.2 s (14) |
| heater, 25 nm | 11.0 M | 1.03 s / 34.0 s (88) | 0.36 s / 12.2 s (15) |
| cube 64^3 / 128^3 / 192^3 | 0.26 / 2.1 / 7.1 M | 0.018 / 0.056 / 0.17 s against 0.19 / 1.66 / 7.75 s (20 / 23 / 27) | - |

The GPU gains 8x (W-cycle, 686k cells) to 46x (cube 192^3) at these sizes. The first solve on the GPU (setup and
compilation) takes 2.9 to 3.7 s (heater) and 0.8 to 2.8 s (cube) with the V-cycle, and 8 to 76 s (heater) with
the W-cycle. The heater results are the same on both devices (waveguide rise at mid-length 20.380 K at 25 nm, 2-D
cross-section 20.380 K).

Transients and kappa(T) (rows in `benchmarks/2026-10-30/thermal-transient.jsonl`; RX 7900 XT against the 16 CPU
threads of the same computer with `THERMAL_GPU_DEVICE=cpu`; the second run, after compilation). Commands (on the GPU
computer, from the repository folder):

```
bash tools/gpu_run.sh thermal_gpu python bench/bench_transient3d.py --case heater3d --dx 0.1e-6 0.05e-6 --compare2d --repeat
bash tools/gpu_run.sh thermal_gpu python bench/bench_transient3d.py --case heater3d --dx 0.05e-6 --cycle W --repeat
bash tools/gpu_run.sh thermal_gpu python bench/bench_transient3d.py --case cross2d --dx 0.05e-6 0.02e-6 --repeat
bash tools/gpu_run.sh thermal_gpu python bench/bench_kappa_t.py --case cross2d --dx 0.05e-6 0.02e-6 0.01e-6 --power 100 200
bash tools/gpu_run.sh thermal_gpu python bench/bench_kappa_t.py --case heater3d --dx 0.1e-6 0.05e-6 --power 100
```

| Transient (BDF2, 200 steps to 200 us) | Cells | GPU | CPU | 10-90 % rise time |
|---|---|---|---|---|
| 3-D heater, 100 nm, V-cycle | 686k | 3.6 s | 44.2 s (12x) | 17.939 us (2-D: 17.939 us) |
| 3-D heater, 50 nm, V-cycle | 2.7 M | 20.4 s | 348.7 s (17x) | 17.938 us |
| 3-D heater, 50 nm, W-cycle | 2.7 M | 13.4 s (40.5 s first run) | 164.1 s (12x) | 17.938 us |
| 2-D cross-section, 50 nm | 29k | 0.91 s | 4.2 s (4.6x) | 17.939 us |
| 2-D cross-section, 20 nm | 183k | 1.95 s | 19.2 s (9.8x) | 17.920 us |

The kappa(T) solves (Picard, 2 to 8 iterations) take 0.16 to 0.30 s on the GPU against 0.6 to 2.1 s on the CPU for
the 2-D cross-section at 10 nm (730k cells), and 1.1 to 1.5 s against 5.4 to 15.4 s for the 3-D heater at 50 nm
(the first solve of each run includes the compilation).

## Known limits and next steps

- kappa(T) is for the steady state only; the transients use a constant kappa and a constant rho c_p. The source of
  a transient is constant from t = 0 (a step response; the fall after switch-off is its complement).
- Picard converges linearly: for Si kappa(T) on the heater 4 iterations (2-D) to 6 (3-D), 7 to 8 with SiO2
  kappa(T), 11 on the 450 K Si slab, 14 on the 765 K Si slab. When kappa changes by a large factor, the first
  hierarchy alone needs 10 to 15 times more CG iterations than a new hierarchy at each iteration (Si slab: 143
  against 14 at 765 K, 245 against 16 at 1186 K). The automatic rebuild (default) gives 47 and 65 with 3
  hierarchies. On a large grid each new hierarchy costs a host setup; `rebuild_factor` sets the balance.
- The valid range of the laws is checked on the final field only (a warning), not during the iteration.
- Crank-Nicolson is not L-stable: use `bdf2` for steps much larger than the fast time constants (the heater, the
  geometric stages). Each stage of the transient builds a multigrid hierarchy on the host and compiles the stage
  loop once per new number of steps.
- The new solvers use the residual of T - T_ref (relative to the heat input). Their default tol is 1e-10; its
  round-off floor on the heater stack is about 1e-12 to 1e-11. The old 2-D `solve_transient` solves for T with tol
  1e-12 relative to (C / dt) T near 300 K, a weaker measure (its difference to `solve_transient2d` is up to 1e-8 of
  the rise).
- 3-D multigrid: the V-cycle needs 41, 62 and 88 iterations on the 3-D heater at 100, 50 and 25 nm (2-D: 17 at
  50 nm). Its levels aggregate only x and y; on such levels the over-correction 1.6 overshoots on modes that change
  only along z, and this compounds per level. CG estimates of the eigenvalues of B A: 0.29 to 4.2 at 100 nm, 0.18
  to 7.0 at 50 nm (up to 29 on random test hierarchies), so the condition number grows at both ends. An
  over-correction of 1.0 on these levels limits the top to 1, but the bottom falls to 0.04 and 0.019 and CG needs
  58 and 83 iterations; so it stays 1.6. The W-cycle (eigenvalues in (0, 1.6]) needs 14, 14 and 15 iterations.
- The axis rule uses one median per axis and level: a grid with a strongly changing aspect ratio along an axis
  (fine and coarse z cells) gets one decision for all its cells.
- The true residual has a round-off floor of about eps |A| |T| / |b|. On a fine 1-D line of cells (200 cells,
  n^2 / 8 = 5000) it is 2e-12 to 7e-12, above the default tolerance 1e-12: `converged` is then False although T
  is correct to 1e-13. Then use tol = 1e-11. The floor is also high when a weak Robin side is the only source
  of b and the cells are thin and conductive (two layers with 1.5 nm Si cells, h = 2e5 W/m^2/K: 6e-11).
- The 3-D heater has no vias, pads or routing; `lead_w` gives a simple metal lead in the heater layer.
