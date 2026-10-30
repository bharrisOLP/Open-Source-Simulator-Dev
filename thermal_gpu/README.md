# thermal-gpu

Steady and transient heat conduction for photonic heaters on the GPU (JAX; AMD ROCm first):
-div(kappa grad T) = q (and rho c_p dT/dt on the left for transients) on a 2-D rectilinear, non-uniform grid (a
device cross-section), with Dirichlet, adiabatic or convective (Robin) boundaries. Steady state also in 3-D
(`solver3d.py`): a finite-length heater with its ends.

```python
from thermal_gpu.solver import solve_steady
from thermal_gpu.structures import heater_stack

x, y, kappa, q, bc, masks = heater_stack(dx=0.02e-6, power_per_length=100.0)   # SI units; 100 W/m = 10 mW / 100 um
res = solve_steady(x, y, kappa, q, bc)                                          # res.T: (nx, ny) in K

from thermal_gpu.solver3d import solve_steady3d
from thermal_gpu.structures import heater_3d

x, y, z, kappa, q, bc, masks = heater_3d(dx=0.05e-6, length=100e-6, power=10e-3)   # z: the waveguide axis
res = solve_steady3d(x, y, z, kappa, q, bc)                                        # res.T: (nx, ny, nz) in K
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

## Known limits and next steps

- Temperature-dependent conductivity and 3-D transients are not done yet.
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
