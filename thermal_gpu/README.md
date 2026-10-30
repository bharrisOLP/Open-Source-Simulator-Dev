# thermal-gpu

Steady and transient heat conduction for photonic heaters on the GPU (JAX; AMD ROCm first):
-div(kappa grad T) = q (and rho c_p dT/dt on the left for transients) on a 2-D rectilinear, non-uniform grid (a
device cross-section), with Dirichlet, adiabatic or convective (Robin) boundaries.

```python
from thermal_gpu.solver import solve_steady
from thermal_gpu.structures import heater_stack

x, y, kappa, q, bc, masks = heater_stack(dx=0.02e-6, power_per_length=100.0)   # SI units; 100 W/m = 10 mW / 100 um
res = solve_steady(x, y, kappa, q, bc)                                          # res.T: (nx, ny) in K
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

## Checks (`tests/test_steady.py`)

| Case | Reference | Agreement |
|---|---|---|
| uniformly heated slab, Dirichlet both sides | exact parabola | 1e-4 relative (200 cells) |
| two layers (Si 148 / SiO2 1.4 W/m/K) | exact series resistance | 1e-9 relative |
| heater over a waveguide, non-uniform grid, Robin top | scipy sparse direct solve of the same system | 1e-7 of the rise |
| slab with T = 0 on both sides, initial sin(pi y / L) (transient) | decay exp(-t/tau), tau = L^2 / (pi^2 alpha) | 1 % |

## Example: thermo-optic phase shifter (`examples/thermo_optic_phase_shifter.py`)

Temperature (this package) -> index change (dn/dT of Si 1.86e-4 /K, SiO2 0.95e-5 /K) -> mode index (meow-gpu) ->
P_pi = lambda / (2 dneff/dP'). For the synthetic SOI stack of `structures.heater_stack` (0.5 x 0.22 um Si
waveguide on 2 um oxide, 2 um heater with its base 2.2 um above the oxide-waveguide interface, so about 2 um above
the waveguide top): waveguide rise 20.3 K at 100 W/m, P_pi = 19.0 mW (50 nm thermal grid: `--dx 0.05e-6`). Published SOI heaters of this type are typically 20 to 30 mW.

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

## Known limits and next steps

- 3-D and temperature-dependent conductivity are not done yet.
