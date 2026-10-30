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
- Jacobi-preconditioned conjugate gradients as one jitted `lax.while_loop` (no host synchronization per
  iteration). Default tolerance 1e-12 relative: the system is ill-conditioned (large contrast, weak convective
  top), and 1e-9 gave errors of 0.5 % of the temperature rise against the direct solve. `SteadyResult.converged`
  is False when CG stops at max_iter.
- Transients (`solve_transient`): implicit Euler, one CG solve per step, started from the previous step.

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

## Known limits and next steps

- Jacobi CG needs O(n) iterations on an n x n grid: a multigrid preconditioner is next.
- 3-D and temperature-dependent conductivity are not done yet.
