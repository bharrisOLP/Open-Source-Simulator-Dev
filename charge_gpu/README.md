# charge-gpu

Carrier distributions of reverse-biased PN junctions on the GPU (JAX; AMD ROCm first), for depletion modulators:
the nonlinear Poisson equation with constant quasi-Fermi levels on a 2-D rectilinear grid, then the plasma
dispersion effect and, with meow-gpu, the change of the mode index and V_pi L.

```python
import jax; jax.config.update("jax_enable_x64", True)   # float64 (in float32 the solve cannot reach its tolerance)
from charge_gpu.modulator import rib_pn
from charge_gpu.poisson import solve

dev = rib_pn()                      # 500 x 220 nm rib, 90 nm slab, lateral PN 3e17 cm^-3 (synthetic)
sol = solve(dev, v_n=2.0, v_p=0.0)  # 2 V reverse bias; sol.n, sol.p (m^-3), sol.psi (V)
```

## Method

- -div(eps grad psi) = q (p - n + N_D - N_A), n = n_i exp((psi - phi_n)/V_T), p = n_i exp((phi_p - psi)/V_T), with
  phi_n and phi_p fixed by the contacts. This is the standard approximation for reverse bias (flat majority
  quasi-Fermi levels, negligible minority carriers); it does not model forward bias or currents (use a
  drift-diffusion code such as DEVSIM for those).
- Cell-centered finite volumes; ohmic contacts as Dirichlet psi = V + V_T asinh(N_net / 2 n_i).
- Newton's method: the Jacobian is symmetric positive definite, so each step is a CG solve as one device loop,
  preconditioned with the aggregation multigrid of thermal-gpu (default; the Newton term is an extra diagonal, so
  the hierarchy is rebuilt per step) or Jacobi; log damping of the steps.
- Plasma dispersion at 1.55 um (Soref and Bennett 1987), from absolute concentrations (the hole term is
  P1^0.8 - P0^0.8). Each contact takes v_n or v_p from the sign of the doping under it.

## Checks (`tests/test_junction.py`)

| Case | Reference | Agreement |
|---|---|---|
| 1-D abrupt junction, 1e18 cm^-3 both sides, zero bias | built-in potential V_T ln(N_A N_D / n_i^2) (a consistency check: the contacts nearly fix it) | 1e-4 V |
| the same at 0, 1 and 3 V reverse bias | depletion width with the Debye term, W^2 ~ V_bi + V_R - 2 V_T (Sze) | 1 % |
| the same | charge neutrality (depleted donors = depleted acceptors) | 1e-3 |
| the same, 2 V reverse bias, 400 cells | an independent scipy solve (sparse Jacobian, direct solves, plain Newton) | 1e-8 V |
| plasma dispersion: changes from zero and from a doped baseline | Soref and Bennett coefficients (P1^0.8 - P0^0.8) | exact |

## Example: depletion modulator (`examples/pn_modulator.py`)

The synthetic rib of `modulator.rib_pn` (3e17 cm^-3 on both sides, contacts 3 um from the rib center), charge-gpu
plus meow-gpu: V_pi L = 1.39 V cm at 1 V, 1.60 V cm at 2 V and 1.96 V cm at 4 V reverse bias; the loss decreases by
1.0 to 2.8 dB/cm (reverse bias removes free carriers). Published depletion modulators of this type are typically
1.5 to 3 V cm. (The first version applied the hole power law to the change of the concentration and reported the
loss change with the wrong sign: 1.17 to 1.83 V cm and "extra loss". The one review found both.)

## Known limits and next steps

- Multigrid cut the CG iterations per bias point from 11k-36k (Jacobi) to 226-560 on 73k cells. The charge solve
  of the modulator example (0, 1, 2, 4 V) took 2.1 s instead of 17.7 s on the Mac CPU (measured by the review of
  this change: `reviews/2026-10-30-b648ae6b7661.md`). With Jacobi CG: 7.4 s on the GPU against 19.5 s on 16 CPU
  threads (`benchmarks/2026-10-30/run-20261030-0626.jsonl`). GPU timings with multigrid: not run yet.
- No forward bias, no currents, no 3-D, no temperature dependence.
