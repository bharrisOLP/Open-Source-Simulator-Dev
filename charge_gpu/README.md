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

from charge_gpu.rc import small_signal
res = small_signal(dev, [1.0, 2.0])  # per bias: C' (F/m), R' (Ohm m), f_3dB
res[1].c, res[1].r, res[1].f3db(), res[1].f3db(length=0.5e-3, r_source=50.0)
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

### Capacitance, series resistance and RC bandwidth (`rc.py`)

All values are per length along the waveguide: C' in F/m, R' in Ohm m. A lumped device of length L has C = C' L
and R = R' / L.

- Capacitance C'(V) = dQ_n/dV, with Q_n the charge on the n side: the sum of q (p - n + N_D - N_A) over the
  n-doped cells. In this model the electron quasi-Fermi level is the n-contact voltage everywhere and the hole
  quasi-Fermi level is the p-contact voltage everywhere, so no conduction current crosses the junction. A change of
  Q_n must come through the n contact: I = dQ_n/dt, so dQ_n/dV is the small-signal capacitance (quasi-static, far
  below the dielectric relaxation frequency). The p side gives -dQ_p/dV, the same value (a check). Central
  differences (h = 20 mV, truncation error below 1e-4) with continuation: each solve starts from the converged
  potential of the previous bias. C' includes the fringe field through the oxide between the p and n silicon in
  the grid.
- Series resistance R'(V): a 1-D path model along x (one lateral junction, contacts on the xmin and xmax sides).
  Each grid column is equipotential in y and has the conductance G(x) = sum_y q (mu_n n + mu_p p) dy, with the
  carriers at the bias. Column x carries the conduction current f(x) I; f(x) is the part of the small-signal
  charge between the junction and the column: 1 in the neutral silicon, 0 deep in the depletion region
  (displacement current). R' = sum_x dx f^2 / G, the dissipation (Re Z for frequency to 0) of this distributed RC
  path. So the undepleted region at each bias sets R', and the depletion edge needs no threshold.
- Mobility: Masetti, Severi and Solmi, IEEE Trans. Electron Devices ED-30, 764 (1983), Table I, Si at 300 K:
  phosphorus parameters for electrons, boron parameters for holes, as a function of |N_D - N_A|.
- RC bandwidth of a lumped device: f_3dB = 1 / (2 pi (R'/L + R_s) C' L). With no source resistance (R_s = 0) it is
  1 / (2 pi R' C'), the same for all lengths.

## Checks (`tests/test_junction.py`, `tests/test_rc.py`)

| Case | Reference | Agreement |
|---|---|---|
| 1-D abrupt junction, 1e18 cm^-3 both sides, zero bias | built-in potential V_T ln(N_A N_D / n_i^2) (a consistency check: the contacts nearly fix it) | 1e-4 V |
| the same at 0, 1 and 3 V reverse bias | depletion width with the Debye term, W^2 ~ V_bi + V_R - 2 V_T (Sze) | 1 % |
| the same | charge neutrality (depleted donors = depleted acceptors) | 1e-3 |
| the same, 2 V reverse bias, 400 cells | an independent scipy solve (sparse Jacobian, direct solves, plain Newton) | 1e-8 V |
| plasma dispersion: changes from zero and from a doped baseline | Soref and Bennett coefficients (P1^0.8 - P0^0.8) | exact |
| 1-D abrupt junction, C = dQ_n/dV at 0, 1 and 3 V reverse bias | depletion approximation C = eps / W per area, W from the Sze check | test 3e-4; measured 9.7e-5, 3.1e-5, 1.2e-5 |
| the same | C decreases with reverse bias (this follows from the row above; not a separate check) | strictly decreasing |
| the same | charge-neutral total: Q_n + Q_p = 0 and -dQ_p/dV = dQ_n/dV | 1.4e-12 and 5e-11 (tests 1e-9 and 1e-8) |
| the same | R'_n and R'_p = (L/2 - W/2) / (q mu N t), the neutral lengths with W from Sze | test 1e-4; measured 8.1e-6, 3.4e-6, 1.7e-6 |
| uniform n and p slab (1e17 cm^-3, 100 nm, 2 um, in oxide), 0 V | sheet resistance: R' = L / (q mu N t); a linear f gives R'/3 | 1e-9; 1e-4 |
| Masetti mobility at 1e16 and 1e18 cm^-3, and the low-doping limit | the formula with the published Table I parameters, evaluated by hand (not with this code): electrons 1183.79 and 276.96, holes 424.91 and 160.31 cm^2/(V s); limits 1414 and 470.5 | 0.01 cm^2/(V s) |

Grid check (not a test; Mac CPU, 0 to 4 V in 9 steps):
- Rib of the example, C' with 10 nm against 5 nm cells: at most 0.09 % (1e17 cm^-3) and 0.27 % (3e17 cm^-3), the
  largest at 0 V, smooth in the bias.
- At 1e18 cm^-3 (Debye length 4.1 nm), 10 nm cells give an error of C' that oscillates with the bias (the
  depletion edges cross the cells): up to 1.9 % against 2.5 nm cells (at 0.5 V). With 5 nm cells the error is at
  most 0.23 % (at 0 V) and smooth. So the example uses 5 nm cells at 1e18 cm^-3. R' changes by at most 0.04 %.
- The 1-D junction (1e18 cm^-3) against eps / W: 10 nm cells -0.7 % to +2.4 % (oscillating), 5 nm cells +0.06 %
  to +0.37 %, 2.5 nm cells +0.02 % to +0.10 % (second order).

## Example: depletion modulator (`examples/pn_modulator.py`)

The synthetic rib of `modulator.rib_pn` (3e17 cm^-3 on both sides, contacts 3 um from the rib center), charge-gpu
plus meow-gpu: V_pi L = 1.39 V cm at 1 V, 1.60 V cm at 2 V and 1.96 V cm at 4 V reverse bias; the loss decreases by
1.0 to 2.8 dB/cm (reverse bias removes free carriers). Published depletion modulators of this type are typically
1.5 to 3 V cm. (The first version applied the hole power law to the change of the concentration and reported the
loss change with the wrong sign: 1.17 to 1.83 V cm and "extra loss". The one review found both.)

## Example: V_pi L, loss and RC bandwidth (`examples/pn_modulator_rc.py`)

The same synthetic rib at three dopings (both sides the same), 0 to 4 V reverse bias; 10 nm cells at 1e17 and
3e17 cm^-3, 5 nm cells at 1e18 cm^-3 (grid check above). charge-gpu gives C', R' and f_3dB. meow-gpu gives the
complex mode index from the absolute carriers (Soref and Bennett from undoped Si), so the loss is the free-carrier
loss of the mode at each bias (no scattering loss). f_3dB: intrinsic (R_s = 0) and for a lumped 0.5 mm device with
a 50 Ohm source.

```
uv run --frozen --project meow_gpu --with-editable ./charge_gpu python charge_gpu/examples/pn_modulator_rc.py
cd charge_gpu && uv run python examples/pn_modulator_rc.py --no-modes     # charge only: C', R', f_3dB
```

Mac CPU run (float64): the charge part (12 solves per doping) 3.4 and 2.6 s for 73k cells, 8.6 s for 293k cells
(1e18 cm^-3); the mode part 1.0 s per doping, 2.4 s for the first doping (with the compile). GPU (RX 7900 XT;
`benchmarks/2026-10-30/charge-rc.jsonl`): the same table to the printed digits; the charge part 4.4 and 2.6 s for
73k cells and 5.3 s for 293k cells; the mode part 1.5 s per doping, 3.3 s for the first (with the compile). So no
gain at 73k cells, and 1.6x at 293k cells.

| Doping (cm^-3) | Bias (V) | V_pi L (V cm) | Loss (dB/cm) | V_pi L x loss (V dB) | C' (fF/um) | R' (Ohm mm) | f_3dB, R_s = 0 (GHz) | f_3dB, 0.5 mm, 50 Ohm (GHz) |
|---|---|---|---|---|---|---|---|---|
| 1e17 | 0 | - | 1.94 | - | 0.214 | 86.2 | 8.6 | 6.7 |
| 1e17 | 1 | 2.09 | 1.40 | 2.9 | 0.151 | 85.2 | 12.4 | 9.6 |
| 1e17 | 2 | 2.51 | 1.04 | 2.6 | 0.125 | 84.6 | 15.1 | 11.7 |
| 1e17 | 4 | 3.31 | 0.58 | 1.9 | 0.098 | 83.7 | 19.5 | 15.0 |
| 3e17 | 0 | - | 7.18 | - | 0.334 | 40.0 | 11.9 | 7.3 |
| 3e17 | 1 | 1.39 | 6.20 | 8.6 | 0.238 | 39.7 | 16.9 | 10.4 |
| 3e17 | 2 | 1.60 | 5.49 | 8.8 | 0.197 | 39.4 | 20.4 | 12.5 |
| 3e17 | 4 | 1.96 | 4.42 | 8.6 | 0.157 | 39.1 | 25.8 | 15.8 |
| 1e18 | 0 | - | 26.9 | - | 0.551 | 19.2 | 15.1 | 6.5 |
| 1e18 | 1 | 0.93 | 25.1 | 23.4 | 0.394 | 19.1 | 21.2 | 9.2 |
| 1e18 | 2 | 1.06 | 23.8 | 25.2 | 0.328 | 19.0 | 25.6 | 11.0 |
| 1e18 | 4 | 1.26 | 21.8 | 27.5 | 0.261 | 18.9 | 32.2 | 13.9 |

- At 3e17 cm^-3 the V_pi L values agree with `examples/pn_modulator.py` (1.39, 1.60, 1.96 V cm) to 0.1 %, and the
  loss decreases by 1.0 to 2.8 dB/cm from 0 V, as there (a consistency check: absolute against relative carriers).
- The index change of the carriers on the 20 nm mode grid is the mean over each mode cell (8 x 8 points), not a
  point sample. At 1e18 cm^-3 each depletion edge moves by 6 to 11 nm per volt, less than one mode cell. With point
  samples, V_pi L at 1 V was 0.93, 0.82 and 0.78 V cm for 10, 5 and 2.5 nm charge cells (no convergence); with cell
  means it is 0.923, 0.930 and 0.932 V cm. At 3e17 cm^-3 the two methods agree to 0.1 % on the same grid
  (`examples/pn_modulator.py` uses point samples).
- More doping gives a lower V_pi L, but more loss and more capacitance. R' decreases faster than C' increases, so
  the intrinsic bandwidth increases with doping.
- With a 50 Ohm source and L = 0.5 mm, the time constant is R'C' + R_s C' L. At 2 V, R'C' decreases with doping
  (10.5, 7.8 and 6.2 ps for 1e17, 3e17 and 1e18 cm^-3), and R_s C' L increases (3.1, 4.9 and 8.2 ps). So the
  bandwidth stays at 11.0 to 12.5 GHz for all three dopings. The device resistance R'/L (169, 79 and 38 Ohm) is
  larger than the source resistance at 1e17 and 3e17 cm^-3.
- C' at 3e17 cm^-3 and 0 V is 24 % above the parallel-plate value of the rib (eps / W x 220 nm = 0.27 fF/um):
  the fringe field through the oxide.
- R' is high because the doping is uniform up to the contacts 3 um away (the p side gives about two thirds: lower
  hole mobility). Real devices add highly doped regions near the contacts.

## Known limits and next steps

- Multigrid cut the CG iterations per bias point from 11k-36k (Jacobi) to 226-560 on 73k cells. The charge solve
  of the modulator example (0, 1, 2, 4 V) took 2.1 s instead of 17.7 s on the Mac CPU (measured by the review of
  this change: `reviews/2026-10-30-b648ae6b7661.md`). With Jacobi CG: 7.4 s on the GPU against 19.5 s on 16 CPU
  threads (`benchmarks/2026-10-30/run-20261030-0626.jsonl`). With multigrid (the RC example, 12 solves per
  doping): 2.6 to 4.4 s per doping for 73k cells on the RX 7900 XT against 2.6 to 3.4 s on the Mac CPU, and 5.3 s
  against 8.6 s for 293k cells (`benchmarks/2026-10-30/charge-rc.jsonl`): no GPU gain at 73k cells.
- No forward bias, no currents, no 3-D, no temperature dependence.
- Capacitance: quasi-static and with constant quasi-Fermi levels, so no diffusion capacitance (forward bias) and
  no generation current. C' does not include the metal electrodes, the pads or the substrate under the oxide (the
  grid has 0.5 um of oxide and closed (zero-flux) outer sides).
- Series resistance: a 1-D path (each column equipotential in y; no current spreading between the slab and the
  rib), one lateral junction only (no PIN, vertical or L-shaped junctions), no contact or metal resistance, the net
  doping only (no compensation), the Masetti mobility at 300 K only.
- Bandwidth: a lumped RC only. A long device needs a traveling-wave model (electrode impedance, velocity match,
  RF loss).
