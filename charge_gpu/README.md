# charge-gpu

Carrier distributions of PN and PIN junctions on the GPU (JAX; AMD ROCm first), for silicon modulators:
- reverse bias (depletion modulators): the nonlinear Poisson equation with constant quasi-Fermi levels on a 2-D
  rectilinear grid (`poisson.py`), the small-signal C', R' and f_3dB (`rc.py`);
- forward and reverse bias with currents (injection phase shifters, diodes): drift-diffusion, the van Roosbroeck
  system (`drift_diffusion.py`);
- then the plasma dispersion effect and, with meow-gpu, the change of the mode index, V_pi L or L_pi, and the loss.

```python
import jax; jax.config.update("jax_enable_x64", True)   # float64 (in float32 the solve cannot reach its tolerance)
from charge_gpu.modulator import rib_pn
from charge_gpu.poisson import solve

dev = rib_pn()                      # 500 x 220 nm rib, 90 nm slab, lateral PN 3e17 cm^-3 (synthetic)
sol = solve(dev, v_n=2.0, v_p=0.0)  # 2 V reverse bias; sol.n, sol.p (m^-3), sol.psi (V)

from charge_gpu.rc import small_signal
res = small_signal(dev, [1.0, 2.0])  # per bias: C' (F/m), R' (Ohm m), f_3dB
res[1].c, res[1].r, res[1].f3db(), res[1].f3db(length=0.5e-3, r_source=50.0)

from charge_gpu import drift_diffusion as dd
from charge_gpu.injection import rib_pin, rib_index_change
pin = rib_pin()                                           # lateral PIN rib, p+ and n+ 1e18 cm^-3 (synthetic)
sols = dd.sweep(pin, [0.0, 0.9], tau_n=1e-9, tau_p=1e-9)  # forward bias V = v_p - v_n, continuation from 0 V
sols[1].current                                           # A/m (= mA/mm) out of the n contact
rib_index_change(pin, sols[1], sols[0])                   # mean n, p, index change and loss change in the rib
```

## Method

- -div(eps grad psi) = q (p - n + N_D - N_A), n = n_i exp((psi - phi_n)/V_T), p = n_i exp((phi_p - psi)/V_T), with
  phi_n and phi_p fixed by the contacts. This is the standard approximation for reverse bias (flat majority
  quasi-Fermi levels, negligible minority carriers); it does not model forward bias or currents (use
  `drift_diffusion.py`, below).
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

### Drift-diffusion for forward and reverse bias (`drift_diffusion.py`, `injection.py`)

The van Roosbroeck system on the same grid, with the same Poisson operator and contacts as `poisson.py` (the API
and the results of `poisson.py` and `rc.py` do not change):

    -div(eps grad psi) = q (p - n + N_D - N_A)
     div J_n =  q R,   J_n = q mu_n (-n grad psi + V_T grad n)
     div J_p = -q R,   J_p = q mu_p (-p grad psi - V_T grad p)
     R = (n p - n_i^2) / (tau_p (n + n_i) + tau_n (p + n_i))

- Scharfetter-Gummel fluxes on each face between two semiconductor cells (D = mu V_T; the face D is the mean of
  the two cells). No carrier flux into the oxide.
- Ohmic contacts: the contact node, half a cell from the cell center, has the charge-neutral equilibrium values
  (psi as in `poisson.py`, both quasi-Fermi levels at the contact voltage). The contact current is the sum of the
  Scharfetter-Gummel currents of the contact faces.
- Mobility: the Masetti mobility of `rc.py` (Si, 300 K, from |N_D - N_A|). It is constant in the field: no velocity
  saturation and no carrier-carrier scattering. A user mobility (per cell or scalar) is an option.
- Recombination: Shockley-Read-Hall with a midgap trap and constant lifetimes. Default tau_n = tau_p = 1 us (a
  usual value for moderately doped bulk Si). The checks and the use case give their own lifetimes: 100 us for the
  Shockley check (see there), 1 ns and 10 ns for the SOI rib (effective lifetime of SOI rib waveguides about 1 ns:
  Dimitropoulos et al., Appl. Phys. Lett. 86, 071115 (2005)).
- Two solvers, and where they run:
  - `method="newton"` (default): the coupled Newton method on (psi, n, p). The Jacobian is assembled on the host and
    solved with the sparse direct LU of scipy (SuperLU) on the CPU. The n and p columns are scaled with n and p (the
    unknowns are the relative changes), the rows are equilibrated. Damping: the step of psi is at most 10 V_T; a
    carrier density changes by at most a factor exp(10) per step and stays positive.
  - `method="gummel"`: the decoupled Gummel iteration. The Poisson step (symmetric positive definite: the nonlinear
    Poisson equation with the quasi-Fermi levels of the last carriers per cell) runs on the JAX device with the
    multigrid-preconditioned CG of `poisson.py`. The two continuity steps (nonsymmetric, linear at fixed psi, with
    R linearized at the last carriers) use the sparse direct LU on the CPU. Fast at low bias, slow at high
    injection (linear convergence).
- `sweep()`: bias continuation from 0 V (started from `poisson.solve`) in steps of at most 50 mV, each solve
  started from the last converged solution; a failed step is halved.
- `injection.py`: a graded 1-D grid, the 1-D diode, the Shockley J0, the lateral PIN rib and the plasma dispersion
  of the rib (`modulator.plasma_dispersion`).

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

## Checks of the drift-diffusion solver (`tests/test_drift_diffusion.py`)

All values from the tests and from the GPU computer (the Newton solves run on its CPU, the Gummel Poisson step on
the RX 7900 XT). The 1-D junction is the one of `test_junction.py` (1e18 cm^-3 both sides, 1 um, 2.5 nm cells).

| Case | Reference | Agreement |
|---|---|---|
| 1-D junction, 0 V, started from a perturbed state (20 mV on psi, 30 % on n and p) | `poisson.solve` (equilibrium) | Newton: psi 3.3e-16 V, n and p 1.4e-14; Gummel: 6.8e-15 V, 7.3e-13 (tests: 1e-11 V, 1e-9) |
| the same | no current | below 8e-11 A/m at every face (the round-off floor, see the limits; test 1e-9 A/m) |
| 1-D junction, 1 V and 3 V reverse, by continuation from 0 V | `poisson.solve` | psi 3.6e-15 and 8.0e-15 V; the majority carriers wherever they exceed n_i: 1.2e-8 and 3.6e-8 (test 1e-7) |
| the same | the minority carriers far from the junction | the equilibrium value n_i^2 / N_A (0.27 % lower in the first cell: the linear decrease toward the depletion edge), so exp(V/V_T) above `poisson.solve`, which keeps phi_n at the n-contact voltage |
| 1-D junction, 0.6 V forward and 1 V reverse | Gummel against Newton (two solvers, two linear solvers; the Gummel Poisson step on rocm:0) | psi 5e-15 V, n and p 8e-13, current 2.7e-8 at 0.6 V |
| 1-D long-base diode, 1e18 cm^-3 both sides, tau 100 us, contacts 6 L from the junction (910 cells, 1 nm at the junction, 3 % growth) | Shockley ideal diode J0 (exp(V/V_T) - 1), J0 = q n_i^2 (D_n / (L_n N_A) + D_p / (L_p N_D)) = 7.55e-15 A/cm^2 (Sze and Ng, Ch. 2) | J / J_Shockley = 1.0099, 1.0041, 1.0015, 0.9994, 0.9908 at 0.55, 0.60, 0.65, 0.70, 0.75 V (tests: 1 % from 0.60 to 0.75 V; 1.5 % at 0.55 V, where the depletion recombination adds the expected 0.9 to 1 %) |
| the same, outside that range | the known deviations | 1.897 at 0.3 V, 1.144 at 0.4 V, 1.024 at 0.5 V: recombination in the depletion region (Sah-Noyce-Shockley, ~ exp(V / 2 V_T): the excess 2.39, 0.99, 0.41 % at 0.50, 0.55, 0.60 V falls by 0.41 per 50 mV against exp(-0.05 V / 2 V_T) = 0.38; test 0.3 to 0.5); 0.939 at 0.8 V: high injection (n_p / N_A = 2.8e-3) and ohmic drop |
| the same | current continuity: the total current through each of the 909 cross-sections and both contacts | contact currents sum to below 1.2e-14 A/m; cross-sections within 2.5e-10 A/m (relative 3.8e-5 at 0.65 V, 1.3e-7 at 0.8 V) |
| the same, 0.65 V, three grids (fine cell 2, 1, 0.5 nm with 6, 3, 1.5 % growth) | Richardson extrapolation | relative differences 2.1e-4 and 4.8e-5 (ratio 4.3: second order); the default grid 6.4e-5 from the extrapolation (test 1e-4) |
| 2-D PIN rib (10 nm cells, contacts 1 um from the center), 0.9 V | current continuity; global balance I = q sum R A + the minority currents of the contacts | 1e-9 and 1e-8 of I (tests) |
| the same, 0.8 and 0.9 V | high injection: quasi-neutral, below the limit n_i exp(V / 2 V_T) | p - n = the background 1e15 cm^-3 to 5 %; n rises, the index falls, the loss rises |
| B(x), B'(x), the Jacobian | identities and central differences | 1e-13; 1e-7; 1e-6 |
| 2-D block with an xmin contact and a ymax contact (L-shaped path), 0.4 V | the contact currents sum to zero | 1.7e-8 of I (round-off; test 1e-7); `x_currents` refuses this device (x faces only) |
| `sweep` with a RuntimeError of the linear solver in one step | the step is halved and the sweep goes on | 0.1 V reached through 0.05 V; `guess=` is refused |
| `rib_pin` with 20 nm cells | the boundaries must be on cell edges | ValueError (a 20 nm grid made the 90 nm slab 80 nm thick) |

Why tau = 100 us for the Shockley check: the ideal-diode current neglects the recombination in the depletion
region. Its ratio to the diffusion current scales as W N / (n_i sqrt(D tau)) exp(-V / 2 V_T). High injection and
the ohmic drop depend on n_p / N, not on tau (the ohmic estimate J rho L / V_T = (n_p / N)(mu_min / mu_maj) has no
tau); here they reduce J by 0.9 % at n_p / N = 4e-4 (0.75 V) and 6 % at 2.8e-3 (0.8 V). With tau = 1 us
(1e17 cm^-3) the two effects
overlap: J / J_Shockley = 1.063, 1.027, 1.009, 0.984, 0.877 at 0.50, 0.55, 0.60, 0.65, 0.70 V (a scratch run on the
GPU computer; not a test). A long lifetime makes the window where the Shockley assumptions hold.

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

## Example: forward-biased PIN phase shifter (`examples/pin_phase_shifter.py`)

The synthetic lateral PIN of `injection.rib_pin`: 500 x 220 nm rib on a 90 nm slab, intrinsic zone (p 1e15 cm^-3,
a usual SOI wafer) up to 0.3 um from the rib sides, p+ and n+ 1e18 cm^-3 beyond, contacts at the slab ends 1.5 um
from the center, 10 nm cells (36.6k cells, 43k unknowns). Drift-diffusion with Newton, 0 to 1.0 V in 50 mV steps.
meow-gpu gives the complex mode index (the mode grid and the cell means of `examples/pn_modulator_rc.py`, Soref and
Bennett from undoped Si): dneff from 0 V, L_pi = lambda / (2 |dneff|), the free-carrier loss of the mode, the loss
over L_pi and P_pi = I V L_pi. Columns "rib": the area means over the rib core (material values).

```
bash tools/remote.sh sh <<'EOF'
bash tools/gpu_run.sh charge_gpu python examples/pin_phase_shifter.py --modes --tau 1e-9 1e-8
EOF
```

GPU computer (the Newton solves on its CPU; `benchmarks/2026-10-30/charge-dd.jsonl`): the charge part 17.1 s and
16.1 s per lifetime (21 continuation steps), the modes 2.8 s and 0.8 s.

| tau (ns) | V | I (mA/mm) | n rib (cm^-3) | p rib (cm^-3) | dn rib | dloss rib (dB/cm) | dneff | L_pi (mm) | loss of the mode (dB/cm) | loss in L_pi (dB) | P_pi (mW) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 0 | 1.7e9 | 1.4e13 | 0 | 0 | - | - | 0.12 | - | - |
| 1 | 0.80 | 0.591 | 4.04e16 | 4.14e16 | -2.02e-4 | 2.57 | -1.97e-4 | 3.94 | 2.62 | 1.03 | 1.86 |
| 1 | 0.85 | 1.55 | 9.43e16 | 9.53e16 | -4.08e-4 | 5.96 | -3.97e-4 | 1.95 | 5.93 | 1.16 | 2.56 |
| 1 | 0.90 | 3.83 | 1.92e17 | 1.93e17 | -7.43e-4 | 12.1 | -7.23e-4 | 1.07 | 12.0 | 1.28 | 3.69 |
| 1 | 0.95 | 8.39 | 3.31e17 | 3.32e17 | -1.18e-3 | 20.9 | -1.14e-3 | 0.677 | 20.5 | 1.39 | 5.40 |
| 1 | 1.00 | 15.9 | 4.94e17 | 4.95e17 | -1.65e-3 | 31.1 | -1.61e-3 | 0.482 | 30.5 | 1.47 | 7.67 |
| 10 | 0.80 | 0.114 | 5.01e16 | 5.11e16 | -2.42e-4 | 3.18 | -2.34e-4 | 3.31 | 3.21 | 1.06 | 0.301 |
| 10 | 0.85 | 0.474 | 1.27e17 | 1.28e17 | -5.25e-4 | 8.05 | -5.10e-4 | 1.52 | 7.95 | 1.21 | 0.613 |
| 10 | 0.90 | 2.05 | 2.96e17 | 2.97e17 | -1.07e-3 | 18.6 | -1.04e-3 | 0.747 | 18.3 | 1.36 | 1.38 |
| 10 | 0.95 | 7.04 | 5.71e17 | 5.72e17 | -1.87e-3 | 36.0 | -1.82e-3 | 0.426 | 35.2 | 1.50 | 2.85 |
| 10 | 1.00 | 17.5 | 9.14e17 | 9.15e17 | -2.79e-3 | 57.6 | -2.72e-3 | 0.285 | 56.2 | 1.60 | 4.99 |

- The rib is in high injection: n and p differ by the background doping (1e15 cm^-3). From 0.80 to 0.85 V the rib
  carriers rise by x2.33 (1 ns), near the x2.63 = exp(0.05 V / 2 V_T) of a full quasi-Fermi split; from 0.95 to
  1.00 V only by x1.49. So the quasi-Fermi split in the rib grows more slowly than V at 1.0 V: more of the voltage
  drops outside the intrinsic zone.
- The loss over L_pi (1.0 to 1.6 dB) changes little with the bias: the index change and the loss both scale with
  the carrier density. A longer lifetime gives the same phase shift with less current (P_pi 1.38 mW against
  3.69 mW at 0.9 V).
- At 1.0 V and 10 ns the current is larger than at 1 ns. Breakdown (a scratch run on the GPU computer with the
  global balance of the test): at 1 ns, q sum R A is 6.58 A/m in the intrinsic zone and 2.99 A/m in the p+ and n+
  regions, and the minority-carrier currents into the contacts are 6.36 A/m (total 15.92 A/m); at 10 ns these are
  1.20, 0.67 and 15.61 A/m. So at 10 ns most of the current is carriers that cross the thin 1e18 cm^-3 end regions
  (0.95 um) into the ohmic contacts. Real devices have more doping there (and Auger recombination and band-gap
  narrowing, which this model does not have).
- Grid check (tau 1 ns; the pin-newton rows of `benchmarks/2026-10-30/charge-dd.jsonl`): 5 nm against 10 nm cells
  change the current by 0.05 % (0.8 and 0.9 V) and 0.09 % (1.0 V), and n rib by 0.06 %, 0.03 % and 0.02 %. So
  10 nm cells are converged to about 0.1 %. Coarser cells cannot place this geometry: the 90 nm slab and the
  250 nm half width need cells of 10 nm or a divisor of it (`rib_pin` raises ValueError otherwise; an earlier
  20 nm comparison had an 80 nm slab, so it measured a change of geometry, not a grid error).

## Drift-diffusion bench (`bench/bench_dd.py`)

One JSON line per case: the long-base diode (Newton, J / J_Shockley at 0.55, 0.65, 0.75 V), the PIN rib with
Newton (0 to 1.0 V), and one Gummel solve of the PIN rib at 0.6 V (the device part: the Poisson step).

```
bash tools/remote.sh sh <<'EOF'
bash tools/gpu_run.sh charge_gpu python bench/bench_dd.py                            # diode and 10 nm (run)
bash tools/gpu_run.sh charge_gpu python bench/bench_dd.py --dx 10e-9 5e-9 --no-diode # larger (run, below)
EOF
```

Default run on the GPU computer: diode 910 cells 2.2 s (J / J_Shockley 1.0099, 1.0015, 0.9908); the PIN rows at
10 nm are the same as in the table below (Newton 16.5 s, Gummel 2.6 s, the same currents).

Larger cases (`benchmarks/2026-10-30/charge-dd.jsonl`; the Gummel Poisson step on the RX 7900 XT, or on the CPU with
`CHARGE_GPU_DEVICE=cpu`; the Newton solve always on the CPU):

| Cells | Newton, 0 to 1.0 V (SuperLU, CPU) | Gummel at 0.6 V, GPU | Gummel at 0.6 V, CPU |
|---|---|---|---|
| 36 600 (10 nm) | 17.2 s | 2.6 s (14 iterations) | 3.9 s |
| 146 400 (5 nm) | 96.3 s | 3.3 s | 6.6 s |

- The Gummel current equals the Newton current to 9.5e-11 relative with the Poisson step on the GPU (10 and 5 nm),
  and to 3.3e-10 (10 nm) and 5.6e-10 (5 nm) with it on the CPU.
- With the Poisson step on the GPU, the whole Gummel solve (14 iterations: the CPU continuity steps and the JIT
  compile of the first call included) is 1.5x (10 nm) and 2.0x (5 nm) faster than with it on the CPU.
- The GPU path is not faster than Newton yet. One Newton continuation step on the CPU costs about 0.8 s at 10 nm
  (17.2 s for 21 steps) and 4.6 s at 5 nm (96.3 s); one Gummel solve costs 2.6 s and 3.3 s. At 0.6 V Gummel
  needs 14 iterations; at high injection it needs many more (linear convergence), so Newton stays the default.
- A device solver for the continuity equations (BiCGSTAB) is the next step.

## Known limits and next steps

- Multigrid cut the CG iterations per bias point from 11k-36k (Jacobi) to 226-560 on 73k cells. The charge solve
  of the modulator example (0, 1, 2, 4 V) took 2.1 s instead of 17.7 s on the Mac CPU (measured by the review of
  this change: `reviews/2026-10-30-b648ae6b7661.md`). With Jacobi CG: 7.4 s on the GPU against 19.5 s on 16 CPU
  threads (`benchmarks/2026-10-30/run-20261030-0626.jsonl`). With multigrid (the RC example, 12 solves per
  doping): 2.6 to 4.4 s per doping for 73k cells on the RX 7900 XT against 2.6 to 3.4 s on the Mac CPU, and 5.3 s
  against 8.6 s for 293k cells (`benchmarks/2026-10-30/charge-rc.jsonl`): no GPU gain at 73k cells.
- No 3-D, no temperature dependence.
- Drift-diffusion: the Newton method and the continuity steps of Gummel use a sparse direct LU on the CPU (v1);
  only the Gummel Poisson step runs on the GPU. Next: a device BiCGSTAB with an incomplete-LU or multigrid
  preconditioner for the nonsymmetric systems. Gummel needs many iterations at high injection; use Newton there.
- Drift-diffusion physics: Boltzmann statistics, SRH with constant lifetimes and a midgap trap, the Masetti
  mobility constant in the field (no velocity saturation, no carrier-carrier scattering), no Auger recombination,
  no band-gap narrowing, no surface recombination at the oxide, 300 K only, ideal ohmic contacts, steady state only
  (no transient, no small-signal AC), 2-D only.
- Current resolution: the face currents are differences of large drift and diffusion terms. In cells of 1 nm at
  1e18 cm^-3 these terms are up to about 1e8 A/m, and the measured floor of a face current is about 2.5e-10 A/m
  (2.5e-8 A/cm^2 for a 1 um high 1-D diode). The contact faces sit in large neutral cells and are much more
  accurate (the contact currents sum to 1e-14 A/m). Currents near the floor (low forward bias, reverse bias) are
  noise.
- `DDSolution.x_currents` counts the x faces only: lateral devices with contacts on xmin and xmax (else
  ValueError). The contact currents work for contacts on every side.
- The PIN example gives the rib means of n, p and the index change as material values; the modal values need
  `--modes` (meow-gpu).
- Capacitance: quasi-static and with constant quasi-Fermi levels, so no diffusion capacitance (forward bias) and
  no generation current. C' does not include the metal electrodes, the pads or the substrate under the oxide (the
  grid has 0.5 um of oxide and closed (zero-flux) outer sides).
- Series resistance: a 1-D path (each column equipotential in y; no current spreading between the slab and the
  rib), one lateral junction only (no PIN, vertical or L-shaped junctions), no contact or metal resistance, the net
  doping only (no compensation), the Masetti mobility at 300 K only.
- Bandwidth: a lumped RC only. A long device needs a traveling-wave model (electrode impedance, velocity match,
  RF loss).
