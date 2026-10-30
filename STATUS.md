# Status

Updated: 2026-10-30.

## In work

**1. meow-gpu (EME): version 0.1 works.** Folder: `meow_gpu/` (`meow_gpu/README.md` has the results).

Done:
- GPU FDE solver in JAX: the tidy3d mode operator (tested equal to machine precision), block Davidson with a
  fast-diagonalization preconditioner, warm start, single/mixed/double precision, drop-in `compute_modes_gpu`.
- Tests: operator, eigenvalues in each precision, warm start, meow backend against tidy3d, device placement.
- Results on the RX 7900 XT: one wide section (200k unknowns, 20 modes) 3.9x faster cold and 11.4x faster warm
  than tidy3d on one core; EME taper (30 cells) 3.3x faster than one core and 2.1x faster than 8 processes,
  |dS21| about 2e-5.

- Correction of meow's post-processing: each mode keeps its own neff (meow 0.15 averages them in Gram-Schmidt;
  this changed the EME of a taper from 0.955 to 0.991, see item 7).

- Batched solves (`solve_batch`, `compute_modes_gpu_batch`; an option): the same modes as the per-cell path, but
  on the RX 7900 XT only 1.2x in single precision and no gain in mixed precision for the 30-cell taper.

- Mixed-precision refinement (the default of "mixed"): complex128 residuals and Rayleigh-Ritz, complex64 GMRES on
  the correction equation; the 30-cell taper in 11.8 s instead of 19.5 s (10.8x fewer complex128 applies), the
  same 1e-10 tolerance; near-degenerate pairs keep their mode identity (tested against ARPACK).

Open (later, in this order): profile the batched path (the work per member is its larger cost); bends and angled
propagation.

**2. Meep on AMD: next.** Findings so far (git-ignored clones in `.local/upstream/`):
- The Erised fork (GPL-2.0-or-later) is a Python add-on: Meep builds the simulation, and the add-on time-steps it
  with an array module (NumPy or CuPy) plus Triton kernels and CUDA C kernels (cupy.RawKernel). Route to AMD: CuPy
  for ROCm (hipRTC compiles RawKernel source) and the AMD backend of Triton. About 250k lines of Python with tests.
- gpmeep: a fork of the Meep C++ core (1.35.0-beta) with about 19k lines of CUDA and C++. Route to AMD: HIPify.
- fdtdx (MIT, JAX): already runs on this card. A Meep-compatible front end on it is the third route.
- Probe result (2026-10-30, `tools/probe_cupy_triton.sh`, `tools/probe/`): YES. PyTorch 2.14.1 for ROCm 7.2 runs
  on gfx1100; Triton 3.8.0 (AMD) runs a kernel; CuPy 14.2 (cupy-rocm-7-0) compiles and runs a RawKernel, with
  versioned links to the HIP libraries of the PyTorch wheel and hipcc from the ROCm 10.1 pip SDK.
- **Blocked, waits for the user:** the next step clones the Erised add-on from GitHub and runs it on the GPU
  computer (`tools/env/fdtd_gpu.sh`). The auto-mode classifier blocked that ("Code from External"). See
  "Decisions for the user". Until then the work continues with item 3.

**3. bands-gpu (the method of MPB): version 0.1 works.** Folder: `bands_gpu/` (`bands_gpu/README.md`).
- Plane-wave operator with FFTs, anisotropic subpixel averaging, block LOBPCG (stabilized), warm start on the k
  path, TE/TM for 2-D cells.
- Checks: exact homogeneous and 1-D Bragg bands; the rod-lattice TM gap agrees with an independent
  finite-difference solver to 2e-4 (both are 0.7 % below the resolution-32 numbers of the MPB tutorial).
- GPU: 11.7x the same code on 16 CPU threads (rods 128 x 128, 16 k points: 5.2 s against 60.3 s;
  `benchmarks/2026-10-30/`).
- Oblique lattices (triangular, hexagonal, fcc) and the analytic interface normal of the objects (TE at second
  order from n = 64 to 128 also in sheared voxels; slower above 128, cause not known); triangular air holes
  128 x 128: 5.6 s on the GPU against 89 s on 16 CPU threads (the CPU run did not converge all k points; 13x per
  iteration).
- find_k (the k of each band at a fixed frequency, as MPB find-k): Newton with the Hellmann-Feynman group
  velocity, parity; a dielectric slab against the exact supercell (1.2e-5 to 4.4e-5 at n 1025; the error depends on the subpixel
  sample count), a photonic-crystal line-defect waveguide against an independent finite-difference solver.
- Next: a comparison with MPB itself.

**4. thermal-gpu: version 0.1 works.** Folder: `thermal_gpu/` (`thermal_gpu/README.md`).
- Cell-centered finite volumes (series face conductance), Jacobi CG as one device loop, steady and implicit
  transient solves.
- Checks: exact parabola, exact two-layer series resistance, sparse direct solve (1e-7 of the rise), slab decay
  time constant (1 %).
- Example `examples/thermo_optic_phase_shifter.py`: thermal-gpu + meow-gpu give P_pi = 19.0 mW for a synthetic SOI
  heater (published devices of this type: typically 20 to 30 mW).
- Multigrid preconditioner (shared, below).
- 3-D steady solver (`solver3d.py`, 3-D aggregation multigrid with semi-coarsening): a 100 um heater with its ends
  (11 M cells, 25 nm) in 1.03 s of CG on the GPU against 34 s on 16 CPU threads; mid-length rise equal to 2-D to
  1e-5 (7e-6 at 100 and 50 nm, 8e-6 at 25 nm), the ends at 0.53 of the 2-D rise, the phase integral 1.0001 of 2-D
  (100 and 50 nm).
- 3-D transients (implicit Euler, BDF2, Crank-Nicolson) and kappa(T) (Picard): the 10-90 % rise time of the 3-D
  heater is 17.94 us (equal to 2-D to 3e-5); kappa(T) of Si and SiO2 lowers the heater rise by 0.9 % at 100 W/m (Si alone: +1.3e-5). 3-D transient
  with 2.7 M cells: 20 s on the GPU against 349 s on 16 CPU threads.
- Next: kappa(T) in the transients; a 3-D heater with vias and pads.

**5. optics-gpu (the coupling use case of ZEMAX POP): version 0.1 works.** Folder: `optics_gpu/`.
- Angular spectrum of the source, real rays through the lens system (Optiland, MIT), image field as a separable
  ray sum on the GPU (PyTorch), coupling by overlap.
- Checks: 4f imaging and analytic Gaussian coupling (1e-3), power conservation (1 %), and an aberrated singlet
  against Optiland's independent Huygens PSF (correlation 0.998).
- Example `examples/chip_to_fiber_coupling.py`: SiN tip mode (meow-gpu) through a 2.5x lens pair into SMF-28:
  eta 0.889 at focus, with a focus scan (14 um mode window; an 8 um window cut the mode tails).
- Optiland's ray trace on the GPU (PyTorch backend, float64; `tools/env/optics_rocm.sh`): equal to the NumPy trace
  (difference 0); at 1e7 rays 7.3x faster than one-thread NumPy (4.5x with the copy to the host); not faster below about 1e5 rays.
- Next: the Jacobian and the weights on the GPU, so that a GPU trace does not go back to the host.

**6. charge-gpu (reverse-biased PN junctions for depletion modulators): version 0.1 works.** Folder:
`charge_gpu/`.
- Nonlinear Poisson with constant quasi-Fermi levels, Newton with device-loop CG; plasma dispersion
  (Soref-Bennett).
- Checks: depletion width (Sze, 1 %), charge neutrality, an independent scipy Newton solve (1e-8 V); the
  built-in potential as a consistency check.
- Example `examples/pn_modulator.py`: V_pi L 1.39, 1.60, 1.96 V cm at 1, 2, 4 V for a synthetic SOI rib (published:
  typically 1.5 to 3 V cm); the loss decreases by 1.0 to 2.8 dB/cm (fewer free carriers).
- Not covered: forward bias and currents (drift-diffusion; DEVSIM stays the tool for that).
- Junction capacitance, series resistance and RC bandwidth (`charge_gpu/src/charge_gpu/rc.py`, example
  `charge_gpu/examples/pn_modulator_rc.py`): C' against eps/W to 1e-4; at 3e17 cm^-3 and 2 V: 0.197 fF/um,
  39.4 Ohm mm, 20.4 GHz intrinsic. No GPU gain at 73k cells (charge part 1.6x at 293k cells, against the Mac CPU).
- Drift-diffusion (Scharfetter-Gummel, SRH; coupled Newton with SuperLU on the CPU, or Gummel with the Poisson step
  on the GPU): the long-base diode within 1 % of Shockley (0.60 to 0.75 V); a PIN phase shifter at 0.9 V with a carrier lifetime of 1 ns:
  L_pi 1.07 mm, P_pi 3.7 mW.
- Next: the continuity equations on the device (BiCGSTAB), so that forward bias runs on the GPU.

**Shared: aggregation multigrid** (`thermal_gpu/src/thermal_gpu/multigrid.py`), used by thermal-gpu (steady and
transient) and charge-gpu (Newton steps). Heater stack: 17 to 27 CG iterations from 50 to 5 nm cells (Jacobi: 809
to 3982 from 50 to 10 nm); cells with a large aspect ratio need more. Heater stack at 5 nm (2.9 million cells):
0.21 s on the GPU, 1.30 s on 16 CPU threads (`benchmarks/2026-10-30/thermal-mg.jsonl`). PN modulator: 226 to 560
CG iterations per bias point instead of 11k to 36k; 2.1 s instead of 17.7 s on the Mac CPU (review record).

**7. photonsuite (one device description for all solvers): version 0.1 works.** Folder: `suite/`.
- `CrossSection` of rectangles with multiphysics materials; `mode_index`, `temperature`, `carriers`,
  `phase_shifter_ppi`, `modulator_vpil`; reproduces both examples within 1e-3. Every solve is checked (a solve
  that does not converge raises `ConvergenceError`); the import enables float64.
- `Polygon` shapes. 3-D FDTD adapter on fdtdx (`suite/src/photonsuite/fdtd.py`) and EME adapter on meow +
  meow-gpu (`suite/src/photonsuite/eme.py`), on the same segment description. FDTD straight strip: T = 0.9999995
  in 10.7 s on the GPU.
- **Finding (resolved): EME against FDTD on a 10 um Si taper (0.5 -> 2.0 um).** meow EME gave T00 = 0.955 and the
  calibrated FDTD 0.997. Cause: meow 0.15 averages the neff of the higher modes in its Gram-Schmidt step
  (`docs/upstream-issues.md`). With the correction in meow-gpu: EME 0.9867 / 0.9900 / 0.9912 at 30 / 60 / 120
  cells; FDTD 0.9961 at 20 nm (0.997 to 0.998 at 25 nm; its flux transmission is 1.0003 to 1.002). Agreement
  within about 0.5 %. Also found: meow drops polygon prisms along z (corrected in the EME adapter).
- `Polygon` in the FDTD adapter (exact cells at the cell centers; staircase fdtdx polygons on grid edges; no
  sub-pixel smoothing). A taper with 80 degree sidewalls: EME 0.9878 (0.10 % below vertical walls), FDTD 0.9975 at
  20 nm (0.14 % above): the sign of this small change is below the accuracy of the comparison.
- FDTD adapter fixes: each shape fills exactly the cells of its centers (segments of 2.5 cells left oxide gaps:
  T_flux 0.44 at 40 nm; sloped polygons moved by up to half a cell), the run on the GPU (after an EME run in the
  same process it was 9 to 13 times slower), a whole number of cells per axis, and clear errors for segments shorter
  than the step. The FDTD rows were run again (`fdtd-v3.jsonl`; old rows in `fdtd-diagnostics.jsonl`).
- Next: layer stacks from a GDS cross-section.

## Queue (in order of work)

The user asked to start with Meep and meow. After those two, the order is by complexity, the simplest first.

| # | Tool | Open-source base | GPU path on AMD | Complexity | Notes |
|---|---|---|---|---|---|
| 1 | meow (EME) | meow (Apache-2) + own FDE solver | JAX on ROCm | medium | v0.1 done; improvements open |
| 2 | Meep (FDTD) | Meep (GPL-2); GPU forks: Erised add-on (Triton/CUDA), gpmeep (CUDA C++) | CuPy-ROCm + Triton-AMD, HIP, or fdtdx | very high | probe passed; **waits for the user** (third-party code) |
| 3 | MPB (band and mode solver) | MPB (GPL-2) | plane-wave basis: FFT + block eigensolver in JAX | medium-high | v0.1 done (own code: `bands_gpu/`) |
| 4 | Thermal | own finite-volume solver (femwell / scikit-fem as references) | CG on GPU (JAX) | medium | v0.1 done (`thermal_gpu/`) |
| 5 | ZEMAX (ray + physical optics) | Optiland (MIT, PyTorch) + own physical optics | PyTorch on ROCm | medium | v0.1 done (`optics_gpu/`) |
| 6 | Charge | own nonlinear Poisson (reverse bias); DEVSIM (Apache-2) for full drift-diffusion | CG on GPU (JAX) | high | v0.1 done for reverse bias (`charge_gpu/`) |
| 7 | Tidy3D-like suite | items 1, 3, 4, 6 + fdtdx | JAX on ROCm | very high | v0.1 for cross-sections; 3-D FDTD (fdtdx) and EME adapters (`suite/`) |

"EME" and "FDTD" in the request of the user are items 1 and 2.

## Decisions for the user

These do not block the work. The agent continues with the default in brackets.

- License of the new code in this public repository. [No LICENSE file yet. Upstream licenses stay with their code.]
- **Meep on AMD (item 2): may the agent clone the third-party GitHub repository Erised-AI-Inc/meep-gpu
  (GPL-2.0-or-later) at commit 8993426, install it and run it on the GPU computer?** The auto-mode classifier
  blocked this. To approve: say so in the session, or allow `bash tools/remote.sh setup-env` with
  `tools/env/fdtd_gpu.sh`. [Waiting. Item 3 continues meanwhile. Other routes: HIPify the gpmeep C++ fork, or a
  front end on fdtdx (PyPI).]

## Log

- 2026-10-30: repository started. Review gate, remote tool, landscape study (`docs/landscape.md`).
- 2026-10-30: meow-gpu 0.1 (GPU FDE solver for meow) with tests and benchmarks.
- 2026-10-30: process lessons. One adversarial review per commit (then a fix list, no second review). No
  permission dialogs (the hook denies instead).
- 2026-10-30: bands-gpu, thermal-gpu 0.1; then optics-gpu, charge-gpu 0.1 and the three examples.
- 2026-10-30: shared multigrid (thermal, charge), photonsuite 0.1, benchmark logs (`benchmarks/2026-10-30/`).
- 2026-10-30: FDTD (fdtdx) and EME adapters in photonsuite; two meow 0.15 errors found and corrected
  (`docs/upstream-issues.md`).
- 2026-10-30: Polygon in the FDTD adapter and exact cell placement. Then four agents in parallel (user request:
  "fan out some agents"), each on one package, each change with its own single review: charge RC bandwidth,
  thermal 3-D, bands oblique lattices, meow batched solves.
- 2026-10-30: second wave of five agents: optics GPU ray trace, drift-diffusion, thermal transients and kappa(T),
  find_k and the mixed-precision refinement of meow-gpu, each with its single review (one review found a
  blocker: the refinement could return the wrong mode of a near-degenerate pair; fixed and tested). New rule
  (user): every test and bench runs on the Ubuntu GPU computer, never on the Mac (`tools/gpu_run.sh`, a GPU lock).
