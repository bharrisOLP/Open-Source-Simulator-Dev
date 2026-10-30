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

Open (later, in this order): batched Davidson over several cells (small sections are overhead-bound);
mixed-precision eigenpair refinement (the FP64 polish is slow on RDNA3); bends and angled propagation.

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
- Next: GPU timings (running), oblique lattices, find_k for waveguide modes, a comparison with MPB itself.

**4. thermal-gpu: version 0.1 works.** Folder: `thermal_gpu/` (`thermal_gpu/README.md`).
- Cell-centered finite volumes (series face conductance), Jacobi CG as one device loop, steady and implicit
  transient solves.
- Checks: exact parabola, exact two-layer series resistance, sparse direct solve (1e-7 of the rise), slab decay
  time constant (1 %).
- Example `examples/thermo_optic_phase_shifter.py`: thermal-gpu + meow-gpu give P_pi = 19.0 mW for a synthetic SOI
  heater (published devices of this type: typically 20 to 30 mW).
- Next: multigrid preconditioner; 3-D; GPU timings.

**5. optics-gpu (the coupling use case of ZEMAX POP): version 0.1 works.** Folder: `optics_gpu/`.
- Angular spectrum of the source, real rays through the lens system (Optiland, MIT), image field as a separable
  ray sum on the GPU (PyTorch), coupling by overlap.
- Checks: 4f imaging and analytic Gaussian coupling (1e-3), power conservation (1 %), and an aberrated singlet
  against Optiland's independent Huygens PSF (correlation 0.998).
- Example `examples/chip_to_fiber_coupling.py`: SiN tip mode (meow-gpu) through a 2.5x lens pair into SMF-28:
  eta 0.889 at focus, with a focus scan (14 um mode window; an 8 um window cut the mode tails).
- Next: Optiland's own ray trace on the GPU (PyTorch backend) for large ray counts; ROCm timings.

**6. charge-gpu (reverse-biased PN junctions for depletion modulators): version 0.1 works.** Folder:
`charge_gpu/`.
- Nonlinear Poisson with constant quasi-Fermi levels, Newton with device-loop CG; plasma dispersion
  (Soref-Bennett).
- Checks: depletion width (Sze, 1 %), charge neutrality, an independent scipy Newton solve (1e-8 V); the
  built-in potential as a consistency check.
- Example `examples/pn_modulator.py`: V_pi L 1.39, 1.60, 1.96 V cm at 1, 2, 4 V for a synthetic SOI rib (published:
  typically 1.5 to 3 V cm); the loss decreases by 1.0 to 2.8 dB/cm (fewer free carriers).
- Not covered: forward bias and currents (drift-diffusion; DEVSIM stays the tool for that).
- Next: multigrid preconditioner (thermal and charge share it).

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
| 7 | Tidy3D-like suite | fdtdx (MIT, JAX, ROCm) + items 1, 3, 4, 6 | JAX on ROCm | very high | One API over FDTD, mode, EME, heat and charge |

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
