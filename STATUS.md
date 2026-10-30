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
- First step: a probe on the GPU computer. Do CuPy for ROCm and Triton for AMD work on gfx1100 with the ROCm pip
  wheels?

## Queue (in order of work)

The user asked to start with Meep and meow. After those two, the order is by complexity, the simplest first.

| # | Tool | Open-source base | GPU path on AMD | Complexity | Notes |
|---|---|---|---|---|---|
| 1 | meow (EME) | meow (Apache-2) + own FDE solver | JAX on ROCm | medium | v0.1 done; improvements open |
| 2 | Meep (FDTD) | Meep (GPL-2); GPU forks: Erised add-on (Triton/CUDA), gpmeep (CUDA C++) | CuPy-ROCm + Triton-AMD, HIP, or fdtdx | very high | **next** |
| 3 | MPB (band and mode solver) | MPB (GPL-2) | plane-wave basis: FFT + block eigensolver in JAX | medium-high | Reuses the eigensolver of item 1 |
| 4 | Thermal | femwell / scikit-fem; torch-fem; jax-fem | FEM with CG + AMG/FFT preconditioner on GPU | medium | Heater cross-sections, 3D transients |
| 5 | ZEMAX (ray + physical optics) | Optiland (MIT, PyTorch); prysm; dLux (JAX) | PyTorch / JAX on ROCm | medium | No open tool replaces Zemax POP; POP is FFT work |
| 6 | Charge (drift-diffusion) | DEVSIM (Apache-2) | GPU sparse solver through the DEVSIM solver callback | high | FP64 and sparse LU: a poor fit for an RDNA3 card |
| 7 | Tidy3D-like suite | fdtdx (MIT, JAX, ROCm) + items 1, 3, 4, 6 | JAX on ROCm | very high | One API over FDTD, mode, EME, heat and charge |

"EME" and "FDTD" in the request of the user are items 1 and 2.

## Decisions for the user

These do not block the work. The agent continues with the default in brackets.

- License of the new code in this public repository. [No LICENSE file yet. Upstream licenses stay with their code.]

## Log

- 2026-10-30: repository started. Review gate, remote tool, landscape study (`docs/landscape.md`).
- 2026-10-30: meow-gpu 0.1 (GPU FDE solver for meow) with tests and benchmarks.
- 2026-10-30: process lessons. One adversarial review per commit (then a fix list, no second review). No
  permission dialogs (the hook denies instead).
