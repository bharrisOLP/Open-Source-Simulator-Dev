# Status

Updated: 2026-10-30.

## In work

**1. meow-gpu: EME (meow) with a GPU mode solver on AMD (ROCm).** Folder: `meow_gpu/`.

Plan:

1. Read the meow 0.15 interface: `Structure` → `Cell` → `CrossSection` → `compute_modes` → `compute_s_matrix`.
2. Write an FDE mode solver in JAX: the same Yee-grid finite-difference operator as the tidy3d local mode solver,
   applied matrix-free on the GPU. Tests: CPU against GPU, against tidy3d on the same grid, against a slab
   waveguide with an analytic solution.
3. Eigen solver: a block eigensolver with an FFT preconditioner, warm-started from the modes of the previous EME
   cell (cells of a taper change slowly), batched over cells and wavelengths.
4. Overlaps and S-matrices on the GPU (batched).
5. A drop-in backend: `meow_gpu.compute_modes(cs, num_modes)` returns `meow.Mode` objects, so that the rest of
   meow (and SAX) works without a change.
6. Benchmark on the RX 7900 XT against the CPU path (one process and 8 processes). Report the result as it is,
   also if the GPU is not faster.

## Queue (in order of work)

The user asked to start with Meep and meow. After those two, the order is by complexity, the simplest first.

| # | Tool | Open-source base | GPU path on AMD | Complexity | Notes |
|---|---|---|---|---|---|
| 1 | meow (EME) | meow (Apache-2) + own FDE solver | JAX on ROCm | medium | **in work** |
| 2 | Meep (FDTD) | Meep (GPL-2); the GPU forks meep-gpu (Triton) and gpmeep (CUDA) | Triton on ROCm, or HIP | very high | Study the forks first; a Triton port can keep the Meep API |
| 3 | MPB (band and mode solver) | MPB (GPL-2) | plane-wave basis: FFT + block eigensolver in JAX | medium-high | Shares the eigensolver of item 1 |
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
