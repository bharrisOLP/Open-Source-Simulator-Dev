# Open-source photonics simulators and their GPU support

Study date: 2026-10-30. Key: **V** = verified at the URL on that date; **M** = from memory, not verified.
Versions and dates come from the GitHub tag pages or PyPI.

## Target hardware

AMD Radeon RX 7900 XT (gfx1100, RDNA3, 20 GB). FP64 runs at 1/32 of the FP32 rate (V, techpowerup.com). Thus:

- Memory-bound kernels (FDTD stencils, sparse matrix-vector products, FFTs) lose only the factor of the extra bytes
  in FP64, about 2x.
- Compute-bound FP64 work (dense LU, dense eigen solves, sparse direct solvers) is slow on this card.
- ROCm 10.0 (2026-08-26) lists gfx1100 on Ubuntu 26.04, with PyTorch 2.11 to 2.13 and JAX 0.10 to 0.11 (V,
  rocm.docs.amd.com compatibility matrix). JAX 0.11.1 with the ROCm 10.1.0 plugin from pip wheels works on this
  card (tested in this repository, `tools/setup_env.sh`).

## FDTD

| Tool | License | GPU | AMD | Last release | URL |
|---|---|---|---|---|---|
| Meep | GPL-2 | none upstream; a CUDA-only design note (2026-08) | no | v1.35.0, 2026-10-09 (V) | github.com/NanoComp/meep |
| meep-gpu (Erised) | GPL-2+ | Triton (CUDA) and Metal; FP32, one GPU, no MPI; 60 of 65 Meep examples | not yet (Triton has a ROCm backend, M) | v1.0.0, 2026 (V) | github.com/Erised-AI-Inc/meep-gpu |
| gpmeep | GPL-2+ | CUDA + MPI, multi-GPU; PML, dispersion, adjoint, DFT, eigenmode sources | no | v1.0.3, 2026 (V) | github.com/Semiconductor-Nanophotonics-Laboratory/gpmeep-releases |
| fdtdx | MIT | JAX: CUDA, ROCm, TPU; differentiable | yes (V) | v0.6.3, 2026-10-12 (V) | github.com/ymahlau/fdtdx |
| Khronos.jl | MIT | KernelAbstractions: CUDA, ROCm, oneAPI, Metal | yes | archived 2026-05-10 (V) | github.com/facebookresearch/Khronos.jl |
| gprMax v4 | GPL-3+ | OpenMP, CUDA, OpenCL, Metal, MPI | OpenCL (M) | v4.0.1, 2026-10-03 (V) | github.com/gprMax/gprMax |
| openEMS | GPL-3 | CPU SIMD only | no | v0.37.0, 2026-10-09 (V) | github.com/thliebig/openEMS-Project |
| fdtd (flaport) | MIT | PyTorch | through PyTorch ROCm (M) | (V) | github.com/flaport/fdtd |
| fdtd-z | MIT | JAX + CUDA; adiabatic absorbers along x and y; no dispersion | no | (V) | github.com/spinsphotonics/fdtdz |
| TorchFDTD | MIT | CuPy CUDA; FP32/FP64, Drude/Lorentz, mode ports, adjoint | no | v1.1.7 (V) | github.com/hyoseokp/TorchFDTD |
| ceviche | MIT | CPU, autograd | no | (V) | github.com/fancompute/ceviche |
| Tidy3D client | LGPL-2+ | the FDTD solver runs only in the vendor cloud | – | 2.12.2, 2026-10-28 (V) | pypi.org/project/tidy3d |

## Mode solvers, EME, RCWA

| Tool | License | GPU | Last release | URL |
|---|---|---|---|---|
| meow | Apache-2 | none; uses the tidy3d local mode solver; S-matrix with SAX | 0.15.0, 2026-03-15 (V) | github.com/gdsfactory/meow |
| tidy3d local mode solver | LGPL | CPU (numpy/scipy) | (V) | docs.flexcompute.com |
| MPB | GPL-2 | none (MPI) | v1.12.0, 2025-04-03 (V) | github.com/NanoComp/mpb |
| femwell | GPL-3 | none; modes, thermal, electrostatics, PN depletion example | v0.1.12, 2025-01-06 (V) | github.com/HelgeGehring/femwell |
| EMEPy | MIT | none | 1.2.3 (V) | github.com/BYUCamachoLab/emepy |
| EMpy | MIT | none | (V) | github.com/lbolla/EMpy |
| torcwa | LGPL | PyTorch CUDA | 0.1.4.2 (V) | github.com/kch3782/torcwa |
| S4 | GPL-2 | none | (V) | github.com/phoebe-p/S4 |
| inkstone | AGPL-3 | none | 0.3 (V) | github.com/alexysong/inkstone |

## Charge (drift-diffusion, TCAD)

| Tool | License | GPU | Status | URL |
|---|---|---|---|---|
| DEVSIM | Apache-2 | none; MKL Pardiso or UMFPACK; custom direct-solver callback | v2.11.2rc5, 2026-10-24 (V) | github.com/devsim/devsim |
| Charon (Sandia) | GPL-3 | not stated (Trilinos, MPI) | (V) | charon.sandia.gov |
| ChargeTransport.jl | AGPL-3 | none | (V) | github.com/WIAS-PDELib/ChargeTransport.jl |
| Genius-TCAD-Open | GPL | none | activity unknown | github.com/cogenda/Genius-TCAD-Open |

## Thermal

| Tool | License | GPU | URL |
|---|---|---|---|
| scikit-fem (femwell thermal) | BSD-3 (M) | none | github.com/kinnala/scikit-fem |
| Elmer | GPL (M) | none found | github.com/ElmerCSC/elmerfem |
| FEniCSx (LGPL) + PETSc (BSD-2) | (M) | PETSc Kokkos; HIP matrices "in development" (V) | petsc.org |
| MOOSE | LGPL (M) | Kokkos heat conduction (V) | mooseframework.inl.gov |
| MFEM | BSD-3 | CUDA, HIP, OCCA, RAJA, libCEED (V) | github.com/mfem/mfem |
| jax-fem | GPL-3 | JAX | github.com/deepmodeling/jax-fem |
| torch-fem | MIT | PyTorch | github.com/meyer-nils/torch-fem |

## Ray and physical optics (ZEMAX-like)

| Tool | License | GPU | URL |
|---|---|---|---|
| Optiland | MIT | PyTorch backend; sequential and non-sequential; reads .zmx; POP only on the roadmap | github.com/HarrisonKramer/optiland |
| rayoptics | BSD-3 | none; reads Zemax and CODE V files | github.com/mjhoptics/ray-optics |
| prysm | MIT | numpy or CuPy backend | github.com/brandondube/prysm |
| dLux | BSD-3 | JAX | github.com/LouisDesdoigts/dLux |
| POPPY | BSD (M) | CuPy CUDA; OpenCL FFTs only | poppy-optics.readthedocs.io |

## Multiphysics, RF, circuits

- gplugins 2.1.5 (gdsfactory): plugins for femwell, elmer, palace, meow, tidy3d, devsim, sax, meep, mpb (V).
- Palace (Apache-2, v0.18.1, 2026-09-21): FEM electromagnetics with CUDA and HIP since v0.13.0 (V). A candidate for
  modulator RF electrodes on AMD.
- SAX 0.18.2 (Apache-2, JAX): circuit S-parameters (V).

## Not verified

- ROCm behaviour of Optiland, torcwa, flaport/fdtd, jax-fem and torch-fem on gfx1100. (fdtdx 0.6.3 with JAX 0.11.1
  runs on gfx1100 in this repository: a 3.3 M-cell straight guide in 10.7 s; with 9943 time steps (computed from the simulated time and the Courant
  step), about
  3e9 cell updates per second; `suite/bench/fdtd_straight.py`, `benchmarks/2026-10-30/fdtd.jsonl`.)
- Complex non-symmetric `eig` on the GPU under PyTorch or JAX on ROCm.
- GPU support in Charon, Elmer, LightPipes and diffractio.
