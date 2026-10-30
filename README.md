# Open-Source Simulator Dev

GPU-accelerated, open-source simulators for photonics, with AMD GPUs (ROCm) as the first target. Tested on a Radeon
RX 7900 XT (gfx1100) with ROCm 10.1 pip wheels and JAX 0.11.1.

| Folder | Physics | Method | Status |
|---|---|---|---|
| `meow_gpu/` | waveguide modes, EME (a backend for meow) | FDE (the tidy3d operator), block Davidson | 0.1: up to 11x one CPU core (warm start); EME taper 3.3x; mixed precision 1.66x faster by refinement; batched solves (option) |
| `bands_gpu/` | photonic band structures (the method of MPB) | plane waves + FFT, LOBPCG | 0.1: 11.7x the same code on 16 CPU threads (rod lattice, 128 x 128); oblique lattices; find_k |
| `thermal_gpu/` | heat conduction (heaters), 2-D and 3-D | finite volumes, multigrid CG | 0.1: 2-D 2.9 million cells in 0.21 s on the GPU; 3-D steady: 11 million cells, CG 1.0 s (33x faster than 16 CPU threads); 3-D transients; kappa(T) |
| `charge_gpu/` | PN junctions in reverse bias (modulators) | nonlinear Poisson, Newton + multigrid CG | 0.1: multigrid CG, 8x faster than Jacobi CG (Mac CPU); capacitance, series resistance, RC bandwidth; drift-diffusion (forward bias) |
| `optics_gpu/` | physical optics through lenses (fiber coupling) | angular spectrum + real rays (Optiland) | 0.1: the ray trace on the GPU, 7.3x one-thread NumPy at 1e7 rays (4.5x with the copy to the host) |
| `suite/` | one device description for all of the above | multiphysics workflows; 3-D FDTD (fdtdx) and EME adapters; rectangles and polygons | 0.1 |
| Meep on AMD | FDTD | ROCm port of a GPU add-on | probe passed; waits for a decision (STATUS.md) |

Examples (`examples/`): a thermo-optic phase shifter (P_pi), a depletion modulator (V_pi L), and chip-to-fiber
coupling through a lens pair (coupling efficiency and focus scan).

- `STATUS.md`: the tool in work, the queue, the decisions and the log.
- `benchmarks/`: the raw JSON rows of the timing runs, with an index of the claims that cite them.
- `tools/gpu_run.sh`: runs a test or a bench on the GPU computer behind a lock (one GPU process at a time). All
  tests run there, not on the development computer.
- `docs/deletion-requests.md`: files outside this repository that the agents want deleted (the user decides).
- `docs/landscape.md`: the open-source tools for each kind of simulation, and their GPU support.
- `docs/upstream-issues.md`: errors found in upstream packages (meow 0.15) and how this repository corrects them.
- `CLAUDE.md`: the rules for the agents that work in this repository (one adversarial review per commit, no
  private data in this public repository).

Each solver has tests against exact solutions, an independent solver, or the upstream tool. The upstream projects
keep their licenses.
