# Open-Source Simulator Dev

GPU-accelerated, open-source simulators for photonics, with AMD GPUs (ROCm) as the first target. Tested on a Radeon
RX 7900 XT (gfx1100) with ROCm 10.1 pip wheels and JAX 0.11.1.

| Folder | Physics | Method | Status |
|---|---|---|---|
| `meow_gpu/` | waveguide modes, EME (a backend for meow) | FDE (the tidy3d operator), block Davidson | 0.1: up to 11x one CPU core (warm start); EME taper 3.3x; mixed precision 1.66x faster by refinement; batched solves (option); bend modes |
| `bands_gpu/` | photonic band structures (the method of MPB) | plane waves + FFT, LOBPCG | 0.1: 11.7x the same code on 16 CPU threads (rod lattice, 128 x 128); oblique lattices; find_k |
| `thermal_gpu/` | heat conduction (heaters), 2-D and 3-D | finite volumes, multigrid CG | 0.1: 2-D 2.9 million cells in 0.21 s on the GPU; 3-D steady: 11 million cells, CG 1.0 s (33x faster than 16 CPU threads); 3-D transients; kappa(T) |
| `charge_gpu/` | PN junctions in reverse and forward bias (modulators, PIN phase shifters) | nonlinear Poisson, Newton + multigrid CG; drift-diffusion (Scharfetter-Gummel) | 0.1: multigrid CG, 8x faster than Jacobi CG (Mac CPU); capacitance, series resistance, RC bandwidth; drift-diffusion (forward bias): the coupled Newton step on the GPU (5.8x to 5.9x faster than SuperLU at 5 nm), Gummel continuity solves on the GPU |
| `optics_gpu/` | physical optics through lenses (fiber coupling) | angular spectrum + real rays (Optiland) | 0.1: the whole pipeline on the GPU (trace, weights, image sum; focus and lateral scans); all focus planes in one pass (3.1x per plane; about 27x in complex64) |
| `suite/` | one device description for all of the above | multiphysics workflows; 3-D FDTD (fdtdx) and EME adapters; rectangles and polygons | 0.1 |
| `fdtd_gpu/` | FDTD (1-D, 2-D, 3-D) with a script interface in the style of Meep's Python interface (no Meep code) | fdtdx (JAX) | 0.1: Bragg, waveguide, ring, dipole and sheet checks |
| Meep on AMD (`addon_gpu/`, GPL) | FDTD | ROCm port of a GPU add-on (GPL-2.0-or-later) | approved and installed on the GPU computer; the port is in work (STATUS.md) |

Examples (`examples/`): a thermo-optic phase shifter (P_pi), a depletion modulator (V_pi L), and chip-to-fiber
coupling through a lens pair (coupling efficiency and focus scan; two runs: the mode solve in `.venv-rocm`, then the
optics in `.venv-optics`, as its docstring shows).

- `STATUS.md`: the tool in work, the queue, the decisions and the log.
- `benchmarks/`: the raw JSON rows of the timing runs, with an index of the claims that cite them.
- `tools/gpu_run.sh`: runs a test or a bench on the GPU computer behind a lock (one GPU process at a time). All
  tests run there, not on the development computer.
- `docs/deletion-requests.md`: files that the agents want deleted but may not delete (outside this repository, and
  review records); the user decides.
- `docs/landscape.md`: the open-source tools for each kind of simulation, and their GPU support.
- `docs/upstream-issues.md`: errors found in upstream packages (meow 0.15) and how this repository corrects them.
- `CLAUDE.md`: the rules for the agents that work in this repository (one adversarial review per commit, no
  private data in this public repository).

Each solver has tests against exact solutions, an independent solver, or the upstream tool.

## License

The code of this repository is MIT-licensed (`LICENSE`), unless a file or folder says otherwise. Some files follow
or adapt code of other open-source projects; `THIRD_PARTY_NOTICES.md` gives them with the copyright notices, and
`LICENSES/` holds the license texts. The upstream projects keep their licenses. They are dependencies, installed
from their own sources, and not part of this repository:

| Package | License | Used by |
|---|---|---|
| meow | Apache-2.0 | `meow_gpu/` (backend), `suite/` |
| tidy3d | LGPL-2.1-or-later (its mode-solver folder: MIT) | the CPU reference and the fallback of `meow_gpu/`; the mode solver of fdtdx |
| fdtdx | MIT | `fdtd_gpu/`, `suite/` |
| Optiland | MIT | `optics_gpu/` |
| JAX, PyTorch | Apache-2.0, BSD-3-Clause | the GPU code |
| CuPy, Triton | MIT | the add-on route (`.local/envs/addon-gpu`) |
| Meep (pymeep 1.35.0, conda-forge) | GPL-2.0-or-later | the add-on route: `tools/env/addon_gpu.sh` installs it into `.local/envs/addon-gpu` |
| the GPU add-on of the add-on route | GPL-2.0-or-later | `tools/env/addon_gpu.sh` clones it into `.local/` at install time |

Notes:
- `meow_gpu/src/meow_gpu/fde/grid.py` (the PML and the stencils) and two CPU functions of `operator.py`
  (`radial_transform`, `assemble`) follow the mode solver of tidy3d closely. That folder of tidy3d has its own MIT
  license (Copyright (c) 2021 Flexcompute, Copyright (c) 2019 Tyler Hughes; parts derived from ceviche). The files
  keep that notice. The device operator and the eigensolvers are new code with the same discretization.
- `post_process_modes` in `meow_gpu/src/meow_gpu/meow_backend.py` adapts the post-processing of meow (Apache-2.0).
- `fdtd_gpu/` follows the published Python interface of Meep (GPL-2.0-or-later). It uses no code of Meep.
- A patch of a GPL project, and code derived from it, keeps the GPL (the patches of the add-on in
  `tools/env/addon_patches/`, and `addon_gpu/`). Such a folder has its own license file.
