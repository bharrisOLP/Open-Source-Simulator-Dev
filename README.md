# Open-Source Simulator Dev

GPU-accelerated, open-source versions of photonics simulators, with AMD GPUs (ROCm) as the first target.

| Target | Physics | Status |
|---|---|---|
| meow-gpu | EME with a GPU finite-difference mode solver | in work |
| Meep on AMD | FDTD | queued |
| MPB on GPU | band and mode solver (plane waves) | queued |
| thermal | heat conduction (FEM) | queued |
| ray and physical optics | ZEMAX-like | queued |
| charge | drift-diffusion | queued |
| suite | Tidy3D-like API over all solvers | queued |

- `STATUS.md`: the tool in work, the queue and the log.
- `docs/landscape.md`: the open-source tools for each kind of simulation, and their GPU support.
- `CLAUDE.md`: the rules for the agents that work in this repository.

Each solver has a CPU reference path and tests that compare the GPU results with the CPU results and with an
independent solver. The upstream projects keep their licenses.
