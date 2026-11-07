# Third-party notices

The code of this repository is MIT-licensed (`LICENSE`). Some files follow or adapt code of other open-source
projects. This file gives those files, the source, and the license and copyright notice of the source. The full
license texts are in `LICENSES/`.

## The mode solver of tidy3d (MIT)

Source: the folder `tidy3d/components/mode/` of tidy3d 2.12.2 (Flexcompute). That folder has its own MIT license
(`LICENSES/tidy3d-mode-MIT.txt`): "Copyright (c) 2021 Flexcompute", "Copyright (c) 2019 Tyler Hughes". Its notice
says that some of its functions are derived from the ceviche package (https://github.com/fancompute/ceviche). The
rest of tidy3d is LGPL-2.1-or-later; this repository uses it only as an installed library.

Files of this repository that follow that code:

| File | Functions | Source |
|---|---|---|
| `meow_gpu/src/meow_gpu/fde/grid.py` | `s_value`, `s_factors`, `pml_speeds`, `axis_stencils` (the PML stretch factors, the PML speeds and the derivative stencils) | `derivatives.py` (`s_value`, `create_sfactor_f`, `create_sfactor_b`, `average_relative_speed`, `make_dxf`, `make_dxb`, `make_dyf`, `make_dyb`) |
| `meow_gpu/src/meow_gpu/fde/operator.py` | `radial_transform` (the bend transform); `assemble` (the CPU reference matrix, the "P Q" form) | `transforms.py` (`radial_transform`); `solver.py` (`solver_diagonal`, the diagonal-media matrix) |

The device operator (`apply` in `operator.py`) and the eigensolvers (`eigensolver.py`) are new code that uses the
same discretization, so that both solvers have the same discrete spectrum.

## meow (Apache-2.0)

Source: meow 0.15.0 (`meow-sim`; author in the package metadata: Floris Laporte), licensed under the Apache License,
Version 2.0 (`LICENSES/Apache-2.0.txt`).

| File | Function | Source | Change |
|---|---|---|---|
| `meow_gpu/src/meow_gpu/meow_backend.py` | `post_process_modes` | `meow.fde.post_process` (`post_process_modes`, `orthonormalize_modes`) | each mode keeps its own neff in the Gram-Schmidt step (meow 0.15 averages them; `docs/upstream-issues.md`) |

## Not in this repository

The GPU add-on of the add-on route of FDTD (GPL-2.0-or-later) and the FDTD solver that it extends (GPL-2.0-or-later)
are installed by `tools/env/addon_gpu.sh` into `.local/` on the GPU computer. They are not part of this repository.
Patches of the add-on and code that is derived from it keep the GPL, in folders with their own license file
(`tools/env/addon_patches/`, `addon_gpu/`).
