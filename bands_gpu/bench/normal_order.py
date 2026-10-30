"""Order of convergence of TE bands with the two normals of the averaging (CPU; about one minute).

    BANDS_GPU_DEVICE=cpu uv run python bench/normal_order.py
Square lattice of rods (r = 0.2 a, eps 12), TE, 4 bands at X, M and (0.2, 0.1). The reference is the rectangular cell
at 256 x 256 (object normal). For n = 32 to 192, the max error of the rectangular cell and of the sheared description
(a1 = (1, 0), a2 = (1, 1)), with the object normal (the default) and with the fitted gradient in all voxels (in a
cubic voxel, the fit has the direction of the first moment: the normal of the code before oblique cells). Prints one
JSON line per grid.
"""
import json

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np  # noqa: E402

from bands_gpu.geometry import Cell, Cylinder  # noqa: E402
from bands_gpu.solver import solve_bands  # noqa: E402


class FitCell(Cell):
    def eps_inverse(self, sub: int = 8, normal: str = "object") -> np.ndarray:
        return Cell.eps_inverse(self, sub, normal="fit")


rod = (Cylinder(radius=0.2, eps=12.0),)
ks = [[0.5, 0, 0], [0.5, 0.5, 0], [0.2, 0.1, 0]]
ref_cell = Cell(size=(1, 1, 1), shape=(256, 256, 1), objects=rod)
ref = solve_bands(ref_cell, ks, 4, "te", tol=1e-10).freqs
for n in (32, 64, 128, 192):
    row = {"n": n}
    for name, cls in (("object", Cell), ("fit", FitCell)):
        rect = cls(size=(1, 1, 1), shape=(n, n, 1), objects=rod)
        sheared = cls(size=(1, np.sqrt(2), 1), shape=(n, n, 1), objects=rod, basis=((1, 0, 0), (1, 1, 0), (0, 0, 1)))
        a = solve_bands(rect, ks, 4, "te", tol=1e-10)
        b = solve_bands(sheared, [sheared.k_fractional(ref_cell.k_cartesian(k)) for k in ks], 4, "te", tol=1e-10)
        row[f"rect_{name}"] = float(np.abs(a.freqs - ref).max())
        row[f"sheared_{name}"] = float(np.abs(b.freqs - ref).max())
        row[f"converged_{name}"] = bool(all(a.converged) and all(b.converged))
    print(json.dumps(row), flush=True)
