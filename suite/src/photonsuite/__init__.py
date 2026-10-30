"""photonsuite: one cross-section description for modes, heat and carriers on the GPU.

The import enables float64 in JAX (jax_enable_x64): the thermal and charge solvers cannot reach their tolerances in
float32, and the functions of photonsuite.physics refuse to run without it.
"""

import jax

jax.config.update("jax_enable_x64", True)

from photonsuite.geometry import CrossSection, Material, Polygon, Rect  # noqa: E402

__all__ = ["CrossSection", "Material", "Polygon", "Rect"]
