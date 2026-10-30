"""One fdtdx static object that gives each cell of the volume its own material (the rasterized geometry)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from fdtdx.core.jax.pytrees import autoinit, frozen_field
from fdtdx.materials import compute_ordered_names
from fdtdx.objects.static_material.static import StaticMultiMaterialObject


class _Frozen:
    """Holds the index array in a frozen field: identity hash, so the array stays out of the JAX trace."""

    def __init__(self, a):
        self.a = a


@autoinit
class VoxelMaterial(StaticMultiMaterialObject):
    """Material index per cell; materials are named m0, m1, ... in the order of the index."""

    voxels: _Frozen = frozen_field(default=None)

    @classmethod
    def from_index(cls, idx: np.ndarray, materials: dict, name: str = "geometry") -> "VoxelMaterial":
        return cls(name=name, materials=materials, voxels=_Frozen(np.asarray(idx, np.int32)),
                   partial_grid_shape=tuple(int(n) for n in idx.shape))

    def get_voxel_mask_for_shape(self):
        return jnp.ones(self.grid_shape, dtype=bool)

    def get_material_mapping(self):
        names = compute_ordered_names(self.materials)
        lut = np.array([names.index(f"m{k}") for k in range(len(self.materials))], np.int32)
        idx = self.voxels.a
        if tuple(idx.shape) != tuple(self.grid_shape):
            raise ValueError(f"voxel array {idx.shape} does not fit the placed object {self.grid_shape}")
        return jnp.asarray(lut[idx])
