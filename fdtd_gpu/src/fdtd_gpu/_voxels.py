"""One fdtdx static object that gives each cell of the volume its own material (the rasterized geometry), or its
own diagonal permittivity (one value per E component: the subpixel average)."""

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
    """Material index per cell; materials are named m0, m1, ... in the order of the index.

    With `from_permittivity` the object holds one material, eps_bg + (1, 1, 2) (diagonal, so fdtdx allocates three
    permittivity components), and a mask of shape (3, *cells): fdtdx sets each component to
    eps_bg + mask (material - eps_bg) (`fdtdx.fdtd.initialization._init_arrays`, the linear blend of a static
    multi-material object over the background eps_bg), so mask = (eps - eps_bg) / (1, 1, 2) gives eps.
    """

    voxels: _Frozen = frozen_field(default=None)
    blend: _Frozen = frozen_field(default=None)

    @classmethod
    def from_index(cls, idx: np.ndarray, materials: dict, name: str = "geometry") -> "VoxelMaterial":
        return cls(name=name, materials=materials, voxels=_Frozen(np.asarray(idx, np.int32)),
                   partial_grid_shape=tuple(int(n) for n in idx.shape))

    @classmethod
    def from_permittivity(cls, eps3: np.ndarray, eps_bg: float, name: str = "geometry") -> "VoxelMaterial":
        """eps3: (3, nx, ny, nz) relative permittivity of E_x, E_y, E_z of each cell; eps_bg: the permittivity of
        the SimulationVolume (the background that fdtdx blends from)."""
        from fdtdx import Material

        step = np.array([1.0, 1.0, 2.0])
        mat = {"m0": Material(permittivity=tuple(float(eps_bg + s) for s in step))}
        mask = (np.asarray(eps3, np.float64) - eps_bg) / step[:, None, None, None]
        shape = tuple(int(n) for n in eps3.shape[1:])
        return cls(name=name, materials=mat, voxels=_Frozen(np.zeros(shape, np.int32)),
                   blend=_Frozen(mask.astype(np.float32)), partial_grid_shape=shape)

    def get_voxel_mask_for_shape(self):
        if self.blend is not None:
            return jnp.asarray(self.blend.a)
        return jnp.ones(self.grid_shape, dtype=bool)

    def get_material_mapping(self):
        names = compute_ordered_names(self.materials)
        lut = np.array([names.index(f"m{k}") for k in range(len(self.materials))], np.int32)
        idx = self.voxels.a
        if tuple(idx.shape) != tuple(self.grid_shape):
            raise ValueError(f"voxel array {idx.shape} does not fit the placed object {self.grid_shape}")
        return jnp.asarray(lut[idx])
