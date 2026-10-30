"""Unit cells and the inverse permittivity on the grid, with anisotropic subpixel averaging.

The cell is rectangular (lattice vectors along x, y, z with lengths a = (ax, ay, az); a 2-D cell has az and nz
unused, a 1-D cell also ay). Objects are filled in order; a later object replaces an earlier one where they overlap.

Subpixel averaging (the method of MPB and Meep; Kottke, Farjadpour and Johnson 2008): in each voxel, with the
supersampled permittivity eps_s, the inverse-permittivity tensor is

    eps_inv = <1/eps> n n^T + (1 / <eps>) (I - n n^T),

where n is the unit normal of the interface in the voxel (here: the direction of the first moment of eps_s about
the voxel center). An E field normal to the interface sees the harmonic mean, a parallel field the arithmetic mean.
Voxels without an interface get the scalar 1/eps.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Cylinder:
    """Infinite along z (in a 2-D cell) or with height ``height`` centered at ``center`` (3-D)."""

    radius: float
    eps: float
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    height: float = np.inf

    def inside(self, x, y, z):
        cx, cy, cz = self.center
        return ((x - cx) ** 2 + (y - cy) ** 2 <= self.radius**2) & (np.abs(z - cz) <= self.height / 2)


@dataclass(frozen=True)
class Sphere:
    radius: float
    eps: float
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def inside(self, x, y, z):
        cx, cy, cz = self.center
        return (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= self.radius**2


@dataclass(frozen=True)
class Block:
    size: tuple[float, float, float]
    eps: float
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def inside(self, x, y, z):
        return ((np.abs(x - self.center[0]) <= self.size[0] / 2) & (np.abs(y - self.center[1]) <= self.size[1] / 2)
                & (np.abs(z - self.center[2]) <= self.size[2] / 2))


@dataclass(frozen=True)
class Cell:
    """A rectangular unit cell. ``shape`` = grid points per axis (1 for an unused axis); ``size`` = lattice
    constants. Coordinates run from -size/2 to size/2 (the cell center is the origin)."""

    size: tuple[float, float, float]
    shape: tuple[int, int, int]
    eps_background: float = 1.0
    objects: tuple = field(default_factory=tuple)

    @property
    def dims(self) -> int:
        return sum(n > 1 for n in self.shape)

    def axes(self, sub: int = 1) -> list[np.ndarray]:
        """Sample coordinates per axis: ``sub`` points per voxel, centered in each voxel (grid points are the voxel
        centers; voxel i spans [x_i - dx/2, x_i + dx/2], with x_0 = -size/2)."""
        out = []
        for a, n in zip(self.size, self.shape):
            if n == 1:
                out.append(np.zeros(1))
                continue
            dx = a / n
            centers = -a / 2 + dx * np.arange(n)
            offsets = (np.arange(sub) + 0.5) / sub * dx - dx / 2
            out.append((centers[:, None] + offsets[None, :]).ravel())
        return out

    def eps_samples(self, sub: int) -> np.ndarray:
        """eps at ``sub`` samples per voxel and axis, shape (nx*s, ny*s, nz*s) (s = 1 on unused axes)."""
        xs, ys, zs = self.axes(sub)
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
        eps = np.full(X.shape, float(self.eps_background))
        for obj in self.objects:
            # periodic images: test the object at the 3^d nearest cell translations
            shifts = [(0.0,) if n == 1 else (-a, 0.0, a) for a, n in zip(self.size, self.shape)]
            hit = np.zeros(X.shape, bool)
            for sx in shifts[0]:
                for sy in shifts[1]:
                    for sz in shifts[2]:
                        hit |= obj.inside(X + sx, Y + sy, Z + sz)
            eps = np.where(hit, obj.eps, eps)
        return eps

    def eps_inverse(self, sub: int = 8) -> np.ndarray:
        """The averaged inverse permittivity tensor, shape (3, 3, nx, ny, nz)."""
        s = [sub if n > 1 else 1 for n in self.shape]
        e = self.eps_samples(sub)
        nx, ny, nz = self.shape
        e = e.reshape(nx, s[0], ny, s[1], nz, s[2])
        mean_eps = e.mean(axis=(1, 3, 5))
        mean_inv = (1.0 / e).mean(axis=(1, 3, 5))
        # first moment of eps about the voxel center (the direction to the high-eps side)
        offs = [(np.arange(k) + 0.5) / k - 0.5 for k in s]
        mom = np.stack([
            (e * offs[0][None, :, None, None, None, None]).mean(axis=(1, 3, 5)),
            (e * offs[1][None, None, None, :, None, None]).mean(axis=(1, 3, 5)),
            (e * offs[2][None, None, None, None, None, :]).mean(axis=(1, 3, 5)),
        ])
        for i, (a, n) in enumerate(zip(self.size, self.shape)):     # moments in physical length units
            mom[i] *= a / n if n > 1 else 0.0
        norm = np.sqrt((mom**2).sum(axis=0))
        interface = (mean_inv * mean_eps > 1 + 1e-12) & (norm > 0)
        nvec = np.where(interface[None], mom / np.where(norm > 0, norm, 1.0)[None], 0.0)
        eye = np.eye(3)[:, :, None, None, None]
        nn = nvec[:, None] * nvec[None, :]
        tensor = mean_inv[None, None] * nn + (1.0 / mean_eps)[None, None] * (eye - nn)
        iso = (1.0 / mean_eps)[None, None] * eye
        return np.where(interface[None, None], tensor, iso)
