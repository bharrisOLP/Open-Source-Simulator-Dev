"""Unit cells and the inverse permittivity on the grid, with anisotropic subpixel averaging.

The cell has three lattice vectors a_i = size[i] * basis[i] / |basis[i]| (the rows of ``Cell.lattice``). Without a
basis, the lattice vectors are along x, y, z (a rectangular cell). A 2-D cell has nz = 1 (axis 3 unused), a 1-D cell
also ny = 1. The lattice vector of an unused axis must be orthogonal to the lattice vectors of the used axes: the
structure is the cut through the origin, extruded along that vector. Objects are in Cartesian coordinates. They are
filled in order; a later object replaces an earlier one where they overlap.

The grid is uniform in the lattice coordinates s_i (fractions of a_i): grid point j_i is at s_i = -1/2 + j_i / n_i,
and the voxel around it is the parallelepiped spanned by the edge vectors d_i = a_i / n_i. The Cartesian point is
r = sum_i s_i a_i. A rectangular cell is the special case of a diagonal lattice (the same q and samples, bit for bit).

Subpixel averaging (the method of MPB and Meep; Kottke, Farjadpour and Johnson 2008): in each voxel, with the
supersampled permittivity eps_s, the inverse-permittivity tensor is

    eps_inv = <1/eps> n n^T + (1 / <eps>) (I - n n^T),

where n is the unit normal of the interface in the voxel. An E field normal to the interface sees the harmonic mean,
a parallel field the arithmetic mean. Voxels without an interface get the scalar 1/eps.

The normal (as MPB): where exactly one object boundary (one periodic image of one object) crosses the voxel, n is the
analytic normal of that object at the voxel center (``normal`` of the shape: radial for a sphere or a cylinder side,
the nearest face for a block or a cylinder cap). Where several boundaries cross the voxel, n is the gradient of eps
that fits the first moments m_i = <eps o_i> (o_i: the offsets in voxel fractions along d_i): for a linear eps,
m_i = (grad eps . d_i) / 12, so grad eps = 12 sum_i m_i n_i b_i (b_i: the reciprocal vectors, b_i . a_j = delta_ij).
The fit is exact only for a linear eps: at a curved step it has a bias in a sheared voxel (TE of a sheared square
lattice, n = 64 to 128: the error fell 2.5x with the fit alone, 4.8x with the analytic normal; tests/test_oblique.py).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

# The triangular (hexagonal) lattice: a1 = (1, 0), a2 = (1/2, sqrt(3)/2), 60 degrees apart (a3 = z for 3-D cells).
TRIANGULAR = ((1.0, 0.0, 0.0), (0.5, math.sqrt(3) / 2, 0.0), (0.0, 0.0, 1.0))


def _small(v, c: int) -> bool:
    """True if the component c of the vector v is zero to rounding (relative to |v|)."""
    return abs(v[c]) <= 1e-12 * float(np.linalg.norm(v))


def _extent(half: tuple, b: np.ndarray) -> float:
    """max |b . v| over the box |v_c| <= half_c (an infinite half-size counts only where b_c is not zero)."""
    return float(sum(h * abs(b[c]) for c, h in enumerate(half) if not _small(b, c)))


@dataclass(frozen=True)
class Cylinder:
    """Axis along z. Infinite (in a 2-D cell) or with height ``height`` centered at ``center`` (3-D)."""

    radius: float
    eps: float
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    height: float = np.inf

    def inside(self, x, y, z):
        cx, cy, cz = self.center
        return ((x - cx) ** 2 + (y - cy) ** 2 <= self.radius**2) & (np.abs(z - cz) <= self.height / 2)

    def extent(self, b: np.ndarray) -> float:
        """max |b . (r - center)| over the points r of the object."""
        return self.radius * math.hypot(b[0], b[1]) + _extent((0.0, 0.0, self.height / 2), b)

    def normal(self, x, y, z):
        """The outward normal (not normalized) of the nearest surface: the side (radial) or a cap (z)."""
        px, py, pz = x - self.center[0], y - self.center[1], z - self.center[2]
        side = np.abs(np.hypot(px, py) - self.radius)
        cap = np.abs(np.abs(pz) - self.height / 2)
        on_side = side <= cap
        return np.where(on_side, px, 0.0), np.where(on_side, py, 0.0), np.where(on_side, 0.0, np.sign(pz))

    def invariant(self, v) -> bool:
        """True if the object does not change along the vector v."""
        return math.isinf(self.height) and _small(v, 0) and _small(v, 1)


@dataclass(frozen=True)
class Sphere:
    radius: float
    eps: float
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def inside(self, x, y, z):
        cx, cy, cz = self.center
        return (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= self.radius**2

    def extent(self, b: np.ndarray) -> float:
        return self.radius * float(np.linalg.norm(b))

    def normal(self, x, y, z):
        return x - self.center[0], y - self.center[1], z - self.center[2]

    def invariant(self, v) -> bool:
        return False


@dataclass(frozen=True)
class Block:
    """A box with its edges along x, y, z (Cartesian, also in an oblique cell)."""

    size: tuple[float, float, float]
    eps: float
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def inside(self, x, y, z):
        return ((np.abs(x - self.center[0]) <= self.size[0] / 2) & (np.abs(y - self.center[1]) <= self.size[1] / 2)
                & (np.abs(z - self.center[2]) <= self.size[2] / 2))

    def extent(self, b: np.ndarray) -> float:
        return _extent(tuple(s / 2 for s in self.size), b)

    def normal(self, x, y, z):
        """The outward normal of the nearest face plane (an infinite size has no face)."""
        p = [x - self.center[0], y - self.center[1], z - self.center[2]]
        d = np.stack(np.broadcast_arrays(*[np.abs(np.abs(pc) - sc / 2) for pc, sc in zip(p, self.size)]))
        face = np.argmin(d, axis=0)
        return tuple(np.where(face == c, np.sign(p[c]), 0.0) for c in range(3))

    def invariant(self, v) -> bool:
        return all(math.isinf(sc) or _small(v, c) for c, sc in enumerate(self.size))


@dataclass(frozen=True)
class Cell:
    """A unit cell. ``shape`` = grid points per lattice vector (1 for an unused axis); ``size`` = the lengths of the
    lattice vectors; ``basis`` = their directions (rows; normalized here, as in MPB), None for x, y, z. Lattice
    coordinates run from -1/2 to 1/2 (the cell center is the origin). Example, the triangular lattice:
    ``Cell(size=(1, 1, 1), shape=(32, 32, 1), basis=TRIANGULAR, objects=...)``."""

    size: tuple[float, float, float]
    shape: tuple[int, int, int]
    eps_background: float = 1.0
    objects: tuple = field(default_factory=tuple)
    basis: tuple | None = None

    def __post_init__(self):
        if self.basis is None:
            return
        e = self.unit_basis
        if e.shape != (3, 3) or not np.all(np.isfinite(e)) or abs(np.linalg.det(e)) < 1e-9:
            raise ValueError("basis: three linearly independent 3-vectors")
        used = [i for i, n in enumerate(self.shape) if n > 1]
        for i in (i for i, n in enumerate(self.shape) if n == 1):
            if any(abs(e[i] @ e[j]) > 1e-12 for j in used):
                raise ValueError(f"basis: the lattice vector of the unused axis {i} must be orthogonal to the "
                                 "lattice vectors of the used axes")

    @property
    def dims(self) -> int:
        return sum(n > 1 for n in self.shape)

    @property
    def unit_basis(self) -> np.ndarray:
        """The unit lattice directions e_i (rows, 3 x 3)."""
        if self.basis is None:
            return np.eye(3)
        e = np.asarray(self.basis, float)
        return e / np.linalg.norm(e, axis=1, keepdims=True)

    @property
    def unit_reciprocal(self) -> np.ndarray:
        """The rows f_i with e_i . f_j = delta_ij (the reciprocal vectors of the unit directions)."""
        if self.basis is None:
            return np.eye(3)
        return np.linalg.inv(self.unit_basis).T

    @property
    def lattice(self) -> np.ndarray:
        """The lattice vectors a_i (rows, Cartesian, units of a)."""
        return np.asarray(self.size, float)[:, None] * self.unit_basis

    @property
    def reciprocal(self) -> np.ndarray:
        """The reciprocal lattice vectors b_i (rows, units of 1/a, without 2 pi): a_i . b_j = delta_ij."""
        return self.unit_reciprocal / np.asarray(self.size, float)[:, None]

    def k_cartesian(self, k_frac) -> np.ndarray:
        """k in reciprocal-lattice coordinates (as MPB) to Cartesian k in units of 2 pi / a: sum_i k_i b_i."""
        return np.asarray(k_frac, float) @ self.reciprocal

    def k_fractional(self, k_cart) -> np.ndarray:
        """Cartesian k (units of 2 pi / a) to reciprocal-lattice coordinates: k_i = k . a_i."""
        return self.lattice @ np.asarray(k_cart, float)

    def axes(self, sub: int = 1) -> list[np.ndarray]:
        """Sample positions along each lattice vector, as lengths (s_i * size[i]): ``sub`` points per voxel, centered
        in each voxel (grid points are the voxel centers; voxel j spans [t_j - dt/2, t_j + dt/2], t_0 = -size/2)."""
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

    def points(self, sub: int = 1) -> list[np.ndarray]:
        """Cartesian coordinates [X, Y, Z] of the samples, r = sum_i t_i e_i, as arrays that broadcast to the shape
        (nx*s, ny*s, nz*s). Each coordinate uses only the axes that change it (in a rectangular cell, X has the shape
        (nx*s, 1, 1)), so that no full 3-D array is made that is not necessary."""
        t = np.meshgrid(*self.axes(sub), indexing="ij", sparse=True)
        e = self.unit_basis
        return [sum(t[i] * e[i, c] for i in range(3) if e[i, c] != 0.0) for c in range(3)]

    def image_shifts(self, obj) -> list[np.ndarray]:
        """The lattice translations T = sum_i n_i a_i (Cartesian) at which images of ``obj`` can touch the samples.
        Per used axis: all n_i with |b_i . (r + T - center)| <= extent for some sample r (|b_i . r| <= 1/2 + 1/(2 n_i)).
        An object that is infinite along b_i must not change along a_i (then n_i in {-1, 0, 1} is exact); else a
        ValueError (the images along a_i would not be complete)."""
        lat, rec = self.lattice, self.reciprocal
        center = np.asarray(obj.center, float)
        ranges = []
        for i, n in enumerate(self.shape):
            if n == 1:
                ranges.append((0,))
                continue
            ext = obj.extent(rec[i])
            if not math.isfinite(ext):
                if not obj.invariant(lat[i]):
                    raise ValueError(f"{type(obj).__name__} is infinite along a direction that is not a lattice "
                                     f"vector: it changes along a{i + 1}, so its images along a{i + 1} are not "
                                     "complete; give it a finite size")
                ranges.append((-1, 0, 1))
                continue
            c = float(rec[i] @ center)
            reach = 0.5 + 0.5 / n
            ranges.append(tuple(range(math.floor(c - reach - ext), math.ceil(c + reach + ext) + 1)))
        return [n0 * lat[0] + n1 * lat[1] + n2 * lat[2] for n0, n1, n2 in itertools.product(*ranges)]

    def eps_samples(self, sub: int) -> np.ndarray:
        """eps at ``sub`` samples per voxel and axis, shape (nx*s, ny*s, nz*s) (s = 1 on unused axes)."""
        return self._sample(sub, track=False)[0]

    def _sample(self, sub: int, track: bool):
        """eps at the samples. With ``track``, also per voxel: the number of object boundaries (periodic images of
        objects) that cross the voxel (the samples of the voxel are partly inside), and the analytic normal of the
        last of them at the voxel center (shape (3, nx, ny, nz))."""
        X, Y, Z = self.points(sub)
        shape = np.broadcast_shapes(X.shape, Y.shape, Z.shape)
        eps = np.full(shape, float(self.eps_background))
        s = [sub if n > 1 else 1 for n in self.shape]
        vox = (self.shape[0], s[0], self.shape[1], s[1], self.shape[2], s[2])
        count = np.zeros(self.shape, int) if track else None
        normal = np.zeros((3, *self.shape)) if track else None
        centers = self.points(1) if track else None
        for obj in self.objects:
            hit = np.zeros(shape, bool)
            for t in self.image_shifts(obj):            # periodic images across all cell faces (also oblique)
                h = np.broadcast_to(obj.inside(X + t[0], Y + t[1], Z + t[2]), shape)
                hit |= h
                if track:
                    hv = h.reshape(vox)
                    cross = hv.any(axis=(1, 3, 5)) & ~hv.all(axis=(1, 3, 5))
                    if cross.any():
                        count += cross
                        nrm = obj.normal(centers[0] + t[0], centers[1] + t[1], centers[2] + t[2])
                        nrm = np.stack([np.broadcast_to(c, self.shape) for c in nrm])
                        normal = np.where(cross[None], nrm, normal)
            eps = np.where(hit, obj.eps, eps)
        return eps, count, normal

    def eps_inverse(self, sub: int = 8, normal: str = "object") -> np.ndarray:
        """The averaged inverse permittivity tensor (Cartesian components), shape (3, 3, nx, ny, nz). ``normal``:
        "object" (the analytic normal where one object boundary crosses the voxel, else the fit) or "fit" (the fitted
        gradient in all voxels; for comparisons)."""
        if normal not in ("object", "fit"):
            raise ValueError("normal: 'object' or 'fit'")
        s = [sub if n > 1 else 1 for n in self.shape]
        e, count, analytic = self._sample(sub, track=True)
        nx, ny, nz = self.shape
        e = e.reshape(nx, s[0], ny, s[1], nz, s[2])
        mean_eps = e.mean(axis=(1, 3, 5))
        mean_inv = (1.0 / e).mean(axis=(1, 3, 5))
        # first moments of eps about the voxel center, in voxel fractions along each lattice vector
        offs = [(np.arange(k) + 0.5) / k - 0.5 for k in s]
        mom = np.stack([
            (e * offs[0][None, :, None, None, None, None]).mean(axis=(1, 3, 5)),
            (e * offs[1][None, None, None, :, None, None]).mean(axis=(1, 3, 5)),
            (e * offs[2][None, None, None, None, None, :]).mean(axis=(1, 3, 5)),
        ])
        # the gradient of eps (Cartesian, to the high-eps side): sum_i m_i n_i b_i (see the module docstring)
        w = [n / a if n > 1 else 0.0 for a, n in zip(self.size, self.shape)]
        f = self.unit_reciprocal
        grad = np.stack([mom[0] * (w[0] * f[0, c]) + mom[1] * (w[1] * f[1, c]) + mom[2] * (w[2] * f[2, c])
                         for c in range(3)])
        # the analytic normal where exactly one object boundary crosses the voxel; else the fitted gradient
        one = (count == 1) & ((analytic**2).sum(axis=0) > 0) & (normal == "object")
        grad = np.where(one[None], analytic, grad)
        norm = np.sqrt((grad**2).sum(axis=0))
        interface = (mean_inv * mean_eps > 1 + 1e-12) & (norm > 0)
        nvec = np.where(interface[None], grad / np.where(norm > 0, norm, 1.0)[None], 0.0)
        eye = np.eye(3)[:, :, None, None, None]
        nn = nvec[:, None] * nvec[None, :]
        tensor = mean_inv[None, None] * nn + (1.0 / mean_eps)[None, None] * (eye - nn)
        iso = (1.0 / mean_eps)[None, None] * eye
        return np.where(interface[None, None], tensor, iso)
