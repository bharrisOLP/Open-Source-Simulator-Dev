"""Geometry in the style of the MIT FDTD package: Vector3, Medium, Block, Cylinder, Sphere, Prism.

Units: um. The cell is centered at the origin. Objects later in the geometry list replace earlier ones. Each cell
takes the material of its center (a staircase; no subpixel averaging).
"""

from __future__ import annotations

import math

import numpy as np

inf = 1.0e20


class Vector3:
    """A 3-vector (x, y, z) with the arithmetic that device scripts use."""

    __slots__ = ("x", "y", "z")

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.x, self.y, self.z = float(x), float(y), float(z)

    def __iter__(self):
        return iter((self.x, self.y, self.z))

    def __getitem__(self, i):
        return (self.x, self.y, self.z)[i]

    def __add__(self, o):
        return Vector3(self.x + o[0], self.y + o[1], self.z + o[2])

    def __sub__(self, o):
        return Vector3(self.x - o[0], self.y - o[1], self.z - o[2])

    def __neg__(self):
        return Vector3(-self.x, -self.y, -self.z)

    def __mul__(self, s):
        return Vector3(self.x * s, self.y * s, self.z * s)

    __rmul__ = __mul__

    def __eq__(self, o):
        return isinstance(o, Vector3) and tuple(self) == tuple(o)

    def __hash__(self):
        return hash(tuple(self))

    def norm(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def __repr__(self):
        return f"Vector3({self.x:g}, {self.y:g}, {self.z:g})"


def _v3(v) -> Vector3:
    if isinstance(v, Vector3):
        return v
    v = tuple(v) + (0.0,) * (3 - len(tuple(v)))
    return Vector3(*v)


class Medium:
    """A lossless, non-dispersive, isotropic material: Medium(epsilon=...) or Medium(index=...)."""

    def __init__(self, epsilon: float | None = None, index: float | None = None, **unsupported):
        if unsupported:
            raise NotImplementedError(f"fdtd_gpu.Medium supports only epsilon and index, not {sorted(unsupported)}")
        if epsilon is not None and index is not None:
            raise ValueError("give epsilon or index, not both")
        eps = index**2 if index is not None else (1.0 if epsilon is None else epsilon)
        if not eps > 0:
            raise ValueError(f"epsilon must be positive, got {eps}")
        self.epsilon = float(eps)

    @property
    def index(self) -> float:
        return math.sqrt(self.epsilon)

    def __repr__(self):
        return f"Medium(epsilon={self.epsilon:g})"


vacuum = air = Medium()


def _axis_of(axis) -> int:
    """The coordinate axis (0, 1, 2) of a direction vector; only x, y and z directions are supported."""
    a = np.abs(np.asarray(tuple(_v3(axis)), float))
    k = int(np.argmax(a))
    if a[k] == 0 or np.count_nonzero(a) != 1:
        raise NotImplementedError(f"only axis-aligned objects are supported (axis {tuple(_v3(axis))})")
    return k


class GeometricObject:
    def __init__(self, material: Medium | None = None, center=Vector3()):
        self.material = material if material is not None else Medium()
        self.center = _v3(center)

    def contains(self, x, y, z):                      # pragma: no cover - abstract
        raise NotImplementedError

    def bounds(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:   # pragma: no cover
        raise NotImplementedError

    def shift(self, d):
        self.center = self.center + _v3(d)
        return self


class Block(GeometricObject):
    """An axis-aligned box: Block(size, center, material). Rotated blocks (e1, e2, e3) are not supported."""

    def __init__(self, size, center=Vector3(), material: Medium | None = None, e1=None, e2=None, e3=None):
        for e, ref in ((e1, (1, 0, 0)), (e2, (0, 1, 0)), (e3, (0, 0, 1))):
            if e is not None and tuple(_v3(e)) != ref:
                raise NotImplementedError("rotated blocks (e1, e2, e3) are not supported")
        super().__init__(material, center)
        self.size = _v3(size)

    def contains(self, x, y, z):
        c, s = self.center, self.size
        return ((np.abs(x - c.x) <= s.x / 2) & (np.abs(y - c.y) <= s.y / 2) & (np.abs(z - c.z) <= s.z / 2))

    def bounds(self):
        c, s = self.center, self.size
        return tuple(c[i] - s[i] / 2 for i in range(3)), tuple(c[i] + s[i] / 2 for i in range(3))


class Cylinder(GeometricObject):
    """A circular cylinder along x, y or z: Cylinder(radius, height=inf, axis=Vector3(0, 0, 1), center, material)."""

    def __init__(self, radius: float, height: float = inf, axis=Vector3(0, 0, 1), center=Vector3(),
                 material: Medium | None = None):
        super().__init__(material, center)
        self.radius, self.height, self.axis = float(radius), float(height), _axis_of(axis)

    def contains(self, x, y, z):
        p = (x - self.center.x, y - self.center.y, z - self.center.z)
        a = self.axis
        u, v = (p[i] for i in range(3) if i != a)
        return (u**2 + v**2 <= self.radius**2) & (np.abs(p[a]) <= self.height / 2)

    def bounds(self):
        half = [self.radius] * 3
        half[self.axis] = self.height / 2
        return tuple(self.center[i] - half[i] for i in range(3)), tuple(self.center[i] + half[i] for i in range(3))


class Sphere(GeometricObject):
    def __init__(self, radius: float, center=Vector3(), material: Medium | None = None):
        super().__init__(material, center)
        self.radius = float(radius)

    def contains(self, x, y, z):
        c = self.center
        return (x - c.x) ** 2 + (y - c.y) ** 2 + (z - c.z) ** 2 <= self.radius**2

    def bounds(self):
        c, r = self.center, self.radius
        return tuple(c[i] - r for i in range(3)), tuple(c[i] + r for i in range(3))


def _in_polygon(u, v, pu, pv):
    """Even-odd rule: True where (u, v) is inside the polygon with vertices (pu, pv)."""
    inside = np.zeros(np.broadcast(u, v).shape, bool)
    n = len(pu)
    for k in range(n):
        u1, v1, u2, v2 = pu[k], pv[k], pu[(k + 1) % n], pv[(k + 1) % n]
        if v1 == v2:
            continue
        crosses = (v1 > v) != (v2 > v)
        u_cross = u1 + (v - v1) * (u2 - u1) / (v2 - v1)
        inside ^= crosses & (u < u_cross)
    return inside


class Prism(GeometricObject):
    """A straight prism: Prism(vertices, height, axis=Vector3(0, 0, 1), center=None, material).

    The vertices lie in a plane normal to the axis (x, y or z; also -x, -y, -z); they give the bottom face, and the
    top face is the bottom face moved by height x axis. With center=None the bottom face is at the plane of the
    vertices. With a center, the prism moves so that the centroid of all its vertices (top and bottom: the mean of
    the vertices, moved by height / 2 along the axis) is at that point, as in the MIT package. Sloped sidewalls
    (sidewall_angle) are not supported.
    """

    def __init__(self, vertices, height: float, axis=Vector3(0, 0, 1), center=None, material: Medium | None = None,
                 sidewall_angle: float = 0.0):
        if sidewall_angle:
            raise NotImplementedError("sloped prism sidewalls are not supported")
        self.axis = _axis_of(axis)
        vs = [tuple(_v3(v)) for v in vertices]
        if len(vs) < 3:
            raise ValueError("a prism needs at least 3 vertices")
        self.height = float(height)
        a = self.axis
        base = vs[0][a]
        if any(abs(v[a] - base) > 1e-12 for v in vs):
            raise ValueError("all prism vertices must lie in one plane normal to the axis")
        t = [i for i in range(3) if i != a]
        self._pu = np.array([v[t[0]] for v in vs])
        self._pv = np.array([v[t[1]] for v in vs])
        sign = 1.0 if _v3(axis)[a] > 0 else -1.0
        c = [0.0] * 3
        c[t[0]] = float(self._pu.mean())                 # the centroid of the vertices
        c[t[1]] = float(self._pv.mean())
        c[a] = base + sign * self.height / 2
        super().__init__(material, Vector3(*c))
        if center is not None:
            d = _v3(center) - self.center
            self._pu = self._pu + d[t[0]]
            self._pv = self._pv + d[t[1]]
            self.center = _v3(center)

    def shift(self, d):
        d = _v3(d)
        t = [i for i in range(3) if i != self.axis]
        self._pu, self._pv = self._pu + d[t[0]], self._pv + d[t[1]]
        return super().shift(d)

    def contains(self, x, y, z):
        p = (x, y, z)
        a = self.axis
        t = [i for i in range(3) if i != a]
        return _in_polygon(p[t[0]], p[t[1]], self._pu, self._pv) & (np.abs(p[a] - self.center[a]) <= self.height / 2)

    def bounds(self):
        a = self.axis
        t = [i for i in range(3) if i != a]
        lo, hi = [0.0] * 3, [0.0] * 3
        lo[t[0]], hi[t[0]] = self._pu.min(), self._pu.max()
        lo[t[1]], hi[t[1]] = self._pv.min(), self._pv.max()
        lo[a], hi[a] = self.center[a] - self.height / 2, self.center[a] + self.height / 2
        return tuple(lo), tuple(hi)


def rasterize(geometry, default_material: Medium, shape, dx: float, trivial=(False, False, False)):
    """Material index of each cell, and the list of materials (index 0 = default_material).

    shape: cells per axis; the cell spans [-n dx / 2, n dx / 2] on each axis. Each cell takes the material of the
    last object that contains its center. On a trivial axis (a cell size of 0: a 2-D or 1-D simulation) the
    coordinate is 0. Returns (idx: int32 array of the shape, materials: list of Medium).
    """
    materials = [default_material]
    idx = np.zeros(shape, np.int32)
    centers = [np.zeros(n) if trivial[i] else (np.arange(n) + 0.5) * dx - n * dx / 2 for i, n in enumerate(shape)]
    for obj in geometry:
        lo, hi = obj.bounds()
        sl = []
        for i in range(3):
            c = centers[i]
            if trivial[i]:
                sl.append(slice(0, len(c)) if lo[i] <= 0.0 <= hi[i] else slice(0, 0))
                continue
            i0 = int(np.searchsorted(c, lo[i] - 1e-9 * dx, side="left"))
            i1 = int(np.searchsorted(c, hi[i] + 1e-9 * dx, side="right"))
            sl.append(slice(i0, i1))
        if any(s.stop <= s.start for s in sl):
            continue
        X, Y, Z = np.meshgrid(centers[0][sl[0]], centers[1][sl[1]], centers[2][sl[2]], indexing="ij")
        mask = np.asarray(obj.contains(X, Y, Z), bool)
        if not mask.any():
            continue
        k = next((j for j, m in enumerate(materials) if m is obj.material), None)
        if k is None:
            materials.append(obj.material)
            k = len(materials) - 1
        sub = idx[tuple(sl)]
        sub[mask] = k
        idx[tuple(sl)] = sub
    return idx, materials
