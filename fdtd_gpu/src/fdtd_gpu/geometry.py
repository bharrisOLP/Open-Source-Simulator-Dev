"""Geometry in the style of the MIT FDTD package: Vector3, Medium, Block, Cylinder, Sphere, Prism.

Units: um. The cell is centered at the origin. Objects later in the geometry list replace earlier ones.
`rasterize` gives each cell the material of its center (a staircase); `Simulation(eps_averaging=True)` then
averages the permittivity at each Yee E component (`fdtd_gpu._averaging`). The `contains` methods take NumPy or
JAX arrays.
"""

from __future__ import annotations

import itertools
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


class _Susceptibility:
    """A susceptibility term with the names and units of the MIT package: frequency f_n and gamma in 1/um (c = 1,
    omega = 2 pi f), sigma dimensionless. Isotropic only: sigma, or sigma_diag with three equal values."""

    def __init__(self, frequency: float = 0.0, gamma: float = 0.0, sigma: float | None = None, sigma_diag=None,
                 sigma_offdiag=None, **unsupported):
        if unsupported:
            raise NotImplementedError(f"{type(self).__name__}: unsupported arguments {sorted(unsupported)}")
        if sigma_offdiag is not None and any(float(s) != 0 for s in _v3(sigma_offdiag)):
            raise NotImplementedError("anisotropic susceptibilities (sigma_offdiag) are not supported")
        if sigma_diag is not None:
            d = tuple(float(s) for s in _v3(sigma_diag))
            if sigma is not None or d[0] != d[1] or d[1] != d[2]:
                raise NotImplementedError("give sigma, or sigma_diag with three equal values (isotropic only)")
            sigma = d[0]
        self.frequency, self.gamma = float(frequency), float(gamma)
        self.sigma = 0.0 if sigma is None else float(sigma)
        if not self.frequency > 0:
            raise ValueError(f"{type(self).__name__}: frequency must be positive, got {self.frequency}")
        if self.gamma < 0:
            raise ValueError(f"{type(self).__name__}: gamma must not be negative, got {self.gamma}")
        if self.sigma < 0:
            raise NotImplementedError(f"{type(self).__name__}: a negative sigma (gain) is not supported")

    def __repr__(self):
        return f"{type(self).__name__}(frequency={self.frequency:g}, gamma={self.gamma:g}, sigma={self.sigma:g})"


class LorentzianSusceptibility(_Susceptibility):
    """A Lorentz oscillator: chi(f) = sigma f_n^2 / (f_n^2 - f^2 - i f gamma) (time dependence exp(-i omega t));
    in the time domain d^2P/dt^2 + 2 pi gamma dP/dt + (2 pi f_n)^2 P = sigma (2 pi f_n)^2 E."""

    def chi(self, f):
        f = np.asarray(f, complex)
        fn = self.frequency
        return self.sigma * fn**2 / (fn**2 - f**2 - 1j * f * self.gamma)


class DrudeSusceptibility(_Susceptibility):
    """A Drude term (free carriers): chi(f) = -sigma f_n^2 / (f^2 + i f gamma); the plasma frequency is
    f_n sqrt(sigma). In the time domain d^2P/dt^2 + 2 pi gamma dP/dt = sigma (2 pi f_n)^2 E."""

    def chi(self, f):
        f = np.asarray(f, complex)
        return -self.sigma * self.frequency**2 / (f**2 + 1j * f * self.gamma)


class Medium:
    """An isotropic material: Medium(epsilon=...) or Medium(index=...), with optional dispersion and loss, in the
    names and units of the MIT package:
    - E_susceptibilities: a list of LorentzianSusceptibility and DrudeSusceptibility terms; epsilon (or index^2)
      is then the high-frequency permittivity eps_inf;
    - D_conductivity: sigma_D in 1/(um/c), the conductivity of the MIT package: eps(f) = eps_inf (1 + i sigma_D /
      (2 pi f)); the fields in the medium decay as exp(-sigma_D t). Not together with E_susceptibilities.
    `epsilon` stays the float eps_inf (the MIT package has the method epsilon(freq); here it is permittivity(f))."""

    def __init__(self, epsilon: float | None = None, index: float | None = None, E_susceptibilities=(),
                 D_conductivity: float = 0.0, **unsupported):
        if unsupported:
            raise NotImplementedError(f"fdtd_gpu.Medium supports epsilon, index, E_susceptibilities and "
                                      f"D_conductivity, not {sorted(unsupported)}")
        if epsilon is not None and index is not None:
            raise ValueError("give epsilon or index, not both")
        eps = index**2 if index is not None else (1.0 if epsilon is None else epsilon)
        if not eps > 0:
            raise ValueError(f"epsilon must be positive, got {eps}")
        self.epsilon = float(eps)
        self.E_susceptibilities = tuple(E_susceptibilities or ())
        for s in self.E_susceptibilities:
            if not isinstance(s, (LorentzianSusceptibility, DrudeSusceptibility)):
                raise NotImplementedError(f"unsupported susceptibility {type(s).__name__} (only "
                                          f"LorentzianSusceptibility and DrudeSusceptibility)")
        self.D_conductivity = float(D_conductivity)
        if self.D_conductivity < 0:
            raise NotImplementedError("a negative D_conductivity (gain) is not supported")
        if self.D_conductivity > 0 and self.dispersive:
            raise NotImplementedError("D_conductivity together with E_susceptibilities is not supported: the MIT "
                                      "package multiplies the whole eps(f) by (1 + i sigma_D / omega), fdtdx adds "
                                      "i sigma / omega. Use a DrudeSusceptibility for the loss")

    @property
    def index(self) -> float:
        """sqrt(eps_inf)."""
        return math.sqrt(self.epsilon)

    @property
    def dispersive(self) -> bool:
        """True if a susceptibility term has sigma > 0."""
        return any(s.sigma > 0 for s in self.E_susceptibilities)

    @property
    def conductive(self) -> bool:
        return self.D_conductivity > 0

    def permittivity(self, f):
        """The complex relative permittivity at the frequencies f (1/um): (1 + i sigma_D / (2 pi f)) (eps_inf +
        sum chi(f)), time dependence exp(-i omega t) (Im > 0: loss)."""
        f = np.asarray(f, complex)
        eps = self.epsilon + sum((s.chi(f) for s in self.E_susceptibilities), np.zeros_like(f))
        return eps * (1 + 1j * self.D_conductivity / (2 * np.pi * f)) if self.conductive else eps

    def __repr__(self):
        extra = "".join([f", E_susceptibilities={list(self.E_susceptibilities)}" if self.E_susceptibilities else "",
                         f", D_conductivity={self.D_conductivity:g}" if self.conductive else ""])
        return f"Medium(epsilon={self.epsilon:g}{extra})"


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
        return (abs(x - c.x) <= s.x / 2) & (abs(y - c.y) <= s.y / 2) & (abs(z - c.z) <= s.z / 2)

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
        return (u**2 + v**2 <= self.radius**2) & (abs(p[a]) <= self.height / 2)

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
    """Even-odd rule: True where (u, v) is inside the polygon with vertices (pu, pv). NumPy or JAX arrays (no
    in-place operations), or floats."""
    inside = False
    n = len(pu)
    for k in range(n):
        u1, v1, u2, v2 = float(pu[k]), float(pv[k]), float(pu[(k + 1) % n]), float(pv[(k + 1) % n])
        if v1 == v2:
            continue
        crosses = (v1 > v) != (v2 > v)
        u_cross = u1 + (v - v1) * (u2 - u1) / (v2 - v1)
        inside = inside ^ (crosses & (u < u_cross))
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
        return _in_polygon(p[t[0]], p[t[1]], self._pu, self._pv) & (abs(p[a] - self.center[a]) <= self.height / 2)

    def bounds(self):
        a = self.axis
        t = [i for i in range(3) if i != a]
        lo, hi = [0.0] * 3, [0.0] * 3
        lo[t[0]], hi[t[0]] = self._pu.min(), self._pu.max()
        lo[t[1]], hi[t[1]] = self._pv.min(), self._pv.max()
        lo[a], hi[a] = self.center[a] - self.height / 2, self.center[a] + self.height / 2
        return tuple(lo), tuple(hi)


def periodic_shifts(obj, periodic, shape, dx: float) -> list[tuple[float, float, float]]:
    """Shifts (um) of the copies of obj that touch the cell [-n dx / 2, n dx / 2] (as in the MIT package, an object
    repeats across periodic faces). On an axis that is not periodic the shift is 0. An object that is infinite along
    a periodic axis (size or height inf) has one copy there. Empty if no copy touches the cell."""
    olo, ohi = obj.bounds()
    per_axis = []
    for a in range(3):
        if not periodic[a] or ohi[a] - olo[a] >= 0.5 * inf:
            per_axis.append([0.0])
            continue
        length = shape[a] * dx
        k0 = math.ceil((-length / 2 - ohi[a]) / length - 1e-9)
        k1 = math.floor((length / 2 - olo[a]) / length + 1e-9)
        if k1 < k0:
            return []
        per_axis.append([k * length for k in range(k0, k1 + 1)])
    return list(itertools.product(*per_axis))


def rasterize(geometry, default_material: Medium, shape, dx: float, trivial=(False, False, False),
              periodic=(False, False, False)):
    """Material index of each cell, and the list of materials (index 0 = default_material).

    shape: cells per axis; the cell spans [-n dx / 2, n dx / 2] on each axis. Each cell takes the material of the
    last object that contains its center. On a trivial axis (a cell size of 0: a 2-D or 1-D simulation) the
    coordinate is 0. On a periodic axis the objects repeat with the period of the cell (`periodic_shifts`).
    Returns (idx: int32 array of the shape, materials: list of Medium).
    """
    materials = [default_material]
    idx = np.zeros(shape, np.int32)
    centers = [np.zeros(n) if trivial[i] else (np.arange(n) + 0.5) * dx - n * dx / 2 for i, n in enumerate(shape)]
    per = tuple(bool(p) and not t for p, t in zip(periodic, trivial))
    for obj in geometry:
        parts = []                           # (cell slices, contains) of each copy that holds a cell center
        for s in periodic_shifts(obj, per, shape, dx):
            lo, hi = obj.bounds()
            lo, hi = [lo[i] + s[i] for i in range(3)], [hi[i] + s[i] for i in range(3)]
            sl = []
            for i in range(3):
                c = centers[i]
                if trivial[i]:
                    sl.append(slice(0, len(c)) if lo[i] <= 0.0 <= hi[i] else slice(0, 0))
                    continue
                i0 = int(np.searchsorted(c, lo[i] - 1e-9 * dx, side="left"))
                i1 = int(np.searchsorted(c, hi[i] + 1e-9 * dx, side="right"))
                sl.append(slice(i0, i1))
            if any(q.stop <= q.start for q in sl):
                continue
            X, Y, Z = np.meshgrid(centers[0][sl[0]] - s[0], centers[1][sl[1]] - s[1], centers[2][sl[2]] - s[2],
                                  indexing="ij")
            parts.append((sl, np.asarray(obj.contains(X, Y, Z), bool)))
        if not parts:
            continue
        # the mask covers only the bounding box of the copies (not the whole grid: many objects on a large grid)
        box = tuple(slice(min(p[0][i].start for p in parts), max(p[0][i].stop for p in parts)) for i in range(3))
        mask = np.zeros(tuple(b.stop - b.start for b in box), bool)
        for sl, m in parts:
            mask[tuple(slice(sl[i].start - box[i].start, sl[i].stop - box[i].start) for i in range(3))] |= m
        if not mask.any():
            continue
        k = next((j for j, m in enumerate(materials) if m is obj.material), None)
        if k is None:
            materials.append(obj.material)
            k = len(materials) - 1
        sub = idx[box]
        sub[mask] = k
    return idx, materials
