"""A device cross-section that each solver samples on its own grid.

Coordinates are in micrometers (x horizontal, y vertical; propagation along z). Shapes are filled in order: a later
shape replaces an earlier one where they overlap. Each material carries the properties of every physics:

    optics: n (at the design wavelength), dn_dT (1/K)        heat: kappa (W/m/K), rho_cp (J/m^3/K)
    charge: eps_r, semiconductor (bool)                       a shape adds doping: n_net = N_D - N_A (cm^-3)
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

EDGE_TOL = 1e-9      # um: a sample within this distance of a rectangle edge counts as on the edge


@dataclass(frozen=True)
class Material:
    name: str
    n: float
    dn_dT: float = 0.0
    kappa: float = 1.0
    rho_cp: float = 1.6e6
    eps_r: float = 1.0
    semiconductor: bool = False


@dataclass(frozen=True)
class Rect:
    """An axis-aligned rectangle (um) of one material, with an optional net doping (cm^-3).

    It holds the half-open region [x0, x1) x [y0, y1), with the edges moved by -EDGE_TOL: a sample on a left or bottom
    edge is inside, a sample on a right or top edge is outside, whatever the rounding of the grid (np.arange gives
    0.22 + 6e-16 for a grid point at 0.22). Mode solvers sample some field components on cell edges, so these
    components see the shape shifted by up to half a cell. There is no sub-pixel averaging yet."""

    x0: float
    x1: float
    y0: float
    y1: float
    material: Material
    doping: float = 0.0
    tag: str = ""

    def contains(self, X, Y):
        return ((X >= self.x0 - EDGE_TOL) & (X < self.x1 - EDGE_TOL)
                & (Y >= self.y0 - EDGE_TOL) & (Y < self.y1 - EDGE_TOL))


@dataclass(frozen=True)
class Polygon:
    """A polygon (vertices (x, y) in um, in order) of one material, for example a rib with sloped sidewalls.

    The same edge rule as Rect: a sample is tested at (X + EDGE_TOL, Y + EDGE_TOL), strictly inside. So a sample on
    a horizontal or vertical edge is inside on the left or bottom side and outside on the right or top side, whatever
    the rounding of the grid; an axis-aligned polygon samples exactly like the Rect of the same corners."""

    vertices: tuple
    material: Material
    doping: float = 0.0
    tag: str = ""

    def __post_init__(self):
        import shapely
        verts = tuple((float(x), float(y)) for x, y in self.vertices)      # hashable, also from a list or an array
        object.__setattr__(self, "vertices", verts)
        poly = shapely.Polygon(verts) if len(verts) >= 3 else None
        if poly is None or not poly.is_valid or poly.area == 0:
            why = shapely.is_valid_reason(poly) if poly is not None else "fewer than 3 vertices"
            raise ValueError(f"Polygon: not a valid simple polygon ({why})")

    def contains(self, X, Y):
        import shapely
        return shapely.contains_xy(shapely.Polygon(self.vertices), np.asarray(X) + EDGE_TOL, np.asarray(Y) + EDGE_TOL)


@dataclass(frozen=True)
class CrossSection:
    background: Material
    shapes: tuple = field(default_factory=tuple)

    def with_shapes(self, *shapes) -> "CrossSection":
        return replace(self, shapes=self.shapes + tuple(shapes))

    def sample(self, X: np.ndarray, Y: np.ndarray):
        """Per point: the material index (0 = background, k = shape k-1... by fill order), the net doping (cm^-3)
        and a dict name -> property arrays."""
        mats = [self.background] + [s.material for s in self.shapes]
        idx = np.zeros(X.shape, int)
        doping = np.zeros(X.shape)
        for k, s in enumerate(self.shapes, start=1):
            inside = s.contains(X, Y)
            idx = np.where(inside, k, idx)
            doping = np.where(inside, s.doping, doping)
        prop = {name: np.array([getattr(m, name) for m in mats])[idx]
                for name in ("n", "dn_dT", "kappa", "rho_cp", "eps_r", "semiconductor")}
        return idx, doping, prop

    def mask(self, tag: str, X, Y):
        """Points inside the shapes with this tag (after the fill order: covered parts do not count)."""
        idx, _, _ = self.sample(X, Y)
        ks = [k for k, s in enumerate(self.shapes, start=1) if s.tag == tag]
        return np.isin(idx, ks)
