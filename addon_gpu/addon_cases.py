# SPDX-License-Identifier: GPL-2.0-or-later
"""Small synthetic test cases for the add-on route (the GPU time stepping of the Erised add-on).

Each builder returns a ``Case``: an ordinary simulation of the MIT FDTD package, the flux monitors, the stop time,
and, for the ring, the point where the field is recorded. All structures use textbook values:

* ``slab2d``: the 2-D dielectric slab waveguide of the add-on examples (permittivity 12, width 1, PML 1), a
  Gaussian line source in the slab and two flux planes.
* ``ring2d``: the ring resonator of the tutorial of the MIT FDTD package (index 3.4, width 1, inner radius 1,
  pad 4, PML 2, Ez source at r + 0.1, Gaussian pulse at 0.15 with width 0.1). The field at the source point is
  recorded every step, and Harminv finds the resonances.
* ``sphere3d``: the 3-D dielectric sphere of the add-on quick start (permittivity 9, radius 0.8, PML 0.8, cell
  4 x 4 x 4) with one flux plane.

The scale argument multiplies the cell of ``slab2d`` and ``sphere3d`` (not the structure), for benches of a useful
size.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

CASES = ("slab2d", "ring2d", "sphere3d")


@dataclass
class Case:
    name: str
    sim: Any
    fluxes: list
    until: float | None = None
    until_after_sources: float | None = None
    point: Any = None                     # mp.Vector3 of the recorded point, or None
    harminv: tuple | None = None          # (component, fcen, df) for Harminv at the point
    shape: tuple = ()                     # cells along each non-zero axis
    extra: dict = field(default_factory=dict)

    @property
    def cells(self) -> int:
        out = 1
        for n in self.shape:
            out *= n
        return out


def _shape(cell, resolution) -> tuple:
    return tuple(int(round(s * resolution)) for s in (cell.x, cell.y, cell.z) if s > 0)


def slab2d(resolution: int | None = None, until: float | None = None, scale: float = 1.0) -> Case:
    import meep as mp

    resolution = resolution or 20
    cell = mp.Vector3(10 * scale, 6 * scale, 0)
    sim = mp.Simulation(
        cell_size=cell,
        resolution=resolution,
        boundary_layers=[mp.PML(1.0)],
        geometry=[mp.Block(mp.Vector3(mp.inf, 1.0, mp.inf), center=mp.Vector3(),
                           material=mp.Medium(epsilon=12))],
        sources=[mp.Source(mp.GaussianSource(frequency=0.25, fwidth=0.3),
                           component=mp.Ez, center=mp.Vector3(-0.5 * cell.x + 1.5, 0))],
    )
    near = sim.add_flux(0.25, 0.3, 5, mp.FluxRegion(center=mp.Vector3(-0.5 * cell.x + 2.5, 0),
                                                    size=mp.Vector3(0, cell.y - 2)))
    far = sim.add_flux(0.25, 0.3, 5, mp.FluxRegion(center=mp.Vector3(0.5 * cell.x - 2.0, 0),
                                                   size=mp.Vector3(0, cell.y - 2)))
    return Case("slab2d", sim, [near, far], until=until or 60.0, shape=_shape(cell, resolution))


def ring2d(resolution: int | None = None, until: float | None = None, scale: float = 1.0) -> Case:
    import meep as mp

    resolution = resolution or 10
    n, w, r, pad, dpml = 3.4, 1.0, 1.0, 4.0, 2.0
    sxy = 2 * (r + w + pad + dpml)
    cell = mp.Vector3(sxy, sxy, 0)
    fcen, df = 0.15, 0.1
    point = mp.Vector3(r + 0.1)
    sim = mp.Simulation(
        cell_size=cell,
        resolution=resolution,
        boundary_layers=[mp.PML(dpml)],
        geometry=[mp.Cylinder(radius=r + w, material=mp.Medium(index=n)), mp.Cylinder(radius=r)],
        sources=[mp.Source(mp.GaussianSource(fcen, fwidth=df), component=mp.Ez, center=point)],
    )
    return Case("ring2d", sim, [], until_after_sources=until or 300.0, point=point,
                harminv=(mp.Ez, fcen, df), shape=_shape(cell, resolution))


def sphere3d(resolution: int | None = None, until: float | None = None, scale: float = 1.0) -> Case:
    import meep as mp

    resolution = resolution or 12
    cell = mp.Vector3(4 * scale, 4 * scale, 4 * scale)
    sim = mp.Simulation(
        cell_size=cell,
        resolution=resolution,
        boundary_layers=[mp.PML(0.8)],
        geometry=[mp.Sphere(radius=0.8, material=mp.Medium(epsilon=9))],
        sources=[mp.Source(mp.GaussianSource(frequency=0.4, fwidth=0.4), component=mp.Ez,
                           center=mp.Vector3(-1.2, 0, 0))],
    )
    flux = sim.add_flux(0.4, 0.4, 3, mp.FluxRegion(center=mp.Vector3(0.9, 0, 0), size=mp.Vector3(0, 2, 2)))
    return Case("sphere3d", sim, [flux], until=until or 35.0, shape=_shape(cell, resolution))


BUILDERS: dict[str, Callable[..., Case]] = {"slab2d": slab2d, "ring2d": ring2d, "sphere3d": sphere3d}


def build(name: str, resolution: int | None = None, until: float | None = None, scale: float = 1.0) -> Case:
    if name not in BUILDERS:
        raise ValueError(f"unknown case {name!r}; known: {', '.join(CASES)}")
    return BUILDERS[name](resolution=resolution, until=until, scale=scale)
