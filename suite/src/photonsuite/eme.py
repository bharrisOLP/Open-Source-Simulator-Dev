"""EME of a device made of cross-section segments along z, with meow and the meow-gpu mode solver.

The same device description as photonsuite.fdtd (Segment(z0, z1, CrossSection)): each Rect of a segment becomes a
meow Box over the z range of its segment, and each Polygon a prism along z; meow cuts the device into cells of length
``cell_length`` (each cell takes the cross-section at its center). meow 0.15 never projects a Prism with axis "z"
inside its z range (``_project_axis_z`` tests ``z < h_max`` where it means ``z > h_max``), so the adapter uses a
subclass with the corrected test. The S-matrix comes from meow (overlaps, interfaces, SAX cascade) with its default
interface settings; ``interface_kwargs`` passes other settings to meow.compute_interface_s_matrices.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np


def _z_prism(vertices, z0: float, z1: float):
    """A meow Prism along z that is present for z0 <= z <= z1 (the upstream test is wrong in meow 0.15)."""
    import meow as mw
    from meow.geometries import Polygon2D

    class ZPrism(mw.Prism):
        def _project_axis_z(self, z):
            return [] if (z < self.h_min or z > self.h_max) else [Polygon2D(poly=self.poly)]

    return ZPrism(poly=np.asarray(vertices, float), h_min=z0, h_max=z1, axis="z")


@dataclass
class EMEResult:
    transmission: float          # |S(left@0 -> right@0)|^2: fundamental to fundamental
    total_right: float           # the sum of |S|^2 over all right ports (power into all kept modes)
    n_cells: int
    modes_kept: list


def structures(segments, x_range, y_range):
    """meow structures of the device. meow fills a point with the structure of the lowest mesh_order, so the
    background gets the highest order and each shape a lower order than every shape before it (the fill order of
    CrossSection: a later shape replaces an earlier one)."""
    import meow as mw

    from photonsuite.fdtd import check_segments

    check_segments(segments)
    mats = {}
    bg = segments[0].xs.background
    mats[bg.name] = mw.IndexMaterial(name=bg.name, n=bg.n)
    z_lo, z_hi = segments[0].z0, segments[-1].z1
    n_shapes = sum(len(seg.xs.shapes) for seg in segments)
    out = [mw.Structure(material=mats[bg.name], mesh_order=n_shapes + 1,
                        geometry=mw.Box(x_min=x_range[0] - 1, x_max=x_range[1] + 1, y_min=y_range[0] - 1,
                                        y_max=y_range[1] + 1, z_min=z_lo - 1, z_max=z_hi + 1))]
    order = n_shapes
    for seg in segments:
        for r in seg.xs.shapes:
            m = mats.setdefault(r.material.name, mw.IndexMaterial(name=r.material.name, n=r.material.n))
            if hasattr(r, "vertices"):
                geom = _z_prism(r.vertices, seg.z0, seg.z1)
            else:
                geom = mw.Box(x_min=r.x0, x_max=r.x1, y_min=r.y0, y_max=r.y1, z_min=seg.z0, z_max=seg.z1)
            out.append(mw.Structure(material=m, mesh_order=order, geometry=geom))
            order -= 1
    return out


def transmission(segments, x_range, y_range, wavelength: float, step: float = 0.02, cell_length: float = 0.25,
                 num_modes: int = 10, precision: str = "mixed", interface_kwargs: dict | None = None) -> EMEResult:
    import jax
    import meow as mw

    from meow_gpu.meow_backend import GPUModeSolver, compute_modes_gpu

    z_lo, z_hi = segments[0].z0, segments[-1].z1
    structures_ = structures(segments, x_range, y_range)
    mesh = mw.Mesh2D(x=np.arange(x_range[0], x_range[1] + step / 2, step),
                     y=np.arange(y_range[0], y_range[1] + step / 2, step))
    n_cells = max(1, int(round((z_hi - z_lo) / cell_length)))
    cells = mw.create_cells(structures_, mesh, np.full(n_cells, (z_hi - z_lo) / n_cells), z_min=z_lo)
    env = mw.Environment(wl=wavelength, T=25.0)
    solver = GPUModeSolver(precision=precision)
    modes = [compute_modes_gpu(mw.CrossSection.from_cell(cell=c, env=env), num_modes=num_modes, solver=solver)
             for c in cells]
    interfaces = (functools.partial(mw.compute_interface_s_matrices, **interface_kwargs) if interface_kwargs
                  else mw.compute_interface_s_matrices)
    with jax.default_device(jax.devices("cpu")[0]):          # SAX with klujax runs on the CPU
        S, pm = mw.compute_s_matrix(modes, cells=cells, interfaces_fn=interfaces)
    S = np.asarray(S)
    t = float(abs(S[pm["left@0"], pm["right@0"]]) ** 2)
    total = float(sum(abs(S[pm["left@0"], pm[k]]) ** 2 for k in pm if k.startswith("right")))
    return EMEResult(t, total, n_cells, [len(m) for m in modes])
