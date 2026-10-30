"""Synthetic lens systems for the checks and the benchmarks (textbook values, lengths in mm, wavelengths in um).

- four_f: two ideal thin lenses (Optiland "paraxial" surfaces) in a 4f arrangement, magnification f2 / f1 (the
  2.5x lens pair of examples/chip_to_fiber_coupling.py with the default values).
- aberrated_singlet: a plano-convex singlet (R = 10 mm, n = 1.5, 3 mm thick) at 2f-2f with strong spherical
  aberration, and a physical stop of 6 mm diameter (RadialAperture) on its first surface.
"""

from __future__ import annotations

import numpy as np
from optiland.optic import Optic


def four_f(f1: float = 2.0, f2: float = 5.0, wl: float = 1.55) -> Optic:
    """Object at the front focus of lens 1 (f1), image at the back focus of lens 2 (f2), no physical aperture."""
    o = Optic()
    o.add_surface(index=0, thickness=f1)
    o.add_surface(index=1, surface_type="paraxial", f=f1, thickness=f1 + f2, is_stop=True)
    o.add_surface(index=2, surface_type="paraxial", f=f2, thickness=f2)
    o.add_surface(index=3)
    o.set_aperture(aperture_type="EPD", value=3.0)
    o.add_wavelength(value=wl, is_primary=True)
    return o


def aberrated_singlet(wl: float = 1.55) -> Optic:
    """Point object 40 mm in front of the singlet; the image surface at the paraxial focus (image_solve)."""
    from optiland.materials import IdealMaterial
    from optiland.physical_apertures import RadialAperture

    o = Optic()
    o.add_surface(index=0, thickness=40.0)
    o.add_surface(index=1, radius=10.0, thickness=3.0, material=IdealMaterial(n=1.5), is_stop=True,
                  aperture=RadialAperture(r_max=3.0))
    o.add_surface(index=2, radius=np.inf, thickness=1.0)
    o.add_surface(index=3)
    o.set_aperture(aperture_type="EPD", value=6.0)
    o.set_field_type(field_type="object_height")
    o.add_field(y=0.0)
    o.add_wavelength(value=wl, is_primary=True)
    o.image_solve()
    return o
