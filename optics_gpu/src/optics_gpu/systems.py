"""Synthetic lens systems for the checks and the benchmarks (textbook values, lengths in mm, wavelengths in um).

- four_f: two ideal thin lenses (Optiland "paraxial" surfaces) in a 4f arrangement, magnification f2 / f1 (the
  2.5x lens pair of examples/chip_to_fiber_coupling.py with the default values).
- aberrated_singlet: a plano-convex singlet (R = 10 mm, n = 1.5, 3 mm thick) at 2f-2f with strong spherical
  aberration, and a physical stop of 6 mm diameter (RadialAperture) on its first surface.
- dispersive_singlet: the same singlet in a glass with dispersion (n_d and V_d of N-BK7), three object heights
  and the wavelengths F, d and C: axial and lateral color.
- ideal_relay: two equal ideal thin lenses (f) in a 4f arrangement with a physical stop of radius r_stop on lens 1:
  a perfect imager of the axis point (magnification -1), image-side NA r / sqrt(r^2 + f^2); the image surface
  can be moved by dz after the back focus (a pure defocus).
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


def dispersive_singlet(fields=(0.0, 0.5, 1.0)) -> Optic:
    """The singlet of aberrated_singlet in a glass with dispersion: AbbeMaterial (Optiland's "polynomial" model,
    its default in 0.6.3) with n_d = 1.5168 and V_d = 64.17 (the catalog values of Schott N-BK7), point objects at
    the heights `fields` (mm) 40 mm in front of it, the wavelengths F, d and C (0.4861, 0.5876 (primary) and
    0.6563 um), the image surface at the paraxial focus of d (image_solve). It has axial and lateral color (the
    lateral color off the axis)."""
    from optiland.materials import AbbeMaterial
    from optiland.physical_apertures import RadialAperture

    glass = AbbeMaterial(n=1.5168, abbe=64.17, model="polynomial")
    o = Optic()
    o.add_surface(index=0, thickness=40.0)
    o.add_surface(index=1, radius=10.0, thickness=3.0, material=glass, is_stop=True, aperture=RadialAperture(r_max=3.0))
    o.add_surface(index=2, radius=np.inf, thickness=1.0)
    o.add_surface(index=3)
    o.set_aperture(aperture_type="EPD", value=6.0)
    o.set_field_type(field_type="object_height")
    for y in fields:
        o.add_field(y=float(y))
    for wl in (0.4861, 0.5876, 0.6563):
        o.add_wavelength(value=wl, is_primary=wl == 0.5876)
    o.image_solve()
    return o


def ideal_relay(f: float = 20.0, r_stop: float = 2.0, dz: float = 0.0, wl: float = 1.55) -> Optic:
    """Object point at the front focus of lens 1, image surface dz mm after the back focus of lens 2 (equal ideal
    thin lenses of focal length f). Lens 1 is the stop, with a physical aperture (RadialAperture) of radius r_stop
    and EPD 2 r_stop. Optiland's thin lens adds the optical path f - sqrt(h^2 + f^2) at the height h, so the rays
    of the axis point meet in one point with equal optical paths; with equal lenses the image-side direction of a
    ray equals its object-side direction (tan u = h / f on both sides)."""
    from optiland.physical_apertures import RadialAperture

    o = Optic()
    o.add_surface(index=0, thickness=f)
    o.add_surface(index=1, surface_type="paraxial", f=f, thickness=2 * f, is_stop=True,
                  aperture=RadialAperture(r_max=r_stop))
    o.add_surface(index=2, surface_type="paraxial", f=f, thickness=f + dz)
    o.add_surface(index=3)
    o.set_aperture(aperture_type="EPD", value=2 * r_stop)
    o.set_field_type(field_type="object_height")
    o.add_field(y=0.0)
    o.add_wavelength(value=wl, is_primary=True)
    return o
