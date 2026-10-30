"""A lateral PN depletion modulator in an SOI rib (synthetic, textbook values) and the plasma dispersion effect.

Plasma dispersion at 1.55 um (Soref and Bennett 1987), concentrations in cm^-3:
    dn     = -(8.8e-22 (N1 - N0) + 8.5e-18 (P1^0.8 - P0^0.8))
    dalpha =  8.5e-18 (N1 - N0) + 6.0e-18 (P1 - P0)          (1/cm)
Reverse bias removes carriers: dalpha < 0, the loss decreases.
"""

from __future__ import annotations

import numpy as np

from charge_gpu.poisson import Contact, Device

EPS_SI, EPS_SIO2 = 11.7, 3.9


def rib_pn(width=0.5e-6, height=0.22e-6, slab=0.09e-6, half_span=3.0e-6, doping=3e23, offset=0.0, dx=10e-9,
           clad=0.5e-6, n_i=1.0e16):
    """A rib (width x height) on a slab (thickness ``slab``, from -half_span to +half_span) in SiO2. p doping
    (``doping`` m^-3) for x < offset, n for x > offset. Ohmic contacts at both slab ends (xmin: p, xmax: n)."""
    x = np.arange(-half_span, half_span + dx / 2, dx)
    y = np.arange(-clad, height + clad + dx / 2, dx)
    xc, yc = 0.5 * (x[1:] + x[:-1]), 0.5 * (y[1:] + y[:-1])
    X, Y = np.meshgrid(xc, yc, indexing="ij")
    semi = ((Y > 0) & (Y < slab)) | ((np.abs(X) < width / 2) & (Y > 0) & (Y < height))
    n_net = np.where(semi, np.where(X < offset, -doping, doping), 0.0)
    eps_r = np.where(semi, EPS_SI, EPS_SIO2)
    end_cells = (yc > 0) & (yc < slab)
    contacts = [Contact("xmin", end_cells, 0.0), Contact("xmax", end_cells, 0.0)]
    return Device(x=x, y=y, eps_r=eps_r, n_net=n_net, semiconductor=semi, contacts=contacts, n_i=n_i)


def plasma_dispersion(n0_m3, p0_m3, n1_m3, p1_m3):
    """(dn, dalpha in 1/m) for a change of the carriers from (n0, p0) to (n1, p1), all absolute concentrations in
    m^-3 (Soref and Bennett 1987, 1.55 um). The hole term is a law of the absolute concentration (P^0.8), so the
    change is P1^0.8 - P0^0.8 (a power law of the change itself would overstate dneff; found by the one review)."""
    to_cm = 1e-6
    n0, p0, n1, p1 = (np.asarray(a, float) * to_cm for a in (n0_m3, p0_m3, n1_m3, p1_m3))
    d_index = -(8.8e-22 * (n1 - n0) + 8.5e-18 * (p1**0.8 - p0**0.8))
    d_alpha = (8.5e-18 * (n1 - n0) + 6.0e-18 * (p1 - p0)) * 100.0
    return d_index, d_alpha
