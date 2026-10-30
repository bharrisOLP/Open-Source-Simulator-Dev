"""Devices and quantities for forward bias (carrier injection), with drift_diffusion.py.

- graded_axis: cell edges that are fine near given points and grow geometrically away from them.
- diode_1d: a 1-D abrupt PN diode (one cell in y) with ohmic contacts at both ends.
- shockley_j0: the saturation current density of the ideal (Shockley) long-base diode.
- rib_pin: a lateral PIN phase shifter in an SOI rib (synthetic, textbook values; no process data).
- rib_index_change: the plasma-dispersion index change and loss (Soref and Bennett, modulator.plasma_dispersion)
  of the carriers in the rib.
"""

from __future__ import annotations

import numpy as np

from charge_gpu.modulator import EPS_SI, EPS_SIO2, plasma_dispersion
from charge_gpu.poisson import Q, Contact, Device, thermal_voltage


def graded_axis(start: float, stop: float, fine: float, ratio: float, points=(), fine_half_width: float = 0.0):
    """Cell edges from start to stop: cells of size ``fine`` within fine_half_width of each point in ``points``,
    growing by ``ratio`` per cell away from them (and never larger than the distance allows)."""
    pts = np.atleast_1d(np.asarray(points, float))
    edges, x = [start], start
    while x < stop - 1e-15 * max(1.0, abs(stop)):
        dist = np.min(np.abs(pts - x)) if pts.size else 0.0
        far = max(dist - fine_half_width, 0.0)
        h = fine + far * (ratio - 1.0)                 # a geometric growth from the fine zone
        if pts.size:                                   # do not step over the fine zone of a point ahead
            ahead = pts[pts > x]
            if ahead.size:
                h = min(h, max(ahead[0] - fine_half_width - x, fine))
        x = min(x + h, stop)
        if stop - x < 0.5 * fine:
            x = stop
        edges.append(x)
    return np.array(edges)


def diode_1d(n_a: float, n_d: float, length_p: float, length_n: float, fine: float = 1e-9,
             fine_half_width: float = 0.1e-6, ratio: float = 1.03, n_i: float = 1.0e16, height: float = 1e-6):
    """A 1-D abrupt PN diode: p (N_A, m^-3) for x < 0, n (N_D) for x > 0, ohmic contacts at x = -length_p (p) and
    x = +length_n (n). One cell of ``height`` in y (per area = per length / height)."""
    x = graded_axis(-length_p, length_n, fine, ratio, points=[0.0], fine_half_width=fine_half_width)
    xc = 0.5 * (x[1:] + x[:-1])
    nx = xc.size
    n_net = np.where(xc < 0, -n_a, n_d)[:, None]
    return Device(x=x, y=np.array([0.0, height]), eps_r=np.full((nx, 1), EPS_SI), n_net=n_net,
                  semiconductor=np.ones((nx, 1), bool),
                  contacts=[Contact("xmin", np.array([True]), 0.0), Contact("xmax", np.array([True]), 0.0)],
                  n_i=n_i)


def shockley_j0(n_a: float, n_d: float, mu_n_p: float, mu_p_n: float, tau_n: float, tau_p: float,
                n_i: float = 1.0e16, T: float = 300.0, length_p: float = np.inf, length_n: float = np.inf):
    """J0 (A/m^2) of the ideal diode (Shockley; Sze and Ng, Physics of Semiconductor Devices, 3rd ed., Ch. 2):
        J0 = q n_i^2 (D_n / (L_n N_A) coth(W_p / L_n) + D_p / (L_p N_D) coth(W_n / L_p)),
    D = mu V_T, L = sqrt(D tau), mu_n_p: the electron mobility in the p side, mu_p_n: the hole mobility in the n
    side. W_p, W_n: the neutral lengths from the depletion edges to the contacts (inf: the long-base diode, coth = 1).
    """
    vt = thermal_voltage(T)
    d_n, d_p = mu_n_p * vt, mu_p_n * vt
    l_n, l_p = np.sqrt(d_n * tau_n), np.sqrt(d_p * tau_p)
    coth = lambda z: 1.0 if np.isinf(z) else 1.0 / np.tanh(z)  # noqa: E731
    return Q * n_i**2 * (d_n / (l_n * n_a) * coth(length_p / l_n) + d_p / (l_p * n_d) * coth(length_n / l_p))


def rib_pin(width=0.5e-6, height=0.22e-6, slab=0.09e-6, half_span=1.5e-6, gap=0.3e-6, doping=1e24,
            background=-1e21, dx=10e-9, clad=0.5e-6, n_i=1.0e16):
    """A lateral PIN phase shifter: a rib (width x height) on a slab (thickness ``slab``, from -half_span to
    +half_span) in SiO2. The intrinsic zone |x| < width/2 + gap has the background doping (default p 1e15 cm^-3, a
    usual SOI wafer); p+ (``doping`` m^-3) for x < -(width/2 + gap), n+ for x > width/2 + gap. Ohmic contacts at both
    slab ends (xmin: p+, xmax: n+). Uniform cells of ``dx``: every boundary (the slab top, the rib sides and top,
    the doping edges, the oxide) must fall on a cell edge, else ValueError (a cell center on a boundary would move
    it by half a cell: with dx = 20 nm the 90 nm slab became 80 nm thick)."""
    lengths = {"width / 2": width / 2, "height": height, "slab": slab, "half_span": half_span, "gap": gap,
               "clad": clad}
    for name, length in lengths.items():
        k = length / dx
        if abs(k - round(k)) > 1e-6:
            raise ValueError(f"rib_pin: {name} = {length:g} m is not a whole number of cells of {dx:g} m")
    x = np.arange(-half_span, half_span + dx / 2, dx)
    y = np.arange(-clad, height + clad + dx / 2, dx)
    xc, yc = 0.5 * (x[1:] + x[:-1]), 0.5 * (y[1:] + y[:-1])
    X, Y = np.meshgrid(xc, yc, indexing="ij")
    semi = ((Y > 0) & (Y < slab)) | ((np.abs(X) < width / 2) & (Y > 0) & (Y < height))
    edge = width / 2 + gap
    n_net = np.where(X < -edge, -doping, np.where(X > edge, doping, background))
    n_net = np.where(semi, n_net, 0.0)
    eps_r = np.where(semi, EPS_SI, EPS_SIO2)
    end_cells = (yc > 0) & (yc < slab)
    contacts = [Contact("xmin", end_cells, 0.0), Contact("xmax", end_cells, 0.0)]
    return Device(x=x, y=y, eps_r=eps_r, n_net=n_net, semiconductor=semi, contacts=contacts, n_i=n_i)


def rib_mask(dev: Device, width=0.5e-6, height=0.22e-6):
    """The cells of the rib core (|x| < width/2, 0 < y < height)."""
    xc, yc = 0.5 * (dev.x[1:] + dev.x[:-1]), 0.5 * (dev.y[1:] + dev.y[:-1])
    X, Y = np.meshgrid(xc, yc, indexing="ij")
    return dev.semiconductor & (np.abs(X) < width / 2) & (Y > 0) & (Y < height)


def rib_index_change(dev: Device, sol, sol0, mask=None):
    """Mean n, p (m^-3) in the rib core, and the area means of the index change and of the change of the absorption
    (1/m) from the solution ``sol0`` (usually 0 V) to ``sol``: Soref and Bennett at 1.55 um, from the absolute
    carriers (modulator.plasma_dispersion). These are material values; the modal values need the mode overlap."""
    mask = rib_mask(dev) if mask is None else mask
    area = (np.diff(dev.x)[:, None] * np.diff(dev.y)[None, :])[mask]
    dn, da = plasma_dispersion(sol0.n[mask], sol0.p[mask], sol.n[mask], sol.p[mask])
    w = area / area.sum()
    return {"n_mean": float(np.sum(w * sol.n[mask])), "p_mean": float(np.sum(w * sol.p[mask])),
            "dn_mean": float(np.sum(w * dn)), "dalpha_mean": float(np.sum(w * da))}
