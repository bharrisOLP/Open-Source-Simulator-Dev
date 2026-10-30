"""Synthetic heater structures: 2-D cross-sections and a 3-D finite-length heater (textbook conductivities;
no process data)."""
import numpy as np

K_SI, K_SIO2, K_METAL = 148.0, 1.4, 100.0


def heater_stack(dx=0.02e-6, width=10e-6, box=2.0e-6, top_ox=2.2e-6, sub=3.0e-6, heater_w=2.0e-6, heater_t=0.1e-6,
                 wg_w=0.5e-6, wg_t=0.22e-6, power_per_length=100.0):
    """Si substrate (thickness sub), buried oxide (box), Si waveguide (wg_w x wg_t) on the box, oxide cladding
    (top_ox above the box), metal heater (heater_w x heater_t) on the cladding. Total heater power per unit length
    power_per_length (W/m). Uniform grid dx. Returns x, y, kappa, q, the boundary conditions and the masks."""
    x = np.arange(-width / 2, width / 2 + dx / 2, dx)
    y = np.arange(-sub - box, top_ox + heater_t + dx / 2, dx)
    xc, yc = 0.5 * (x[1:] + x[:-1]), 0.5 * (y[1:] + y[:-1])
    X, Y = np.meshgrid(xc, yc, indexing="ij")
    kappa = np.where(Y < -box, K_SI, K_SIO2)
    wg = (np.abs(X) < wg_w / 2) & (Y > 0) & (Y < wg_t)
    kappa = np.where(wg, K_SI, kappa)
    heater = (np.abs(X) < heater_w / 2) & (Y > top_ox) & (Y < top_ox + heater_t)
    kappa = np.where(heater, K_METAL, kappa)
    if not heater.any():
        raise ValueError("the heater covers no grid cell: use a grid step dx smaller than the heater thickness")
    area = heater.sum() * dx * dx
    q = np.where(heater, power_per_length / area, 0.0)
    bc = {"ymin": ("dirichlet", 300.0), "ymax": ("robin", 10.0, 300.0)}
    return x, y, kappa, q, bc, {"waveguide": wg, "heater": heater}


def graded_edges(a: float, b: float, h_a: float, h_b: float, h_max: float, growth: float = 1.25) -> np.ndarray:
    """Cell edges on [a, b]: cells of size h_a at a and h_b at b that grow by the factor growth toward the middle,
    up to h_max. The sizes are then scaled so that the cells fill b - a exactly (both end points are edges)."""
    if not (b > a and min(h_a, h_b, h_max) > 0 and growth >= 1):
        raise ValueError("graded_edges needs b > a, positive cell sizes and growth >= 1")
    left, right, ha, hb = [], [], h_a, h_b
    while sum(left) + sum(right) < (b - a) * (1 - 1e-12):
        if ha <= hb:
            left.append(ha)
            ha = min(ha * growth, h_max)
        else:
            right.append(hb)
            hb = min(hb * growth, h_max)
    sizes = np.array(left + right[::-1])
    edges = a + np.r_[0.0, np.cumsum(sizes * ((b - a) / sizes.sum()))]
    edges[-1] = b
    return edges


def heater_3d(dx=0.05e-6, length=100e-6, margin=20e-6, dz_end=0.25e-6, dz_max=2.0e-6, growth=1.25, power=10e-3,
              lead_w=0.0, **stack):
    """A finite-length phase-shifter heater: the cross-section of heater_stack (keyword arguments **stack: width,
    box, top_ox, sub, heater_w, heater_t, wg_w, wg_t; uniform grid dx in x and y) along z. The heater (power in W)
    covers |z| < length / 2; the waveguide and the layers continue for margin beyond each heater end. The z grid
    is fine (dz_end) at the heater ends and grows to dz_max; the heater ends are cell edges. lead_w > 0 adds a
    metal lead (no heat) of that width in the heater layer beyond each end. The z ends are adiabatic.
    With power = P' length, the cross-section and q are those of heater_stack(power_per_length=P'), so the
    mid-length result compares directly with the 2-D one.
    Returns x, y, z, kappa, q, the boundary conditions and the masks (nx, ny, nz)."""
    if "power_per_length" in stack:
        raise ValueError("give the total heater power (W) with power=")
    x, y, kappa2, _, bc, masks2 = heater_stack(dx=dx, power_per_length=power / length, **stack)
    h = length / 2
    z = graded_edges(-h, h, dz_end, dz_end, dz_max, growth)
    if margin > 0:
        z = np.concatenate([graded_edges(-h - margin, -h, dz_max, dz_end, dz_max, growth)[:-1], z,
                            graded_edges(h, h + margin, dz_end, dz_max, dz_max, growth)[1:]])
    zc = 0.5 * (z[1:] + z[:-1])
    on = (np.abs(zc) < h)[None, None, :]
    h2, wg2 = masks2["heater"][:, :, None], masks2["waveguide"][:, :, None]
    heater = h2 & on
    kappa = np.where(h2 & ~on, K_SIO2, kappa2[:, :, None] * np.ones(zc.size))
    if lead_w > 0:
        xc = 0.5 * (x[1:] + x[:-1])
        lead = (np.abs(xc)[:, None] < lead_w / 2) & masks2["heater"].any(axis=0)[None, :]   # the heater layer
        kappa = np.where(lead[:, :, None] & ~on, K_METAL, kappa)
    vol = (np.diff(x)[:, None, None] * np.diff(y)[None, :, None] * np.diff(z)[None, None, :])
    q = np.where(heater, power / vol[heater].sum(), 0.0)
    return x, y, z, kappa, q, dict(bc), {"waveguide": wg2 & np.ones(zc.size, bool), "heater": heater}
