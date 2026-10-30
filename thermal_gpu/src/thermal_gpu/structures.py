"""Synthetic heater cross-sections (textbook conductivities; no process data)."""
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
