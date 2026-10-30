"""Synthetic test cross-sections (public textbook values, no process data)."""
import numpy as np

N_SI, N_SIO2, N_SIN = 3.476, 1.444, 1.996


def strip(width=0.5, height=0.22, n_core=N_SI, n_clad=N_SIO2, w_dom=3.0, h_dom=2.0, dx=0.02, dy=0.02):
    """Rectangular core in a uniform cladding. eps at the Ex, Ey, Ez positions of the Yee cell (Ex at the
    x-centre of the cell edge, Ey at the y-centre, Ez at the corner), sampled without averaging."""
    x = np.arange(-w_dom / 2, w_dom / 2 + dx / 2, dx)
    y = np.arange(-h_dom / 2, h_dom / 2 + dy / 2, dy)
    xc, yc = 0.5 * (x[1:] + x[:-1]), 0.5 * (y[1:] + y[:-1])
    xe, ye = x[:-1], y[:-1]

    def eps(xx, yy):
        X, Y = np.meshgrid(xx, yy, indexing="ij")
        inside = (np.abs(X) < width / 2) & (np.abs(Y) < height / 2)
        return np.where(inside, n_core**2, n_clad**2).astype(complex)

    return eps(xc, ye), eps(xe, yc), eps(xe, ye), x, y


def coupler(core=11, height=0.22, gap=40, margin=25, n_core=N_SI, n_clad=N_SIO2, h_dom=1.6, dx=0.04):
    """Two identical cores, far apart (a weak directional coupler): the even and odd supermodes have very close
    eigenvalues (a cluster). Widths, gap and margins in cells of dx; the structure is mirror-symmetric on the grid
    (core = 11 cells: 0.44 um at dx = 0.04). Ex at the cell centres in x; Ey and Ez at the cell edges in x, in the
    core when the edge is between two core cells."""
    nx = 2 * (core + margin) + gap
    x = (np.arange(nx + 1) - nx / 2) * dx
    y = np.arange(-h_dom / 2, h_dom / 2 + dx / 2, dx)
    yc, ye = 0.5 * (y[1:] + y[:-1]), y[:-1]
    cell = np.zeros(nx, bool)
    cell[margin: margin + core] = True
    cell[margin + core + gap: margin + 2 * core + gap] = True
    edge = np.zeros(nx, bool)
    edge[1:] = cell[1:] & cell[:-1]

    def eps(in_x, yy):
        inside = in_x[:, None] & (np.abs(yy) < height / 2)[None, :]
        return np.where(inside, n_core**2, n_clad**2).astype(complex)

    return eps(cell, ye), eps(edge, yc), eps(edge, ye), x, y
