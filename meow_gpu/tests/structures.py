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


def sin_tip(half=7.0, dx=0.1, side=0.3, n_core=N_SIN, n_clad=N_SIO2):
    """A small SiN core (about side x side um) in SiO2, in a square window of +-half um: a weakly guided mode (neff
    about 1.448 at 1.55 um, cladding 1.444) just above a dense band of window modes. The structure of
    examples/chip_to_fiber_coupling.py before 2026-11-07, when that example also used this construction (since then
    it uses integer cell indices and a symmetric core with partial cells).

    The real core is not symmetric. A sample is in the core when its coordinate c (a cell centre or a cell edge, by
    the eps component) satisfies |c| < side / 2 in floating point, and the coordinates of np.arange are not exact:
    the sample at +side/2 is in the core and the one at -side/2 is not. With the defaults (side 0.3 um):

    - dx = 0.05 (280 x 280 cells): the edge samples of the core run from -0.10 to +0.15 um (6 samples: x of eps_yy
      and eps_zz, y of eps_xx and eps_zz); the centre samples from -0.125 to +0.125 um (6 samples).
    - dx = 0.1 (140 x 140 cells): the centre samples run from -0.05 to +0.15 um (3 samples: x of eps_xx, y of
      eps_yy); the edge samples from -0.1 to +0.1 um (3 samples).

    So the core has no mirror symmetry, and its centre is not at the origin. It is a valid synthetic structure (each
    test compares solvers on the same eps arrays), and the tests of the refinement need it: its near-degenerate
    pairs of window modes gave the regressions of 2026-10-30 and 2026-11-07. ``tests/test_refine.py`` checks this
    description (``test_sin_tip_core_is_the_documented_one``)."""
    g = np.arange(-half, half + 1e-9, dx)
    xc, xe = 0.5 * (g[1:] + g[:-1]), g[:-1]

    def eps(xx, yy):
        X, Y = np.meshgrid(xx, yy, indexing="ij")
        core = (np.abs(X) < side / 2) & (np.abs(Y) < side / 2)
        return np.where(core, n_core**2, n_clad**2).astype(complex)

    return eps(xc, xe), eps(xe, xc), eps(xe, xe), g, g
