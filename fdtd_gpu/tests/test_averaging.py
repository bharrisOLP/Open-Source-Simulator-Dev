"""Subpixel averaging (eps_averaging=True): the averaged permittivity against exact fill fractions and the formula of
Kottke et al. (no FDTD run), the transfer to fdtdx, and FDTD checks against independent references (averaged and
staircase on the same grids)."""

import numpy as np
import pytest

import cases
import fdtd_gpu as fg
from fdtd_gpu._averaging import LATTICE_N, averaged_permittivity, lattice, normal_grid
from fdtd_gpu.geometry import rasterize

E1, E2 = 1.0, 12.0          # background and object permittivity of the checks without FDTD


def _avg(geometry, shape, dx, trivial, backend="numpy"):
    idx, mats = rasterize(geometry, fg.Medium(epsilon=E1), shape, dx, trivial)
    return averaged_permittivity(geometry, mats, idx, dx, trivial, backend=backend)[0]


def _fill_below(lo, hi, y0):
    """Fill fraction of the interval [lo, hi] below y0."""
    return np.clip((y0 - lo) / (hi - lo), 0.0, 1.0)


def test_lattice_projections_and_symmetry():
    """Each coordinate of the lattice takes the midpoint values of n points (exact fill fraction of a plane normal to
    an axis to 1 / (2 n)); the 2-D and 3-D sets are centrally symmetric; the normal grid is mirror symmetric."""
    for d in (1, 2, 3):
        u, n = lattice(d), LATTICE_N[d]
        mid = (np.arange(n) + 0.5) / n - 0.5
        for k in range(d):
            vals, counts = np.unique(np.round(u[:, k] * 2 * n).astype(int), return_counts=True)
            assert np.array_equal(vals, np.round(mid * 2 * n).astype(int)) and np.all(counts == (1 if d == 1 else 2))
        if d > 1:
            assert np.allclose(np.sort(u, axis=0), np.sort(-u, axis=0))
        g, w = normal_grid(d)
        assert np.allclose(g.sum(axis=0), 0) and np.allclose((w[:, None] * g).sum(axis=0), 0) and np.all(w >= 0)


@pytest.mark.parametrize("three_d", [False, True])
def test_plane_normal_to_axis(three_d):
    """A plane between grid lines (normal along y): E parallel to the plane takes the arithmetic mean, E normal the
    harmonic mean, both with the exact fill fraction (to 1 / (2 n) of the lattice); the two parallel components are
    equal (the normal grid gives no false tangential part)."""
    dx, y0 = 0.1, 0.0337
    shape = (6, 16, 6) if three_d else (6, 16, 3)
    trivial = (False, False, not three_d)
    blk = fg.Block(size=fg.Vector3(fg.inf, 10.0, fg.inf), center=fg.Vector3(0, y0 - 5.0),
                   material=fg.Medium(epsilon=E2))
    eps = _avg([blk], shape, dx, trivial)
    yc = (np.arange(shape[1]) + 0.5) * dx - shape[1] * dx / 2           # cell centers; E_y at yc + dx / 2
    tol = 1 / (2 * LATTICE_N[3 if three_d else 2]) + 1e-9
    f_par = _fill_below(yc - dx / 2, yc + dx / 2, y0)
    f_nor = _fill_below(yc, yc + dx, y0)
    meas_par = (eps[2, 2, :, 2] - E1) / (E2 - E1)
    meas_nor = (1 / eps[1, 2, :, 2] - 1 / E1) / (1 / E2 - 1 / E1)
    assert np.max(np.abs(meas_par - f_par)) < tol, (meas_par, f_par)
    assert np.max(np.abs(meas_nor - f_nor)) < tol, (meas_nor, f_nor)
    assert np.allclose(eps[0], eps[2], rtol=1e-12)
    assert np.any((f_par > 0.05) & (f_par < 0.95)) and np.any((f_nor > 0.05) & (f_nor < 0.95))


def _kottke_diag(f, n2, forward=False):
    """The averaged eps of a component for the fill fraction f of E2 in E1 and n_a^2 = n2: the diagonal of the
    inverse tensor of Kottke et al. (or of the forward tensor)."""
    mean, inv_mean = f * E2 + (1 - f) * E1, f / E2 + (1 - f) / E1
    if forward:
        return n2 / inv_mean + (1 - n2) * mean
    return 1 / (n2 * inv_mean + (1 - n2) / mean)


@pytest.mark.parametrize("diagonal", ["inverse", "forward"])
def test_slanted_plane_kottke_diagonal(diagonal, monkeypatch):
    """A plane at 30 degrees in a 2-D cell: each averaged component against the diagonal of the inverse Kottke
    tensor, 1 / eps_aa = n_a^2 <1/eps> + (1 - n_a^2) / <eps> (or of the forward tensor with DIAGONAL = "forward"),
    with the exact fill fraction and the exact normal."""
    import fdtd_gpu._averaging as av
    monkeypatch.setattr(av, "DIAGONAL", diagonal)
    dx, th, c0 = 0.1, np.deg2rad(30.0), 0.0123
    nx, ny = np.cos(th), np.sin(th)
    big = 50.0                                     # half-plane nx x + ny y < c0 as a large triangle
    t = np.array([-ny, nx])
    p0 = np.array([nx, ny]) * c0
    verts = [p0 + big * t, p0 - big * t, p0 - big * np.array([nx, ny]) * 2]
    tri = fg.Prism([fg.Vector3(*v) for v in verts], height=fg.inf, material=fg.Medium(epsilon=E2))
    shape = (12, 12, 3)
    eps = _avg([tri], shape, dx, (False, False, True))
    cc = (np.arange(12) + 0.5) * dx - 0.6
    ys = (np.arange(20000) + 0.5) / 20000

    def fill(x0, y0):                              # exact: integrate the x extent below the line over y
        y = y0 - dx / 2 + ys * dx
        xe = (c0 - ny * y) / nx
        return np.clip((xe - (x0 - dx / 2)) / dx, 0, 1).mean()

    errs = []
    for a, (sx, sy) in enumerate(((dx / 2, 0), (0, dx / 2), (0, 0))):
        for i in range(12):
            for j in range(12):
                f = fill(cc[i] + sx, cc[j] + sy)
                if 0.02 < f < 0.98:
                    errs.append(abs(eps[a, i, j, 0] / _kottke_diag(f, (nx**2, ny**2, 0.0)[a], diagonal == "forward")
                                    - 1))
    assert len(errs) > 20
    assert max(errs) < 2e-2 and np.median(errs) < 5e-3, (max(errs), np.median(errs))     # the 24 x 24 normal grid


def test_slanted_plane_3d():
    """A plane with the normal (2, 3, 6) / 7 in a 3-D cell (a sphere of radius 1e4 um: its sagitta over the cell is
    below 3e-5 um): each averaged component against the diagonal of the inverse Kottke tensor with the exact fill
    fraction and the exact normal (the 12 x 12 x 12 normal grid)."""
    dx, c0, rad = 0.1, 0.0123, 1.0e4
    nv = np.array([2.0, 3.0, 6.0]) / 7.0
    ctr = nv * (c0 - rad)
    sph = fg.Sphere(radius=rad, center=fg.Vector3(*ctr), material=fg.Medium(epsilon=E2))
    shape = (8, 8, 8)
    eps = _avg([sph], shape, dx, (False, False, False))
    cc = (np.arange(8) + 0.5) * dx - 0.4
    g = (np.arange(300) + 0.5) / 300 - 0.5
    gu, gv = np.meshgrid(g * dx, g * dx, indexing="ij")

    def fill(q):                                   # exact: the z extent below the plane, over a 300 x 300 grid in x, y
        zc = (c0 - nv[0] * (q[0] + gu) - nv[1] * (q[1] + gv)) / nv[2]
        return np.clip((zc - (q[2] - dx / 2)) / dx, 0, 1).mean()

    errs = []
    for a in range(3):
        for i, j, k in np.ndindex(*shape):
            q = np.array([cc[i], cc[j], cc[k]])
            q[a] += dx / 2
            f = fill(q)
            if 0.02 < f < 0.98:
                errs.append(abs(eps[a, i, j, k] / _kottke_diag(f, nv[a] ** 2) - 1))
    assert len(errs) > 50
    assert max(errs) < 3e-2 and np.median(errs) < 1e-2, (len(errs), max(errs), np.median(errs))


def test_periodic_faces_repeat_objects():
    """On a periodic axis the objects repeat, as in the MIT package: a block of the length of the period gives the
    same staircase and averaged permittivity as an infinite block, and a block across the periodic face the same as
    that block moved by one period. A Simulation with k_point and a PML only along y has x periodic."""
    dx, shape, trivial, per = 1 / 40, (20, 80, 3), (False, False, True), (True, False, False)
    si = fg.Medium(index=3.45)

    def run(geom):
        idx, mats = rasterize(geom, fg.Medium(index=1.45), shape, dx, trivial, per)
        return idx, averaged_permittivity(geom, mats, idx, dx, trivial, per, backend="numpy")[0]

    a = run([fg.Block(size=fg.Vector3(fg.inf, 0.37, fg.inf), material=si)])
    b = run([fg.Block(size=fg.Vector3(0.5, 0.37, fg.inf), material=si)])
    c = run([fg.Block(size=fg.Vector3(0.2137, 0.37, fg.inf), center=fg.Vector3(0.25, 0.1), material=si)])
    d = run([fg.Block(size=fg.Vector3(0.2137, 0.37, fg.inf), center=fg.Vector3(-0.25, 0.1), material=si)])
    for u, v in ((a, b), (c, d)):
        assert np.array_equal(u[0], v[0]) and np.allclose(u[1], v[1], rtol=1e-9, atol=0)
    assert c[0][0, 44, 0] == 1 and c[0][-1, 44, 0] == 1               # the block is at both faces
    assert abs(c[1][0, -1, 44, 0] - 3.45**2) < 1e-5     # E_x on the face (inside the block and its copy)
    cell = fg.Vector3(0.5, 2.0, 0)
    assert fg.Simulation(cell_size=cell, resolution=40, k_point=fg.Vector3(),
                         boundary_layers=[fg.PML(0.5, direction=fg.Y)])._periodic_axes() == (True, False, False)
    assert fg.Simulation(cell_size=cell, resolution=40, boundary_layers=[fg.PML(0.5)])._periodic_axes() == (
        False, False, False)
    # a material face between the last and the first cell: the block [0.003, 0.253] crosses x = 0.25 (its copy ends
    # at x = -0.247). The same block moved by -4 cells has no face there: its arrays, rolled by 4 cells, must agree.
    # The E_x voxel of the last cell needs the wrapped index of the first cell (this fails with mode="edge").
    e = run([fg.Block(size=fg.Vector3(0.25, 0.37, fg.inf), center=fg.Vector3(0.128, 0.1), material=si)])
    f = run([fg.Block(size=fg.Vector3(0.25, 0.37, fg.inf), center=fg.Vector3(0.028, 0.1), material=si)])
    assert np.array_equal(e[0], np.roll(f[0], 4, axis=0))
    assert np.allclose(e[1], np.roll(f[1], 4, axis=1), rtol=1e-9, atol=0)
    assert 2.2 < e[1][0, -1, 44, 0] < 11.8                            # the E_x voxel on the face x = 0.25 is mixed
    with pytest.raises(NotImplementedError):                          # fdtdx would wrap the whole x axis
        fg.Simulation(cell_size=fg.Vector3(0.5, 2.0, 0), resolution=40, k_point=fg.Vector3(),
                      boundary_layers=[fg.PML(0.5, direction=fg.Y), fg.PML(0.2, direction=fg.X, side=fg.High)])


def test_numpy_and_jax_agree():
    """The jit-compiled JAX path (GPU, float32) gives the NumPy (float64) result for a mixed 3-D geometry."""
    geom = [fg.Block(size=fg.Vector3(fg.inf, 0.43, 0.22), material=fg.Medium(index=3.45)),
            fg.Cylinder(radius=0.37, height=0.3, center=fg.Vector3(0.1, 0.05), material=fg.Medium(index=2.0)),
            fg.Prism([fg.Vector3(-0.5, -0.4), fg.Vector3(0.2, -0.3), fg.Vector3(-0.3, 0.3)], height=0.15,
                     material=fg.Medium(index=1.6))]
    shape, dx, trivial = (24, 20, 16), 0.05, (False, False, False)
    e_np = _avg(geom, shape, dx, trivial, backend="numpy")
    e_jx = _avg(geom, shape, dx, trivial, backend="jax")
    d = np.abs(e_jx / e_np - 1)
    # float32 sample points within about 1e-7 um of a surface can fall on the other side: one of the 2042 lattice
    # points of a voxel changes eps by up to (11.9 - 1) / 2042 / eps
    assert np.mean(d > 1e-5) < 0.05 and np.max(d) < 3e-3, (np.mean(d > 1e-5), np.max(d))     # bench: 1.3 %, 1.3e-3


def test_fdtdx_gets_the_components_and_get_epsilon():
    """fdtdx receives one inverse permittivity per E component (3 components), also for objects below the background;
    the staircase keeps one component. setup_seconds is NaN before the setup. get_epsilon
    removes the trivial axis."""
    core = fg.Block(size=fg.Vector3(fg.inf, 0.37, fg.inf), material=fg.Medium(index=3.45))
    for avg in (True, False):
        sim = fg.Simulation(cell_size=fg.Vector3(1.0, 2.0, 0), resolution=20, geometry=[core],
                            boundary_layers=[fg.PML(0.5)], default_material=fg.Medium(index=1.45), eps_averaging=avg)
        assert sim.get_epsilon().shape == (20, 40)
        import jax
        with jax.enable_x64(False):
            _, arrays, _, _ = sim._build(4, [])
        inv = np.asarray(arrays.inv_permittivities)
        if avg:
            assert inv.shape == (3, 20, 40, 3) and np.max(np.abs(inv * sim.eps3 - 1)) < 1e-6
            assert sim.n_averaged > 0 and not np.allclose(sim.eps3[1], sim.eps3[2])
        else:
            assert inv.shape == (1, 20, 40, 3) and np.max(np.abs(inv[0] * sim.eps - 1)) < 1e-6
    # objects below the background (eps 11.9): the blend of fdtdx has a mask down to (1 - 11.9) / 1 = -10.9 for E_x and
    # E_y (step 1) and -5.45 for E_z (step 2), which increases its float32 rounding (bound below: 3e-5)
    geom = [fg.Block(size=fg.Vector3(fg.inf, 0.37, fg.inf), material=fg.Medium(index=2.0)),
            fg.Cylinder(radius=0.3, center=fg.Vector3(0.1, 0.3), material=fg.Medium(index=1.0))]
    sim = fg.Simulation(cell_size=fg.Vector3(1.0, 2.0, 0), resolution=20, geometry=geom,
                        boundary_layers=[fg.PML(0.5)], default_material=fg.Medium(index=3.45))
    sim.get_epsilon()
    assert np.isnan(fg.Simulation(cell_size=fg.Vector3(1, 1, 0), resolution=10).setup_seconds)
    import jax
    with jax.enable_x64(False):
        _, arrays, _, _ = sim._build(4, [])
    inv = np.asarray(arrays.inv_permittivities)
    assert abs(sim.eps3.min() - 1.0) < 1e-6 and abs(sim.eps3.max() - 3.45**2) < 1e-5
    assert np.max(np.abs(inv * sim.eps3 - 1)) < 3e-5, np.max(np.abs(inv * sim.eps3 - 1))




@pytest.mark.gpu
@pytest.mark.parametrize("e_normal", [False, True])
def test_slab_frequency_against_exact_mode(e_normal):
    """Periodic slab (width 0.37 um: faces between grid lines), res 40: the frequency of the guided mode with
    beta = 2 pi / 0.5 um against the exact slab mode with the exact Yee dispersion along x and in time. Averaged:
    below 1e-3; staircase: at least 4 x larger (bench: 2.3e-4 against 2.4e-3 (Ez), 2.0e-4 against 4.0e-3 (Hz))."""
    avg = cases.slab(resolution=40, e_normal=e_normal)
    stair = cases.slab(resolution=40, e_normal=e_normal, eps_averaging=False)
    assert abs(avg["rel_err"]) < 1e-3, avg
    assert abs(stair["rel_err"]) > 4 * abs(avg["rel_err"]), (avg["rel_err"], stair["rel_err"])


@pytest.mark.gpu
def test_bragg_faces_between_grid_lines():
    """Bragg stack for lam0 = 1.03 um (layer faces between grid lines) against exact transfer matrices: averaged,
    max |R - R_exact| falls at second order (res 80 to 160) and is far below the staircase, which stops converging
    (bench, res 40 / 80 / 160: averaged 0.114 / 0.0265 / 0.0062, staircase 0.257 / 0.049 / 0.047)."""
    a80, a160 = cases.bragg(resolution=80, lam0=1.03), cases.bragg(resolution=160, lam0=1.03)
    s160 = cases.bragg(resolution=160, lam0=1.03, eps_averaging=False)
    assert np.log2(a80["max_dR"] / a160["max_dR"]) > 1.8, (a80["max_dR"], a160["max_dR"])
    assert a160["max_dR"] < 1e-2 and s160["max_dR"] > 4 * a160["max_dR"], (a160["max_dR"], s160["max_dR"])
    for r in (a80, a160, s160):
        assert r["max_RT"] < 1e-3, r["max_RT"]


def _errs(r, ms):
    by_m = {m["m"]: m for m in r["modes"]}
    return np.array([abs(by_m[m]["rel_err"]) for m in ms]), np.array([abs(by_m[m]["Q"]) for m in ms])


@pytest.mark.gpu
def test_ring_tm_converges_at_second_order():
    """2-D ring, Ez (E parallel to the interfaces: the arithmetic mean): the resonance error against the exact Bessel
    solution falls at second order from res 40 to 80 (bench, m = 18: -0.255 %, -0.115 %, -0.063 % at res 40, 60, 80;
    staircase -0.496 %, -0.173 %, -0.111 %), and the rough staircase circle no longer limits Q (bench, res 40:
    1.2e5 to 1.4e7 averaged, 5.6e3 to 1.5e4 staircase)."""
    r40, r80 = cases.ring(resolution=40), cases.ring(resolution=80)
    ms = [m for m in (17, 18, 19, 20) if m in {q["m"] for q in r40["modes"]} & {q["m"] for q in r80["modes"]}]
    assert len(ms) >= 3, (r40["modes"], r80["modes"])
    e40, q40 = _errs(r40, ms)
    e80, _ = _errs(r80, ms)
    assert np.all(np.log2(e40 / e80) > 1.7), (ms, e40, e80)
    assert np.all(e80 < 1.2e-3) and np.all(q40 > 5e4), (e80, q40)


@pytest.mark.gpu
def test_disk_te_q_and_known_first_order_limit():
    """2-D disk, Hz (in-plane E; the normal turns through all angles). The diagonal rule drops the off-diagonal
    terms of the Kottke tensor, so the frequency error stays first order, of the size of the staircase error (bench,
    m = 8 to 10 at res 80: +0.067 % to +0.092 % averaged, +0.035 % to +0.115 % staircase). The averaged circle has no
    staircase roughness: Q of the m = 8, 9 modes at res 60 is 3.1e5 and 9.0e5 (staircase 9.3e3 and 1.6e4)."""
    a, s = cases.disk(resolution=60), cases.disk(resolution=60, eps_averaging=False)
    ms = [8, 9]
    ea, qa = _errs(a, ms)
    es, qs = _errs(s, ms)
    assert np.all(ea < 2e-3) and np.all(es < 2e-3), (ea, es)
    assert np.all(qa > 10 * qs), (qa, qs)
