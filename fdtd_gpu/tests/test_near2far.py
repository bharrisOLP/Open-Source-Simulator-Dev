"""Near-to-far-field transform: the transform of exact dipole and line-source fields on a closed surface (no FDTD),
the API, and FDTD checks against the analytic far fields, the flux and the analytic power (cases.transform_exact,
cases.near2far_dipole, cases.near2far_line)."""

import numpy as np
import pytest

import cases
import fdtd_gpu as fg
from fdtd_gpu.near2far import farfield_grid


@pytest.mark.parametrize("dims,component", [(3, "Ez"), (3, "Ex"), (2, "Ez"), (2, "Ex")])
def test_transform_of_exact_fields(dims, component):
    """The exact fields of a point dipole (3-D, Balanis form) or of a line current (2-D, Hankel form) on a closed
    surface give back the exact fields outside it (at 1000 um and 3 um; second order in the spacing of the surface
    points) and close to zero inside it (extinction theorem). 3-D: the JAX (GPU, float64) and NumPy sums agree."""
    r1 = cases.transform_exact(dims, component, h=0.1)
    r2 = cases.transform_exact(dims, component, h=0.05, compare_backends=True)
    assert max(r2["far_E"], r2["far_H"]) < 3e-3 and r2["near_E"] < 4e-3, r2
    assert np.log2(r1["far_E"] / r2["far_E"]) > 1.9 and np.log2(r1["near_E"] / r2["near_E"]) > 1.9, (r1, r2)
    assert r2["inside"] < 2e-3, r2
    if dims == 3:
        assert r2["jax_vs_numpy"] < 1e-11, r2["jax_vs_numpy"]


def test_farfield_grid_and_errors():
    """The points of get_farfields (floor(size x resolution) per axis, ends included, one point at the center below
    2), the trapezoidal weights, and the unsupported cases."""
    pts, shape, w = farfield_grid(fg.Vector3(1, 2, 3), fg.Vector3(2, 0.1, 0), 5, (0, 1, 2))
    assert shape == (10,) and pts.shape == (10, 3)
    assert np.allclose(pts[:, 0], np.linspace(0, 2, 10)) and np.all(pts[:, 1] == 2) and np.all(pts[:, 2] == 3)
    assert abs(w.sum() - 2.0) < 1e-12
    pts, shape, w = farfield_grid(fg.Vector3(), fg.Vector3(1, 2, 7), 4, (0, 1))      # 2-D: z is not an axis
    assert shape == (4, 8) and np.all(pts[:, 2] == 0) and abs(w.sum() - 2.0) < 1e-12
    assert np.allclose(pts.reshape(4, 8, 3)[:, 0, 0], np.linspace(-0.5, 0.5, 4))
    sim1 = fg.Simulation(cell_size=fg.Vector3(0, 0, 4), resolution=10, boundary_layers=[fg.PML(1)])
    with pytest.raises(NotImplementedError):
        sim1.add_near2far(0.5, 0.1, 3, fg.Near2FarRegion(center=fg.Vector3(0, 0, 1)))
    sim = fg.Simulation(cell_size=fg.Vector3(4, 4, 0), resolution=10, boundary_layers=[fg.PML(1)])
    reg = fg.Near2FarRegion(center=fg.Vector3(1), size=fg.Vector3(0, 2))
    with pytest.raises(NotImplementedError):
        sim.add_near2far(0.5, 0.1, 3, reg, nperiods=2)
    for d in (fg.Z, 5):                                       # a normal along the axis of size 0, or no axis
        with pytest.raises(ValueError, match="normal along an axis of size > 0"):
            sim.add_near2far(0.5, 0.1, 3, reg, fg.Near2FarRegion(center=fg.Vector3(1), size=fg.Vector3(0, 2),
                                                                 direction=d))
    assert sim._monitors == []                                # nothing is added on an error
    n2f = sim.add_near2far(0.5, 0.1, 3, reg)
    assert isinstance(n2f, fg.DftNear2Far) and np.allclose(sim.get_near2far_freqs(n2f), [0.45, 0.5, 0.55])
    with pytest.raises(RuntimeError):
        sim.get_farfield(n2f, fg.Vector3(10))


def _worst(r):
    """The largest error of each FDTD check over both boxes and all frequencies."""
    return dict(flux=np.max(np.abs(r["far_over_flux"] - 1)), field=np.max(r["field_err"]),
                box=np.max(r["box_diff"]))


def _check_fdtd(r, other, three_d):
    """The bounds of the FDTD checks at res 20 (n = 1; run r) and the order of convergence between r and a run at
    another resolution. Measured values: README, check 9."""
    assert r["resolution"] == 20 and other["resolution"] != 20
    assert np.max(np.abs(r["far_over_flux"] - 1)) < 6e-3, r["far_over_flux"]          # far-field power = flux
    assert np.max(np.abs(r["far_over_exact"] - 1)) < 1e-2, r["far_over_exact"]        # against the analytic power
    assert np.max(r["pattern_err"]) < 1.5e-2, r["pattern_err"]
    assert np.max(r["field_err"]) < 7e-3, r["field_err"]                              # complex, one global phase
    assert np.max(r["near_err"]) < 7e-3, r["near_err"]
    assert np.max(r["box_diff"]) < 7e-3, r["box_diff"]                                # two box sizes
    assert np.max(np.abs(r["phase"] + r["omega_dt_half"])) < 4e-3, (r["phase"], r["omega_dt_half"])
    key = "cube_over_far" if three_d else "square_over_far"
    assert np.max(np.abs(r[key] - 1)) < 1e-3, r[key]                                   # DftNear2Far.flux
    assert r["eps"] == r["n"] ** 2
    lo, hi = sorted((r, other), key=lambda q: q["resolution"])
    w1, w2 = _worst(lo), _worst(hi)
    for k in w1:                                                   # the FDTD discretization: second order
        order = np.log(w1[k] / w2[k]) / np.log(hi["resolution"] / lo["resolution"])
        assert order > 1.8, (k, order, w1, w2)


@pytest.mark.gpu
def test_near2far_dipole_3d():
    """3-D Ez dipole: far fields from two closed boxes against the analytic dipole fields, the flux and the analytic
    power, at res 10 and 20; the API (get_farfield, get_farfields, DftNear2Far.flux)."""
    r = cases.near2far_dipole(resolution=20)
    assert r["device"].startswith(("rocm", "cuda", "gpu"))
    _check_fdtd(r, cases.near2far_dipole(resolution=10), True)
    assert r["api_point"] < 1e-12 and r["plane_shape"] == [20, 10, 5], (r["api_point"], r["plane_shape"])


@pytest.mark.gpu
@pytest.mark.parametrize("component", ["Ez", "Ex"])
def test_near2far_line_2d(component):
    """2-D line source with E normal to (Ez) or in (Ex) the plane: far fields (Hankel form) from two closed squares
    against the analytic fields, the flux and the analytic power, at res 20 and 40."""
    _check_fdtd(cases.near2far_line(resolution=20, component=component),
                cases.near2far_line(resolution=40, component=component), False)


@pytest.mark.gpu
def test_near2far_minus_data_and_medium_warning():
    """load_minus_near2far_data subtracts the DFT fields of another run (here a second, equal run: zero far fields;
    test_near2far_subtract_incident_fields subtracts fields that differ); a surface that crosses a second material
    gives a warning, and the far fields use the material of most surface cells."""
    def run(minus=None):
        gs = fg.GaussianSource(0.5, fwidth=0.2)
        block = fg.Block(size=fg.Vector3(0.4, 0.4, fg.inf), center=fg.Vector3(1.0, 0), material=fg.Medium(index=2))
        sim = fg.Simulation(cell_size=fg.Vector3(5, 5, 0), resolution=10, geometry=[block],
                            sources=[fg.Source(gs, fg.Ez)], boundary_layers=[fg.PML(1.0)])
        n2f = sim.add_near2far(0.5, 0.2, 3, *cases.box_regions(fg.Near2FarRegion, 1.0, (0, 1)))
        if minus is not None:
            sim.load_minus_near2far_data(n2f, minus)
        sim.run(until_after_sources=20)
        with pytest.warns(UserWarning, match="crosses 2 materials"):
            ff = sim.get_farfields(n2f, 1, center=fg.Vector3(0, 100), size=fg.Vector3(4, 0))
        return sim, n2f, ff

    sim, n2f, ff = run()
    assert n2f.eps == 1.0 and ff["Ez"].shape == (4, 3) and np.abs(ff["Ez"]).min() > 0
    _, _, ff0 = run(sim.get_near2far_data(n2f))
    assert np.abs(ff0["Ez"]).max() < 1e-6 * np.abs(ff["Ez"]).max(), np.abs(ff0["Ez"]).max()


@pytest.mark.gpu
def test_near2far_close_to_the_surface():
    """3-D Ez dipole, res 20: the error of the transform at 5 and 10 cells outside the box of the DFT points (both
    from the FDTD fields and, the quadrature alone, from the exact fields on the same points). README: keep the
    points at least 5 cells away."""
    r = cases.near2far_distance(resolution=20, cells=(1, 5, 10))
    assert r["surface_points_match"], r
    for k in ("err_E", "err_H"):
        assert max(r[k][1:]) < 1e-2, r                      # measured: 4.6e-3 and 4.0e-3 (E)
    for k in ("exact_E", "exact_H"):
        assert max(r[k][1:]) < 4e-3, r                      # measured: 2.1e-3 (H, 5 cells)
    assert min(r["err_E"][0], r["exact_E"][0]) > 5e-2, r      # 1 cell: the quadrature fails (0.13)


@pytest.mark.gpu
def test_near2far_subtract_incident_fields():
    """2-D: a source outside the closed square (run A) and the same source plus one inside it (run B). The far fields
    of B - A equal the field of the inner source; the transform of A is close to zero outside the square and close to
    minus the field of the outer source inside it (res 20). At the inner points the transform of B - A is close to
    zero and that of B close to the size of the outer field: a subtraction that does nothing, or that also changes
    the object without load_minus_near2far_data, fails by a factor of about 100."""
    r = cases.near2far_subtract(resolution=20)
    assert np.max(r["err_sub"]) < 3e-3, r["err_sub"]                       # measured 1.3e-3
    assert np.max(r["err_plain"]) < 1.5e-2, r["err_plain"]                 # 6.5e-3: plus the residual of A
    assert np.max(r["outside_source_far"]) < 1.5e-2, r["outside_source_far"]   # 6.8e-3
    assert np.max(r["inside_err"]) < 7e-3, r["inside_err"]                 # 3.3e-3
    assert np.max(r["inside_sub"]) < 1e-2, r["inside_sub"]                 # B - A: only S1 (inside): zero
    assert np.min(r["inside_plain"]) > 0.9, r["inside_plain"]              # B: the -S2 part is left
