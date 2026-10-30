"""compute_modes_gpu gives the meow modes of compute_modes_tidy3d: the same neff and (after meow's own
normalization and phase) the same fields."""
import meow as mw
import numpy as np
import pytest

from meow_gpu.meow_backend import GPUModeSolver, compute_modes_gpu


def strip_cross_section(width=0.5, wl=1.55, dx=0.04):
    oxide = mw.IndexMaterial(name="sio2", n=1.444)
    si = mw.IndexMaterial(name="si", n=3.476)
    clad = mw.Structure(material=oxide, geometry=mw.Box(x_min=-2, x_max=2, y_min=-2, y_max=2, z_min=-1, z_max=2))
    core = mw.Structure(material=si, mesh_order=1,
                        geometry=mw.Box(x_min=-width / 2, x_max=width / 2, y_min=0, y_max=0.22, z_min=-1, z_max=2))
    mesh = mw.Mesh2D(x=np.arange(-1.2, 1.2 + dx / 2, dx), y=np.arange(-0.7, 0.92 + dx / 2, dx))
    cell = mw.Cell(structures=[clad, core], mesh=mesh, z_min=0.0, z_max=1.0)
    return mw.CrossSection.from_cell(cell=cell, env=mw.Environment(wl=wl, T=25.0))


@pytest.mark.parametrize("precision,tol", [("mixed", 1e-8), ("single", 2e-5)])
def test_modes_match_tidy3d(precision, tol):
    cs = strip_cross_section()
    ref = mw.compute_modes_tidy3d(cs, num_modes=4)
    got = compute_modes_gpu(cs, num_modes=4, solver=GPUModeSolver(precision=precision, extra=4))
    assert len(got) == len(ref)
    for a, b in zip(got, ref):
        assert abs(a.neff - b.neff) < tol
        overlap = abs(mw.inner_product(a, b))      # both normalized to 1 by meow's post-processing
        assert overlap == pytest.approx(1.0, abs=10 * tol + 1e-6)


def test_solver_arrays_ignore_the_cpu_default_of_klujax():
    """klujax (imported by meow through sax) sets the JAX default platform to cpu; meow_gpu must still use the
    accelerator when there is one."""
    import jax
    import klujax  # noqa: F401
    from meow_gpu.device import solver_device
    from meow_gpu.fde.operator import make_problem
    from structures import strip
    try:
        gpu = jax.devices("gpu")[0]
    except RuntimeError:
        gpu = None
    p = make_problem(*strip(dx=0.1, dy=0.1)[:5], 1.55)
    leaves = jax.tree_util.tree_leaves(p.device())
    assert {d for a in leaves for d in a.devices()} == {solver_device()}
    if gpu is not None:
        assert solver_device() == gpu


def _below_cutoff_eme(solver):
    """A width step with more modes than the window guides: several modes are below cutoff (evanescent)."""
    oxide = mw.IndexMaterial(name="sio2", n=1.444)
    si = mw.IndexMaterial(name="si", n=3.476)
    clad = mw.Structure(material=oxide, geometry=mw.Box(x_min=-2, x_max=2, y_min=-2, y_max=2, z_min=-1, z_max=3))
    narrow = mw.Structure(material=si, mesh_order=1,
                          geometry=mw.Box(x_min=-0.2, x_max=0.2, y_min=0, y_max=0.22, z_min=-1, z_max=1))
    wide = mw.Structure(material=si, mesh_order=1,
                        geometry=mw.Box(x_min=-0.35, x_max=0.35, y_min=0, y_max=0.22, z_min=1, z_max=3))
    mesh = mw.Mesh2D(x=np.arange(-0.5, 0.5 + 0.02, 0.04), y=np.arange(-0.3, 0.52 + 0.02, 0.04))
    cells = mw.create_cells([clad, narrow, wide], mesh, np.array([1.0, 1.0]))
    css = [mw.CrossSection.from_cell(cell=c, env=mw.Environment(wl=1.55, T=25.0)) for c in cells]
    if solver is None:
        modes = [mw.compute_modes_tidy3d(cs, num_modes=16) for cs in css]
    else:
        modes = [compute_modes_gpu(cs, num_modes=16, solver=solver) for cs in css]
    S, pm = mw.compute_s_matrix(modes, cells=cells)
    return abs(np.asarray(S)[pm["left@0"], pm["right@0"]]), modes


def test_modes_below_cutoff_follow_the_tidy3d_direction_rule():
    """Review finding (2026-10-30): with modes below cutoff, a power-flow sign rule gave seed-dependent S21."""
    ref, ref_modes = _below_cutoff_eme(None)
    assert any(abs(np.imag(m.neff)) > 1e-6 for ms in ref_modes for m in ms), "the case needs evanescent modes"
    for seed in (0, 2):
        got, modes = _below_cutoff_eme(GPUModeSolver(precision="double", extra=4, seed=seed, warm_start=False))
        assert all(np.imag(m.neff) >= -1e-9 for ms in modes for m in ms), "no growing modes"
        assert got == pytest.approx(ref, abs=3e-6), seed   # before the fix: up to 3e-3, seed-dependent
