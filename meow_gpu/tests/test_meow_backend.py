"""compute_modes_gpu gives the meow modes of compute_modes_tidy3d: the same neff and (after meow's own
normalization and phase) the same fields. Both use the post-processing of meow_gpu, which keeps the neff of each
mode (meow 0.15 averages them in its Gram-Schmidt step)."""
import meow as mw
import numpy as np
import pytest

from meow_gpu.meow_backend import GPUModeSolver, compute_modes_gpu, compute_modes_gpu_batch, post_process_modes


def strip_cross_section(width=0.5, wl=1.55, dx=0.04, bend_radius=None, num_pml=(0, 0)):
    oxide = mw.IndexMaterial(name="sio2", n=1.444)
    si = mw.IndexMaterial(name="si", n=3.476)
    clad = mw.Structure(material=oxide, geometry=mw.Box(x_min=-2, x_max=2, y_min=-2, y_max=2, z_min=-1, z_max=2))
    core = mw.Structure(material=si, mesh_order=1,
                        geometry=mw.Box(x_min=-width / 2, x_max=width / 2, y_min=0, y_max=0.22, z_min=-1, z_max=2))
    extra = {} if bend_radius is None else {"bend_radius": bend_radius}
    mesh = mw.Mesh2D(x=np.arange(-1.2, 1.2 + dx / 2, dx), y=np.arange(-0.7, 0.92 + dx / 2, dx), num_pml=num_pml,
                     **extra)
    cell = mw.Cell(structures=[clad, core], mesh=mesh, z_min=0.0, z_max=1.0)
    return mw.CrossSection.from_cell(cell=cell, env=mw.Environment(wl=wl, T=25.0))


@pytest.mark.parametrize("precision,tol", [("mixed", 1e-8), ("single", 2e-5)])
def test_modes_match_tidy3d(precision, tol):
    cs = strip_cross_section()
    ref = mw.compute_modes_tidy3d(cs, num_modes=4, post_process=post_process_modes)
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


def _below_cutoff_eme(solver, batch=None):
    """A width step with more modes than the window guides: several modes are below cutoff (evanescent).
    ``batch``: solve the cells with ``compute_modes_gpu_batch``."""
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
        modes = [mw.compute_modes_tidy3d(cs, num_modes=16, post_process=post_process_modes) for cs in css]
    elif batch:
        modes = compute_modes_gpu_batch(css, num_modes=16, solver=solver, batch=batch)
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


def test_post_processing_keeps_the_neff_of_each_mode():
    """meow 0.15: Mode.__add__ averages neff, so its Gram-Schmidt gives the TM0 mode of this strip about 2.12 (the
    mean of TE0 and TM0). The post-processing of meow_gpu keeps the eigenvalue of each mode."""
    cs = strip_cross_section()
    raw = mw.compute_modes_tidy3d(cs, num_modes=4, post_process=lambda modes: modes)
    meow_pp = mw.compute_modes_tidy3d(cs, num_modes=4)
    ours = compute_modes_gpu(cs, num_modes=4, solver=GPUModeSolver(precision="double", extra=4))
    assert [complex(m.neff) for m in ours] == pytest.approx([complex(m.neff) for m in raw[: len(ours)]], abs=1e-9)
    assert abs(meow_pp[1].neff - raw[1].neff) > 0.1          # the upstream averaging (remove when meow fixes it)



def test_post_processing_equals_meow_except_the_neff():
    """The same fields as meow's own post-processing (filter, normalization, Gram-Schmidt); with PML (the filter
    drops PML modes) and with a duplicate mode (the Gram-Schmidt drop branch)."""
    from meow.fde.post_process import post_process_modes as meow_post_process

    for cs in (strip_cross_section(), strip_cross_section(num_pml=(5, 5))):
        raw = mw.compute_modes_tidy3d(cs, num_modes=8, post_process=lambda modes: modes)
        for modes in (raw, raw + raw[:1]):
            ours, theirs = post_process_modes(modes), meow_post_process(modes)
            assert len(ours) == len(theirs)
            for a, b in zip(ours, theirs):
                for c in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
                    assert np.allclose(np.asarray(getattr(a, c)), np.asarray(getattr(b, c)), rtol=0, atol=1e-12)
        assert len(post_process_modes(raw + raw[:1])) == len(post_process_modes(raw))   # the duplicate is dropped


def test_angled_propagation_is_not_supported():
    from types import SimpleNamespace

    from meow_gpu.meow_backend import _supported
    assert _supported(SimpleNamespace(mesh=SimpleNamespace(angle_theta=0.1))) == "angled propagation"
    assert _supported(strip_cross_section(bend_radius=20.0)) is None      # bends: on the GPU (tests/test_bend.py)


def test_the_tidy3d_fallback_keeps_the_neff(monkeypatch):
    """A cross-section that the GPU solver does not support falls back to tidy3d; the fallback uses the same
    post-processing. The real case (angled propagation) needs the tensorial solver of tidy3d-extras, so the test
    marks a bend as not supported (tidy3d solves bends)."""
    import meow_gpu.meow_backend as backend
    monkeypatch.setattr(backend, "_supported", lambda cs: "angled propagation")
    cs = strip_cross_section(bend_radius=20.0)
    with pytest.warns(UserWarning, match="angled propagation"):
        got = compute_modes_gpu(cs, num_modes=3)
    raw = mw.compute_modes_tidy3d(cs, num_modes=3, post_process=lambda modes: modes)
    assert [complex(m.neff) for m in got] == pytest.approx([complex(m.neff) for m in raw[: len(got)]], abs=1e-9)
