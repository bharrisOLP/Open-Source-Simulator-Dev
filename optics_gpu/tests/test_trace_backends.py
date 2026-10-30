"""Checks of the ray trace on Optiland's PyTorch backend (float64) against its NumPy backend.

The same rays (a grid of directions that overfills the stop, so that the physical aperture of the singlet clips a
part of them) go through the 4f system of two ideal lenses and the aberrated singlet. The positions, directions,
optical path (OPL) and intensity at the image surface must agree to TOL; the coupling efficiencies and the PSF of
tests/test_pop.py must not change. The "cuda" cases run only where PyTorch has a GPU (CUDA or ROCm).
"""
import warnings

import numpy as np
import optiland.backend as be
import pytest
import torch

from optics_gpu.pop import gaussian, optiland_torch, overlap_efficiency, propagate_through, trace_rays
from optics_gpu.systems import aberrated_singlet, four_f

warnings.filterwarnings("ignore", category=RuntimeWarning)
TOL = 1e-12      # mm (positions, OPL) and direction cosines; bench rows: GPU 0, CPU of the GPU computer 7e-14 mm
DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU"))]


def directions(half=0.09, n=101):
    s = np.linspace(-half, half, n)
    L, M = np.meshgrid(s, s, indexing="ij")
    return L.ravel(), M.ravel()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("make", [four_f, aberrated_singlet], ids=["4f", "singlet"])
def test_torch_trace_equals_numpy_trace(make, device):
    L, M = directions()
    ref = trace_rays(make(), L, M, 1.55)
    got = trace_rays(make(), L, M, 1.55, device=device)
    assert isinstance(got.x, torch.Tensor) and got.x.device.type == device and got.x.dtype == torch.float64
    assert be.get_backend() == "numpy"
    clipped = to_np(ref.i) == 0
    if make is aberrated_singlet:
        assert 0.1 < clipped.mean() < 0.9                       # the stop clips a part of the rays
    for name in ("x", "y", "z", "L", "M", "N", "opd", "i"):
        r, g = to_np(getattr(ref, name)), to_np(getattr(got, name))
        assert np.array_equal(np.isnan(r), np.isnan(g)), name
        fin = np.isfinite(r)
        assert np.max(np.abs(r[fin] - g[fin])) <= TOL, name


def to_np(a):
    return a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a)


@pytest.mark.parametrize("device", DEVICES)
def test_coupling_and_psf_do_not_change(device):
    """The low-angle 4f coupling check and the singlet PSF of tests/test_pop.py, traced by NumPy and by PyTorch."""
    xs = np.arange(-48, 48) * 0.5
    xi = np.arange(-64, 64) * 0.75
    src = gaussian(xs, xs, 6.0)
    a = propagate_through(four_f(), src, 0.5, 1.55, xi, xi, n_pad=4)
    b = propagate_through(four_f(), src, 0.5, 1.55, xi, xi, n_pad=4, trace_device=device)
    assert a.trace_device == "numpy" and b.trace_device.startswith(device) and a.n_rays == b.n_rays
    for wf in (15.0, 12.0, 19.5):
        fiber = gaussian(xi, xi, wf)
        assert overlap_efficiency(b.E, fiber) == pytest.approx(overlap_efficiency(a.E, fiber), abs=1e-12)
    pt = np.zeros((64, 64), complex)
    pt[32, 32] = 1.0
    xp = np.arange(-40, 40) * 1.0
    kw = {"na_cut": 0.5, "amp_cut": 0.0, "n_pad": 16}
    a = propagate_through(aberrated_singlet(), pt, 1.0, 1.55, xp, xp, **kw)
    b = propagate_through(aberrated_singlet(), pt, 1.0, 1.55, xp, xp, trace_device=device, **kw)
    assert a.n_rays == b.n_rays
    assert np.max(np.abs(b.E - a.E)) <= 1e-9 * np.max(np.abs(a.E))


def test_chunked_image_sum_equals_one_sum():
    xs = np.arange(-48, 48) * 0.5
    xi = np.arange(-64, 64) * 0.75
    src = gaussian(xs, xs, 6.0)
    one = propagate_through(four_f(), src, 0.5, 1.55, xi, xi, n_pad=4, ray_chunk=10**9)
    many = propagate_through(four_f(), src, 0.5, 1.55, xi, xi, n_pad=4, ray_chunk=401)
    assert one.n_rays > 10 * 401
    assert np.max(np.abs(many.E - one.E)) <= 1e-12 * np.max(np.abs(one.E))


def test_ray_chunk_below_one_is_refused():
    """A ray_chunk below 1 gave an empty loop and an all-zero field (the one review found it)."""
    xs = np.arange(-48, 48) * 0.5
    xi = np.arange(-64, 64) * 0.75
    for bad in (0, -1):
        with pytest.raises(ValueError, match="ray_chunk"):
            propagate_through(four_f(), gaussian(xs, xs, 6.0), 0.5, 1.55, xi, xi, n_pad=4, ray_chunk=bad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_image_sum_keeps_one_chunk_of_ex_and_ey_in_gpu_memory():
    """The peak GPU memory of the image sum is one chunk of ex and ey, (Nx + Ny) x ray_chunk complex128 values,
    and some small arrays (rays, weights, E); not 2x (the old ex and ey and the temporaries; the one review)."""
    xs = np.arange(-48, 48) * 0.5
    xi = np.arange(-64, 64) * 0.75
    src = gaussian(xs, xs, 6.0)
    chunk = 2048
    propagate_through(four_f(), src, 0.5, 1.55, xi, xi, n_pad=4, ray_chunk=chunk)     # warm-up (BLAS workspace)
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    img = propagate_through(four_f(), src, 0.5, 1.55, xi, xi, n_pad=4, ray_chunk=chunk)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    one_chunk = (xi.size + xi.size) * chunk * 16
    assert img.device.startswith("cuda") and img.n_rays > 2 * chunk
    assert one_chunk < peak < 1.3 * one_chunk


def test_float32_torch_optic_is_refused_with_the_cause():
    """An optic made under Optiland's PyTorch backend in float32 (its default) gives float32 check rays: the error
    must name the precision, not the dictionary (the one review found the wrong diagnosis)."""
    be.set_backend("torch")
    try:
        assert be.get_precision() == 32
        o = four_f()
        L, M = directions(n=11)
        with pytest.raises(ValueError, match="float32"):
            trace_rays(o, L, M, 1.55, device="cpu")
        assert be.get_backend() == "torch" and be.get_precision() == 32
    finally:
        be.set_backend("numpy")


def test_numpy_optic_traces_while_the_torch_backend_is_active():
    """A NumPy optic, with Optiland's PyTorch backend (float32) active: the check rays run on NumPy."""
    o = aberrated_singlet()
    L, M = directions(n=21)
    ref = trace_rays(o, L, M, 1.55)
    be.set_backend("torch")
    try:
        got = trace_rays(o, L, M, 1.55, device="cpu")
        assert be.get_backend() == "torch" and be.get_precision() == 32
    finally:
        be.set_backend("numpy")
    for name in ("x", "y", "L", "M", "opd", "i"):
        r, g = to_np(getattr(ref, name)), to_np(getattr(got, name))
        assert np.array_equal(np.isnan(r), np.isnan(g)), name
        fin = np.isfinite(r)
        assert np.max(np.abs(r[fin] - g[fin])) <= TOL, name


def test_backend_state_is_restored():
    """optiland_torch sets the earlier backend, device and precision again, also after an error in the block."""
    with optiland_torch("cpu"):
        assert be.get_backend() == "torch" and be.get_precision() == 64
    assert be.get_backend() == "numpy"
    be.set_backend("torch")
    try:
        assert be.get_precision() == 32 and be.get_device() == "cpu"           # Optiland's PyTorch defaults
    finally:
        be.set_backend("numpy")
    with pytest.raises(KeyError), optiland_torch("cpu"):
        raise KeyError("an error in the block")
    assert be.get_backend() == "numpy"
    with pytest.raises(ValueError, match="trace device"):
        trace_rays(four_f(), [0.0], [0.0], 1.55, device="mps")                   # no float64 on Apple GPUs
    assert be.get_backend() == "numpy"


def test_a_copy_that_traces_differently_is_refused(monkeypatch):
    """trace_rays compares the PyTorch copy of the optic with the original on check rays."""
    from optiland.optic import Optic

    real = Optic.from_dict

    def wrong(data):
        o = real(data)
        o.surfaces.surfaces[-1].geometry.cs.z = o.surfaces.surfaces[-1].geometry.cs.z + 0.01   # image 10 um off
        return o

    monkeypatch.setattr(Optic, "from_dict", staticmethod(wrong))
    L, M = directions(n=11)
    with pytest.raises(RuntimeError, match="traces differently"):
        trace_rays(four_f(), L, M, 1.55, device="cpu")
    assert be.get_backend() == "numpy"
