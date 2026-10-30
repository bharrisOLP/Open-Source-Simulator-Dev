"""Checks of image_planes, the batched image sum of the device pipeline (all planes of a focus scan in one pass
over the rays, in groups of GROUP_RAYS rays): against a direct NumPy sum of the plane waves, against the per-plane
image sum of the NumPy pipeline (_image_sum_direct), the complex64 option against complex128, and the GPU memory
of one chunk. The "cuda" cases run only where PyTorch has a GPU (CUDA or ROCm).
"""
import warnings

import numpy as np
import pytest
import torch

from optics_gpu import pop
from optics_gpu.pop import (RayBundle, _image_sum_direct, _planes_step, gaussian, image_planes,
                            image_sum, overlap_efficiency, overlap_efficiency_device, trace_bundle)
from optics_gpu.systems import aberrated_singlet, four_f

warnings.filterwarnings("ignore", category=RuntimeWarning)
GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
DEVICES = ["cpu", pytest.param("cuda", marks=GPU)]
WL = 1.55
K = 2 * np.pi / WL
XS = np.arange(-48, 48) * 0.5          # the source grid of the 4f checks (um)
XI = np.arange(-64, 64) * 0.75         # the image grid of the 4f checks (um)
PSF = {"na_cut": 0.5, "amp_cut": 0.0, "n_pad": 16}


def rel(a, b):
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def random_bundle(device, sizes=(1700, 1300), seed=3):
    """Rays with random directions (|L|, |M| < 0.3) and random complex weights, in blocks of the given sizes."""
    rng = np.random.default_rng(seed)
    blocks = []
    for n in sizes:
        L, M = rng.uniform(-0.3, 0.3, n), rng.uniform(-0.3, 0.3, n)
        a = rng.normal(size=n) + 1j * rng.normal(size=n)
        t = lambda v: torch.as_tensor(v, device=device)
        blocks.append((t(L), t(M), t(np.sqrt(1 - L**2 - M**2)), t(a)))
    return RayBundle(blocks, WL, sum(sizes), sum(sizes), torch.device(device), torch.zeros(()))


def gaussian_focus_bundle(device, theta=0.25, l_max=0.6, step=0.005):
    """A Gaussian beam focused at dz = 0 in image space: one ray per direction on a grid |L|, |M| <= l_max with the
    weight exp(-(L^2 + M^2) / theta^2) (a Gaussian waist of radius WL / (pi theta) = 2.0 um). The rays go up to
    the direction cosine 0.6, so that the grid phase k L (x - xc) is large on a wide grid."""
    v = np.arange(-l_max, l_max + 1e-12, step)
    L, M = (g.ravel() for g in np.meshgrid(v, v, indexing="ij"))
    a = np.exp(-(L**2 + M**2) / theta**2).astype(complex)
    t = lambda u: torch.as_tensor(u, device=device)
    return RayBundle([(t(L), t(M), t(np.sqrt(1 - L**2 - M**2)), t(a))], WL, L.size, L.size,
                     torch.device(device), torch.zeros(()))


def curved_gaussian_bundle(device):
    """The 4f image of a 6 um Gaussian with a curved wavefront (R 1.5 mm; the focus is off the image surface), as in
    tests/test_device_pipeline.py."""
    X, Y = np.meshgrid(XS, XS, indexing="ij")
    src = gaussian(XS, XS, 6.0) * np.exp(1j * K * (X**2 + Y**2) / (2 * 1500.0))
    return trace_bundle(four_f(), src, 0.5, WL, n_pad=4, device=device)


def psf_bundle(device):
    src = np.zeros((64, 64), complex)
    src[32, 32] = 1.0
    return trace_bundle(aberrated_singlet(), src, 1.0, WL, device=device, **PSF)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("ray_chunk", [None, 1000, 700])
@pytest.mark.parametrize("outputs", [pop.GROUP_OUTPUTS, 1], ids=["batched-groups", "group-loop"])
def test_planes_equal_a_direct_numpy_sum(device, ray_chunk, outputs, monkeypatch):
    """image_planes against the plane-wave sum E_z(x, y) = sum_r a_r exp(i k (L x + M y + N dz_z)) in NumPy (all
    rays, no chunks, no phase reference), for random rays in two blocks, a non-uniform x grid and a y grid that is
    not centered on 0. Groups of 256 rays (GROUP_RAYS patched) and chunks of 1000 and 700 rays give full groups
    and short last groups; GROUP_OUTPUTS 1 gives the path for large grids (complex128: addmm_ of the whole chunk,
    complex64: one product per group). Observed on the GPU computer (CPU and GPU): 4.1e-14 (complex128; the random
    sum has no focus, so its largest |E| is small) and 5.9e-7 (complex64)."""
    monkeypatch.setattr(pop, "GROUP_RAYS", 256)
    monkeypatch.setattr(pop, "GROUP_OUTPUTS", outputs)
    bundle = random_bundle(device)
    rng = np.random.default_rng(7)
    x = np.sort(rng.uniform(-20, 25, 23))
    y = 5.0 + np.arange(17) * 1.3
    dz = np.array([-50.0, 0.0, 30.0, 120.0])
    L, M, N, a = (np.concatenate([b[i].cpu().numpy() for b in bundle.blocks]) for i in range(4))
    ph = (K * (L[None, None, None] * x[None, :, None, None] + M[None, None, None] * y[None, None, :, None]
               + N[None, None, None] * dz[:, None, None, None]))
    ref = (a * np.exp(1j * ph)).sum(-1)                                          # (Z, Nx, Ny)
    got = image_planes(bundle, x, y, dz, ray_chunk=ray_chunk)
    assert got.shape == (4, 23, 17) and got.dtype == torch.complex128 and got.device.type == device
    assert rel(got.cpu().numpy(), ref) <= 1e-12
    for i, d in enumerate(dz):
        assert rel(image_sum(bundle, x, y, dz=d, ray_chunk=ray_chunk).cpu().numpy(), ref[i]) <= 1e-12
        assert rel(_image_sum_direct(bundle, x, y, dz=d).cpu().numpy(), ref[i]) <= 1e-12
    got32 = image_planes(bundle, x, y, dz, ray_chunk=ray_chunk, dtype=torch.complex64)
    assert got32.dtype == torch.complex128 and rel(got32.cpu().numpy(), ref) <= 1e-5


@pytest.mark.parametrize("device", DEVICES)
def test_batched_focus_scan_equals_per_plane_sums(device):
    """A focus scan of 21 planes (dz -80 to 80 um) in one pass (image_planes) against one image sum per plane with
    the image sum of the NumPy pipeline (_image_sum_direct: one matrix product per chunk, no phase reference), for
    the 4f image of a Gaussian with a curved wavefront and for the singlet PSF; and the coupling efficiency of the
    stack (overlap_efficiency_device on (Z, Nx, Ny)) against the overlap of each plane. Observed on the GPU computer:
    field 1.0e-14 (4f) and 1.3e-14 (singlet) relative to the largest |E| (CPU: 2.0e-15, 2.8e-15); eta 1.4e-14 or
    less."""
    dz = np.linspace(-80.0, 80.0, 21)
    for bundle, x in ((curved_gaussian_bundle(device), XI), (psf_bundle(device), np.arange(-40, 40) * 1.0)):
        stack = image_planes(bundle, x, x, dz)
        ref = torch.stack([_image_sum_direct(bundle, x, x, dz=d) for d in dz])
        assert rel(stack.cpu().numpy(), ref.cpu().numpy()) <= 1e-12
        fiber = torch.as_tensor(gaussian(x, x, 6.0), device=bundle.device)
        p0 = torch.vdot(ref[10].ravel(), ref[10].ravel()).real
        for power in (None, p0):
            eta = overlap_efficiency_device(stack, fiber, power=power).cpu().numpy()
            eta_ref = np.array([float(overlap_efficiency_device(e, fiber, power=power)) for e in ref])
            assert eta.shape == (21,) and np.max(np.abs(eta - eta_ref)) <= 1e-12
        assert float(overlap_efficiency_device(ref[0], fiber)) == pytest.approx(
            overlap_efficiency(ref[0].cpu().numpy(), gaussian(x, x, 6.0)), abs=1e-13)


def uniform_pupil_bundle(device, na=0.6, step=0.004):
    """A focus with a uniform pupil of NA 0.6 (equal weights up to the direction cosine 0.6): the rays at the
    largest angles carry as much field as the axial rays."""
    v = np.arange(-na, na + 1e-12, step)
    L, M = (g.ravel() for g in np.meshgrid(v, v, indexing="ij"))
    inside = L**2 + M**2 <= na**2
    L, M = L[inside], M[inside]
    t = lambda u: torch.as_tensor(u, device=device)  # noqa: E731
    return RayBundle([(t(L), t(M), t(np.sqrt(1 - L**2 - M**2)), t(np.ones(L.size, complex)))], WL, L.size, L.size,
                     torch.device(device), torch.zeros(()))


@pytest.mark.parametrize("device", DEVICES)
def test_complex64_coupling_is_within_1e_6_of_complex128(device):
    """The complex64 option (float32 grid phases and float32 sums of groups of GROUP_RAYS rays, float64 phase
    reference and plane phases) against complex128: the coupling efficiency into a Gaussian mode for each plane of
    a focus scan, normalized by the power of each plane and by the power at dz = 0, for the 4f image with a curved
    wavefront, the singlet PSF, a Gaussian focus (weights exp(-(L^2 + M^2) / 0.25^2), rays to 0.6) and a focus with
    a uniform pupil of NA 0.6 (equal weights to 0.6, 7.1e4 rays). The last two on a 201 x 201 um grid: the largest
    grid phase is 243 rad, near PHASE32_MAX; in the uniform pupil the rays at large phases carry their full weight.
    Observed on the GPU computer (largest |d eta| with the power of each plane and with the power at dz = 0;
    largest field difference relative to the largest |E|): 4f 3.7e-9, 8.3e-8, 1.5e-6; singlet 1.2e-8, 1.6e-8,
    1.1e-6; Gaussian focus 1.2e-8, 1.2e-8, 4.0e-7; uniform pupil 9.9e-8, 3.3e-7, 1.8e-6 (CPU 9.1e-9, 4.9e-8,
    4.9e-7). These values hold for these fields only (see test_complex64_error_of_one_plane_wave_is_within_the_bound).
    """
    cases = [(curved_gaussian_bundle(device), XI, 15.0, np.linspace(-80.0, 80.0, 9)),
             (psf_bundle(device), np.arange(-40, 40) * 1.0, 6.0, np.linspace(-60.0, 60.0, 9)),
             (gaussian_focus_bundle(device), np.arange(-100, 101) * 1.0, 2.0, np.linspace(-100.0, 100.0, 9)),
             (uniform_pupil_bundle(device), np.arange(-100, 101) * 1.0, 1.5, np.array([-10.0, 0.0, 10.0]))]
    for bundle, x, w, dz in cases:
        e64 = image_planes(bundle, x, x, dz)
        e32 = image_planes(bundle, x, x, dz, dtype=torch.complex64)
        assert rel(e32.cpu().numpy(), e64.cpu().numpy()) <= 2e-5
        mode = torch.as_tensor(gaussian(x, x, w), device=bundle.device)
        p0 = torch.vdot(e64[len(dz) // 2].ravel(), e64[len(dz) // 2].ravel()).real
        for power in (None, p0):
            d = (overlap_efficiency_device(e32, mode, power) - overlap_efficiency_device(e64, mode, power)).abs()
            assert float(d.max()) <= 1e-6, (x.size, float(d.max()))
        if bundle.n_rays == 58081:                                                # the Gaussian focus
            eta = overlap_efficiency_device(e64, mode).cpu().numpy()
            assert eta[len(dz) // 2] > 0.999 and eta[0] < 0.05                     # a real focus scan
    assert bundle.n_rays > 7e4 and float(overlap_efficiency_device(e64, mode)[1]) > 0.5   # the uniform pupil


@pytest.mark.parametrize("device", DEVICES)
def test_complex64_error_of_one_plane_wave_is_within_the_bound(device):
    """One plane wave (2e4 equal rays, L' 0.5, M' 0.3) on a grid of +-120 um: the grid phase is 243 rad, and all
    rays have the same rounding errors, so the float32 errors add with one sign (the worst case of the tested
    fields). The field error of complex64 must stay below the bound of the image_planes docstring,
    (GROUP_RAYS + 6 PHASE32_MAX) 2^-24 sum_r |a_r| = 1.5e-4 sum_r |a_r|. Observed on the GPU computer: 3.3e-5 of
    sum_r |a_r| (CPU 2.1e-5)."""
    n = 20000
    t = lambda v: torch.as_tensor(np.full(n, v), device=device)  # noqa: E731
    bundle = RayBundle([(t(0.5), t(0.3), t(np.sqrt(1 - 0.34)), t(1.0 + 0j))], WL, n, n, torch.device(device),
                       torch.zeros(()))
    x, y = np.arange(-120, 121) * 1.0, np.array([-120.0, -60.0, 0.0, 60.0, 120.0])
    e64 = image_planes(bundle, x, y, [0.0, 50.0])
    e32 = image_planes(bundle, x, y, [0.0, 50.0], dtype=torch.complex64)
    bound = (pop.GROUP_RAYS + 6 * pop.PHASE32_MAX) * 2.0**-24
    err = float((e32 - e64).abs().max()) / n
    assert float(e64.abs().max()) == pytest.approx(n, rel=1e-12) and err <= bound, (err, bound)


@pytest.mark.parametrize("device", DEVICES)
def test_complex64_refuses_large_grid_phases(device):
    """complex64 stops with a ValueError when the grid phase k max|L'| max|x - xc| is above PHASE32_MAX (Gaussian
    focus, rays up to 0.6, grid +-130 um: 316 rad); complex128 has no such limit."""
    bundle = gaussian_focus_bundle(device, step=0.02)
    x = np.arange(-130, 131) * 1.0
    with pytest.raises(ValueError, match="complex64"):
        image_planes(bundle, x, x, [0.0], dtype=torch.complex64)
    assert torch.isfinite(image_planes(bundle, x, x, [0.0]).abs()).all()


C128, C64 = torch.complex128, torch.complex64
MEMORY_CASES = [          # (planes, Nx, Ny, dtype, SUM_ELEMENTS): the cases of the one review and the two paths
    (41, 128, 128, C128, 1 << 22), (41, 128, 128, C64, 1 << 22),        # batched groups, chunk below GROUP_RAYS
    (21, 128, 128, C128, 1 << 25), (21, 128, 128, C64, 1 << 25),        # batched groups
    (9, 512, 256, C128, 1 << 22), (9, 512, 256, C64, 1 << 22),          # large grid: addmm_, one product per group
    (41, 512, 512, C64, 1 << 25), (101, 512, 512, C128, 1 << 25),       # the review: 1.13 and 1.45 before the fix
    (1000, 1, 1, C64, 1 << 25),                                          # an axial scan (the review: 1.28)
]


def _case_id(c):
    return f"{c[0]}x{c[1]}x{c[2]}-{str(c[3]).split('.')[-1]}-2^{c[4].bit_length() - 1}"


@GPU
@pytest.mark.parametrize("case", MEMORY_CASES, ids=_case_id)
def test_image_planes_memory_is_within_the_budget(case, monkeypatch):
    """The peak GPU memory of image_planes with the default chunk, minus the result E, is at most SUM_ELEMENTS
    complex128 values (no tolerance), and at least half of it (the model of _planes_memory is not too large).
    Several chunks of random rays; SUM_ELEMENTS is patched smaller for some cases, so that they run in a short
    time. Observed on the GPU computer (peak minus E over the budget, in the order of MEMORY_CASES): 0.67, 0.63,
    0.97, 0.95, 0.91, 0.74, 0.82, 0.99, 0.96."""
    nz, nx, ny, dtype, budget = case
    monkeypatch.setattr(pop, "SUM_ELEMENTS", budget)
    step = _planes_step(None, nz, nx, ny, dtype)
    bundle = random_bundle("cuda", sizes=(max(3 * step, 3000),))
    x, y, dz = np.arange(nx) * 0.5, np.arange(ny) * 0.5, np.linspace(-50.0, 50.0, nz)
    image_planes(bundle, x, y, dz, dtype=dtype)                                 # warm-up (BLAS workspace)
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    E = image_planes(bundle, x, y, dz, dtype=dtype)
    torch.cuda.synchronize()
    ratio = (torch.cuda.max_memory_allocated() - base - E.numel() * 16) / (16 * budget)
    assert 0.5 <= ratio <= 1.0, ratio


def test_image_planes_refuses_bad_inputs():
    bundle = random_bundle("cpu", sizes=(50,))
    x = np.arange(-4, 4) * 1.0
    with pytest.raises(ValueError, match="dtype"):
        image_planes(bundle, x, x, [0.0], dtype=torch.float64)
    with pytest.raises(ValueError, match="ray_chunk"):
        image_planes(bundle, x, x, [0.0], ray_chunk=0)
    X, Y = np.meshgrid(x, x, indexing="ij")
    for args in ((X, x, [0.0]), (x, Y, [0.0]), (x, x, np.zeros((2, 2)))):        # the earlier code failed, too
        with pytest.raises(ValueError, match="1-D"):
            image_planes(bundle, *args)
    with pytest.raises(ValueError, match="1-D"):
        image_sum(bundle, X, Y)
    assert image_planes(bundle, x, x, []).shape == (0, 8, 8)
    assert image_planes(bundle, x, x, 5.0).shape == (1, 8, 8)
    one = image_planes(bundle, x, x, [5.0])[0]
    for dz in (5.0, np.float64(5.0), torch.tensor(5.0, dtype=torch.float64)):
        assert torch.equal(image_sum(bundle, x, x, dz=dz), one)
    with pytest.raises(RuntimeError):
        image_sum(bundle, x, x, dz=torch.tensor([1.0, 2.0]))                 # one plane only
