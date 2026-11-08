"""Checks of the non-uniform FFT of the image sum (image_planes and image_sum with method "nufft"): against a direct
NumPy sum of the plane waves (random rays, one ray, a grid that does not resolve the rays), against the direct sum
of image_planes for the 4f, singlet and focus cases (focus scans of 41 planes) and for the coupling into a fiber,
the fallback for a non-uniform grid, the batches of planes and the copies of the fine grid, bad inputs, and the GPU
memory. The "cuda" cases run only where PyTorch has a GPU (CUDA or ROCm).

The error measure of the non-uniform FFT is max |E_nufft - E| / sum_r |a_r| (the field of one ray has the size
|a_r| everywhere, so this is the error of the worst ray). Relative to the largest |E|, the error is larger by
sum_r |a_r| / max |E| (1 for a focus of rays in phase, more for random rays or a plane out of focus).
"""
import warnings

import numpy as np
import pytest
import torch

from optics_gpu import pop
from optics_gpu.pop import (RayBundle, _nufft_steps, gaussian, image_planes, image_sum, nufft_width,
                            overlap_efficiency_device, trace_bundle)
from optics_gpu.systems import aberrated_singlet, four_f

warnings.filterwarnings("ignore", category=RuntimeWarning)
GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
DEVICES = ["cpu", pytest.param("cuda", marks=GPU)]
WL = 1.55
K = 2 * np.pi / WL
XS = np.arange(-48, 48) * 0.5          # the source grid of the 4f checks (um)
XI = np.arange(-64, 64) * 0.75         # the image grid of the 4f checks (um)
PSF = {"na_cut": 0.5, "amp_cut": 0.0, "n_pad": 16}
TOLS = [1e-2, 1e-6, 1e-9, 1e-12]


def bundle_of(device, L, M, a):
    t = lambda v: torch.as_tensor(np.asarray(v), device=device)  # noqa: E731
    return RayBundle([(t(L), t(M), t(np.sqrt(1 - L**2 - M**2)), t(a))], WL, L.size, L.size, torch.device(device),
                     torch.zeros(()))


def random_bundle(device, sizes=(1700, 1300), l_max=0.3, seed=3):
    """Rays with random directions (|L|, |M| < l_max) and random complex weights, in blocks of the given sizes."""
    rng = np.random.default_rng(seed)
    blocks = []
    for n in sizes:
        L, M = rng.uniform(-l_max, l_max, n), rng.uniform(-l_max, l_max, n)
        a = rng.normal(size=n) + 1j * rng.normal(size=n)
        blocks.append(bundle_of(device, L, M, a).blocks[0])
    return RayBundle(blocks, WL, sum(sizes), sum(sizes), torch.device(device), torch.zeros(()))


def rays_of(bundle):
    return (np.concatenate([b[i].cpu().numpy() for b in bundle.blocks]) for i in range(4))


def numpy_planes(bundle, x, y, dz):
    """E_z(x, y) = sum_r a_r exp(i k (L x + M y + N dz_z)) in NumPy: all rays at one time, no phase reference."""
    L, M, N, a = rays_of(bundle)
    ph = K * (L * x[:, None, None] + M * y[None, :, None])                       # (Nx, Ny, R)
    return np.stack([(a * np.exp(1j * (ph + K * N * d))).sum(-1) for d in np.atleast_1d(dz)])


def sum_abs(bundle):
    return float(sum(b[3].abs().sum() for b in bundle.blocks))


def curved_gaussian_bundle(device):
    """The 4f image of a 6 um Gaussian with a curved wavefront (R 1.5 mm), as in tests/test_image_planes.py."""
    X, Y = np.meshgrid(XS, XS, indexing="ij")
    src = gaussian(XS, XS, 6.0) * np.exp(1j * K * (X**2 + Y**2) / (2 * 1500.0))
    return trace_bundle(four_f(), src, 0.5, WL, n_pad=4, device=device)


def psf_bundle(device):
    src = np.zeros((64, 64), complex)
    src[32, 32] = 1.0
    return trace_bundle(aberrated_singlet(), src, 1.0, WL, device=device, **PSF)


def grid_bundle(device, l_max, step, theta=None):
    """One ray per direction on a grid |L|, |M| <= l_max: a Gaussian focus (weights exp(-(L^2 + M^2) / theta^2))
    or, with theta None, a uniform pupil of NA l_max (equal weights inside the circle)."""
    v = np.arange(-l_max, l_max + 1e-12, step)
    L, M = (g.ravel() for g in np.meshgrid(v, v, indexing="ij"))
    if theta is None:
        inside = L**2 + M**2 <= l_max**2
        L, M = L[inside], M[inside]
        a = np.ones(L.size, complex)
    else:
        a = np.exp(-(L**2 + M**2) / theta**2).astype(complex)
    return bundle_of(device, L, M, a)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("tol", TOLS)
@pytest.mark.parametrize("l_max, h", [(0.3, 0.9), (0.6, 2.0)], ids=["resolved", "aliased"])
def test_nufft_equals_a_direct_numpy_sum(device, tol, l_max, h):
    """image_planes(method="nufft") against the NumPy plane-wave sum for random rays in two blocks, 4 planes, an x
    grid of 24 points and a y grid of 17 points (both uniform, with other steps and not centered on 0), in one chunk
    and in chunks of 500 rays: max |dE| / sum_r |a_r| <= tol. "aliased": rays to 0.6 on a 2 um grid, up to 0.77
    cycles per grid step (above the Nyquist limit 0.5), so the rays wrap around the fine grid; on the grid points
    the field is the same. Observed on the GPU computer (resolved, aliased; the same on CPU and GPU, and in one
    chunk and in chunks): tol 1e-2: 2.2e-5, 1.5e-5; 1e-6: 2.7e-9, 1.8e-9; 1e-9: 4.4e-12, 3.0e-12; 1e-12: 5.4e-15,
    4.3e-15."""
    bundle = random_bundle(device, l_max=l_max)
    x = -7.3 + np.arange(24) * h
    y = 5.0 + np.arange(17) * 1.3 * h
    dz = np.array([-50.0, 0.0, 30.0, 120.0])
    ref = numpy_planes(bundle, x, y, dz)
    for ray_chunk in (None, 500):
        got = image_planes(bundle, x, y, dz, ray_chunk=ray_chunk, method="nufft", tol=tol)
        assert got.shape == (4, 24, 17) and got.dtype == torch.complex128 and got.device.type == device
        err = float(np.max(np.abs(got.cpu().numpy() - ref))) / sum_abs(bundle)
        print("observed", device, tol, l_max, ray_chunk, f"{err:.1e}")
        assert err <= tol, (tol, ray_chunk, err)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("tol", TOLS)
def test_nufft_of_one_ray(device, tol):
    """One ray (|a| = 1) in 20 random directions with |L|, |M| < 0.6 on a 101 x 101 grid of pitch 0.5 um:
    max |dE| <= tol. Observed on the GPU computer (largest of the 20; the same on CPU and GPU): tol 1e-2: 2.4e-4;
    1e-6: 3.7e-8; 1e-9: 9.4e-11; 1e-12: 1.3e-13. The worst position: test_nufft_of_one_ray_at_the_worst_position."""
    rng = np.random.default_rng(1)
    x = np.arange(-50, 51) * 0.5
    worst = 0.0
    for _ in range(20):
        L, M = rng.uniform(-0.6, 0.6, 2)
        bundle = bundle_of(device, np.array([L]), np.array([M]), np.array([1.0 + 0j]))
        ref = np.exp(1j * K * (L * x[:, None] + M * x[None, :]))
        got = image_sum(bundle, x, x, method="nufft", tol=tol).cpu().numpy()
        worst = max(worst, float(np.max(np.abs(got - ref))))
    print("observed", device, tol, f"{worst:.1e}")
    assert worst <= tol, (tol, worst)


WORST_X = (np.arange(128) - 64) * 2.0      # 128 points of pitch 2 um: the fine grid is exactly 2 Nx = 256 points


def one_ray_error(device, f, tol, width=None, monkeypatch=None):
    """max |dE| of image_sum(method="nufft") for one ray (|a| = 1) with f = L h / lambda = M h / lambda cycles per
    grid step on WORST_X x WORST_X; width: a kernel width instead of nufft_width(tol) (beta = 2.30 w)."""
    L = f * WL / 2.0
    bundle = bundle_of(device, np.array([L]), np.array([L]), np.array([1.0 + 0j]))
    ref = np.exp(1j * K * L * (WORST_X[:, None] + WORST_X[None, :]))
    if width is not None:
        monkeypatch.setattr(pop, "nufft_width", lambda t: (width, pop.NUFFT_BETA * width))
    got = image_sum(bundle, WORST_X, WORST_X, method="nufft", tol=tol).cpu().numpy()
    return float(np.max(np.abs(got - ref)))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("tol", TOLS)
def test_nufft_of_one_ray_at_the_worst_position(device, tol, monkeypatch):
    """The worst case of the error per ray: one ray on a 128 x 128 grid (pitch 2 um), so that the fine grid is
    exactly 2 Nx and the edge modes reach |m| / n1 = 1/4, with L = M, so that the x and y errors add at a corner of
    the grid. The 1-D error of one ray depends only on the position of the ray in its fine-grid cell (a shift by one
    cell multiplies the spread grid and the exact field by the same phase), so 64 positions in one cell (f = (-41 +
    j / 64) / 256, near the worst f -0.16 of the review) and the two worst directions of the review (L = M =
    -0.12671 and -0.28210) cover it. max |dE| <= tol with nufft_width. For tol 1e-9 and 1e-12, the two narrower
    widths (the width of Barnett et al., ceil(log10(1 / tol)) + 1, and that plus 1) are above tol. Observed on the
    GPU computer (worst / tol, the same on CPU and GPU): tol 1e-2 (w 5): 0.08; 1e-6 (w 9): 0.10; 1e-9 (w 12): 0.16,
    with w 11: 1.68 and w 10: 14.58; 1e-12 (w 15): 0.26, with w 14: 1.89 and w 13: 14.31."""
    fs = list((-41 + np.arange(64) / 64) / 256) + [-0.12671 * 2.0 / WL, -0.28210 * 2.0 / WL]
    w = nufft_width(tol)[0]
    worst = max(one_ray_error(device, f, tol) for f in fs)
    narrow = {}
    if tol <= 1e-9:
        for dw in (1, 2):
            with monkeypatch.context() as m:
                narrow[w - dw] = max(one_ray_error(device, f, tol, w - dw, m) for f in fs)
    print("observed", device, tol, w, f"{worst / tol:.2f}", {k: f"{v / tol:.2f}" for k, v in narrow.items()})
    assert worst <= tol, (tol, worst)
    assert all(v > tol for v in narrow.values()), narrow


def test_kernel_transform_is_exact():
    """_es_transform (Gauss-Legendre in t, z = sin t, NUFFT_QUAD nodes) against mpmath (tanh-sinh quadrature of the
    integral in z, 30 digits; mpmath comes with torch through sympy), at the modes m = 0, 16, 32, 48, 64 of a fine
    grid of 256 points (|m| <= n / 4), for the widths of tol 1e-2 to 1e-12: the largest relative difference is at
    most 1e-14. The rule in z with 128 nodes (before the one review) is above 1e-12 at w 5 (tol 1e-2): the
    square-root ends of phi. Observed on the GPU computer: 6.4e-15; the rule in z: 8.6e-11 (w 5), 2.7e-14 (w 9),
    6.6e-15 (w 12), 7.0e-15 (w 15)."""
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 30
    m = np.array([0, 16, 32, 48, 64])
    z, wq = np.polynomial.legendre.leggauss(pop.NUFFT_QUAD)
    worst, z_rule = 0.0, {}
    for tol in TOLS:
        w, beta = nufft_width(tol)
        ref = np.array([float(0.5 * w * mp.quad(lambda z, k=int(k): mp.exp(beta * (mp.sqrt(1 - z * z) - 1))
                                                 * mp.cos(mp.pi * k * w * z / 256), [-1, 0, 1])) for k in m])
        worst = max(worst, float(np.max(np.abs(pop._es_transform(m, 256, w, beta) / ref - 1))))
        in_z = 0.5 * w * (np.cos(np.outer(np.pi * m * w / 256, z)) @ (wq * np.exp(beta * (np.sqrt(1 - z**2) - 1))))
        z_rule[w] = float(np.max(np.abs(in_z / ref - 1)))
    print("observed", f"{worst:.1e}", {k: f"{v:.1e}" for k, v in z_rule.items()})
    assert worst <= 1e-14 and z_rule[5] > 1e-12


def optical_cases(device):
    """(name, bundle, grid, mode radius, planes): focus scans of 41 planes for the 4f image with a curved wavefront
    and the singlet PSF; 9 planes for a Gaussian focus with rays to 0.6; 3 planes for a uniform pupil of NA 0.6
    (the last two on a 201 x 201 um grid)."""
    return [("4f", curved_gaussian_bundle(device), XI, 15.0, np.linspace(-100.0, 100.0, 41)),
            ("singlet", psf_bundle(device), np.arange(-40, 40) * 1.0, 6.0, np.linspace(-60.0, 60.0, 41)),
            ("gaussian-focus", grid_bundle(device, 0.6, 0.005, theta=0.25), np.arange(-100, 101) * 1.0, 2.0,
             np.linspace(-100.0, 100.0, 9)),
            ("uniform-pupil", grid_bundle(device, 0.6, 0.004), np.arange(-100, 101) * 1.0, 1.5,
             np.array([-10.0, 0.0, 10.0]))]


@pytest.mark.parametrize("device", DEVICES)
def test_nufft_focus_scans_equal_the_direct_sum(device):
    """The non-uniform FFT against the direct sum of image_planes (complex128) for the cases of optical_cases:
    the field per sum_r |a_r| (<= tol), and the coupling efficiency into a Gaussian mode of each plane, normalized
    by the power of each plane and by the power at the middle plane (as in a focus scan). tol 1e-12: |d eta| <=
    1e-11; tol 1e-9: |d eta| <= 1e-8. Observed on the GPU computer (field per sum |a| at tol 1e-12 and 1e-9; |d eta|
    at 1e-12 and 1e-9, the larger of the two powers): 4f (4501 rays) 1.4e-14, 4.0e-13, 2.4e-14, 1.0e-13; singlet
    (7505 rays) 5.8e-15, 3.5e-12, 5.2e-15, 3.1e-13; Gaussian focus (58081 rays) 1.2e-14, 1.6e-12, 2.4e-14,
    7.4e-15; uniform pupil (70667 rays) 1.3e-14, 1.7e-12, 1.1e-14, 7.2e-15. CPU: the same fields to 2 digits, the
    same |d eta| within a factor of 2."""
    for name, bundle, x, w, dz in optical_cases(device):
        ref = image_planes(bundle, x, x, dz)
        mode = torch.as_tensor(gaussian(x, x, w), device=bundle.device)
        p0 = torch.vdot(ref[len(dz) // 2].ravel(), ref[len(dz) // 2].ravel()).real
        for tol, eta_tol in ((1e-12, 1e-11), (1e-9, 1e-8)):
            got = image_planes(bundle, x, x, dz, method="nufft", tol=tol)
            err = float((got - ref).abs().max()) / sum_abs(bundle)
            assert err <= tol, (name, tol, err)
            for power in (None, p0):
                d = (overlap_efficiency_device(got, mode, power) - overlap_efficiency_device(ref, mode, power)).abs()
                print("observed", device, name, bundle.n_rays, tol, f"field {err:.1e}", power is None,
                      f"eta {float(d.max()):.1e}")
                assert float(d.max()) <= eta_tol, (name, tol, float(d.max()))
        if name == "gaussian-focus":
            eta = overlap_efficiency_device(ref, mode).cpu().numpy()
            assert eta[len(dz) // 2] > 0.999 and eta[0] < 0.05                     # a real focus scan


@pytest.mark.parametrize("device", DEVICES)
def test_nufft_coupling_into_a_fiber(device):
    """The 4f coupling of tests/test_pop.py (a 6 um Gaussian imaged to 15 um, into Gaussian modes of radius 15, 12
    and 19.5 um) at the image surface: image_sum with the non-uniform FFT against the direct sum, |d eta| <= 1e-8
    for tol 1e-6 and <= 1e-12 for tol 1e-12; and against the analytic Gaussian-mismatch value (2e-3, the limit of
    the ideal-lens model, as in tests/test_pop.py). Observed on the GPU computer (radius 15, 12, 19.5 um): eta
    0.99998, 0.95324, 0.93319; |d eta| 7.1e-13, 4.0e-11, 4.0e-11 (tol 1e-6) and 0, 1.1e-16, 0 (tol 1e-12; CPU
    3.3e-16, 6.7e-16, 1.1e-15)."""
    bundle = trace_bundle(four_f(), gaussian(XS, XS, 6.0), 0.5, WL, n_pad=4, device=device)
    ref = image_sum(bundle, XI, XI)
    for wf in (15.0, 12.0, 19.5):
        mode = torch.as_tensor(gaussian(XI, XI, wf), device=bundle.device)
        eta_ref = float(overlap_efficiency_device(ref, mode))
        for tol, eta_tol in ((1e-6, 1e-8), (1e-12, 1e-12)):
            eta = float(overlap_efficiency_device(image_sum(bundle, XI, XI, method="nufft", tol=tol), mode))
            print("observed", device, wf, tol, eta, f"{abs(eta - eta_ref):.1e}")
            assert abs(eta - eta_ref) <= eta_tol, (wf, tol, eta - eta_ref)
        assert eta == pytest.approx((2 * 15.0 * wf / (15.0**2 + wf**2)) ** 2, abs=2e-3)


@pytest.mark.parametrize("device", DEVICES)
def test_nufft_grids(device):
    """The grids of the non-uniform FFT: np.linspace axes (steps that differ in the last bits) are uniform (no
    warning; tol 1e-12 against NumPy); a one-point axis (a line, a point); a descending axis; and the fallback for a
    non-uniform axis: a UserWarning, and the same field as the direct sum (bit for bit). image_sum is one plane of
    image_planes (on the GPU only to rounding: index_add_ adds in a varying order)."""
    bundle = random_bundle(device, sizes=(900,))
    s = sum_abs(bundle)
    grids = [(np.linspace(-20.0, 20.0, 41), np.linspace(3.0, 9.0, 13)), (np.array([2.5]), np.arange(-8, 9) * 0.7),
             (np.arange(-8, 9) * 0.7, np.array([-1.0])), (np.array([0.3]), np.array([0.0])),
             (np.arange(10, -11, -1) * 0.8, np.arange(-6, 6) * 1.1)]
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        for x, y in grids:
            got = image_planes(bundle, x, y, [0.0, 40.0], method="nufft")
            assert got.shape == (2, x.size, y.size)
            assert float(np.max(np.abs(got.cpu().numpy() - numpy_planes(bundle, x, y, [0.0, 40.0])))) / s <= 1e-12
    x_bad = np.array([-3.0, -1.0, 0.0, 2.5, 4.0])
    with pytest.warns(UserWarning, match="not uniform"):
        got = image_planes(bundle, x_bad, grids[0][1], [0.0, 40.0], method="nufft")
    assert torch.equal(got, image_planes(bundle, x_bad, grids[0][1], [0.0, 40.0]))
    with pytest.warns(UserWarning, match="not uniform"):
        image_sum(bundle, grids[0][0], np.array([0.0, 1.0, 1.0 + 1e-9]), method="nufft")
    assert image_planes(bundle, grids[0][0], grids[0][1], [], method="nufft").shape == (0, 41, 13)
    one = image_planes(bundle, grids[0][0], grids[0][1], [5.0], method="nufft")[0]
    again = image_sum(bundle, grids[0][0], grids[0][1], dz=5.0, method="nufft")
    assert float((again - one).abs().max()) <= 1e-15 * s               # the GPU adds in a varying order


@pytest.mark.parametrize("device", DEVICES)
def test_nufft_batches_and_copies(device, monkeypatch):
    """Batches of planes, copies of the fine grid and chunks of rays do not change the field (beyond rounding).
    7 planes on a 24 x 21 grid (fine grid 48 x 48, w 15): the default is one batch, 16 copies and one chunk per
    block; patched: batches of 3 planes, one copy and chunks of 31 rays (SUM_ELEMENTS 45000, ALLOC_SLACK 0); 3
    copies and chunks of 164 rays (SUM_ELEMENTS 400000, ALLOC_SLACK 0); one copy and ray_chunk 97 (NUFFT_COPIES 1);
    one plane of the fine grid above the budget (SUM_ELEMENTS 12000 < 6 n1 n2, ALLOC_SLACK 0): batches of one plane
    and chunks of NUFFT_MIN_CHUNK = 4096 rays, not 12. Against the default: max |dE| / sum |a| <= 1e-14. Observed on
    the GPU computer: 1.9e-16, 2.0e-16, 2.8e-16, 2.4e-16 (CPU 2.7e-16, 1.5e-16, 2.7e-16, 2.7e-16)."""
    bundle = random_bundle(device, sizes=(700, 333))
    x, y, dz = np.arange(-12, 12) * 0.9, np.arange(-10, 11) * 1.1, np.linspace(-40.0, 40.0, 7)
    n1, n2 = pop._fine_size(24, 15), pop._fine_size(21, 15)
    assert (n1, n2) == (48, 48) and _nufft_steps(None, 7, n1, n2, 15) == (7, 16, 15404)
    ref = image_planes(bundle, x, y, dz, method="nufft")
    cases = [({"SUM_ELEMENTS": 45000, "ALLOC_SLACK": 0}, None, (3, 1, 31)),
             ({"SUM_ELEMENTS": 400000, "ALLOC_SLACK": 0}, None, (7, 3, 164)),
             ({"NUFFT_COPIES": 1}, 97, (7, 1, 97)),
             ({"SUM_ELEMENTS": 12000, "ALLOC_SLACK": 0}, None, (1, 1, 4096))]
    for patch, ray_chunk, steps in cases:
        with monkeypatch.context() as m:
            for name, value in patch.items():
                m.setattr(pop, name, value)
            assert _nufft_steps(ray_chunk, 7, n1, n2, 15) == steps
            got = image_planes(bundle, x, y, dz, ray_chunk=ray_chunk, method="nufft")
        err = float((got - ref).abs().max()) / sum_abs(bundle)
        print("observed", device, patch, f"{err:.1e}")
        assert err <= 1e-14, patch


def test_nufft_steps_of_large_grids():
    """The chunk of the spread for large image grids with the default budget: Nx 2048 (fine grid 4096^2, one plane
    above the budget: batches of one plane) keeps the formula (21333 rays); Nx 4096 (fine grid 8192^2, more than
    SUM_ELEMENTS alone) gets NUFFT_MIN_CHUNK rays, for 1 and 21 planes. Before the one review it was 1 ray."""
    assert pop._fine_size(2048, 15) == 4096 and pop._fine_size(4096, 15) == 8192
    assert _nufft_steps(None, 1, 4096, 4096, 15) == (1, 1, 21333)
    for nz in (1, 21):
        assert _nufft_steps(None, nz, 8192, 8192, 15) == (1, 1, pop.NUFFT_MIN_CHUNK)


def test_nufft_width_and_bad_inputs():
    """nufft_width: w = ceil(log10(1 / tol)) + 3, beta = 2.30 w; tol outside 1e-12 .. 1e-2, an unknown method,
    complex64 with the non-uniform FFT and ray_chunk 0 give a ValueError."""
    assert [nufft_width(t)[0] for t in (1e-2, 1e-6, 1e-9, 1e-12, 3e-7)] == [5, 9, 12, 15, 10]
    assert nufft_width(1e-12)[1] == pytest.approx(2.30 * 15)
    bundle = random_bundle("cpu", sizes=(50,))
    x = np.arange(-4, 4) * 1.0
    for tol in (0.0, 1e-13, 0.1, float("nan")):
        with pytest.raises(ValueError, match="tol"):
            image_planes(bundle, x, x, [0.0], method="nufft", tol=tol)
    with pytest.raises(ValueError, match="method"):
        image_planes(bundle, x, x, [0.0], method="fft")
    with pytest.raises(ValueError, match="complex128"):
        image_planes(bundle, x, x, [0.0], method="nufft", dtype=torch.complex64)
    with pytest.raises(ValueError, match="ray_chunk"):
        image_planes(bundle, x, x, [0.0], method="nufft", ray_chunk=0)


MEMORY_CASES = [          # (planes, Nx, Ny, SUM_ELEMENTS): one batch with copies; batches of planes; one plane
    (1, 128, 128, 1 << 22), (21, 128, 128, 1 << 22), (41, 256, 256, 1 << 23), (1, 512, 512, 1 << 23),
    (5, 1024, 1024, 1 << 25),
]


@GPU
@pytest.mark.parametrize("case", MEMORY_CASES, ids=lambda c: f"{c[0]}x{c[1]}x{c[2]}-2^{c[3].bit_length() - 1}")
def test_nufft_memory_is_within_the_budget(case, monkeypatch):
    """The peak GPU memory of image_planes with the non-uniform FFT (default tol), minus the result E, is at most
    SUM_ELEMENTS complex128 values (no tolerance) and at least a quarter of it. Several chunks of random rays.
    Observed on the GPU computer (peak minus E over the budget, in the order of MEMORY_CASES): 0.60, 0.83, 0.80,
    0.63, 0.65."""
    nz, nx, ny, budget = case
    monkeypatch.setattr(pop, "SUM_ELEMENTS", budget)
    w = nufft_width(1e-12)[0]
    n1, n2 = pop._fine_size(nx, w), pop._fine_size(ny, w)
    step = _nufft_steps(None, nz, n1, n2, w)[2]
    bundle = random_bundle("cuda", sizes=(max(3 * step, 3000),))
    x, y, dz = np.arange(nx) * 0.5, np.arange(ny) * 0.5, np.linspace(-50.0, 50.0, nz)
    image_planes(bundle, x, y, dz, method="nufft")                              # warm-up (FFT plans)
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    E = image_planes(bundle, x, y, dz, method="nufft")
    torch.cuda.synchronize()
    ratio = (torch.cuda.max_memory_allocated() - base - E.numel() * 16) / (16 * budget)
    print("observed", case, f"{ratio:.3f}")
    assert 0.25 <= ratio <= 1.0, ratio
