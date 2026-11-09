"""Checks of the float32 spread of the non-uniform FFT (image_planes with method "nufft" and dtype complex64) against
the float64 path (the same non-uniform FFT in complex128) and against its stated bound (nufft32_bound), the chunks
and flushes of the float32 spread (_spread32_segments: at most NUFFT_TERMS32 terms in a float32 value), and method
"auto" (the rule _auto_nufft; the fields stay within the stated tolerances). The "cuda" cases run only where
PyTorch has a GPU (CUDA or ROCm).

The error measure is |dE| / sum_r |a_r| (as in tests/test_nufft.py); against the bound it is the largest ratio
|E32 - E64| / (B sum_r |a_r|) over the grid points (at most 1).
"""
import warnings

import numpy as np
import pytest
import torch
from test_nufft import (DEVICES, GPU, TOLS, WL, bundle_of, grid_bundle, numpy_planes, optical_cases, random_bundle,
                        sum_abs)

from optics_gpu import pop
from optics_gpu.pop import (NUFFT_TERMS32, _auto_nufft, _nufft_steps, gaussian, image_planes, image_sum,
                            nufft32_bound, nufft_width, overlap_efficiency_device)

warnings.filterwarnings("ignore", category=RuntimeWarning)
C64 = {"method": "nufft", "dtype": torch.complex64}


def ratio_to_bound(e32, e64, bound, s):
    """The largest |E32 - E64| / (B s) over the grid points of all planes (e32, e64: (Z, Nx, Ny) tensors)."""
    d = (e32 - e64).abs().cpu().numpy()
    return float(np.max(d / (bound[None] * s)))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("tol", [1e-2, 1e-6, 1e-12])
def test_nufft32_equals_the_float64_path(device, tol):
    """Random rays in two blocks (as in tests/test_nufft.py), 4 planes, grids of 24 x 17 points, in one chunk and
    in chunks of 500 rays: |E32 - E64| <= B sum_r |a_r| at each grid point (B = nufft32_bound), and E32 against the
    NumPy sum within (max B + tol) sum_r |a_r|."""
    bundle = random_bundle(device)
    x, y, dz = -7.3 + np.arange(24) * 0.9, 5.0 + np.arange(17) * 1.17, np.array([-50.0, 0.0, 30.0, 120.0])
    s, bound = sum_abs(bundle), nufft32_bound(24, 17, tol)
    ref = numpy_planes(bundle, x, y, dz)
    for ray_chunk in (None, 500):
        e64 = image_planes(bundle, x, y, dz, ray_chunk=ray_chunk, method="nufft", tol=tol)
        e32 = image_planes(bundle, x, y, dz, ray_chunk=ray_chunk, tol=tol, **C64)
        assert e32.shape == (4, 24, 17) and e32.dtype == torch.complex128 and e32.device.type == device
        r = ratio_to_bound(e32, e64, bound, s)
        err = float(np.max(np.abs(e32.cpu().numpy() - ref))) / s
        print("observed", device, tol, ray_chunk, f"ratio {r:.2e}", f"err {err:.1e}", f"bound {bound.max():.1e}")
        assert r <= 1.0 and err <= bound.max() + tol


@pytest.mark.parametrize("device", DEVICES)
def test_nufft32_of_rays_in_phase(device):
    """The worst case of a float32 sum: 60000 equal rays (a = 1) in one direction (L' 0.3, M' -0.2), so all terms
    of a fine-grid value are equal and in phase, and each value of a copy gets 3750 terms (60000 / 16 copies): more
    than NUFFT_TERMS32, so the float32 spread must flush. The field is 60000 exp(i k (L' x + M' y + N' dz)).
    |E32 - E64| <= B sum_r |a_r| (B = nufft32_bound), and against the analytic field within (max B + tol)."""
    n = 60000
    L, M = np.full(n, 0.3), np.full(n, -0.2)
    bundle = bundle_of(device, L, M, np.ones(n, complex))
    x, y, dz = np.arange(-16, 16) * 0.7, np.arange(-12, 12) * 0.9, np.array([0.0, 25.0])
    tol = pop.NUFFT_TOL32
    bound = nufft32_bound(32, 24, tol)
    e64 = image_planes(bundle, x, y, dz, method="nufft", tol=tol)
    e32 = image_planes(bundle, x, y, dz, **C64)
    k, N = 2 * np.pi / WL, np.sqrt(1 - 0.3**2 - 0.2**2)
    exact = n * np.exp(1j * k * (0.3 * x[None, :, None] - 0.2 * y[None, None, :] + N * dz[:, None, None]))
    r = ratio_to_bound(e32, e64, bound, n)
    err = float(np.max(np.abs(e32.cpu().numpy() - exact))) / n
    print("observed", device, f"ratio {r:.2e}", f"err {err:.1e}", f"bound {bound.max():.1e}")
    assert r <= 1.0 and err <= bound.max() + tol


def exact_counts(bundle, segs, n1, n2, w, copies, fx, fy):
    """The terms of each value of each copy between two flushes, counted as the spread adds them: the largest count
    of each flush period, and the rays of each block in the order of the segments."""
    beta = nufft_width(pop.NUFFT_TOL32)[1]
    counts, worst, seen = torch.zeros(copies * n1 * n2, dtype=torch.int64), [], [[] for _ in bundle.blocks]
    for seg in segs:
        if seg is None:
            worst.append(int(counts.max()))
            counts.zero_()
            continue
        i, s, e = seg
        Lo, Mo, _, ao = bundle.blocks[i]
        sel = torch.arange(s, e).mul_(pop._golden_stride(ao.numel())).remainder_(ao.numel())
        seen[i].append(sel)
        _, ix = pop._es_spread_1d(Lo[sel] * fx, n1, w, beta)
        _, iy = pop._es_spread_1d(Mo[sel] * fy, n2, w, beta)
        ix += (torch.arange(e - s) % copies)[:, None] * n1
        counts += torch.bincount((ix[:, :, None] * n2 + iy[:, None, :]).reshape(-1), minlength=counts.numel())
    return worst, seen


def test_spread32_segments_bound_the_terms():
    """_spread32_segments on the CPU, for 50000 rays in two blocks: 40000 rays near 2 directions (a few fine-grid
    cells: many terms per value; a chunk of 20000 rays gives each copy about 625 rays of one direction) and 10000
    random rays; a 20 x 20 grid of pitch 1 um, tol 1e-6 (w 9, fine grid 40 x 40), 16 copies, chunks of 20000 rays.
    The exact counts (the indices of the spread, counted with bincount): every value of every copy gets at most
    NUFFT_TERMS32 terms between two flushes; the segments take every ray exactly once; there are flushes. The
    bound of the count (2 x 2 blocks) is not far above the exact count: the largest exact count of a flush period
    is above NUFFT_TERMS32 / 8."""
    rng = np.random.default_rng(5)
    dirs = np.array([[0.1, 0.05], [-0.2, 0.15]])
    pick = rng.integers(0, 2, 40000)
    L1, M1 = dirs[pick, 0] + rng.normal(0, 1e-3, 40000), dirs[pick, 1] + rng.normal(0, 1e-3, 40000)
    L2, M2 = rng.uniform(-0.4, 0.4, 10000), rng.uniform(-0.4, 0.4, 10000)
    b1, b2 = bundle_of("cpu", L1, M1, np.ones(40000, complex)), bundle_of("cpu", L2, M2, np.ones(10000, complex))
    bundle = pop.RayBundle([b1.blocks[0], b2.blocks[0]], WL, 50000, 50000, torch.device("cpu"), torch.zeros(()))
    w = nufft_width(pop.NUFFT_TOL32)[0]
    n1 = n2 = pop._fine_size(20, w)
    fx = fy = 1.0 / WL
    segs = pop._spread32_segments(bundle, fx, fy, n1, n2, w, 16, 20000) + [None]
    worst, seen = exact_counts(bundle, segs, n1, n2, w, 16, fx, fy)
    sizes = [seg[2] - seg[1] for seg in segs if seg is not None]
    print("observed", n1, len(sizes), "chunks", min(sizes), max(sizes), "flushes", segs.count(None), "worst", worst)
    assert (n1, w) == (40, 9) and max(worst) <= NUFFT_TERMS32 and max(worst) > NUFFT_TERMS32 / 8
    assert segs.count(None) >= 3
    for i, b in enumerate(bundle.blocks):
        got = torch.cat(seen[i]).sort().values
        assert torch.equal(got, torch.arange(b[3].numel()))


def test_nufft32_bound_values():
    """nufft32_bound: rho(m) = Phi_max / psi(m) is 1 at the center to 1e-4 and largest at the grid edge, for tol
    1e-2 to 1e-12; Phi (the sum of the w kernel values of one ray) changes by less than 1e-4 of its value over a
    cell (the docstring; then the 4096 samples of Phi_max are within 1e-10 of the largest value). The value for a
    128 x 128 grid at the default tol 1e-6 is printed."""
    for tol in TOLS:
        w, beta = nufft_width(tol)
        rho = pop._es_rho(128, w, beta)
        d = np.arange(4096) / 4096.0
        z = (2 * d[:, None] + 2 * np.arange(w)[None, :] - w) / w
        phi = np.exp(beta * (np.sqrt(np.clip(1.0 - z * z, 0.0, None)) - 1.0)).sum(1)
        change = (phi.max() - phi.min()) / phi.max()
        print("observed", tol, w, f"rho center {rho[64]:.8f} edge {rho[0]:.3f}", f"Phi change {change:.1e}")
        assert abs(rho[64] - 1) <= 1e-4 and np.argmax(rho) == 0 and change < 1e-4
    b = nufft32_bound(128, 128)
    print("observed", f"B 128 x 128 tol 1e-6: center {b[64, 64]:.2e}, largest {b.max():.2e}")
    assert b[64, 64] == pytest.approx((NUFFT_TERMS32 + pop.NUFFT_COPIES + 3) * 2.0**-24, rel=1e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_nufft32_focus_scans(device):
    """The float32 spread for the cases of tests/test_nufft.py (focus scans of the 4f image and the singlet PSF, 41
    planes; a Gaussian focus, 9 planes; a uniform pupil of NA 0.6, 3 planes, on a 201 x 201 um grid): against the
    float64 path at the same tol (1e-6), |dE| <= B sum_r |a_r| at each point; the coupling into a Gaussian mode
    against the direct complex128 sum, with the power of each plane and with one power: |d eta| <= 1e-6 (the limit
    of the direct complex64 checks in tests/test_image_planes.py)."""
    for name, bundle, x, wm, dz in optical_cases(device):
        tol = pop.NUFFT_TOL32
        bound, s = nufft32_bound(x.size, x.size, tol), sum_abs(bundle)
        ref = image_planes(bundle, x, x, dz)
        e64 = image_planes(bundle, x, x, dz, method="nufft", tol=tol)
        e32 = image_planes(bundle, x, x, dz, **C64)
        r = ratio_to_bound(e32, e64, bound, s)
        mode = torch.as_tensor(gaussian(x, x, wm), device=bundle.device)
        p0 = torch.vdot(ref[len(dz) // 2].ravel(), ref[len(dz) // 2].ravel()).real
        d_eta = max(float((overlap_efficiency_device(e32, mode, p) - overlap_efficiency_device(ref, mode, p))
                          .abs().max()) for p in (None, p0))
        err = float((e32 - ref).abs().max()) / s
        print("observed", device, name, bundle.n_rays, f"ratio {r:.2e}", f"field {err:.1e}", f"eta {d_eta:.1e}")
        assert r <= 1.0 and d_eta <= 1e-6


def test_auto_rule():
    """_auto_nufft: the non-uniform FFT from AUTO_POINTS image points (4 times that for fewer than AUTO_PLANES
    planes), AUTO_RAYS rays x planes and AUTO_WORK rays x planes x points (per dtype), else the direct sum: each
    limit alone keeps the direct sum."""
    z = pop.AUTO_PLANES
    for dt in (torch.complex128, torch.complex64):
        p, q, work = pop.AUTO_POINTS[dt], pop.AUTO_RAYS[dt], pop.AUTO_WORK[dt]
        rays = max(q, int(np.ceil(work / p)))                   # the fewest rays x planes on a grid of p points
        n = int(np.ceil(rays / z))                              # rays for z planes
        assert _auto_nufft(n, z, p, 1, dt) and _auto_nufft(n, z, 1, p, dt)
        assert not _auto_nufft((rays - 1) // z, z, p, 1, dt)
        assert not _auto_nufft(10**12, 50, p - 1, 1, dt)
        assert not _auto_nufft(10**12, z - 1, 4 * p - 1, 1, dt) and _auto_nufft(10**12, z - 1, 4 * p, 1, dt)
        big = int(np.ceil(work / q)) + 4 * p                    # a grid where the ray limit decides (one plane)
        assert _auto_nufft(q, 1, big, 1, dt) and not _auto_nufft(q - 1, 1, big, 1, dt)


@pytest.mark.parametrize("device", DEVICES)
def test_auto_selects_and_stays_within_the_tolerance(device, monkeypatch):
    """image_planes(method="auto") gives the field of the method that the rule selects: the direct sum (bit for
    bit) where the rule says direct, the non-uniform FFT with the complex128 spread where it says nufft (also for
    complex64, at its default tol 1e-6; bit for bit on the CPU; on the GPU the atomic adds change order, so within
    1e-15 sum_r |a_r|); a non-uniform grid gives the direct sum without a warning; and image_sum takes method
    "auto". The limits are patched so that both sides of the rule occur for the same rays (random rays, 3 planes)."""
    bundle = random_bundle(device, sizes=(900, 400))
    s = sum_abs(bundle)
    x, y, dz = np.arange(-12, 12) * 0.9, np.arange(-8, 9) * 1.1, [-20.0, 0.0, 15.0]
    monkeypatch.setattr(pop, "AUTO_PLANES", 1)                       # no larger grid limit for 3 planes
    for dt in (torch.complex128, torch.complex64):
        ref_direct = image_planes(bundle, x, y, dz, dtype=dt)
        tol = pop.NUFFT_TOL if dt == torch.complex128 else pop.NUFFT_TOL32
        ref_nufft = image_planes(bundle, x, y, dz, method="nufft", tol=tol)                # the complex128 spread
        runs = 1e-15                                                                     # two GPU runs (atomic order)
        for points, want in ((24 * 17, ref_nufft), (24 * 17 + 1, ref_direct)):
            monkeypatch.setitem(pop.AUTO_POINTS, dt, points)
            monkeypatch.setitem(pop.AUTO_WORK, dt, 1.0)
            monkeypatch.setitem(pop.AUTO_RAYS, dt, 1)
            got = image_planes(bundle, x, y, dz, dtype=dt, method="auto")
            if want is ref_direct or device == "cpu":
                assert torch.equal(got, want)
            else:
                assert float((got - want).abs().max()) <= runs * s
        monkeypatch.setitem(pop.AUTO_POINTS, dt, 1)
        x_bad = np.array([-3.0, -1.0, 0.0, 2.5, 4.0])
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            got = image_planes(bundle, x_bad, y, dz, dtype=dt, method="auto")
        assert torch.equal(got, image_planes(bundle, x_bad, y, dz, dtype=dt))
        one = image_sum(bundle, x, y, dz=15.0, dtype=dt, method="auto")
        assert float((one - image_planes(bundle, x, y, [15.0], method="nufft", tol=tol)[0]).abs().max()) <= runs * s


@pytest.mark.parametrize("device", DEVICES)
def test_auto_of_a_large_grid_phase(device, monkeypatch):
    """complex64 on a grid whose phase is above PHASE32_MAX: the direct sum gives a ValueError; "auto" that selects
    the non-uniform FFT computes the field (its phases are in float64; the complex128 spread), within 1e-6
    sum_r |a_r| (its default tol) of the complex128 direct sum (a uniform pupil of NA 0.6 on a 251 x 251 um grid:
    grid phase 305 rad)."""
    bundle = grid_bundle(device, 0.6, 0.02)
    x = np.arange(-125, 126) * 1.0
    with pytest.raises(ValueError, match="grid phase"):
        image_planes(bundle, x, x, [0.0], dtype=torch.complex64)
    monkeypatch.setitem(pop.AUTO_POINTS, torch.complex64, 1)
    monkeypatch.setattr(pop, "AUTO_PLANES", 1)
    monkeypatch.setitem(pop.AUTO_WORK, torch.complex64, 1.0)
    monkeypatch.setitem(pop.AUTO_RAYS, torch.complex64, 1)
    got = image_planes(bundle, x, x, [0.0], dtype=torch.complex64, method="auto")
    err = float((got - image_planes(bundle, x, x, [0.0])).abs().max()) / sum_abs(bundle)
    print("observed", device, bundle.n_rays, f"{err:.1e}")
    assert err <= 1e-6


@GPU
@pytest.mark.parametrize("case", [(1, 128, 128, 1 << 22), (21, 128, 128, 1 << 22), (1, 512, 512, 1 << 23)],
                         ids=lambda c: f"{c[0]}x{c[1]}x{c[2]}-2^{c[3].bit_length() - 1}")
def test_nufft32_memory_is_within_the_budget(case, monkeypatch):
    """The peak GPU memory of the float32 spread (default tol 1e-6), minus the result E, is at most SUM_ELEMENTS
    complex128 values (the budget of the complex128 path; the float32 copies and terms take half the bytes, the
    complex128 fine grid and the flush temporaries are added)."""
    nz, nx, ny, budget = case
    monkeypatch.setattr(pop, "SUM_ELEMENTS", budget)
    w = nufft_width(pop.NUFFT_TOL32)[0]
    step = _nufft_steps(None, nz, pop._fine_size(nx, w), pop._fine_size(ny, w), w)[2]
    bundle = random_bundle("cuda", sizes=(max(3 * step, 3000),))
    x, y, dz = np.arange(nx) * 0.5, np.arange(ny) * 0.5, np.linspace(-50.0, 50.0, nz)
    image_planes(bundle, x, y, dz, **C64)                                        # warm-up (FFT plans)
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    E = image_planes(bundle, x, y, dz, **C64)
    torch.cuda.synchronize()
    ratio = (torch.cuda.max_memory_allocated() - base - E.numel() * 16) / (16 * budget)
    print("observed", case, f"{ratio:.3f}")
    assert ratio <= 1.0, ratio
