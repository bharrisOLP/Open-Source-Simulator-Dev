"""Checks of the device pipeline (trace_bundle, image_sum): angular spectrum, trace, Jacobian, weights and image sum
on one PyTorch device in float64, against the NumPy reference pipeline (propagate_through with trace_device None),
against analytic Gaussian-beam results and against Optiland's Huygens PSF. The "cuda" cases run only where PyTorch
has a GPU (CUDA or ROCm).
"""
import warnings

import numpy as np
import pytest
import torch
from torch.overrides import TorchFunctionMode

from optics_gpu.pop import (N_CHECK, _np_gradient0, _uniform_step, gaussian, image_sum, lateral_scan,
                            overlap_efficiency, overlap_efficiency_device, propagate_through, trace_bundle)
from optics_gpu.systems import aberrated_singlet, four_f

warnings.filterwarnings("ignore", category=RuntimeWarning)
GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
DEVICES = ["cpu", pytest.param("cuda", marks=GPU)]
WL = 1.55
XS = np.arange(-48, 48) * 0.5          # the source grid of the 4f checks (um)
XI = np.arange(-64, 64) * 0.75         # the image grid of the 4f checks (um)
PSF = {"na_cut": 0.5, "amp_cut": 0.0, "n_pad": 16}


def point_source():
    src = np.zeros((64, 64), complex)
    src[32, 32] = 1.0
    return src


def four_f_moved(d_mm: float):
    """systems.four_f (f 2 mm and 5 mm) with the image surface d_mm after the back focus of lens 2."""
    from optiland.optic import Optic

    o = Optic()
    o.add_surface(index=0, thickness=2.0)
    o.add_surface(index=1, surface_type="paraxial", f=2.0, thickness=7.0, is_stop=True)
    o.add_surface(index=2, surface_type="paraxial", f=5.0, thickness=5.0 + d_mm)
    o.add_surface(index=3)
    o.set_aperture(aperture_type="EPD", value=3.0)
    o.add_wavelength(value=WL, is_primary=True)
    return o


def rel(a, b):
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


@pytest.mark.parametrize("device", DEVICES)
def test_gradient_has_the_numpy_formulas(device):
    """_np_gradient0 against np.gradient (edge_order 1) for a uniform axis, the fftshift(fftfreq) axis of the
    angular spectrum (NumPy takes it as non-uniform: its steps differ in the last bits) and a random axis."""
    rng = np.random.default_rng(1)
    f = rng.normal(size=(9, 7))
    axes = [np.arange(9) * 0.25, np.fft.fftshift(WL * np.fft.fftfreq(9, d=0.5)), np.sort(rng.uniform(0, 1, 9))]
    assert _uniform_step(axes[0]) == 0.25 and _uniform_step(axes[2]) is None
    for x in axes:
        ref = np.gradient(f, x, axis=0)
        t = lambda v: torch.as_tensor(v, dtype=torch.float64, device=device)  # noqa: E731
        got = _np_gradient0(t(f), t(x), _uniform_step(x)).cpu().numpy()
        assert np.max(np.abs(got - ref)) <= 1e-13 * np.max(np.abs(ref))
        ref1 = np.gradient(f.T, x, axis=1)
        got1 = _np_gradient0(t(f.T).T, t(x), _uniform_step(x)).T.cpu().numpy()
        assert np.max(np.abs(got1 - ref1)) <= 1e-13 * np.max(np.abs(ref1))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("trace_chunk", [None, 500], ids=["one-block", "blocks-of-500"])
def test_device_pipeline_equals_numpy_pipeline(device, trace_chunk):
    """The 4f coupling into 3 fiber radii and the singlet PSF of tests/test_pop.py: the device pipeline (also in
    blocks of about 500 rays, which checks the halo rows of the Jacobian) against the NumPy pipeline. Observed
    largest field differences (relative to the largest |E|; the device pipeline sums with image_planes, the NumPy
    pipeline with _add_rays): GPU 9.8e-15 and 8.0e-15 (4f, one block and blocks), 1.2e-14 and 1.3e-14 (singlet);
    CPU 8.6e-15 (4f) and 5.6e-12 (singlet: the PyTorch CPU trace differs from NumPy by up to 7e-14 mm of OPL); eta
    4.4e-16."""
    src = gaussian(XS, XS, 6.0)
    a = propagate_through(four_f(), src, 0.5, WL, XI, XI, n_pad=4)
    b = propagate_through(four_f(), src, 0.5, WL, XI, XI, n_pad=4, trace_device=device, trace_chunk=trace_chunk)
    assert a.trace_device == "numpy" and b.trace_device.startswith(device) and b.device.startswith(device)
    assert a.n_rays == b.n_rays > 8 * 500
    assert np.max(np.abs(b.E - a.E)) <= 1e-12 * np.max(np.abs(a.E))
    for wf in (15.0, 12.0, 19.5):
        fiber = gaussian(XI, XI, wf)
        assert overlap_efficiency(b.E, fiber) == pytest.approx(overlap_efficiency(a.E, fiber), abs=1e-12)
    xp = np.arange(-40, 40) * 1.0
    a = propagate_through(aberrated_singlet(), point_source(), 1.0, WL, xp, xp, **PSF)
    b = propagate_through(aberrated_singlet(), point_source(), 1.0, WL, xp, xp, trace_device=device,
                          trace_chunk=trace_chunk, **PSF)
    assert a.n_rays == b.n_rays
    assert np.max(np.abs(b.E - a.E)) <= 1e-9 * np.max(np.abs(a.E))


@pytest.mark.parametrize("device", DEVICES)
def test_device_psf_matches_optiland_huygens_psf(device):
    """The PSF check of tests/test_pop.py on the device pipeline (the stop clips a part of the rays, and blocks of
    about 4000 rays split the direction grid)."""
    from optiland.psf import HuygensPSF

    o = aberrated_singlet()
    xi = np.arange(-40, 40) * 1.0
    img = propagate_through(o, point_source(), 1.0, WL, xi, xi, trace_device=device, trace_chunk=4000, **PSF)
    ours = np.abs(img.E) ** 2
    ref = np.asarray(HuygensPSF(o, (0, 0), WL, num_rays=128, image_size=80, pixel_pitch=1e-3).psf)
    corr = (ours * ref).sum() / np.sqrt((ours**2).sum() * (ref**2).sum())
    assert corr > 0.99


def _is_cpu(v) -> bool:
    """True for a CPU tensor, a CPU device or the name of a CPU device (the targets of Tensor.to)."""
    if isinstance(v, torch.Tensor):
        return v.device.type == "cpu"
    if isinstance(v, torch.device):
        return v.type == "cpu"
    if isinstance(v, str):
        try:
            return torch.device(v).type == "cpu"
        except RuntimeError:
            return False
    return False


class _Watch(TorchFunctionMode):
    """Records each copy of an array from a device to the host (cpu, numpy, tolist, __array__; to with a CPU
    device or a CPU tensor in any argument; copy_ into a CPU tensor), each scalar that comes to the host (item, int,
    float, bool, index), and each large tensor that a torch operation makes on the CPU."""

    BIG = 1000

    def __init__(self):
        super().__init__()
        self.to_host, self.scalars, self.cpu_tensors = [], [], []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        name = getattr(func, "__name__", str(func))
        src = args[0] if args and isinstance(args[0], torch.Tensor) else None
        if src is not None and src.device.type != "cpu":
            if name in ("cpu", "numpy", "tolist", "__array__") or (
                    name == "to" and any(_is_cpu(v) for v in (*args[1:], *kwargs.values()))):
                self.to_host.append((name, src.numel()))
            elif name in ("item", "__int__", "__float__", "__bool__", "__index__"):
                self.scalars.append((name, src.numel()))
        if name == "copy_" and src is not None and src.device.type == "cpu" and len(args) > 1 and \
                isinstance(args[1], torch.Tensor) and args[1].device.type != "cpu":
            self.to_host.append((name, args[1].numel()))
        for t in out if isinstance(out, tuple | list) else (out,):
            if isinstance(t, torch.Tensor) and t.device.type == "cpu" and t.numel() >= self.BIG:
                self.cpu_tensors.append((name, tuple(t.shape)))
        return out


@GPU
def test_watch_sees_each_form_of_host_copy():
    """The watch of test_pipeline_data_stay_on_the_device must see each form of a copy to the host, also of a small
    array (30 values, below BIG) and with a CPU tensor as the target of to (the one review found that form)."""
    x = torch.arange(5000, dtype=torch.float64, device="cuda")
    cpu = torch.empty(1, dtype=torch.float64)
    forms = {
        "cpu": lambda: x[:30].cpu(), "to str": lambda: x[:30].to("cpu"),
        "to device": lambda: x[:30].to(torch.device("cpu")), "to device kw": lambda: x[:30].to(device="cpu"),
        "to str dtype": lambda: x[:30].to("cpu", torch.float32), "to tensor": lambda: x[:30].to(cpu),
        "tolist": lambda: x[:30].tolist(), "numpy": lambda: x[:30].detach().cpu().numpy(),
        "copy_": lambda: torch.empty(30, dtype=torch.float64).copy_(x[:30]),
        "Tensor.cpu": lambda: torch.Tensor.cpu(x[:30]),
    }
    for form, fn in forms.items():
        with _Watch() as w:
            fn()
        assert [n for _, n in w.to_host] and max(n for _, n in w.to_host) == 30, form
    for form, fn in {"item": lambda: x[0].item(), "float": lambda: float(x[0]), "int": lambda: int(x[1]),
                     "bool": lambda: bool(x[1] > 0)}.items():
        with _Watch() as w:
            fn()
        assert w.scalars and not w.to_host, form


@GPU
def test_pipeline_data_stay_on_the_device():
    """Inside trace_bundle and image_sum, no torch operation makes a large tensor on the CPU, and the only arrays
    that come to the host are those of the N_CHECK check rays of torch_tracer (a fixed count, not one per block);
    the rest are scalars (Optiland's own tests in its trace, and the bounding box), at most 60 per block and 20 more
    (a loop that reads one scalar per ray fails this). The 4f system and a point source: 5.5e4 rays in 8 blocks
    (observed: 20 arrays of 25 values; 421 scalars, about 53 per block)."""
    with _Watch() as w:
        bundle = trace_bundle(four_f(), point_source(), 1.0, WL, na_cut=0.4, amp_cut=0.0, n_pad=8, device="cuda",
                              trace_chunk=10000)
        E = image_sum(bundle, XI, XI, dz=50.0)
    assert bundle.n_rays > 50000 and len(bundle.blocks) > 3
    assert all(t.device.type == "cuda" for b in bundle.blocks for t in b) and E.device.type == "cuda"
    assert w.cpu_tensors == []
    arrays = [n for _, n in w.to_host if n > 1]
    assert set(arrays) <= {N_CHECK} and len(arrays) <= 40, w.to_host
    assert all(n == 1 for _, n in w.scalars)
    assert len(w.scalars) <= 60 * len(bundle.blocks) + 20, len(w.scalars)


@GPU
def test_trace_memory_is_bounded_by_the_block():
    """Peak GPU memory of trace_bundle, the 4f system and a point source (flat spectrum; dx 2.5 um, so that the
    bounding box of |L|, |M| < 0.3 nearly fills the spectrum grid): 2.45e5 rays traced, 1.93e5 kept. With blocks
    of 4096 rays, the peak is the box (A and keep: 17 bytes per traced ray), the bundle (40 bytes per kept ray)
    and one block (observed: 53 bytes per traced ray); with one block of all rays the temporaries of the trace of
    all rays are in memory at one time (observed: 249 bytes per traced ray)."""
    kw = {"na_cut": 0.3, "amp_cut": 0.0, "n_pad": 8, "device": "cuda"}
    peaks, blocks = {}, {}
    for chunk in (4096, 10**9, 4096):                       # the first run is a warm-up
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        bundle = trace_bundle(four_f(), point_source(), 2.5, WL, trace_chunk=chunk, **kw)
        torch.cuda.synchronize()
        peaks[chunk] = torch.cuda.max_memory_allocated() - base
        n_traced, n_kept, blocks[chunk] = bundle.n_traced, bundle.n_rays, len(bundle.blocks)
        del bundle
    assert n_traced > 2e5 and n_kept > 0.75 * n_traced and blocks[10**9] == 1 and blocks[4096] > 50
    assert peaks[4096] < 64 * n_traced
    assert peaks[10**9] - peaks[4096] > 150 * n_traced


@pytest.mark.parametrize("device", DEVICES)
def test_focus_and_lateral_scans_match_gaussian_beams(device):
    """The 4f image of a 6 um Gaussian is a 15 um Gaussian (half-angle 0.033 rad in image space). Its coupling into
    a 15 um Gaussian mode moved by dz along z is 1 / (1 + (lambda dz / (2 pi w^2))^2), and moved by s laterally
    exp(-s^2 / w^2) (two equal Gaussian beams; textbook results, e.g. Saleh and Teich, Fundamentals of Photonics,
    ch. 3). The focus scan (image_sum with dz: one plane wave per ray) must also agree with the FFT angular-
    spectrum propagation of the field at dz = 0 (the method of examples/chip_to_fiber_coupling.py), and the
    lateral scan (two matrix products) with overlap_efficiency of the moved mode. The FFT method keeps all power
    on its periodic grid; the ray sum lets wide-angle power leave the grid. Normalized by the power at dz = 0
    (power=p0) the two agree more closely. Observed on the GPU computer (CPU and GPU the same): 1.4e-3 from the
    focus formula (the image of the ideal-lens model is not an exact Gaussian, see README), 1.5e-6 from the FFT
    propagation normalized by the power at each plane and 2.7e-7 normalized by p0 (at dz = 400 um, where the
    20 um beam starts to wrap on the 96 um FFT grid), 8.1e-4 from the lateral formula, 3e-16 from
    overlap_efficiency."""
    w = 15.0
    bundle = trace_bundle(four_f(), gaussian(XS, XS, 6.0), 0.5, WL, n_pad=4, device=device)
    fiber = torch.as_tensor(gaussian(XI, XI, w), device=bundle.device)
    E0 = image_sum(bundle, XI, XI).cpu().numpy()
    p0 = float(np.vdot(E0, E0).real)
    FX, FY = np.meshgrid(*(2 * [np.fft.fftfreq(XI.size, d=0.75)]), indexing="ij")
    kz = 2 * np.pi * np.sqrt(np.clip(1 / WL**2 - FX**2 - FY**2, 0, None))
    d_beam, d_fft, d_fft_p0 = [], [], []
    for dz in (-300.0, -100.0, 0.0, 200.0, 400.0):
        E = image_sum(bundle, XI, XI, dz=dz)
        eta = float(overlap_efficiency_device(E, fiber))
        d_beam.append(abs(eta - 1 / (1 + (WL * dz / (2 * np.pi * w**2)) ** 2)))
        e_fft = np.fft.ifft2(np.fft.fft2(E0) * np.exp(1j * kz * dz))
        eta_fft = overlap_efficiency(e_fft, gaussian(XI, XI, w))
        d_fft.append(abs(eta - eta_fft))
        d_fft_p0.append(abs(float(overlap_efficiency_device(E, fiber, power=p0)) - eta_fft))
    s = np.array([-10.0, -4.0, 0.0, 3.0, 8.0])
    scan = lateral_scan(torch.as_tensor(E0, device=bundle.device), XI, XI, w, s, s).cpu().numpy()
    S, T = np.meshgrid(s, s, indexing="ij")
    d_lat = np.max(np.abs(scan - scan[2, 2] * np.exp(-(S**2 + T**2) / w**2)))
    X, Y = np.meshgrid(XI, XI, indexing="ij")
    moved = np.exp(-((X - 8.0) ** 2 + (Y + 4.0) ** 2) / w**2)
    d_ovl = abs(scan[4, 1] - overlap_efficiency(E0, moved))
    assert max(d_beam) <= 2e-3 and max(d_fft) <= 1e-5 and d_lat <= 2e-3 and d_ovl <= 1e-12
    assert max(d_fft_p0) <= 1e-6, d_fft_p0


@pytest.mark.parametrize("device", DEVICES)
def test_image_sum_dz_equals_a_moved_image_surface(device):
    """image_sum(dz=d) of the original optic against the NumPy pipeline of the same optic with its image surface
    moved by d (an independent path: Optiland traces the rays to the moved surface). The sources are not symmetric
    in dz: a Gaussian with a curved wavefront (R = 1.5 mm, so the focus is off the image surface) through the 4f
    system, and the point source through the singlet (spherical aberration). A wrong sign of dz in image_sum gives
    a field difference near 1 (the one review found that no other test sees the sign). Observed on the GPU
    computer: 2.1e-12 and 2.9e-12 (4f, d = -80 and +80 um; CPU and GPU), 2.2e-11 and 2.6e-11 (singlet, -60 and
    +60 um, GPU; CPU 2.2e-11 and 3.0e-11); with the opposite sign 1.0 (4f), 1.7 and 2.1 (singlet)."""
    k = 2 * np.pi / WL
    X, Y = np.meshgrid(XS, XS, indexing="ij")
    src = gaussian(XS, XS, 6.0) * np.exp(1j * k * (X**2 + Y**2) / (2 * 1500.0))
    bundle = trace_bundle(four_f_moved(0.0), src, 0.5, WL, n_pad=4, device=device)
    fiber = gaussian(XI, XI, 15.0)
    etas = []
    for d in (-80.0, 80.0):
        ref = propagate_through(four_f_moved(d / 1e3), src, 0.5, WL, XI, XI, n_pad=4).E
        assert rel(image_sum(bundle, XI, XI, dz=d).cpu().numpy(), ref) <= 1e-10
        assert rel(image_sum(bundle, XI, XI, dz=-d).cpu().numpy(), ref) > 0.5
        etas.append(overlap_efficiency(ref, fiber))
    assert abs(etas[0] - etas[1]) > 1e-3                       # the case is not symmetric in dz (0.9959, 0.9875)
    xp = np.arange(-40, 40) * 1.0
    bundle = trace_bundle(aberrated_singlet(), point_source(), 1.0, WL, device=device, **PSF)
    for d in (-60.0, 60.0):
        o = aberrated_singlet()
        image = o.surfaces.surfaces[-1]
        image.geometry.cs.z = image.geometry.cs.z + d / 1e3
        ref = propagate_through(o, point_source(), 1.0, WL, xp, xp, **PSF).E
        assert rel(image_sum(bundle, xp, xp, dz=d).cpu().numpy(), ref) <= 1e-9
        assert rel(image_sum(bundle, xp, xp, dz=-d).cpu().numpy(), ref) > 0.5


@pytest.mark.parametrize("device", DEVICES)
def test_bundle_power_is_the_source_power(device):
    """RayBundle.power (Parseval: the power of the kept rays) of a Gaussian through the lossless 4f system equals the
    power of the source (all directions are kept except those below amp_cut), and the power of the ray-sum field
    on the 96 um image grid equals RayBundle.power (power conservation of the sqrt(J) weights; the grid holds the
    15 um image). Observed on the GPU computer (CPU and GPU): -1e-12 and 5.5e-5."""
    src = gaussian(XS, XS, 6.0)
    bundle = trace_bundle(four_f(), src, 0.5, WL, n_pad=4, device=device)
    p_src = np.sum(np.abs(src) ** 2) * 0.25
    E = image_sum(bundle, XI, XI).cpu().numpy()
    assert float(bundle.power) == pytest.approx(p_src, rel=1e-9)
    assert np.sum(np.abs(E) ** 2) * 0.75**2 == pytest.approx(float(bundle.power), rel=1e-4)


def test_device_pipeline_refuses_bad_inputs():
    src = gaussian(XS, XS, 6.0)
    with pytest.raises(ValueError, match="trace device"):
        trace_bundle(four_f(), src, 0.5, WL, device="mps")
    with pytest.raises(ValueError, match="trace_chunk"):
        trace_bundle(four_f(), src, 0.5, WL, device="cpu", trace_chunk=0)
    with pytest.raises(ValueError, match="no direction"):
        trace_bundle(four_f(), src, 0.5, WL, device="cpu", amp_cut=2.0)
    with pytest.raises(ValueError, match="ray_chunk"):
        propagate_through(four_f(), src, 0.5, WL, XI, XI, trace_device="cpu", ray_chunk=0)
