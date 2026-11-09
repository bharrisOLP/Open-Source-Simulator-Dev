"""Checks of the batches of optics_gpu.analysis (items 7 to 10 of its module docstring): many fields and wavelengths
in one trace, the FFTs of many planes in one call, through focus by the change of the reference sphere (defocus),
the polychromatic PSF and MTF on a common grid, and image_quality (all of them in one call).

- Batch against one call for each field and wavelength: the wavefronts, the FFT PSFs and the MTFs.
- defocus against a new trace to the moved image surface (Optiland moves the surface; the same ray aiming); on the
  perfect lens also against the exact OPD of a spherical wave and against the Huygens PSF of the image sum (an
  independent method: a plane-wave sum with the propagation factor exp(i k N' dz)).
- The polychromatic PSF against Optiland's MMDFTPSF (a matrix Fourier transform, for each wavelength on the
  common grid), against the weighted sum of the per-call FFT PSFs resampled by exact trigonometric interpolation
  (their DFT, then the trigonometric polynomial at the shifted points), and on the perfect lens against the
  weighted sum of Airy patterns; the polychromatic MTF against the diffraction-limited MTFs and against the
  analytic MTF of two equal PSFs at a distance s, MTF |cos(pi nu s)|.
The "cuda" cases run only where PyTorch has a GPU (CUDA or ROCm). The observed values are in README.md.
"""
import dataclasses
import math
import warnings

import numpy as np
import pytest
import torch

from optics_gpu import analysis
from optics_gpu.analysis import (airy, clear_copy_cache, defocus, diffraction_limited_mtf, fft_mtf, fft_mtfs, fft_psf,
                                 fft_psfs, huygens_psf, image_quality, polychromatic_mtf, polychromatic_psf,
                                 pupil_wavefront, pupil_wavefronts)
from optics_gpu.systems import aberrated_singlet, dispersive_singlet, ideal_relay

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
DEVICES = ["cpu", pytest.param("cuda", marks=GPU)]
FCD = (0.4861, 0.5876, 0.6563)
FIELDS = ((0.0, 0.0), (0.0, 0.5), (0.0, 1.0))
TENSORS = ("opd", "intensity", "inside", "xp", "yp", "zp", "ldir", "mdir", "ndir", "valid")
NUMBERS = ("radius", "center", "wavelength", "fno", "freq_step", "n_rays", "n_image", "exit_pupil", "chief_dir",
           "field", "dz", "z_spread")


def report(name, **values):
    print(f"\n[{name}] " + ", ".join(f"{k} {v:.3g}" if isinstance(v, float) else f"{k} {v}"
                                     for k, v in values.items()))


def np_(t):
    return t.detach().cpu().numpy()


def same_wavefront(a, b) -> bool:
    return (all(torch.equal(getattr(a, q), getattr(b, q)) for q in TENSORS)
            and all(getattr(a, q) == getattr(b, q) for q in NUMBERS) and np.array_equal(a.p, b.p))


def singlet_at_infinity():
    """The singlet of systems.aberrated_singlet with the object at infinity and the angle fields 0 and 2 degrees."""
    from optiland.materials import IdealMaterial
    from optiland.optic import Optic
    from optiland.physical_apertures import RadialAperture

    o = Optic()
    o.add_surface(index=0, radius=np.inf, thickness=np.inf)
    o.add_surface(index=1, radius=10.0, thickness=3.0, material=IdealMaterial(n=1.5), is_stop=True,
                  aperture=RadialAperture(r_max=3.0))
    o.add_surface(index=2, radius=np.inf, thickness=17.0)
    o.add_surface(index=3)
    o.set_aperture(aperture_type="EPD", value=6.0)
    o.set_field_type(field_type="angle")
    o.add_field(y=0.0)
    o.add_field(y=2.0)
    o.add_wavelength(value=1.55, is_primary=True)
    o.image_solve()
    return o


def moved(make, dz_um: float):
    """The optic of make() with its image surface dz_um um farther from the last lens surface."""
    o = make()
    k = len(o.surfaces.surfaces) - 2
    o.updater.set_thickness(float(o.surfaces[k].thickness) + dz_um * 1e-3, k)
    return o


# ---------------------------------------------------------------------------------------------------------------
# Batch against one call per field and wavelength
# ---------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("device", DEVICES)
def test_batch_of_fields_and_wavelengths_equals_one_call_each(device):
    """3 fields x 3 wavelengths of the dispersive singlet in one batch (blocks of 1000 rays, so that a block holds
    rays of two fields) against pupil_wavefront for each pair (one block each): bit for bit. The same for the
    singlet with the object at infinity (the launch-plane phase of item 3), 2 angle fields."""
    o = dispersive_singlet()
    batch = pupil_wavefronts(o, FIELDS, FCD, n_pupil=48, device=device, trace_chunk=1000)
    for f, field in enumerate(FIELDS):
        for i, wl in enumerate(FCD):
            assert same_wavefront(batch[f][i], pupil_wavefront(o, field, wl, n_pupil=48, device=device))
    lateral = batch[2][0].center[1] - batch[2][2].center[1]
    report(f"batch dispersive singlet {device}", lateral_color_mm=lateral, rays=batch[0][0].n_rays)
    assert abs(lateral) > 1e-4                                          # the system has lateral color
    inf = singlet_at_infinity()
    two = pupil_wavefronts(inf, [(0.0, 0.0), (0.0, 1.0)], n_pupil=40, device=device)
    for f, field in enumerate([(0.0, 0.0), (0.0, 1.0)]):
        assert same_wavefront(two[f][0], pupil_wavefront(inf, field, n_pupil=40, device=device))


@pytest.mark.parametrize("device", DEVICES)
def test_batched_ffts_equal_one_fft_each(device):
    """fft_psfs and fft_mtfs of 9 wavefronts (one batched FFT each) against fft_psf and fft_mtf of each."""
    o = dispersive_singlet()
    wfs = [w for row in pupil_wavefronts(o, FIELDS, FCD, n_pupil=64, device=device) for w in row]
    psfs = fft_psfs(wfs, 160)
    mtfs = fft_mtfs(wfs, psfs)
    d_psf = d_mtf = 0.0
    bitwise = True
    for w, p, m in zip(wfs, psfs, mtfs):
        q = fft_psf(w, 160)
        r = fft_mtf(w, q)
        d_psf = max(d_psf, float((p.psf - q.psf).abs().max()))
        d_mtf = max(d_mtf, float((m.tangential - r.tangential).abs().max()),
                    float((m.sagittal - r.sagittal).abs().max()))
        bitwise &= torch.equal(p.psf, q.psf) and torch.equal(m.tangential, r.tangential)
        assert p.dx == q.dx and np.array_equal(p.x, q.x) and np.array_equal(m.freq_tan, r.freq_tan)
    report(f"batched ffts {device}", d_psf=d_psf, d_mtf=d_mtf, bitwise=bitwise)
    assert d_psf <= 1e-14 and d_mtf <= 1e-14


def test_copy_cache_reuses_one_copy_per_state():
    """The second call uses the kept copy (the same object) and gives the same wavefront; a changed optic gets a
    new copy; clear_copy_cache forgets them."""
    clear_copy_cache()
    o = aberrated_singlet()
    a = pupil_wavefront(o, n_pupil=32, device="cpu")
    assert len(analysis._copies) == 1
    first = next(iter(analysis._copies.values()))
    b = pupil_wavefront(o, n_pupil=32, device="cpu")
    assert len(analysis._copies) == 1 and next(iter(analysis._copies.values())) is first
    assert same_wavefront(a, b)
    o.updater.set_thickness(float(o.surfaces[2].thickness) + 0.05, 2)
    c = pupil_wavefront(o, n_pupil=32, device="cpu")
    assert len(analysis._copies) == 2 and not torch.equal(a.opd, c.opd)
    clear_copy_cache()
    assert len(analysis._copies) == 0


# ---------------------------------------------------------------------------------------------------------------
# Through focus
# ---------------------------------------------------------------------------------------------------------------

def singlet_off_axis():
    o = aberrated_singlet()
    o.add_field(y=1.0)
    return o


MOVED = {"singlet": (aberrated_singlet, (0.0, 0.0)), "singlet_off_axis": (singlet_off_axis, (0.0, 1.0)),
         "singlet_at_infinity": (singlet_at_infinity, (0.0, 1.0))}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", list(MOVED))
def test_defocus_equals_a_trace_to_the_moved_image_surface(device, name):
    """defocus(wf, dz) against pupil_wavefront of the optic whose image surface is moved by dz (Optiland traces to
    the moved surface): the OPD map, the intensities, the sphere (center and radius) and its points, the FFT PSF."""
    make, field = MOVED[name]
    wf = pupil_wavefront(make(), field, n_pupil=96, device=device)
    worst = {"d_opd": 0.0, "d_psf": 0.0, "d_center_mm": 0.0, "d_points_mm": 0.0}
    w_pv = []
    for dz in (-50.0, 50.0):
        got = defocus(wf, dz)
        ref = pupil_wavefront(moved(make, dz), field, n_pupil=96, device=device)
        ok = np_(ref.valid)
        assert np.array_equal(ok, np_(got.valid)) and torch.equal(got.intensity, ref.intensity)
        assert got.radius == pytest.approx(ref.radius, rel=1e-12)
        worst["d_opd"] = max(worst["d_opd"], float(np.max(np.abs(np_(got.opd)[ok] - np_(ref.opd)[ok]))))
        worst["d_center_mm"] = max(worst["d_center_mm"], max(abs(u - v) for u, v in zip(got.center, ref.center)))
        for q in ("xp", "yp", "zp"):
            worst["d_points_mm"] = max(worst["d_points_mm"], float(np.max(np.abs(np_(getattr(got, q))[ok]
                                                                                   - np_(getattr(ref, q))[ok]))))
        worst["d_psf"] = max(worst["d_psf"], float((fft_psf(got).psf - fft_psf(ref).psf).abs().max()))
        w = np_(got.opd)[ok & (np_(got.intensity) > 0)]
        w_pv.append(float(w.max() - w.min()))
    report(f"defocus vs moved surface {name} {device}", **worst, opd_pv_minus=w_pv[0], opd_pv_plus=w_pv[1])
    assert worst["d_opd"] < 1e-9 and worst["d_center_mm"] < 1e-12 and worst["d_points_mm"] < 1e-9
    assert worst["d_psf"] < 1e-9
    assert defocus(wf, 0.0) is wf


def exact_defocus_waves(p, r_stop, f, dz_mm, R, wl):
    """The exact OPD W(u) = dz (1 - cos u) + sqrt(R^2 - dz^2 sin^2 u) - R of ideal_relay with its image surface
    dz after the focus (tests/test_analysis.py), in waves."""
    PX, PY = np.meshgrid(p, p, indexing="ij")
    h = np.hypot(PX, PY) * r_stop
    s, c = h / np.hypot(h, f), f / np.hypot(h, f)
    return (dz_mm * (1 - c) + np.sqrt(R**2 - dz_mm**2 * s**2) - R) / (wl * 1e-3)


@pytest.mark.parametrize("device", DEVICES)
def test_through_focus_of_the_perfect_lens(device):
    """ideal_relay (f 20 mm, stop radius 2 mm, NA 0.0995) through focus, dz = -150, 0, 60, 150 um (W020 =
    NA^2 dz / 2: -0.48, 0, 0.19, 0.48 waves), pupil 256, grid 1024, on 49 x 49 samples of the FFT PSF:
    - the OPD by defocus against the exact spherical-wave OPD of the moved plane (with its new reference sphere),
      and the first-order (Debye) phase n dz (1 - D . D_c) / (N_c lambda) against that full change;
    - the Huygens PSF of the image sum at the same dz (directions up to 0.15; an independent method, a plane-wave
      (Debye) sum with the propagation factor exp(i k N' dz)) against: the FFT PSF of the pupil with the
      first-order phase only; a Debye sum over the pupil rays with that phase, the traced image-side directions
      (L', M') and the weight J = |d(L', M') / d(Px, Py)| (uniform in direction cosine, as the image sum of this
      relay); and the FFT PSF of defocus (the reference sphere of the moved plane, as a trace to the moved surface).
    The image sum and the first-order pupil are symmetric in dz; defocus is not: its second-order rest
    -dz^2 sin^2 u / (2 R') (the change of the reference sphere) has the same sign for both signs of dz."""
    f, r, wl = 20.0, 2.0, 1.55
    relay = ideal_relay(f=f, r_stop=r)
    wf = pupil_wavefront(relay, n_pupil=256, device=device)
    dzs = (-150.0, 0.0, 60.0, 150.0)
    # the traced image-side directions on the full grid (NumPy trace of the original), for J
    PX, PY = np.meshgrid(wf.p, wf.p, indexing="ij")
    rays = relay.ray_tracer.trace_generic(0.0, 0.0, PX.ravel().copy(), PY.ravel().copy(), wl, record=False)
    Lp, Mp = (np.asarray(v).reshape(PX.shape) for v in (rays.L, rays.M))
    J = np.abs(np.gradient(Lp, wf.p, axis=0) * np.gradient(Mp, wf.p, axis=1)
               - np.gradient(Lp, wf.p, axis=1) * np.gradient(Mp, wf.p, axis=0))
    use = np_(wf.inside & (wf.intensity > 0))
    t = lambda v: torch.as_tensor(np.asarray(v), dtype=torch.float64, device=wf.device)  # noqa: E731
    k = 2 * np.pi / wl
    out = {}
    c, h = 512, 24
    ins = np_(wf.inside)
    for dz in dzs:
        got = defocus(wf, dz)
        d_map = float(np.max(np.abs(np_(got.opd)[ins] - exact_defocus_waves(wf.p, r, f, dz * 1e-3, f - dz * 1e-3,
                                                                              wl)[ins])))
        first = torch.where(wf.valid, dz * 1e-3 * (1 - wf.ndir) / (wl * 1e-3), 0)   # on the axis: D_c = (0, 0, 1)
        d_first = float(np.max(np.abs(np_(got.opd - wf.opd - first)[ins])))
        one = dataclasses.replace(wf, opd=wf.opd + first)                # the pupil with the first-order phase
        psf, psf1 = fft_psfs([got, one], 1024)
        xs = psf.x[c - h:c + h + 1]
        I = np_(huygens_psf(relay, wl, xs, xs, dz=dz, na_cut=0.15, device=device))
        F = np_(psf.psf)[c - h:c + h + 1, c - h:c + h + 1]
        F1 = np_(psf1.psf)[c - h:c + h + 1, c - h:c + h + 1]
        ex = torch.exp(1j * k * t(xs)[:, None] * t(Lp[use])[None, :])
        ey = torch.exp(1j * k * t(xs)[:, None] * t(Mp[use])[None, :])
        w_use = one.opd[torch.as_tensor(use, device=wf.device)]
        E = (ex * (t(J[use]) * torch.polar(torch.ones_like(w_use), -2 * np.pi * w_use))[None, :]) @ ey.T
        debye = np_(E.abs() ** 2) / float(J[use].sum()) ** 2
        out[dz] = {"d_map": d_map, "d_first": d_first, "first_vs_sum": float(np.max(np.abs(I - F1))),
                   "debyeJ_vs_sum": float(np.max(np.abs(I - debye))), "defocus_vs_sum": float(np.max(np.abs(I - F))),
                   "peak_defocus": float(F.max()), "peak_first": float(F1.max()), "peak_sum": float(I.max())}
        assert d_map < 1e-9
        # the second-order rest: dz^2 sin^2 u / (2 R') at the edge, 3.6e-3 (150 um) and 5.8e-4 (60 um) waves
        assert d_first <= 1.1 * (dz * 1e-3) ** 2 * 0.0099 / (2 * (f - dz * 1e-3)) / (wl * 1e-3)
    for dz, v in out.items():
        report(f"through focus perfect lens {device} dz {dz}", **v)
    for v in out.values():
        assert v["first_vs_sum"] < 2e-3 and v["debyeJ_vs_sum"] < 2e-3      # the Debye model: two methods agree
        assert v["defocus_vs_sum"] < 1e-2                                  # plus the change of the sphere
    lo, hi = out[-150.0], out[150.0]
    assert abs(hi["peak_sum"] - lo["peak_sum"]) < 1e-3 and abs(hi["peak_first"] - lo["peak_first"]) < 1e-3
    assert hi["peak_defocus"] - lo["peak_defocus"] > 5e-3                # the asymmetry of the sphere change
    assert hi["peak_defocus"] < 0.5 and out[60.0]["peak_defocus"] > 0.85  # the defocus lowers the peak


# ---------------------------------------------------------------------------------------------------------------
# Polychromatic PSF and MTF
# ---------------------------------------------------------------------------------------------------------------

def trig_resample(psf: np.ndarray, x_rel: np.ndarray, y_rel: np.ndarray) -> np.ndarray:
    """The FFT PSF (M x M [ix, iy], the chief-ray point at M // 2) at the points (x_rel, y_rel) (in its samples, from
    the chief-ray point) by trigonometric interpolation in NumPy: the DFT of the samples, then the trigonometric
    polynomial. Exact for M >= 2 N - 1: the PSF of the sampled pupil has the frequencies -(N - 1) to N - 1 in units
    of 1 / (M dx), and M samples resolve them."""
    M = psf.shape[0]
    otf = np.fft.fft2(np.fft.ifftshift(psf))
    k = np.fft.fftfreq(M, 1.0 / M)
    Ex = np.exp(2j * np.pi * np.outer(x_rel, k) / M)
    Ey = np.exp(2j * np.pi * np.outer(y_rel, k) / M)
    return np.real(Ex @ otf @ Ey.T) / M**2


def test_polychromatic_psf_on_its_own_grid_is_the_fft_psf():
    """One wavelength, the default grid and step (the grid of fft_psf): the matrix Fourier transform gives the FFT
    PSF; the polychromatic MTF of one wavelength is fft_mtf."""
    wf = pupil_wavefront(aberrated_singlet(), n_pupil=64, device="cpu")
    q = fft_psf(wf)
    poly = polychromatic_psf([wf])
    assert poly.dx == pytest.approx(q.dx, rel=1e-14) and poly.weights.tolist() == [1.0]
    d = float((poly.psf - q.psf).abs().max())
    m, r = polychromatic_mtf([wf], [q]), fft_mtf(wf, q)
    d_m = max(float((m.tangential - r.tangential).abs().max()), float((m.sagittal - r.sagittal).abs().max()))
    report("poly one wavelength", d_psf=d, d_mtf=d_m)
    assert d < 1e-13 and d_m < 1e-12


@pytest.mark.parametrize("device", DEVICES)
def test_polychromatic_psf_against_optiland_mmdft_and_the_resampled_fft_psfs(device):
    """The dispersive singlet at F, d, C: (a) on the axis, each monochromatic PSF on the common grid against
    Optiland's MMDFTPSF with the same pupil (num_rays N), image_size and pixel_pitch; (b) at the field 1 mm
    (lateral color), the polychromatic PSF with the weights 1, 2, 1 against the weighted sum of the per-call FFT
    PSFs (fft_psf of pupil_wavefront), each resampled at x - s_l by trig_resample, with the weights
    v_l = w_l K_l^2 / (lambda_l F_l)^2 (sum 1) made here from the per-call data."""
    from optiland.psf.mmdft import MMDFTPSF

    o = dispersive_singlet()
    N = 64
    wfs = pupil_wavefronts(o, [(0.0, 0.0), (0.0, 1.0)], FCD, n_pupil=N, device=device)
    poly = polychromatic_psf(wfs[0], grid_size=100)
    d_mm = 0.0
    for i, wl in enumerate(FCD):
        # 98 points (index 1 .. 98 of the 100 here): Optiland refuses image_size above lambda F (N - 1) / pixel_pitch
        ref = np.asarray(MMDFTPSF(o, (0.0, 0.0), wl, num_rays=N, image_size=98, pixel_pitch=poly.dx).psf).T / 100
        d_mm = max(d_mm, float(np.max(np.abs(np_(poly.mono[i])[1:99, 1:99] - ref))))
    w = np.array([1.0, 2.0, 1.0])
    off = polychromatic_psf(wfs[1], w, grid_size=2 * N)
    want = np.zeros((2 * N, 2 * N))
    v = []
    psfs = []
    for wl in FCD:
        one = pupil_wavefront(o, (0.0, 1.0), wl, n_pupil=N, device=device)
        psfs.append((one, fft_psf(one)))
        K = int((one.intensity > 0).sum())
        v.append(K**2 / (wl * one.fno) ** 2)
    v = w * np.array(v) / np.sum(w * np.array(v))
    ref_c = wfs[1][0].center
    xs = (np.arange(2 * N) - N) * off.dx
    for (one, q), vl in zip(psfs, v):
        sx, sy = ((one.center[0] - ref_c[0]) * 1e3, (one.center[1] - ref_c[1]) * 1e3)
        want += vl * trig_resample(np_(q.psf), (xs - sx) / q.dx, (xs - sy) / q.dx)
    d_sum = float(np.max(np.abs(np_(off.psf) - want)))
    report(f"poly vs mmdft and resampled fft {device}", d_mmdft=d_mm, d_weighted_sum=d_sum, dx=off.dx,
           shift_um=float(off.shift[:, 1].max() - off.shift[:, 1].min()), peak=float(off.psf.max()))
    assert np.allclose(off.weights, v, rtol=1e-12, atol=0)
    assert d_mm < 1e-11 and d_sum < 1e-12
    assert float(off.shift[:, 1].max() - off.shift[:, 1].min()) > 0.1     # lateral color of more than 0.1 um


@pytest.mark.parametrize("device", DEVICES)
def test_polychromatic_perfect_lens_is_the_weighted_sum_of_airy_patterns(device):
    """ideal_relay (no dispersion; NA 0.0995) at 1.31, 1.55 and 1.65 um with the weights 1, 2, 1: the PSF is
    sum_l v_l Airy(r; NA, lambda_l), v_l ~ w_l / lambda_l^2 (equal K and F), peak 1; the MTF is
    sum_l u_l MTF_dl(nu; NA, lambda_l), u_l = w_l / sum(w) (equal pupil power). Pupil 256, grid 1024."""
    relay = ideal_relay()
    wls, w = (1.31, 1.55, 1.65), np.array([1.0, 2.0, 1.0])
    na = 2.0 / math.hypot(2.0, 20.0)
    wfs = pupil_wavefronts(relay, [(0.0, 0.0)], wls, n_pupil=256, device=device)[0]
    poly = polychromatic_psf(wfs, w, grid_size=1024)
    X, Y = np.meshgrid(poly.x, poly.y, indexing="ij")
    r = np.hypot(X, Y)
    v = w / np.array(wls) ** 2
    v = v / v.sum()
    want = sum(vl * airy(r, na, wl) for vl, wl in zip(v, wls))
    near = r < 3 * 0.61 * 1.65 / na * 1.7
    d_airy = float(np.max(np.abs(np_(poly.psf)[near] - want[near])))
    psfs = fft_psfs(wfs, 1024)
    mtf = polychromatic_mtf(wfs, psfs, w)
    u = w / w.sum()
    want_m = sum(ul * diffraction_limited_mtf(mtf.freq_tan, na, wl) for ul, wl in zip(u, wls))
    d_tan = float(np.max(np.abs(np_(mtf.tangential) - want_m)))
    d_sag = float(np.max(np.abs(np_(mtf.sagittal) - want_m)))
    report(f"poly perfect lens {device}", peak=float(poly.psf.max()), d_airy=d_airy, d_mtf_tan=d_tan,
           d_mtf_sag=d_sag, dx=poly.dx)
    assert np.allclose(poly.weights, v, rtol=1e-12)
    assert float(poly.psf[512, 512]) == pytest.approx(1.0, abs=1e-12)
    assert d_airy < 1e-3 and d_tan < 1e-3 and d_sag < 1e-3


def test_polychromatic_shift_two_equal_psfs():
    """The same wavefront twice, the second with its chief-ray point moved by s = 3 um in y and 2 um in x: the PSF
    is (PSF(x) + PSF(x - s)) / 2 and the MTF is MTF |cos(pi nu s)| on each axis (|1 + exp(-i 2 pi nu s)| / 2)."""
    wf = pupil_wavefront(aberrated_singlet(), n_pupil=64, device="cpu")
    sx, sy = 2e-3, 3e-3
    w2 = dataclasses.replace(wf, center=(wf.center[0] + sx, wf.center[1] + sy, wf.center[2]))
    q = fft_psf(wf)
    poly = polychromatic_psf([wf, w2], grid_size=96)
    xs = (np.arange(96) - 48) * poly.dx
    p = np_(q.psf)
    want = 0.5 * (trig_resample(p, xs / q.dx, xs / q.dx) + trig_resample(p, (xs - sx * 1e3) / q.dx,
                                                                         (xs - sy * 1e3) / q.dx))
    d_psf = float(np.max(np.abs(np_(poly.psf) - want)))
    m = polychromatic_mtf([wf, w2], [q, q])
    r = fft_mtf(wf, q)
    d_tan = float(np.max(np.abs(np_(m.tangential) - np_(r.tangential) * np.abs(np.cos(np.pi * m.freq_tan * sy)))))
    d_sag = float(np.max(np.abs(np_(m.sagittal) - np_(r.sagittal) * np.abs(np.cos(np.pi * m.freq_sag * sx)))))
    report("poly shift", d_psf=d_psf, d_mtf_tan=d_tan, d_mtf_sag=d_sag)
    assert d_psf < 1e-13 and d_tan < 1e-12 and d_sag < 1e-12


# ---------------------------------------------------------------------------------------------------------------
# One batch for all, and bad inputs
# ---------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("device", DEVICES)
def test_image_quality_equals_the_per_call_functions(device):
    """image_quality (3 fields, 3 wavelengths, 3 planes, weights 1 2 1) against pupil_wavefront, defocus, fft_psf,
    fft_mtf, polychromatic_psf and polychromatic_mtf called for each field, wavelength and plane."""
    o = dispersive_singlet()
    dz = (-40.0, 0.0, 40.0)
    w = (1.0, 2.0, 1.0)
    res = image_quality(o, FIELDS, FCD, dz, w, n_pupil=48, device=device)
    d_psf = d_mtf = d_poly = d_pmtf = 0.0
    for f, field in enumerate(FIELDS):
        per = [pupil_wavefront(o, field, wl, n_pupil=48, device=device) for wl in FCD]
        for z, d in enumerate(dz):
            planes = [defocus(q, d) for q in per]
            psfs = [fft_psf(q) for q in planes]
            for i in range(3):
                assert same_wavefront(res.wavefronts[f][i][z], planes[i])
                d_psf = max(d_psf, float((res.psfs[f][i][z].psf - psfs[i].psf).abs().max()))
                m = fft_mtf(planes[i], psfs[i])
                d_mtf = max(d_mtf, float((res.mtfs[f][i][z].tangential - m.tangential).abs().max()))
            poly = polychromatic_psf(planes, w)
            d_poly = max(d_poly, float((res.poly_psfs[f][z].psf - poly.psf).abs().max()))
            pm = polychromatic_mtf(planes, psfs, w)
            d_pmtf = max(d_pmtf, float((res.poly_mtfs[f][z].sagittal - pm.sagittal).abs().max()))
    report(f"image_quality {device}", d_psf=d_psf, d_mtf=d_mtf, d_poly=d_poly, d_poly_mtf=d_pmtf)
    assert res.wavelengths == list(FCD) and res.fields == list(FIELDS) and res.dz.tolist() == list(dz)
    assert d_psf <= 1e-14 and d_mtf <= 1e-14 and d_poly <= 1e-14 and d_pmtf <= 1e-14


def test_bad_inputs():
    o = aberrated_singlet()
    wf = pupil_wavefront(o, n_pupil=16, device="cpu")
    with pytest.raises(ValueError):
        defocus(dataclasses.replace(wf, ldir=None), 10.0)                # no ray data
    from optiland.materials import IdealMaterial
    from optiland.optic import Optic
    from optiland.physical_apertures import RadialAperture

    curved = Optic()                                                   # the singlet with a curved image surface
    curved.add_surface(index=0, thickness=40.0)
    curved.add_surface(index=1, radius=10.0, thickness=3.0, material=IdealMaterial(n=1.5), is_stop=True,
                       aperture=RadialAperture(r_max=3.0))
    curved.add_surface(index=2, radius=np.inf, thickness=30.0)
    curved.add_surface(index=3, radius=-50.0)
    curved.set_aperture(aperture_type="EPD", value=6.0)
    curved.set_field_type(field_type="object_height")
    curved.add_field(y=0.0)
    curved.add_wavelength(value=1.55, is_primary=True)
    with pytest.raises(ValueError):
        defocus(pupil_wavefront(curved, n_pupil=16, device="cpu"), 10.0)
    with pytest.raises(ValueError):
        pupil_wavefronts(o, [], device="cpu")
    with pytest.raises(ValueError):
        fft_psfs([wf, pupil_wavefront(o, n_pupil=20, device="cpu")])   # two pupil samplings
    with pytest.raises(ValueError):
        fft_mtfs([wf], fft_psfs([wf], 30))                             # 30 < 2 N - 1
    with pytest.raises(ValueError):
        polychromatic_psf([wf], dx=2 * fft_psf(wf).dx)                 # the window is above the period
    with pytest.raises(ValueError):
        polychromatic_psf([wf, wf], [1.0, -1.0])
    with pytest.raises(ValueError):
        polychromatic_psf([wf], [1.0, 1.0])
    with pytest.raises(ValueError):
        polychromatic_mtf([wf], [fft_psf(wf)], freq=[-1.0])
    with pytest.raises(ValueError):
        image_quality(o, dz=[], device="cpu")


# ---------------------------------------------------------------------------------------------------------------
# Review 2026-11-08-de7c55fed642: a glass image space, the sign of the lateral-color phase, the paths of the batch
# ---------------------------------------------------------------------------------------------------------------

def singlet_in_glass(after):
    """The singlet of systems.aberrated_singlet with its last gap in a glass of index 1.4 (IdealMaterial), the
    object heights 0 and 1 mm, 1.55 um; the image surface has air after it (after None, Optiland's default) or the
    index `after`."""
    from optiland.materials import IdealMaterial
    from optiland.optic import Optic
    from optiland.physical_apertures import RadialAperture

    o = Optic()
    o.add_surface(index=0, thickness=40.0)
    o.add_surface(index=1, radius=10.0, thickness=3.0, material=IdealMaterial(n=1.5), is_stop=True,
                  aperture=RadialAperture(r_max=3.0))
    o.add_surface(index=2, radius=np.inf, thickness=1.0, material=IdealMaterial(n=1.4))
    if after is None:
        o.add_surface(index=3)
    else:
        o.add_surface(index=3, material=IdealMaterial(n=after))
    o.set_aperture(aperture_type="EPD", value=6.0)
    o.set_field_type(field_type="object_height")
    o.add_field(y=0.0)
    o.add_field(y=1.0)
    o.add_wavelength(value=1.55, is_primary=True)
    o.image_solve()
    return o


@pytest.mark.parametrize("device", DEVICES)
def test_defocus_with_a_glass_image_space(device):
    """Image space in a glass (n 1.4). With air after the image surface (Optiland's default), Optiland's paraxial
    exit pupil is computed in that air and moves with the image surface by dz (1 - 1 / 1.4): defocus refuses the
    wavefront (ValueError). With the index 1.4 also after the image surface, the exit pupil stays fixed and defocus
    equals a new trace to the image surface moved by -80 and +80 um, on the axis and at the object height 1 mm."""
    air = singlet_in_glass(None)
    wf = pupil_wavefront(air, n_pupil=48, device=device)
    assert not wf.exit_pupil_fixed
    with pytest.raises(ValueError):
        defocus(wf, 80.0)
    z_moved = pupil_wavefront(moved(lambda: singlet_in_glass(None), 80.0), n_pupil=48, device=device).exit_pupil[2]
    shift = z_moved - wf.exit_pupil[2]                                  # mm
    r_air = wf.radius
    assert shift == pytest.approx(0.080 * (1 - 1 / 1.4), rel=1e-6)
    glass = singlet_in_glass(1.4)
    r_glass = 0.0
    worst = {"d_opd": 0.0, "d_radius_rel": 0.0, "d_exit_pupil_mm": 0.0}
    w_pv = []
    for field in ((0.0, 0.0), (0.0, 1.0)):
        wf = pupil_wavefront(glass, field, n_pupil=64, device=device)
        assert wf.exit_pupil_fixed and wf.n_image == pytest.approx(1.4)
        if field == (0.0, 0.0):
            r_glass = wf.radius
        for dz in (-80.0, 80.0):
            got = defocus(wf, dz)
            ref = pupil_wavefront(moved(lambda: singlet_in_glass(1.4), dz), field, n_pupil=64, device=device)
            ok = np_(ref.valid)
            assert np.array_equal(ok, np_(got.valid))
            worst["d_opd"] = max(worst["d_opd"], float(np.max(np.abs(np_(got.opd)[ok] - np_(ref.opd)[ok]))))
            worst["d_radius_rel"] = max(worst["d_radius_rel"], abs(got.radius / ref.radius - 1))
            worst["d_exit_pupil_mm"] = max(worst["d_exit_pupil_mm"],
                                           max(abs(u - v) for u, v in zip(ref.exit_pupil, wf.exit_pupil)))
            w = np_(got.opd)[ok & (np_(got.intensity) > 0)]
            w_pv.append(float(w.max() - w.min()))
    report(f"defocus glass image space {device}", exit_pupil_shift_mm=shift, radius_air_after_mm=r_air,
           radius_glass_after_mm=r_glass, gap_air_mm=float(air.surfaces[2].thickness),
           gap_glass_mm=float(glass.surfaces[2].thickness), **worst, opd_pv_min=min(w_pv),
           opd_pv_max=max(w_pv))
    assert worst["d_opd"] < 1e-9 and worst["d_radius_rel"] < 1e-12 and worst["d_exit_pupil_mm"] < 1e-9


def test_polychromatic_mtf_sign_of_the_lateral_color_phase():
    """Two PSFs of one wavelength: A of the singlet on the axis, B with 0.4 waves of Fringe Z7 and Z8 (coma along x
    and y: its OTF is complex on both axes) and the chief-ray point moved by s = (2, 3) um. The reference is the
    OTF of PSF_A(x) + PSF_B(x - s) on the period of the FFT PSF: B moved by trigonometric interpolation (exact),
    then the DFT. polychromatic_mtf (exp(-i 2 pi nu s)) agrees; the same reference with B moved by -s (the other
    sign) does not."""
    from optics_gpu.analysis import zernike_basis

    wa = pupil_wavefront(aberrated_singlet(), n_pupil=64, device="cpu")
    PX, PY = np.meshgrid(wa.p, wa.p, indexing="ij")
    Z = np_(zernike_basis(torch.as_tensor(PX.ravel()), torch.as_tensor(PY.ravel()), 8, "fringe"))
    coma = torch.as_tensor(0.4 * (Z[:, 6] + Z[:, 7]).reshape(PX.shape)) * wa.inside
    sx, sy = 2e-3, 3e-3                                                 # mm
    wb = dataclasses.replace(wa, opd=wa.opd + coma, center=(wa.center[0] + sx, wa.center[1] + sy, wa.center[2]))
    qa, qb = fft_psfs([wa, wb])
    m = polychromatic_mtf([wa, wb], [qa, qb])
    M, c = qa.psf.shape[0], qa.psf.shape[0] // 2
    xs = (np.arange(M) - M // 2) * qa.dx

    def reference(sign):
        total = np_(qa.psf) + trig_resample(np_(qb.psf), (xs - sign * sx * 1e3) / qa.dx,
                                            (xs - sign * sy * 1e3) / qa.dx)
        otf = np.fft.fft2(np.fft.ifftshift(total))
        otf = otf / otf[0, 0]
        return np.abs(otf[0, :c]), np.abs(otf[:c, 0])                  # tangential (along y), sagittal (along x)

    tan, sag = reference(+1)
    tan_f, sag_f = reference(-1)
    d = max(float(np.max(np.abs(np_(m.tangential) - tan))), float(np.max(np.abs(np_(m.sagittal) - sag))))
    d_flip = min(float(np.max(np.abs(np_(m.tangential) - tan_f))), float(np.max(np.abs(np_(m.sagittal) - sag_f))))
    report("poly mtf sign", d=d, d_other_sign=d_flip)
    assert d < 1e-5 and d_flip > 1e-3


@pytest.mark.parametrize("device", DEVICES)
def test_fft_batches_split_above_fft_elements(device, monkeypatch):
    """FFT_ELEMENTS set to 2 planes: fft_psfs and fft_mtfs of 5 wavefronts in 3 calls give the results of one call."""
    wfs = [w for row in pupil_wavefronts(dispersive_singlet(), FIELDS[:1], FCD, n_pupil=32, device=device)
           for w in row]
    wfs += [defocus(wfs[0], 30.0), defocus(wfs[1], -30.0)]
    one = fft_psfs(wfs, 64)
    m_one = fft_mtfs(wfs, one)
    monkeypatch.setattr(analysis, "FFT_ELEMENTS", 2 * 64 * 64)
    split = fft_psfs(wfs, 64)
    m_split = fft_mtfs(wfs, split)
    d_psf = max(float((a.psf - b.psf).abs().max()) for a, b in zip(one, split))
    d_mtf = max(float((a.tangential - b.tangential).abs().max()) for a, b in zip(m_one, m_split))
    report(f"fft split {device}", d_psf=d_psf, d_mtf=d_mtf)
    assert d_psf <= 1e-14 and d_mtf <= 1e-14


def test_copy_cache_eviction_and_no_key(monkeypatch):
    """COPY_CACHE set to 2: after 3 states the first copy is gone and the last two are kept. A state whose
    dictionary cannot be pickled (_state_key None) keeps no copy and gives the same wavefront."""
    clear_copy_cache()
    monkeypatch.setattr(analysis, "COPY_CACHE", 2)
    o = aberrated_singlet()
    keys = []
    for k in range(3):
        o.updater.set_thickness(float(o.surfaces[2].thickness) + 0.01, 2)
        kept = pupil_wavefront(o, n_pupil=16, device="cpu")
        keys.append(analysis._state_key(o.to_dict()))
    assert [key for key, _ in analysis._copies] == keys[1:]
    monkeypatch.setattr(analysis, "_state_key", lambda data: None)
    clear_copy_cache()
    again = pupil_wavefront(o, n_pupil=16, device="cpu")
    assert len(analysis._copies) == 0 and same_wavefront(kept, again)


@GPU
def test_copy_cache_holds_a_cpu_and_a_cuda_copy_of_one_optic():
    clear_copy_cache()
    o = aberrated_singlet()
    a = pupil_wavefront(o, n_pupil=32, device="cpu")
    b = pupil_wavefront(o, n_pupil=32, device="cuda")
    devices = sorted(dev for _, dev in analysis._copies)
    assert len(devices) == 2 and devices[0].startswith("cpu") and devices[1].startswith("cuda:")
    assert float((a.opd - b.opd.cpu()).abs().max()) < 1e-10


def test_image_quality_without_mtf():
    o = dispersive_singlet()
    full = image_quality(o, FIELDS[:2], FCD, (-20.0, 20.0), (1.0, 2.0, 1.0), n_pupil=32, device="cpu")
    part = image_quality(o, FIELDS[:2], FCD, (-20.0, 20.0), (1.0, 2.0, 1.0), n_pupil=32, device="cpu", mtf=False)
    assert part.mtfs is None and part.poly_mtfs is None and part.poly_psfs is not None
    for f in range(2):
        for z in range(2):
            assert torch.equal(part.poly_psfs[f][z].psf, full.poly_psfs[f][z].psf)
            for i in range(3):
                assert torch.equal(part.psfs[f][i][z].psf, full.psfs[f][i][z].psf)
