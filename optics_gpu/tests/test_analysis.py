"""Checks of the image-quality analyses (optics_gpu.analysis): the OPD map with its reference sphere, the Zernike fit,
the FFT PSF, the MTF and the Huygens PSF of the image sum.

- A perfect lens (systems.ideal_relay: two equal ideal thin lenses with a physical stop): the OPD is zero, the FFT
  PSF and the Huygens PSF are the Airy pattern, the MTF is the diffraction-limited MTF (analytic).
- A pure defocus (the image surface of the perfect lens moved by dz): the traced OPD against the exact OPD of a
  spherical wave on the reference sphere, W(u) = dz (1 - cos u) + sqrt(R^2 - dz^2 sin^2 u) - R (u: the image-side
  ray angle, R: the radius of the reference sphere), and the Zernike defocus term against W020 = NA^2 dz / 2.
- The aberrated singlet (and two off-axis systems) against Optiland's CPU functions: Wavefront, ZernikeFit,
  ZernikeOPD, ScalarFFTPSF, ScalarFFTMTF and HuygensPSF.
- The same code on the GPU against the CPU.
The "cuda" cases run only where PyTorch has a GPU (CUDA or ROCm). The observed values are in README.md (Checks).
"""
import math
import warnings

import numpy as np
import pytest
import torch

from optics_gpu.analysis import (airy, diffraction_limited_mtf, fft_mtf, fft_psf, huygens_psf, pupil_wavefront,
                                 zernike_basis, zernike_fit, zernike_indices)
from optics_gpu.systems import aberrated_singlet, ideal_relay

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
DEVICES = ["cpu", pytest.param("cuda", marks=GPU)]
WL = 1.55


def report(name, **values):
    print(f"\n[{name}] " + ", ".join(f"{k} {v:.3g}" if isinstance(v, float) else f"{k} {v}"
                                     for k, v in values.items()))


def np_(t):
    return t.detach().cpu().numpy()


# ---------------------------------------------------------------------------------------------------------------
# Zernike polynomials
# ---------------------------------------------------------------------------------------------------------------

def test_zernike_indices_and_values_are_those_of_optiland():
    """The (n, m) of the Fringe and Noll numbering, and the values of the terms at random points, against
    Optiland's ZernikeFringe and ZernikeNoll (the same definitions; tests/test_analysis compares the fits too)."""
    from optiland.zernike import ZernikeFringe, ZernikeNoll

    assert zernike_indices(9, "fringe") == [(0, 0), (1, 1), (1, -1), (2, 0), (2, 2), (2, -2), (3, 1), (3, -1), (4, 0)]
    assert zernike_indices(11, "noll")[3] == (2, 0) and zernike_indices(11, "noll")[10] == (4, 0)
    rng = np.random.default_rng(3)
    r, phi = np.sqrt(rng.uniform(0, 1, 500)), rng.uniform(-np.pi, np.pi, 500)
    x, y = r * np.cos(phi), r * np.sin(phi)
    for kind, cls in (("fringe", ZernikeFringe), ("noll", ZernikeNoll)):
        ref = cls(np.ones(45))
        assert [tuple(map(int, t)) for t in ref.indices] == zernike_indices(45, kind)
        ours = np_(zernike_basis(torch.as_tensor(x), torch.as_tensor(y), 45, kind))
        theirs = np.stack(ref.terms(r, phi), axis=1)
        assert np.max(np.abs(ours - theirs)) < 1e-11
    assert zernike_indices(37, "fringe")[36] == (6, 6)


def test_noll_terms_are_orthonormal_on_the_unit_disk():
    """<Z_i Z_j> over the unit disk = delta_ij for Noll's normalization (Noll 1976), and the Fringe terms are
    orthogonal with <Z_j^2> = 1 / (n + 1) (m = 0) or 1 / (2 (n + 1)) (m != 0); midpoint rule on a 2000 x 2000
    grid (the error of the rule at the edge of the disk is about 1e-3)."""
    g = (np.arange(2000) + 0.5) / 1000 - 1
    X, Y = np.meshgrid(g, g, indexing="ij")
    ins = X**2 + Y**2 <= 1
    x, y = torch.as_tensor(X[ins]), torch.as_tensor(Y[ins])
    for kind in ("noll", "fringe"):
        A = zernike_basis(x, y, 28, kind)
        G = np_(A.T @ A) / ins.sum()
        if kind == "noll":
            want = np.eye(28)
        else:
            want = np.diag([1 / (n + 1) if m == 0 else 1 / (2 * (n + 1)) for n, m in zernike_indices(28, kind)])
        report(f"orthogonality {kind}", max_err=float(np.max(np.abs(G - want))))
        assert np.max(np.abs(G - want)) < 1e-3


# ---------------------------------------------------------------------------------------------------------------
# The perfect lens
# ---------------------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def relay():
    return ideal_relay(f=20.0, r_stop=2.0)


NA_RELAY = 2.0 / math.hypot(2.0, 20.0)       # image-side NA of ideal_relay(20, 2): sin(atan(2 / 20))


@pytest.mark.parametrize("device", DEVICES)
def test_perfect_lens_gives_zero_opd_the_airy_pattern_and_the_diffraction_limited_mtf(relay, device):
    N, M = 256, 1024
    wf = pupil_wavefront(relay, n_pupil=N, device=device)
    assert wf.radius == pytest.approx(20.0, abs=1e-9)                  # exit pupil 2f after lens 2, image at f
    assert wf.fno == pytest.approx(1 / (2 * NA_RELAY), rel=1e-12)
    assert wf.freq_step[0] == pytest.approx(2 * NA_RELAY / ((N - 1) * WL * 1e-3), rel=1e-12)
    assert wf.freq_step[1] == pytest.approx(wf.freq_step[0], rel=1e-12)
    w_max = float(wf.opd.abs().max())
    psf = fft_psf(wf, M)
    p = np_(psf.psf)
    c = M // 2
    peak = float(p[c, c])
    X, Y = np.meshgrid(psf.x, psf.y, indexing="ij")
    r = np.hypot(X, Y)
    near = r < 3 * 0.61 * WL / NA_RELAY * 1.7                          # the first 5 rings
    d_airy = float(np.max(np.abs(p[near] - airy(r[near], NA_RELAY, WL))))
    mtf = fft_mtf(wf, psf)
    dl = diffraction_limited_mtf(mtf.freq_tan, NA_RELAY, WL)
    d_tan = float(np.max(np.abs(np_(mtf.tangential) - dl)))
    d_sag = float(np.max(np.abs(np_(mtf.sagittal) - dl)))
    report(f"perfect lens {device}", w_max=w_max, peak=peak, d_airy=d_airy, d_mtf_tan=d_tan, d_mtf_sag=d_sag,
           dx_um=psf.dx, rays=wf.n_rays)
    assert w_max < 1e-9
    assert peak == pytest.approx(1.0, abs=1e-12)
    assert d_airy < 1e-3
    assert d_tan < 1e-3 and d_sag < 1e-3


@pytest.mark.parametrize("device", DEVICES)
def test_huygens_psf_of_the_image_sum_is_the_airy_pattern(relay, device):
    """The image sum of a point source through the perfect lens: a Debye sum over the image-side directions,
    uniform in direction cosine up to the NA (equal lenses), so its PSF is the Airy pattern; on the samples of the
    FFT PSF it agrees with the FFT PSF."""
    N, M = 256, 1024
    psf = fft_psf(pupil_wavefront(relay, n_pupil=N, device=device), M)
    c, h = M // 2, 24
    xs = psf.x[c - h:c + h + 1]
    I = np_(huygens_psf(relay, WL, xs, xs, na_cut=0.15, device=device))
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    d_airy = float(np.max(np.abs(I - airy(np.hypot(X, Y), NA_RELAY, WL))))
    d_fft = float(np.max(np.abs(I - np_(psf.psf)[c - h:c + h + 1, c - h:c + h + 1])))
    report(f"huygens perfect lens {device}", peak=float(I[h, h]), d_airy=d_airy, d_fft=d_fft)
    assert I[h, h] == pytest.approx(1.0, abs=1e-9)
    assert d_airy < 2e-3 and d_fft < 2e-3


# ---------------------------------------------------------------------------------------------------------------
# Pure defocus
# ---------------------------------------------------------------------------------------------------------------

def exact_defocus_waves(p, r_stop, f, dz, R):
    """W(u) = dz (1 - cos u) + sqrt(R^2 - dz^2 sin^2 u) - R in waves on the pupil grid of ideal_relay: the ray at
    the normalized pupil radius rho has tan u = rho r_stop / f (mm)."""
    PX, PY = np.meshgrid(p, p, indexing="ij")
    h = np.hypot(PX, PY) * r_stop
    s, c = h / np.hypot(h, f), f / np.hypot(h, f)
    return (dz * (1 - c) + np.sqrt(R**2 - dz**2 * s**2) - R) / (WL * 1e-3), PX, PY


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("r_stop,dz", [(2.0, 0.02), (2.0, -0.02), (1.0, 0.02)])
def test_pure_defocus_against_the_exact_spherical_wave_and_w020(device, r_stop, dz):
    f, N = 20.0, 128
    wf = pupil_wavefront(ideal_relay(f=f, r_stop=r_stop, dz=dz), n_pupil=N, device=device)
    assert wf.radius == pytest.approx(f - dz, abs=1e-9)               # exit pupil 2f after lens 2
    W, PX, PY = exact_defocus_waves(wf.p, r_stop, f, dz, f - dz)
    ins = np_(wf.inside)
    d_map = float(np.max(np.abs(np_(wf.opd)[ins] - W[ins])))
    na = r_stop / math.hypot(r_stop, f)
    w020 = na**2 * dz / 2 / (WL * 1e-3)                                 # waves
    out = {}
    for kind, z4 in (("fringe", w020 / 2), ("noll", w020 / (2 * math.sqrt(3)))):
        fit = zernike_fit(wf, 36, kind)
        A = np_(zernike_basis(torch.as_tensor(PX[ins]), torch.as_tensor(PY[ins]), 36, kind))
        exact = np.linalg.lstsq(A, W[ins], rcond=None)[0]
        c = np_(fit.coeffs)
        out[kind] = (c[3], exact[3], z4)
        assert abs(c[3] - exact[3]) < 1e-9                              # the fit of the traced map = of the exact
        others = np.delete(c - exact, 3)
        assert np.max(np.abs(others)) < 1e-9
        assert c[3] / z4 - 1 == pytest.approx(exact[3] / z4 - 1, abs=1e-8)
        # small NA: c4 / (W020 / 2) - 1 = t^2 / 4 - dz / R to first order (t = r_stop / f = tan u_max; README)
        assert c[3] / z4 - 1 == pytest.approx((r_stop / f) ** 2 / 4 - dz / (f - dz), abs=1e-4)
    report(f"defocus {device} r {r_stop} dz {dz}", d_map=d_map, w020=w020,
           c4_fringe=out["fringe"][0], rel_fringe=out["fringe"][0] / out["fringe"][2] - 1,
           c4_noll=out["noll"][0], rel_noll=out["noll"][0] / out["noll"][2] - 1)
    assert d_map < 1e-9


# ---------------------------------------------------------------------------------------------------------------
# Against Optiland (CPU, NumPy backend)
# ---------------------------------------------------------------------------------------------------------------

def singlet_off_axis():
    """systems.aberrated_singlet with a second field: object height 1 mm."""
    o = aberrated_singlet()
    o.add_field(y=1.0)
    return o


def singlet_at_infinity():
    """The singlet of systems.aberrated_singlet with the object at infinity and an angle field of 2 degrees."""
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
    o.add_wavelength(value=WL, is_primary=True)
    o.image_solve()
    return o


SYSTEMS = {"singlet": (aberrated_singlet, (0.0, 0.0)), "singlet_off_axis": (singlet_off_axis, (0.0, 1.0)),
           "singlet_at_infinity": (singlet_at_infinity, (0.0, 1.0))}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", list(SYSTEMS))
def test_against_optiland_wavefront_zernike_fft_psf_and_mtf(device, name):
    from optiland.mtf import ScalarFFTMTF
    from optiland.psf.fft import ScalarFFTPSF
    from optiland.wavefront import Wavefront
    from optiland.zernike import ZernikeFit as OptilandZernikeFit

    make, field = SYSTEMS[name]
    o = make()
    N, M = 128, 256
    wf = pupil_wavefront(o, field=field, n_pupil=N, device=device)
    data = Wavefront(o, fields=[field], wavelengths=[WL], num_rays=N, distribution="uniform").get_data(field, WL)
    ins = np_(wf.inside).T                                              # Optiland's [iy, ix] order
    opd, inten = np_(wf.opd).T[ins], np_(wf.intensity).T[ins]
    d_opd = float(np.max(np.abs(opd - np.asarray(data.opd))))
    assert np.array_equal(inten, np.asarray(data.intensity))
    assert wf.radius == pytest.approx(float(data.radius), rel=1e-12)
    for v, ref in ((wf.xp, data.pupil_x), (wf.yp, data.pupil_y), (wf.zp, data.pupil_z)):
        assert np.max(np.abs(np_(v).T[ins] - np.asarray(ref))) < 1e-9
    # Zernike fits on the same points
    p = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(p, p)
    use = inten > 0
    d_z = {}
    for kind in ("fringe", "noll"):
        ours = np_(zernike_fit(wf, 37, kind).coeffs)
        ref = np.asarray(OptilandZernikeFit(X[ins][use], Y[ins][use], np.asarray(data.opd)[use], kind, 37).coeffs)
        d_z[kind] = float(np.max(np.abs(ours - ref)))
        assert d_z[kind] < 1e-11
    # FFT PSF and MTF
    psf = fft_psf(wf, M)
    ref_psf = np.asarray(ScalarFFTPSF(o, field, WL, num_rays=N, grid_size=M).psf) / 100
    d_psf = float(np.max(np.abs(np_(psf.psf).T - ref_psf)))
    mtf = fft_mtf(wf, psf)
    ref_mtf = ScalarFFTMTF(o, [field], WL, num_rays=N, grid_size=M)
    d_mtf = max(float(np.max(np.abs(np_(mtf.tangential) - np.asarray(ref_mtf.mtf[0][0])))),
                float(np.max(np.abs(np_(mtf.sagittal) - np.asarray(ref_mtf.mtf[0][1])))))
    assert np.allclose(mtf.freq_tan, np.asarray(ref_mtf.freq_tang[0]), rtol=1e-12)
    assert np.allclose(mtf.freq_sag, np.asarray(ref_mtf.freq_sag[0]), rtol=1e-12)
    report(f"optiland {name} {device}", d_opd=d_opd, w_pv=float(opd.max() - opd.min()), d_fringe=d_z["fringe"],
           d_noll=d_z["noll"], strehl=float(np_(psf.psf).max()), d_psf=d_psf, d_mtf=d_mtf, fno=wf.fno)
    assert d_opd < 1e-10
    assert d_psf < 1e-11 and d_mtf < 1e-11


@pytest.mark.parametrize("device", DEVICES)
def test_singlet_zernike_against_optiland_zernike_opd_on_its_hexapolar_rings(device):
    """ZernikeOPD (Optiland's default: 15 hexapolar rings, Fringe, 37 terms) samples the pupil on other points
    than the uniform grid: the coefficients differ by the projection of the residual of the fit."""
    from optiland.wavefront import ZernikeOPD

    o = aberrated_singlet()
    fit = zernike_fit(pupil_wavefront(o, n_pupil=128, device=device), 37, "fringe")
    ref = np.asarray(ZernikeOPD(o, (0.0, 0.0), WL, num_rings=15, zernike_type="fringe", num_terms=37).coeffs)
    ours = np_(fit.coeffs)
    d = float(np.max(np.abs(ours - ref)))
    report(f"zernike_opd {device}", d=d, c4=float(ours[3]), c9=float(ours[8]), c16=float(ours[15]),
           rms_residual=fit.rms_residual, cond=fit.cond)
    assert d < 1e-5


@pytest.mark.parametrize("device", DEVICES)
def test_singlet_huygens_psf_against_optiland_and_the_fft_psf(device):
    """The Huygens PSF of the image sum against Optiland's HuygensPSF (pupil-sphere wavelets; 80 x 80 points of
    1 um) and against the FFT PSF of this module on its own samples (pupil 256, grid 1024)."""
    from optiland.psf import HuygensPSF

    o = aberrated_singlet()
    ref = HuygensPSF(o, (0, 0), WL, num_rays=128, image_size=80, pixel_pitch=1e-3)
    xs = np.linspace(float(ref.cx) - 0.04, float(ref.cx) + 0.04, 80) * 1e3
    I = np_(huygens_psf(o, WL, xs, xs, device=device))
    R = np.asarray(ref.psf).T / 100
    corr = float((I * R).sum() / np.sqrt((I**2).sum() * (R**2).sum()))
    d_ref = float(np.max(np.abs(I - R)) / R.max())
    N, M = 256, 1024
    psf = fft_psf(pupil_wavefront(o, n_pupil=N, device=device), M)
    c, h = M // 2, 20
    xf = psf.x[c - h:c + h + 1]
    If = np_(huygens_psf(o, WL, xf, xf, device=device))
    F = np_(psf.psf)[c - h:c + h + 1, c - h:c + h + 1]
    d_fft = float(np.max(np.abs(If - F)) / F.max())
    corr_fft = float((If * F).sum() / np.sqrt((If**2).sum() * (F**2).sum()))
    report(f"huygens singlet {device}", corr=corr, d_ref=d_ref, peak=float(I.max()), peak_ref=float(R.max()),
           d_fft=d_fft, corr_fft=corr_fft, peak_fft=float(F.max()), peak_on_fft=float(If.max()), dx_fft=psf.dx)
    assert corr > 0.995 and corr_fft > 0.995
    assert 0.85 < I.max() / R.max() < 1 and 0.85 < If.max() / F.max() < 1     # the peaks differ (see README)


# ---------------------------------------------------------------------------------------------------------------
# GPU against CPU, and bad inputs
# ---------------------------------------------------------------------------------------------------------------

@GPU
def test_gpu_matches_cpu():
    o = aberrated_singlet()
    res = {}
    for dev in ("cpu", "cuda"):
        wf = pupil_wavefront(o, n_pupil=200, device=dev, trace_chunk=7000)
        psf = fft_psf(wf, 512)
        mtf = fft_mtf(wf, psf)
        res[dev] = (np_(wf.opd), np_(zernike_fit(wf, 37).coeffs), np_(psf.psf), np_(mtf.tangential),
                    np_(mtf.sagittal))
    d = [float(np.max(np.abs(a - b))) for a, b in zip(res["cpu"], res["cuda"])]
    report("gpu vs cpu", d_opd=d[0], d_zernike=d[1], d_psf=d[2], d_mtf_tan=d[3], d_mtf_sag=d[4])
    assert max(d) < 1e-10


def test_one_block_equals_blocks():
    o = aberrated_singlet()
    a = pupil_wavefront(o, n_pupil=64, device="cpu")
    b = pupil_wavefront(o, n_pupil=64, device="cpu", trace_chunk=500)
    assert torch.equal(a.opd, b.opd) and torch.equal(a.intensity, b.intensity)


def test_bad_inputs():
    o = aberrated_singlet()
    with pytest.raises(ValueError):
        pupil_wavefront(o, device="mps")
    with pytest.raises(ValueError):
        pupil_wavefront(o, n_pupil=1, device="cpu")
    with pytest.raises(ValueError):
        pupil_wavefront(o, n_pupil=16, device="cpu", trace_chunk=0)
    wf = pupil_wavefront(o, n_pupil=16, device="cpu")
    with pytest.raises(ValueError):
        fft_psf(wf, 8)
    with pytest.raises(ValueError):
        zernike_fit(wf, 36, "standard")
    with pytest.raises(ValueError):
        zernike_fit(wf, 400)
