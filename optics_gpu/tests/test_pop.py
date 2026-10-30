"""Checks of the physical-optics propagation through a lens system.

A 4f system of two ideal thin lenses (Optiland "paraxial" surfaces, f1 = 2 mm, f2 = 5 mm) images a Gaussian source
of radius w0 to w0 f2/f1, and the coupling into a Gaussian mode of radius wf is (2 w1 wf / (w1^2 + wf^2))^2.
Optiland's thin lens maps ray slopes (tan), not direction cosines (sin): it does not meet the sine condition, so
the image of a beam with half-angle theta is wider by about theta^2 / 2 (tan - sin), not a solver error. The
low-angle case checks the method; the high-angle case checks that the error follows theta^2.
"""
import warnings

import numpy as np
import pytest

from optics_gpu.pop import gaussian, overlap_efficiency, propagate_through
from optics_gpu.systems import aberrated_singlet, four_f

warnings.filterwarnings("ignore", category=RuntimeWarning)


def image(w0, dx, ni, di):
    xs = np.arange(-ni // 2, ni // 2) * dx
    xi = np.arange(-64, 64) * di
    img = propagate_through(four_f(), gaussian(xs, xs, w0), dx, 1.55, xi, xi, n_pad=4)
    I = np.abs(img.E) ** 2
    X, _ = np.meshgrid(xi, xi, indexing="ij")
    return img, xi, 2 * np.sqrt((I * X**2).sum() / I.sum())


def test_low_angle_gaussian_imaging_and_coupling():
    img, xi, w_img = image(6.0, 0.5, 96, 0.75)                 # half-angle 0.082 rad
    w1 = 15.0
    assert w_img == pytest.approx(w1, rel=5e-3)
    for wf in (w1, 0.8 * w1, 1.3 * w1):
        analytic = (2 * w1 * wf / (w1**2 + wf**2)) ** 2
        assert overlap_efficiency(img.E, gaussian(xi, xi, wf)) == pytest.approx(analytic, abs=2e-3)


def test_slope_mapping_error_grows_as_theta_squared():
    _, _, w_lo = image(6.0, 0.5, 96, 0.75)                    # theta 0.082
    _, _, w_hi = image(2.0, 0.25, 64, 0.25)                   # theta 0.247
    err_lo, err_hi = w_lo / 15.0 - 1, w_hi / 5.0 - 1
    assert err_hi / err_lo == pytest.approx((0.247 / 0.082) ** 2, rel=0.25)


def test_power_is_conserved_through_a_lossless_system():
    """The sqrt(J) weight conserves power (the first version used 1/sqrt(J); this test found it: power x 39): the integral of |E|^2 over the image equals that of the source
    (the 4f system has no physical aperture, so every ray reaches the image; low angle, so the obliquity factor
    that the method omits is below 1e-2)."""
    dx, di = 0.5, 0.75
    xs = np.arange(-48, 48) * dx
    xi = np.arange(-64, 64) * di
    src = gaussian(xs, xs, 6.0)
    img = propagate_through(four_f(), src, dx, 1.55, xi, xi, n_pad=4)
    p_src = np.sum(np.abs(src) ** 2) * dx * dx
    p_img = np.sum(np.abs(img.E) ** 2) * di * di
    assert p_img / p_src == pytest.approx(1.0, abs=1e-2)


def test_aberrated_singlet_matches_optiland_huygens_psf():
    """A point source through a plano-convex singlet at 2f-2f with strong spherical aberration (6 mm stop at 40 mm):
    the image of a single-pixel source (flat angular spectrum) is the PSF; it must match Optiland's own scalar
    Huygens PSF (an independent method: pupil sampling and Huygens-Fresnel summation). The stop must be a physical
    aperture (RadialAperture): the EPD setting aims the ray generator but does not clip rays."""
    from optiland.psf import HuygensPSF

    o = aberrated_singlet()
    xi = np.arange(-40, 40) * 1.0
    src = np.zeros((64, 64), complex)
    src[32, 32] = 1.0
    img = propagate_through(o, src, 1.0, 1.55, xi, xi, na_cut=0.5, amp_cut=0.0, n_pad=16)
    ours = np.abs(img.E) ** 2
    ref = np.asarray(HuygensPSF(o, (0, 0), 1.55, num_rays=128, image_size=80, pixel_pitch=1e-3).psf)
    corr = (ours * ref).sum() / np.sqrt((ours**2).sum() * (ref**2).sum())
    assert corr > 0.99
