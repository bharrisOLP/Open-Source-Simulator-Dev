"""Image-quality analyses of a lens system on the GPU: the wavefront (OPD) map in the exit pupil, its Zernike fit,
the FFT PSF, the MTF, and the Huygens PSF of the image sum of optics_gpu.pop as the second PSF method.

All steps run on one PyTorch device ("cpu" or "cuda"; PyTorch on ROCm reports an AMD GPU as "cuda") in float64.
The trace uses Optiland's PyTorch backend on a checked copy of the optic (as optics_gpu.pop.trace_rays) and
Optiland's ray aiming (Optic.ray_tracer.trace_generic): one ray per point of the normalized pupil grid.

Definitions (lengths of Optiland in mm, wavelengths in um, the OPD in waves):

1. Pupil grid: Px and Py = np.linspace(-1, 1, N) (normalized entrance-pupil coordinates of Optiland's ray
   aiming); the points with Px^2 + Py^2 <= 1 are traced. The arrays are indexed [ix, iy] (x first, as the image
   fields of optics_gpu.pop); Optiland's arrays are indexed [iy, ix] (their transpose).
2. Reference sphere: the center is the image point of the chief ray (Px = Py = 0) on the image surface; the radius
   R is the distance from that point to the paraxial exit-pupil point (Optiland: paraxial.exit_pupil_point_gcs).
   Each ray goes back from its point on the image surface along its direction to the sphere (the root t > 0 of
   |X - t D - C|^2 = R^2 nearest to the image, else the other root); its optical path to the sphere is
   OPL - n t (n: the index of image space).
3. OPD: W = (P_chief - P_ray) / lambda, with P the optical path from the object to the reference sphere. W > 0
   where the ray has a shorter optical path than the chief ray. For an object at infinity and an angle field, the
   phase of the incident plane wave across the launch plane is added (n_obj D0 . (X0 - X0_chief)). The pupil
   function is P = sqrt(I) exp(-i 2 pi W) (I: the ray intensity, 0 for a clipped ray).
4. Zernike fit: least squares over the traced points with I > 0, unweighted, in (rho, phi) of (Px, Py):
   rho = sqrt(Px^2 + Py^2), phi = atan2(Py, Px); each term R_n^|m|(rho) cos(m phi) for m >= 0 and
   R_n^|m|(rho) sin(|m| phi) for m < 0. kind "fringe": the Fringe (University of Arizona) numbering
   j = (1 + (n + |m|)/2)^2 - 2|m| + floor((1 - sgn m)/2), no normalization (Z4 = 2 rho^2 - 1, Z9 = 6 rho^4 - 6 rho^2
   + 1); for j <= 36 this is the 37-term Fringe set of lens-design tools, j = 37 is (6, 6) here and (12, 0) there.
   kind "noll": the numbering of Noll (J. Opt. Soc. Am. 66, 207 (1976)), normalized to unit RMS over the unit disk
   (sqrt(n + 1) for m = 0, sqrt(2 (n + 1)) else; Z4 = sqrt(3) (2 rho^2 - 1)). The coefficients are in waves.
5. FFT PSF: the N x N pupil function in the middle of an M x M grid of zeros (M >= N; (M - N) // 2 zeros before),
   PSF = |fftshift(fft2(P))|^2 / K^2, K = the number of samples with |P| > 0: a diffraction-limited pupil of
   uniform amplitude has the peak 1 (the Strehl ratio is the peak of an aberrated PSF). Image sampling
   dx = lambda F (N - 1) / M, with the working F-number F = 1 / (2 sqrt(<(n sin a)^2>)) of the four marginal rays
   (a: the angle between a marginal ray and the chief ray; mean over the rays that are not clipped).
6. MTF: |fftshift(fft2(PSF))| normalized at zero frequency; tangential along y (pupil Py), sagittal along x, M // 2
   values each. One pupil sample of lag is the frequency n |D(+1) - D(-1)| / ((N - 1) lambda) (D: the direction of
   the marginal ray at pupil coordinate +1 or -1; for a flat image surface normal to z). With M >= 2 N the
   autocorrelation of the pupil does not wrap.

Items 2 to 6 use the conventions of Optiland 0.6.3 (MIT; its Wavefront "chief_ray" strategy, ZernikeFit,
ScalarFFTPSF and ScalarFFTMTF), so that the results compare one to one (tests/test_analysis.py). The code is
written from the equations above.

The Huygens PSF (huygens_psf) is the image sum of optics_gpu.pop for a point source (one source sample, a flat
angular spectrum): E(x, y) = sum_r a_r exp(i k (L'_r x + M'_r y + N'_r dz)), normalized by (sum_r |a_r|)^2, the
peak for all rays in phase. It is a Debye-type sum over the image-side directions, independent of the pupil grid
of the FFT method; it needs an object at a finite distance on the axis.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import torch

from .pop import COPY_TOL, TRACE_RAYS, _max_difference, image_sum, optiland_torch, trace_bundle

MAX_FNUM = 10000.0      # the largest working F-number (as Optiland)
CHECK_P = (np.array([0.0, 0.0, 0.7, -0.3, 0.5, -0.9]), np.array([0.0, 0.6, 0.0, -0.5, 0.5, 0.1]))   # check rays


@dataclass
class PupilWavefront:
    """The wavefront of one field point and one wavelength on the N x N pupil grid, tensors on one device."""
    p: np.ndarray            # the pupil axis (N,), np.linspace(-1, 1, N), the same for Px and Py
    opd: torch.Tensor        # W in waves (N, N) [ix, iy]; 0 outside the unit circle and for rays that failed
    intensity: torch.Tensor  # the ray intensity I (N, N); 0 outside, for failed and for clipped rays
    inside: torch.Tensor     # Px^2 + Py^2 <= 1 (N, N), bool
    xp: torch.Tensor         # the points of the rays on the reference sphere, mm (N, N) each (0 outside)
    yp: torch.Tensor
    zp: torch.Tensor
    radius: float            # the radius R of the reference sphere, mm
    center: tuple[float, float, float]   # the chief-ray image point, mm (global coordinates)
    wavelength: float        # um
    fno: float               # the working F-number
    freq_step: tuple[float, float]       # cycles/mm per pupil sample of lag: (tangential, sagittal)
    n_rays: int              # the traced rays (the points inside the unit circle)

    @property
    def n(self) -> int:
        return self.p.size

    @property
    def device(self) -> torch.device:
        return self.opd.device


@contextlib.contextmanager
def torch_optic(optic, wavelength: float, field: tuple[float, float], device: str) -> Iterator:
    """A copy of optic under Optiland's PyTorch backend in float64 on device (optiland_torch, torch.no_grad).
    The check rays CHECK_P (normalized pupil coordinates) of the field go through the original (on the NumPy
    backend if it keeps NumPy arrays) and through the copy; if they differ by more than COPY_TOL, a RuntimeError
    stops it. The original must trace in float64 (else a ValueError)."""
    import optiland.backend as be
    from optiland.backend.utils import is_torch_tensor, to_numpy
    from optiland.optic import Optic

    active = be.get_backend()
    if not any(is_torch_tensor(s.geometry.cs.z) for s in optic.surfaces.surfaces):   # made under NumPy
        be.set_backend("numpy")
    try:
        data = optic.to_dict()
        ref = optic.ray_tracer.trace_generic(field[0], field[1], CHECK_P[0].copy(), CHECK_P[1].copy(), wavelength,
                                             record=False)
    finally:
        be.set_backend(active)
    if to_numpy(ref.x).dtype != np.float64:
        raise ValueError("the check rays of the original optic are not float64: make the optic under the NumPy "
                         "backend or in float64")
    with optiland_torch(device), torch.no_grad():
        copy = Optic.from_dict(data)
        t = lambda v: be.array(v.copy())  # noqa: E731
        got = copy.ray_tracer.trace_generic(field[0], field[1], t(CHECK_P[0]), t(CHECK_P[1]), wavelength,
                                            record=False)
        diff = _max_difference(ref, got)
        if not diff <= COPY_TOL:
            raise RuntimeError(f"the PyTorch copy of the optic traces differently from the original (largest "
                               f"difference {diff:.3g})")
        yield copy


def _path_to_sphere(rays, center, R: float) -> torch.Tensor:
    """The length t (mm) from each ray point on the image surface back along its direction to the reference
    sphere (center, R): the root of |X - t D - C|^2 = R^2 (|D| = 1) as in item 2 of the module docstring."""
    xc, yc, zc = center
    dx, dy, dz = rays.x - xc, rays.y - yc, rays.z - zc       # the differences first (no cancellation of z^2)
    L, M, N = rays.L, rays.M, rays.N
    a = L * L + M * M + N * N
    b = -2 * (L * dx + M * dy + N * dz)
    c = dx * dx + dy * dy + dz * dz - R * R
    d = torch.clamp(b * b - 4 * a * c, min=0)
    t1 = (-b - torch.sqrt(d)) / (2 * a)
    t2 = (-b + torch.sqrt(d)) / (2 * a)
    return torch.where(t1 < 0, t2, t1)


def _float(v) -> float:
    from optiland.backend.utils import to_numpy

    return float(np.ravel(to_numpy(v))[0])


def pupil_wavefront(optic, field: tuple[float, float] = (0.0, 0.0), wavelength: float | None = None,
                    n_pupil: int = 128, device: str = "cuda", trace_chunk: int | None = None) -> PupilWavefront:
    """The OPD map W (waves) of the field point (Hx, Hy) (normalized field coordinates of the optic) on the
    n_pupil x n_pupil pupil grid, with the reference sphere of item 2 of the module docstring. The rays go in
    blocks of trace_chunk (None: TRACE_RAYS) through Optiland's PyTorch backend on device (float64)."""
    from optiland.fields.field_types import AngleField

    if device not in ("cpu", "cuda"):
        raise ValueError(f"device {device!r}: use 'cpu' or 'cuda' (the analyses need float64)")
    if n_pupil < 2:
        raise ValueError(f"n_pupil {n_pupil}: use at least 2")
    if trace_chunk is not None and trace_chunk < 1:
        raise ValueError(f"trace_chunk {trace_chunk}: use an integer of at least 1, or None")
    wl = float(optic.primary_wavelength if wavelength is None else wavelength)
    hx, hy = float(field[0]), float(field[1])
    # host values of the original optic: the exit-pupil point, the indices of image and object space
    xpp = [_float(v) for v in optic.paraxial.exit_pupil_point_gcs()]
    n_img = _float(optic.surfaces.n(optic.primary_wavelength)[-1])      # as Optiland's wavefront strategy
    n_img_wl = abs(_float(optic.image_surface.material_post.n(wl)))     # as Optiland's MTF frequency steps
    launch = optic.object_surface.is_infinite and isinstance(optic.fields.field_definition, AngleField)
    n_obj = _float(optic.object_surface.material_post.n(wl)) if launch else 1.0
    p = np.linspace(-1.0, 1.0, n_pupil)
    PX, PY = np.meshgrid(p, p, indexing="ij")
    inside_np = PX**2 + PY**2 <= 1
    dev = torch.device(device)
    if dev.type == "cuda":
        dev = torch.device("cuda", torch.cuda.current_device())
    step = trace_chunk if trace_chunk is not None else TRACE_RAYS
    with torch_optic(optic, wl, (hx, hy), device) as copy:
        tr = copy.ray_tracer.trace_generic
        f64 = {"dtype": torch.float64, "device": dev}
        # the chief ray and the four marginal rays (Px, Py) = (0, +-1), (+-1, 0)
        five = tr(hx, hy, torch.tensor([0.0, 0.0, 0.0, 1.0, -1.0], **f64), torch.tensor([0.0, 1.0, -1.0, 0.0, 0.0],
                  **f64), wl, record=False, retain_launch=launch)
        center = (five.x[0].item(), five.y[0].item(), five.z[0].item())
        R = math.dist(center, xpp)
        t0 = _path_to_sphere(five, center, R)
        p_chief = (five.opd[0] - n_img * t0[0]).item()
        chief_launch = five._launch_state if launch else None
        # the working F-number (Optiland's get_working_FNO) and the MTF frequency steps (flat image normal to z)
        D = torch.stack([five.L, five.M, five.N], dim=1)
        cosang = torch.clamp(D[1:] @ D[0], -1.0, 1.0)
        na2 = (n_img_wl * torch.sin(torch.arccos(cosang))) ** 2
        ok = five.i[1:] > 0
        mean_na2 = float((na2[ok].mean() if bool(ok.any()) else na2.mean()).item())
        fno = math.inf if mean_na2 <= 0 else 1.0 / (2.0 * math.sqrt(mean_na2))
        fno = min(fno, MAX_FNUM)
        band = torch.stack([D[1, :2] - D[2, :2], D[3, :2] - D[4, :2]]).norm(dim=1)
        steps = n_img_wl * band / ((n_pupil - 1) * wl * 1e-3)
        freq_step = (float(steps[0].item()), float(steps[1].item()))
        # all rays inside the unit circle, in blocks
        Px = torch.as_tensor(PX[inside_np], **f64)
        Py = torch.as_tensor(PY[inside_np], **f64)
        parts = []
        for s in range(0, Px.numel(), step):
            rays = tr(hx, hy, Px[s:s + step].clone(), Py[s:s + step].clone(), wl, record=False,
                      retain_launch=launch)
            t = _path_to_sphere(rays, center, R)
            path = rays.opd - n_img * t
            if launch:
                ls, lc = rays._launch_state, chief_launch
                path = path + n_obj * (ls.L * (ls.x - lc.x[0]) + ls.M * (ls.y - lc.y[0]) + ls.N * (ls.z - lc.z[0]))
            w = (p_chief - path) / (wl * 1e-3)
            xp, yp, zp = rays.x - t * rays.L, rays.y - t * rays.M, rays.z - t * rays.N
            ok = torch.isfinite(w) & torch.isfinite(xp) & torch.isfinite(yp) & torch.isfinite(zp)
            ok &= torch.isfinite(rays.i)
            zero = torch.zeros_like(w)
            parts.append([torch.where(ok, v, zero) for v in (w, rays.i, xp, yp, zp)])
            del rays, t, path
    inside = torch.as_tensor(inside_np, device=dev)
    out = []
    for k in range(5):
        full = torch.zeros((n_pupil, n_pupil), **f64)
        full[inside] = torch.cat([q[k] for q in parts])
        out.append(full)
    return PupilWavefront(p, out[0], out[1], inside, out[2], out[3], out[4], R, center, wl, fno, freq_step,
                          int(inside_np.sum()))


# ---------------------------------------------------------------------------------------------------------------
# Zernike polynomials
# ---------------------------------------------------------------------------------------------------------------

def zernike_indices(n_terms: int, kind: str = "fringe") -> list[tuple[int, int]]:
    """The (n, m) of the terms j = 1 .. n_terms (item 4 of the module docstring)."""
    if kind not in ("fringe", "noll"):
        raise ValueError(f"kind {kind!r}: use 'fringe' or 'noll'")
    if n_terms < 1:
        raise ValueError(f"n_terms {n_terms}: use at least 1")

    def number(n: int, m: int) -> int:
        if kind == "fringe":
            return (1 + (n + abs(m)) // 2) ** 2 - 2 * abs(m) + (1 if m < 0 else 0)
        c = 0 if (m > 0 and n % 4 <= 1) or (m < 0 and n % 4 >= 2) else 1
        return n * (n + 1) // 2 + abs(m) + c

    table = {}
    n = 0
    while len([j for j in table if j <= n_terms]) < n_terms:
        for m in range(-n, n + 1, 2):
            table[number(n, m)] = (n, m)
        n += 1
    return [table[j] for j in range(1, n_terms + 1)]


def _radial(n: int, m: int, rho: torch.Tensor) -> torch.Tensor:
    m = abs(m)
    out = torch.zeros_like(rho)
    for k in range((n - m) // 2 + 1):
        c = (-1) ** k * math.factorial(n - k) / (math.factorial(k) * math.factorial((n + m) // 2 - k)
                                                 * math.factorial((n - m) // 2 - k))
        out = out + c * rho ** (n - 2 * k)
    return out


def zernike_basis(px: torch.Tensor, py: torch.Tensor, n_terms: int, kind: str = "fringe") -> torch.Tensor:
    """The terms Z_1 .. Z_n_terms at the points (px, py) of the unit disk: a (points, n_terms) tensor."""
    rho, phi = torch.sqrt(px * px + py * py), torch.atan2(py, px)
    cols = []
    for n, m in zernike_indices(n_terms, kind):
        norm = 1.0 if kind == "fringe" else math.sqrt((2 * n + 2) / (2 if m == 0 else 1))
        ang = torch.cos(m * phi) if m >= 0 else torch.sin(-m * phi)
        cols.append(norm * _radial(n, m, rho) * ang)
    return torch.stack(cols, dim=1)


@dataclass
class ZernikeFit:
    coeffs: torch.Tensor     # (n_terms,) waves, on the device of the wavefront
    kind: str
    indices: list[tuple[int, int]]
    rms_residual: float      # the RMS of W minus the fit over the fitted points, waves
    cond: float              # the condition number of the scaled normal matrix


def zernike_fit(wf: PupilWavefront, n_terms: int = 36, kind: str = "fringe") -> ZernikeFit:
    """The least-squares Zernike fit of the OPD map (item 4 of the module docstring). The normal matrix A^T A and
    A^T W are made on the device; the small n_terms x n_terms system is solved on the host (NumPy lstsq, columns
    scaled to unit norm)."""
    PX, PY = np.meshgrid(wf.p, wf.p, indexing="ij")
    use = wf.inside & (wf.intensity > 0)
    if int(use.sum().item()) < n_terms:
        raise ValueError(f"{int(use.sum().item())} pupil points for {n_terms} terms: use a finer pupil grid")
    f64 = {"dtype": torch.float64, "device": wf.device}
    px, py = torch.as_tensor(PX, **f64)[use], torch.as_tensor(PY, **f64)[use]
    A = zernike_basis(px, py, n_terms, kind)
    z = wf.opd[use]
    G = (A.T @ A).cpu().numpy()
    b = (A.T @ z).cpu().numpy()
    d = np.sqrt(np.diag(G))
    Gs = G / np.outer(d, d)
    y = np.linalg.lstsq(Gs, b / d, rcond=None)[0]
    c = torch.as_tensor(y / d, **f64)
    res = z - A @ c
    return ZernikeFit(c, kind, zernike_indices(n_terms, kind), float(torch.sqrt((res * res).mean()).item()),
                      float(np.linalg.cond(Gs)))


# ---------------------------------------------------------------------------------------------------------------
# PSF and MTF
# ---------------------------------------------------------------------------------------------------------------

@dataclass
class FFTPSF:
    psf: torch.Tensor        # (M, M) [ix, iy], the diffraction-limited peak is 1 (Strehl units)
    dx: float                # um, the image sampling on both axes
    x: np.ndarray            # um, the image coordinates of the samples on x (the chief-ray point at index M // 2)
    y: np.ndarray
    pupil: torch.Tensor      # the N x N pupil function (complex128)


def pupil_function(wf: PupilWavefront) -> torch.Tensor:
    """P = sqrt(I) exp(-i 2 pi W) inside the unit circle, 0 outside (complex128, (N, N) [ix, iy])."""
    amp = torch.sqrt(wf.intensity)
    P = torch.polar(amp, -2 * math.pi * wf.opd)
    return torch.where(wf.inside, P, torch.zeros_like(P))


def fft_psf(wf: PupilWavefront, grid_size: int | None = None) -> FFTPSF:
    """The FFT PSF of item 5 of the module docstring on an M x M grid (grid_size M; None: 2 N, M >= N)."""
    N = wf.n
    M = 2 * N if grid_size is None else int(grid_size)
    if M < N:
        raise ValueError(f"grid_size {M} is below the pupil sampling {N}")
    P = pupil_function(wf)
    count = int((P.abs() > 0).sum().item())
    if count == 0:
        raise ValueError("no ray reaches the image: the pupil function is zero")
    b = (M - N) // 2
    big = torch.zeros((M, M), dtype=torch.complex128, device=P.device)
    big[b:b + N, b:b + N] = P
    a = torch.fft.fftshift(torch.fft.fft2(big))
    psf = (a.real**2 + a.imag**2) / float(count) ** 2
    dx = wf.wavelength * wf.fno * (N - 1) / M
    k = (np.arange(M) - M // 2) * dx
    return FFTPSF(psf, dx, wf.center[0] * 1e3 + k, wf.center[1] * 1e3 + k, P)


@dataclass
class MTF:
    freq_tan: np.ndarray     # cycles/mm (M // 2 values)
    freq_sag: np.ndarray
    tangential: torch.Tensor  # along y (pupil Py)
    sagittal: torch.Tensor    # along x (pupil Px)


def fft_mtf(wf: PupilWavefront, psf: FFTPSF) -> MTF:
    """The MTF of item 6 of the module docstring from the FFT PSF of wf."""
    M = psf.psf.shape[0]
    otf = torch.fft.fftshift(torch.fft.fft2(psf.psf)).abs()
    c = M // 2
    dc = otf[c, c]
    tan = torch.clamp(otf[c, c:][:c] / dc, 0.0, 1.0)
    sag = torch.clamp(otf[c:, c][:c] / dc, 0.0, 1.0)
    j = np.arange(c)
    return MTF(j * wf.freq_step[0], j * wf.freq_step[1], tan, sag)


def diffraction_limited_mtf(freq, na: float, wavelength: float) -> np.ndarray:
    """The MTF of an aberration-free circular pupil of uniform amplitude (incoherent):
    (2 / pi) (acos s - s sqrt(1 - s^2)), s = freq / (2 NA / lambda), 0 for s >= 1 (freq in cycles/mm, lambda in
    um; Goodman, Introduction to Fourier Optics, 3rd ed., eq. 6-32)."""
    s = np.clip(np.asarray(freq, float) / (2 * na / (wavelength * 1e-3)), 0.0, 1.0)
    return 2 / np.pi * (np.arccos(s) - s * np.sqrt(1 - s**2))


def airy(r_um, na: float, wavelength: float) -> np.ndarray:
    """The Airy pattern (2 J1(v) / v)^2, v = 2 pi NA r / lambda, peak 1 (Born and Wolf, Principles of Optics,
    section 8.5.2)."""
    from scipy.special import j1

    v = 2 * np.pi * na * np.asarray(r_um, float) / wavelength
    out = np.ones_like(v)
    nz = v != 0
    out[nz] = (2 * j1(v[nz]) / v[nz]) ** 2
    return out


def huygens_psf(optic, wavelength: float, x_img, y_img, dz: float = 0.0, n_src: int = 64, dx_src: float = 1.0,
                n_pad: int = 16, na_cut: float = 0.5, device: str = "cuda",
                trace_chunk: int | None = None) -> torch.Tensor:
    """The PSF of the on-axis object point by the image sum of optics_gpu.pop (trace_bundle and image_sum):
    |E|^2 / (sum_r |a_r|)^2 on the grid (x_img, y_img) in um, dz um after the image surface. The point source is
    one sample of an n_src x n_src source of pitch dx_src um, padded n_pad times: the direction step is
    lambda / (n_src n_pad dx_src) and the image field repeats with the period n_src n_pad dx_src um (keep the image
    grid inside it). Directions up to na_cut are traced; the stop of the optic must be a physical aperture."""
    src = np.zeros((n_src, n_src), complex)
    src[n_src // 2, n_src // 2] = 1.0
    bundle = trace_bundle(optic, src, dx_src, wavelength, na_cut, 0.0, n_pad, device, trace_chunk)
    E = image_sum(bundle, x_img, y_img, dz=dz)
    peak = sum(b[3].abs().sum() for b in bundle.blocks) ** 2
    return (E.real**2 + E.imag**2) / peak
