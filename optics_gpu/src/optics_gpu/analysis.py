# SPDX-License-Identifier: MIT
# Parts of this file follow Optiland 0.6.3 (MIT; Copyright (c) 2024 Kramer Harrison): pupil_wavefronts and
# _chief_and_marginal follow ChiefRayStrategy.compute_wavefront_data step by step (the reference sphere, the optical
# path to it, the restored launch-plane phase of an object at infinity, the mask of valid rays), get_working_FNO
# (the working F-number) and ScalarFFTMTF._get_mtf_frequency_steps (the MTF frequency step); _path_to_sphere
# follows SphericalReference.path_length; fft_psf and fft_psfs follow the padding and the normalization of
# ScalarFFTPSF; the kernels of polychromatic_psf use the conventions of MMDFTPSF._compute_kernels (the pupil index
# arange(N) - N // 2, the image index arange(M) - M // 2, the period lambda F (N - 1), left kernel x pupil x right
# kernel, the normalization by K^2; the matrix DFT itself is that of Soummer et al. 2007). The index of image space
# differs from Optiland (item 2 of the module docstring). See THIRD_PARTY_NOTICES.md and LICENSES/Optiland-MIT.txt.
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
   OPL - n t. n is the index of image space (the medium before the image surface) at the traced wavelength; the
   same n is in the working F-number (item 5) and the MTF frequency step (item 6). Optiland 0.6.3 uses the medium
   after the image surface (air by default): at the primary wavelength for the OPD, at the traced wavelength for
   the working F-number and the MTF frequency step. The results are the same when the two media are the same and,
   for the OPD, that medium is not dispersive or the traced wavelength is the primary one. For an image in a
   medium (an immersed image) with air after the image surface they differ: the OPD, and F, the PSF step dx and
   the MTF axis by the factor n.
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
   the marginal ray at pupil coordinate +1 or -1; for a flat image surface normal to z). The full autocorrelation
   of the pupil has 2 N - 1 lags: fft_mtf needs M >= 2 N - 1 (a ValueError below), so that the whole
   autocorrelation is free of wrap. The returned values (up to the cutoff) stay exact down to M = 2 N - 3 (a check
   in the review reviews/2026-11-08-0e8cf1910c4b.md); the bound keeps a margin.

Items 2 to 6 use the conventions of Optiland 0.6.3 (MIT; its Wavefront "chief_ray" strategy, ZernikeFit,
ScalarFFTPSF and ScalarFFTMTF), so that the results compare one to one (tests/test_analysis.py, systems with
air before and after the image surface). The parts named in the notice at the top of this file follow Optiland's
code. Differences: the index n of item 2 (the conditions of agreement are there), and the PSF peak 1 for a perfect
pupil (Optiland: 100).

The Huygens PSF (huygens_psf) is the image sum of optics_gpu.pop for a point source (one source sample, a flat
angular spectrum): E(x, y) = sum_r a_r exp(i k (L'_r x + M'_r y + N'_r dz)), normalized by (sum_r |a_r|)^2, the
peak for all rays in phase. It is a Debye-type sum over the image-side directions, independent of the pupil grid
of the FFT method, with the energy-conserving weights a_r ~ sqrt(J_r) (J: the area ratio of the image-side to
the object-side direction cosines). The FFT PSF (and Optiland) give each entrance-pupil sample the same amplitude;
for an aberrated system the two weights give different peaks (tests/test_analysis.py). It needs an object at a
finite distance on the axis (else a ValueError).

Batches (tests/test_analysis_batch.py):

7. One trace for each wavelength (pupil_wavefronts): the rays of all fields at one wavelength, with the 6 check rays
   and the 5 rays of item 5 (the chief ray and the four marginal rays) of each field, go through Optiland's
   PyTorch backend in the same trace_generic calls (one call for each block of trace_chunk rays). The check rays
   are compared with a NumPy trace of the original optic (one trace for each wavelength, all fields) in each call,
   as before. The PyTorch copy of the optic (Optic.to_dict, then Optic.from_dict) is made one time for each state
   and device and kept (COPY_CACHE copies, keyed by the SHA-256 of the pickled dictionary; clear_copy_cache forgets
   them); its paraxial path (SurfaceGroup.build_paraxial_path) is made one time and kept on the copy, because
   nothing changes the copy. The ray computations are elementwise, so a batch gives the same numbers as one call
   for each field and wavelength (tests: bit for bit for the systems of the tests; an optic whose surface
   intersections iterate over all rays of a call can differ to its iteration tolerance). pupil_wavefront is
   pupil_wavefronts for one field and one wavelength.
8. Through focus (defocus): the image plane moved by dz along +z. Each ray of the pupil grid is a straight line in
   image space (index n), from its point Xp on the reference sphere with its direction D: it meets the moved plane
   after s = (z_c + dz - z_p) / N. The new sphere has its center at the chief-ray point on the moved plane,
   C' = C + dz D_c / N_c, and the radius R' = |C' - exit pupil|; t' is the path back to it (item 2). The new OPD is
   W' = W + n ((R + dz / N_c - R') - (s - t')) / lambda: the optical path to the new sphere, relative to the chief
   ray (whose s is R + dz / N_c and t' is R'). This is the same calculation as a trace to the moved image surface
   (a new trace gives the same rays after the last surface), without the trace, when the exit-pupil point stays
   where it is. Optiland's paraxial exit pupil (item 2) is computed in the medium after the image surface: it
   stays fixed only when that medium has the index of image space at the primary wavelength (for example air on
   both sides of the image surface). Else it moves with the image surface by dz (1 - n_after / n_image) (a new
   trace then uses another reference sphere), and defocus refuses the wavefront (exit_pupil_fixed False: a
   ValueError). To first order in dz it is the
   defocus phase n dz (1 - D . D_c) / (N_c lambda) of a plane-wave (Debye) sum; the full form keeps the change of
   the reference sphere. The rest (-dz^2 sin^2 u / (2 R') for a perfect lens) has the same sign for both signs of
   dz: the through-focus PSF is not symmetric about the focus, while the plane-wave sum of image_sum with dz is.
   It needs a flat image surface normal to z.
9. Polychromatic PSF (polychromatic_psf): sum_l v_l PSF_l(x - s_l) on one common grid, PSF_l in Strehl units
   (item 5), s_l the chief-ray point of wavelength l minus that of the reference wavelength (lateral color). The
   weights v_l = w_l K_l^2 / (lambda_l F_l)^2, normalized to the sum 1 (w_l: the spectral power; K_l: item 5) give
   each wavelength the energy w_l sum(I_l) (the energy of a Strehl-unit PSF is (lambda F (N - 1))^2 sum(I) / K^2 by
   Parseval), and the peak 1 for a system without aberration and lateral color. The resampling is exact: each
   monochromatic PSF is the Fraunhofer sum |sum_p P_p exp(-i 2 pi p (x - s_l) / T_l)|^2 / K_l^2 (T_l = lambda_l F_l
   (N - 1), the period of the FFT PSF of item 5) evaluated at the points of the common grid by two matrix products
   (a matrix Fourier transform: Soummer et al., Opt. Express 15, 15935 (2007)); on the grid of fft_psf it gives
   the FFT PSF. The grid must stay inside half the period of each wavelength, |x - s_l| <= T_l / 2 (else a
   ValueError): there the PSF of the sampled pupil repeats.
   Polychromatic MTF (polychromatic_mtf): |sum_l u_l OTF_l(nu) exp(-i 2 pi nu s_l)|, u_l = w_l sum(I_l) normalized
   to the sum 1, OTF_l the complex OTF of item 6 (the PSF about its chief-ray point, normalized at frequency 0)
   at the frequency nu, linearly interpolated between the lags of that wavelength (lag = nu / its frequency step;
   0 above its last lag); tangential: nu along y and s_l in y, sagittal: along x.
10. One batch (image_quality): pupil_wavefronts, defocus for each plane, fft_psfs and fft_mtfs (the FFTs of all
   fields, wavelengths and planes in one batched call, split above FFT_ELEMENTS complex values), and with spectral
   weights polychromatic_psf (grid_size of the PSFs, dx by its default) and polychromatic_mtf for each field and
   plane.
"""

from __future__ import annotations

import hashlib
import math
import pickle
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
import torch

from .pop import COPY_TOL, TRACE_RAYS, _max_difference, image_sum, optiland_torch, trace_bundle

MAX_FNUM = 10000.0      # the largest working F-number (as Optiland)
CHECK_P = (np.array([0.0, 0.0, 0.7, -0.3, 0.5, -0.9]), np.array([0.0, 0.6, 0.0, -0.5, 0.5, 0.1]))   # check rays
FIVE_P = (np.array([0.0, 0.0, 0.0, 1.0, -1.0]), np.array([0.0, 1.0, -1.0, 0.0, 0.0]))   # chief and marginal rays
COPY_CACHE = 8          # PyTorch copies of optics kept for reuse (item 7 of the module docstring)
FFT_ELEMENTS = 1 << 25  # complex128 values of one batched FFT (512 MB): fft_psfs and fft_mtfs split larger batches
FLAT_TOL = 1e-9         # mm: defocus needs the rays on the image surface within this distance of one z plane
_copies: OrderedDict = OrderedDict()


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
    # for defocus (item 8 of the module docstring); pupil_wavefronts sets them
    ldir: torch.Tensor | None = None     # the image-side direction cosines of the rays (N, N) each (0 where not valid)
    mdir: torch.Tensor | None = None
    ndir: torch.Tensor | None = None
    valid: torch.Tensor | None = None    # (N, N) bool: the samples with finite ray data (opd, xp, yp, zp: not 0-filled)
    n_image: float = 1.0                 # the index of image space at the wavelength (item 2)
    exit_pupil: tuple[float, float, float] | None = None   # the paraxial exit-pupil point, mm (item 2)
    chief_dir: tuple[float, float, float] | None = None    # the image-side direction cosines of the chief ray
    field: tuple[float, float] = (0.0, 0.0)                # (Hx, Hy)
    dz: float = 0.0          # um: the plane of the sphere center after the image surface (0: the image surface)
    z_spread: float = 0.0    # mm: the largest |z - z_chief| of the valid rays on the image surface
    # True: the media before and after the image surface have the same index at the primary wavelength, so the
    # paraxial exit pupil of Optiland does not move with the image surface (item 8; defocus needs it)
    exit_pupil_fixed: bool = True

    @property
    def n(self) -> int:
        return self.p.size

    @property
    def device(self) -> torch.device:
        return self.opd.device


def clear_copy_cache() -> None:
    """Forget the PyTorch copies of optics that pupil_wavefronts keeps for reuse (item 7 of the module docstring)."""
    _copies.clear()


def _state_key(data: dict) -> str | None:
    """The key of the state of an optic: the SHA-256 of its pickled dictionary (Optic.to_dict). None if the
    dictionary cannot be pickled (then the copy is not kept)."""
    try:
        return hashlib.sha256(pickle.dumps(data, protocol=4)).hexdigest()
    except (pickle.PicklingError, TypeError, AttributeError, ValueError):
        return None


def _torch_copy(data: dict, dev: torch.device):
    """The PyTorch copy (float64 on dev) of the optic of data: made one time for each state and device, the last
    COPY_CACHE copies kept (the least recently used goes first). Nothing changes the copy after it is made, so its
    paraxial path (SurfaceGroup.build_paraxial_path, the data of the geometry that Optiland's paraxial ray aiming
    builds 3 times for each trace) is made one time and kept on the copy. Call it inside optiland_torch."""
    from optiland.optic import Optic

    key = _state_key(data)
    if key is not None and (key, str(dev)) in _copies:
        _copies.move_to_end((key, str(dev)))
        return _copies[(key, str(dev))]
    copy = Optic.from_dict(data)
    path = copy.surfaces.build_paraxial_path()
    copy.surfaces.build_paraxial_path = lambda: path
    if key is not None:
        _copies[(key, str(dev))] = copy
        while len(_copies) > COPY_CACHE:
            _copies.popitem(last=False)
    return copy


def _path_to_sphere(x, y, z, L, M, N, center, R) -> torch.Tensor:
    """The length t (mm) from each ray point (x, y, z) back along its direction (L, M, N) to the reference sphere
    (center, R): the root of |X - t D - C|^2 = R^2 (|D| = 1) as in item 2 of the module docstring. center and R are
    numbers, or tensors with one value for each ray."""
    xc, yc, zc = center
    dx, dy, dz = x - xc, y - yc, z - zc                      # the differences first (no cancellation of z^2)
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


class _Slice:
    """The ray data (x, y, z, L, M, N, opd, i) of the rays start .. stop - 1 of Optiland rays (for the check)."""

    def __init__(self, rays, start: int, stop: int):
        for q in ("x", "y", "z", "L", "M", "N", "opd", "i"):
            setattr(self, q, getattr(rays, q)[start:stop])


def pupil_wavefronts(optic, fields: Sequence[tuple[float, float]] = ((0.0, 0.0),),
                     wavelengths: Sequence[float] | None = None, n_pupil: int = 128, device: str = "cuda",
                     trace_chunk: int | None = None) -> list[list[PupilWavefront]]:
    """The OPD maps W (waves) of the field points (Hx, Hy) (normalized field coordinates of the optic) at the
    wavelengths (um; None: the primary wavelength) on the n_pupil x n_pupil pupil grid, with the reference sphere of
    item 2 of the module docstring: out[i_field][i_wavelength]. Optiland's PyTorch backend traces the rays on device
    (float64) in blocks of trace_chunk rays (None: TRACE_RAYS), one trace for each block: the rays of all fields at
    one wavelength go in the same blocks (item 7 of the module docstring)."""
    import optiland.backend as be
    from optiland.backend.utils import is_torch_tensor, to_numpy
    from optiland.fields.field_types import AngleField

    if device not in ("cpu", "cuda"):
        raise ValueError(f"device {device!r}: use 'cpu' or 'cuda' (the analyses need float64)")
    if n_pupil < 2:
        raise ValueError(f"n_pupil {n_pupil}: use at least 2")
    if trace_chunk is not None and trace_chunk < 1:
        raise ValueError(f"trace_chunk {trace_chunk}: use an integer of at least 1, or None")
    fields = [(float(h[0]), float(h[1])) for h in fields]
    wls = [float(optic.primary_wavelength)] if wavelengths is None else [float(w) for w in wavelengths]
    if not fields or not wls:
        raise ValueError("give at least one field and one wavelength")
    nf = len(fields)
    hx, hy = np.array([f[0] for f in fields]), np.array([f[1] for f in fields])
    # host values of the original optic: the exit-pupil point, the indices of image and object space
    xpp = tuple(_float(v) for v in optic.paraxial.exit_pupil_point_gcs())
    # the index of image space (the medium before the image surface) at the traced wavelength, for the OPD, F and
    # the MTF step; Optiland 0.6.3 uses the medium after the image surface (see item 2 of the module docstring)
    n_img = [abs(_float(optic.image_surface.material_pre.n(wl))) for wl in wls]
    # Optiland's paraxial exit pupil (at the primary wavelength) stays fixed when the image surface moves only if the
    # media on both sides of the image surface have the same index there (item 8 of the module docstring)
    wl_p = float(optic.primary_wavelength)
    n_pre_p = abs(_float(optic.image_surface.material_pre.n(wl_p)))
    n_post_p = abs(_float(optic.image_surface.material_post.n(wl_p)))
    pupil_fixed = abs(n_post_p - n_pre_p) <= 1e-12 * n_pre_p
    launch = optic.object_surface.is_infinite and isinstance(optic.fields.field_definition, AngleField)
    n_obj = [_float(optic.object_surface.material_post.n(wl)) if launch else 1.0 for wl in wls]
    p = np.linspace(-1.0, 1.0, n_pupil)
    PX, PY = np.meshgrid(p, p, indexing="ij")
    inside_np = PX**2 + PY**2 <= 1
    n_in = int(inside_np.sum())
    dev = torch.device(device)
    if dev.type == "cuda":
        dev = torch.device("cuda", torch.cuda.current_device())
    step = trace_chunk if trace_chunk is not None else TRACE_RAYS
    f64 = {"dtype": torch.float64, "device": dev}
    # the check rays of all fields through the original (on the NumPy backend if it keeps NumPy arrays)
    active = be.get_backend()
    if not any(is_torch_tensor(s.geometry.cs.z) for s in optic.surfaces.surfaces):   # made under NumPy
        be.set_backend("numpy")
    try:
        data = optic.to_dict()
        refs = [optic.ray_tracer.trace_generic(np.repeat(hx, 6), np.repeat(hy, 6), np.tile(CHECK_P[0], nf),
                                               np.tile(CHECK_P[1], nf), wl, record=False) for wl in wls]
    finally:
        be.set_backend(active)
    if to_numpy(refs[0].x).dtype != np.float64:
        raise ValueError("the check rays of the original optic are not float64: make the optic under the NumPy "
                         "backend or in float64")
    # the rays of one wavelength: the 6 check rays of each field, the 5 rays of each field (FIVE_P), then the pupil
    # rays of field 0, field 1, ... (n_in each); the first block holds the 11 nf special rays and step pupil rays
    n_sp, total = 11 * nf, nf * n_in
    sp_h = [np.concatenate([np.repeat(h, 6), np.repeat(h, 5)]) for h in (hx, hy)]
    sp_p = [np.concatenate([np.tile(CHECK_P[k], nf), np.tile(FIVE_P[k], nf)]) for k in (0, 1)]
    fid_np = np.repeat(np.arange(nf), n_in)
    out: list[list[PupilWavefront]] = [[] for _ in range(nf)]
    with optiland_torch(device), torch.no_grad():
        copy = _torch_copy(data, dev)
        tr = copy.ray_tracer.trace_generic
        t = lambda v: torch.as_tensor(v, **f64)  # noqa: E731
        H_all = [t(h[fid_np]) for h in (hx, hy)]
        P_all = [t(np.tile(q[inside_np], nf)) for q in (PX, PY)]
        H_sp, P_sp = [t(h) for h in sp_h], [t(q) for q in sp_p]
        fid = torch.as_tensor(fid_np, device=dev)
        inside = torch.as_tensor(inside_np, device=dev)
        for iw, wl in enumerate(wls):
            n_i = n_img[iw]
            flat = {q: torch.zeros(total, **f64) for q in ("w", "i", "xp", "yp", "zp", "L", "M", "N")}
            valid = torch.zeros(total, dtype=torch.bool, device=dev)
            spread = torch.zeros(nf, **f64)
            per_field = None
            for s in range(0, max(total, 1), step):
                e = min(s + step, total)
                hxb, hyb, pxb, pyb = (v[s:e] for v in (*H_all, *P_all))
                off = n_sp if s == 0 else 0
                if s == 0:
                    hxb, hyb, pxb, pyb = (torch.cat([a, b]) for a, b in zip((*H_sp, *P_sp), (hxb, hyb, pxb, pyb)))
                # new tensors (clone): Optiland can change the ray arrays in place
                rays = tr(hxb.clone(), hyb.clone(), pxb.clone(), pyb.clone(), wl, record=False,
                          retain_launch=launch)
                if s == 0:
                    diff = _max_difference(refs[iw], _Slice(rays, 0, 6 * nf))
                    if not diff <= COPY_TOL:
                        raise RuntimeError(f"the PyTorch copy of the optic traces differently from the original "
                                           f"(largest difference {diff:.3g})")
                    per_field = _chief_and_marginal(rays, nf, xpp, n_i, wl, launch, n_pupil)
                    cx, cy, cz, R, pc = (t([q[k] for q in per_field]) for k in range(5))
                    lc = [torch.stack([q[5][k] for q in per_field]) for k in range(3)] if launch else None
                fb = fid[s:e]
                x, y, z, L, M, N = (getattr(rays, q)[off:] for q in ("x", "y", "z", "L", "M", "N"))
                tt = _path_to_sphere(x, y, z, L, M, N, (cx[fb], cy[fb], cz[fb]), R[fb])
                path = rays.opd[off:] - n_i * tt
                if launch:
                    ls = rays._launch_state
                    path = path + n_obj[iw] * (ls.L[off:] * (ls.x[off:] - lc[0][fb]) + ls.M[off:] * (ls.y[off:]
                                               - lc[1][fb]) + ls.N[off:] * (ls.z[off:] - lc[2][fb]))
                w = (pc[fb] - path) / (wl * 1e-3)
                xp, yp, zp = x - tt * L, y - tt * M, z - tt * N
                inten = rays.i[off:]
                ok = torch.isfinite(w) & torch.isfinite(xp) & torch.isfinite(yp) & torch.isfinite(zp)
                ok &= torch.isfinite(inten)
                zero = torch.zeros_like(w)
                for q, v in (("w", w), ("i", inten), ("xp", xp), ("yp", yp), ("zp", zp), ("L", L), ("M", M),
                             ("N", N)):
                    flat[q][s:e] = torch.where(ok, v, zero)
                valid[s:e] = ok
                spread.scatter_reduce_(0, fb, torch.where(ok, (z - cz[fb]).abs(), zero), reduce="amax")
                del rays, tt, path, w, xp, yp, zp, ok
            spread_h = spread.tolist()
            for f in range(nf):
                sl = slice(f * n_in, (f + 1) * n_in)
                grids = {}
                for q, v in flat.items():
                    full = torch.zeros((n_pupil, n_pupil), **f64)
                    full[inside] = v[sl]
                    grids[q] = full
                vgrid = torch.zeros((n_pupil, n_pupil), dtype=torch.bool, device=dev)
                vgrid[inside] = valid[sl]
                cxf, cyf, czf, Rf, _, _, fno, fstep, cdir = per_field[f]
                out[f].append(PupilWavefront(p, grids["w"], grids["i"], inside, grids["xp"], grids["yp"],
                                             grids["zp"], Rf, (cxf, cyf, czf), wl, fno, fstep, n_in, grids["L"],
                                             grids["M"], grids["N"], vgrid, n_i, xpp, cdir, fields[f], 0.0,
                                             spread_h[f], pupil_fixed))
    return out


def _chief_and_marginal(rays, nf: int, xpp, n_i: float, wl: float, launch: bool, n_pupil: int) -> list[tuple]:
    """For each field, from its 5 rays FIVE_P (after the 6 nf check rays of the first block): the chief-ray image
    point (the center of the reference sphere), the radius R, the optical path of the chief ray to the sphere, the
    launch point of the chief ray (an object at infinity), the working F-number and the MTF frequency steps
    (Optiland's get_working_FNO and _get_mtf_frequency_steps; a flat image surface normal to z), and the direction
    cosines of the chief ray."""
    out = []
    for f in range(nf):
        j = 6 * nf + 5 * f
        x, y, z, L, M, N, opd, inten = (getattr(rays, q)[j:j + 5] for q in ("x", "y", "z", "L", "M", "N", "opd",
                                                                               "i"))
        center = (x[0].item(), y[0].item(), z[0].item())
        R = math.dist(center, xpp)
        t0 = _path_to_sphere(x, y, z, L, M, N, center, R)
        p_chief = (opd[0] - n_i * t0[0]).item()
        chief_launch = None
        if launch:
            ls = rays._launch_state
            chief_launch = (ls.x[j], ls.y[j], ls.z[j])
        D = torch.stack([L, M, N], dim=1)
        cosang = torch.clamp(D[1:] @ D[0], -1.0, 1.0)
        na2 = (n_i * torch.sin(torch.arccos(cosang))) ** 2
        ok = inten[1:] > 0
        mean_na2 = float((na2[ok].mean() if bool(ok.any()) else na2.mean()).item())
        fno = math.inf if mean_na2 <= 0 else 1.0 / (2.0 * math.sqrt(mean_na2))
        fno = min(fno, MAX_FNUM)
        band = torch.stack([D[1, :2] - D[2, :2], D[3, :2] - D[4, :2]]).norm(dim=1)
        steps = n_i * band / ((n_pupil - 1) * wl * 1e-3)
        freq_step = (float(steps[0].item()), float(steps[1].item()))
        cdir = (L[0].item(), M[0].item(), N[0].item())
        out.append((*center, R, p_chief, chief_launch, fno, freq_step, cdir))
    return out


def pupil_wavefront(optic, field: tuple[float, float] = (0.0, 0.0), wavelength: float | None = None,
                    n_pupil: int = 128, device: str = "cuda", trace_chunk: int | None = None) -> PupilWavefront:
    """The OPD map W (waves) of the field point (Hx, Hy) (normalized field coordinates of the optic) on the
    n_pupil x n_pupil pupil grid, with the reference sphere of item 2 of the module docstring: pupil_wavefronts for
    one field and one wavelength (None: the primary wavelength)."""
    wls = None if wavelength is None else [wavelength]
    return pupil_wavefronts(optic, [field], wls, n_pupil, device, trace_chunk)[0][0]


def defocus(wf: PupilWavefront, dz: float) -> PupilWavefront:
    """The wavefront of wf in the image plane moved by dz um along +z (away from the optic when the light goes along
    +z in image space), with the reference sphere of that plane (item 8 of the module docstring): the same OPD map
    as a trace to the moved image surface, without a new trace. It needs a flat image surface normal to z
    (z_spread <= FLAT_TOL), the same index on both sides of the image surface at the primary wavelength
    (exit_pupil_fixed; else Optiland's exit pupil moves with the image surface), and the ray data of
    pupil_wavefronts; else a ValueError. dz = 0 gives wf itself (no condition)."""
    dz = float(dz)
    if dz == 0.0:
        return wf
    if wf.ldir is None or wf.exit_pupil is None or wf.chief_dir is None:
        raise ValueError("defocus needs the ray data of pupil_wavefronts (ldir, mdir, ndir, valid, exit_pupil)")
    if not wf.z_spread <= FLAT_TOL:
        raise ValueError(f"the rays on the image surface spread over {wf.z_spread:.3g} mm in z: defocus needs a "
                         f"flat image surface normal to z")
    if not wf.exit_pupil_fixed:
        raise ValueError("the medium after the image surface has another index than image space (at the primary "
                         "wavelength): Optiland's paraxial exit pupil moves with the image surface, so a trace to "
                         "the moved surface has another reference sphere. Give the image surface the medium of "
                         "image space (its material) to use defocus")
    dzm = dz * 1e-3
    lc, mc, nc = wf.chief_dir
    c0 = wf.center
    c1 = (c0[0] + dzm * lc / nc, c0[1] + dzm * mc / nc, c0[2] + dzm)
    R1 = math.dist(c1, wf.exit_pupil)
    ok = wf.valid
    nd = torch.where(ok, wf.ndir, torch.ones_like(wf.ndir))
    s = (c0[2] + dzm - wf.zp) / nd                             # from the old sphere to the moved plane
    X, Y, Z = wf.xp + s * wf.ldir, wf.yp + s * wf.mdir, wf.zp + s * wf.ndir
    t1 = _path_to_sphere(X, Y, Z, wf.ldir, wf.mdir, nd, c1, R1)    # back to the new sphere
    s_chief = wf.radius + dzm / nc                             # the chief ray: from c0 - R D_c to the plane
    dW = wf.n_image * ((s_chief - R1) - (s - t1)) / (wf.wavelength * 1e-3)
    zero = torch.zeros_like(wf.opd)
    return replace(wf, opd=torch.where(ok, wf.opd + dW, zero), xp=torch.where(ok, X - t1 * wf.ldir, zero),
                   yp=torch.where(ok, Y - t1 * wf.mdir, zero), zp=torch.where(ok, Z - t1 * wf.ndir, zero),
                   radius=R1, center=c1, dz=wf.dz + dz)


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
    """The MTF of item 6 of the module docstring from the FFT PSF of wf. The grid of the PSF must have at least
    2 N - 1 points (N: the pupil sampling; else a ValueError): the full autocorrelation of the pupil needs them.
    The returned values stay exact down to 2 N - 3 points (the check in the module docstring)."""
    M = psf.psf.shape[0]
    if M < 2 * wf.n - 1:
        raise ValueError(f"grid_size {M} is below 2 N - 1 = {2 * wf.n - 1}: the full autocorrelation of the "
                         f"pupil needs 2 N - 1 points (use fft_psf(wf) with the default grid 2 N)")
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
                n_pad: int = 16, na_cut: float = 0.5, device: str = "cuda", trace_chunk: int | None = None,
                return_bundle: bool = False):
    """The PSF of the on-axis object point by the image sum of optics_gpu.pop (trace_bundle and image_sum):
    |E|^2 / (sum_r |a_r|)^2 on the grid (x_img, y_img) in um, dz um after the image surface. The point source is
    one sample of an n_src x n_src source of pitch dx_src um, padded n_pad times: the direction step is
    lambda / (n_src n_pad dx_src) and the image field repeats with the period n_src n_pad dx_src um (keep the image
    grid inside it). Directions up to na_cut are traced; the stop of the optic must be a physical aperture. An
    object at infinity gives a ValueError. return_bundle: also return the RayBundle (its n_rays: the kept rays)."""
    if optic.object_surface.is_infinite:
        raise ValueError("huygens_psf needs an object at a finite distance (the image sum traces from an object "
                         "point); for an object at infinity use fft_psf")
    src = np.zeros((n_src, n_src), complex)
    src[n_src // 2, n_src // 2] = 1.0
    bundle = trace_bundle(optic, src, dx_src, wavelength, na_cut, 0.0, n_pad, device, trace_chunk)
    E = image_sum(bundle, x_img, y_img, dz=dz)
    peak = sum(b[3].abs().sum() for b in bundle.blocks) ** 2
    psf = (E.real**2 + E.imag**2) / peak
    return (psf, bundle) if return_bundle else psf


# ---------------------------------------------------------------------------------------------------------------
# Batches: many planes in one FFT, through focus, polychromatic PSF and MTF
# ---------------------------------------------------------------------------------------------------------------

def _same_grid(wfs: Sequence[PupilWavefront]) -> tuple[int, torch.device]:
    if len(wfs) == 0:
        raise ValueError("give at least one wavefront")
    n, dev = wfs[0].n, wfs[0].device
    if any(w.n != n or w.device != dev for w in wfs):
        raise ValueError("all wavefronts of a batch need the same pupil sampling N and the same device")
    return n, dev


def _per_call(M: int) -> int:
    """The planes of M x M complex values in one batched FFT (at least 1, at most FFT_ELEMENTS values)."""
    return max(1, FFT_ELEMENTS // (M * M))


def fft_psfs(wfs: Sequence[PupilWavefront], grid_size: int | None = None) -> list[FFTPSF]:
    """fft_psf of each wavefront (the same N and device; grid_size M, None: 2 N), with the FFTs of all planes in one
    batched call (torch.fft.fft2 over the last two dimensions of a (B, M, M) stack); a batch above FFT_ELEMENTS
    complex values goes in several calls. The PSF tensors of a call are views of one (B, M, M) tensor."""
    N, dev = _same_grid(wfs)
    M = 2 * N if grid_size is None else int(grid_size)
    if M < N:
        raise ValueError(f"grid_size {M} is below the pupil sampling {N}")
    b = (M - N) // 2
    out = []
    per = _per_call(M)
    for s in range(0, len(wfs), per):
        part = wfs[s:s + per]
        P = torch.stack([pupil_function(w) for w in part])
        counts = (P.abs() > 0).sum(dim=(1, 2)).tolist()
        if min(counts) == 0:
            raise ValueError("no ray reaches the image in a wavefront of the batch: its pupil function is zero")
        big = torch.zeros((len(part), M, M), dtype=torch.complex128, device=dev)
        big[:, b:b + N, b:b + N] = P
        a = torch.fft.fftshift(torch.fft.fft2(big), dim=(-2, -1))
        del big
        k2 = torch.tensor([float(c) ** 2 for c in counts], dtype=torch.float64, device=dev)
        psf = (a.real**2 + a.imag**2) / k2[:, None, None]
        del a
        for w, q, pk in zip(part, psf, P):
            dx = w.wavelength * w.fno * (N - 1) / M
            k = (np.arange(M) - M // 2) * dx
            out.append(FFTPSF(q, dx, w.center[0] * 1e3 + k, w.center[1] * 1e3 + k, pk))
    return out


def fft_mtfs(wfs: Sequence[PupilWavefront], psfs: Sequence[FFTPSF]) -> list[MTF]:
    """fft_mtf of each pair (wavefront, its FFT PSF), with the FFTs of all PSFs in one batched call (split above
    FFT_ELEMENTS complex values). All PSFs need the same grid M >= 2 N - 1."""
    N, _ = _same_grid(wfs)
    if len(psfs) != len(wfs):
        raise ValueError(f"{len(wfs)} wavefronts and {len(psfs)} PSFs: give one PSF for each wavefront")
    M = psfs[0].psf.shape[0]
    if any(q.psf.shape != (M, M) for q in psfs):
        raise ValueError("all PSFs of a batch need the same grid")
    if M < 2 * N - 1:
        raise ValueError(f"grid_size {M} is below 2 N - 1 = {2 * N - 1}: the full autocorrelation of the pupil needs "
                         f"2 N - 1 points (use fft_psfs(wfs) with the default grid 2 N)")
    c = M // 2
    out = []
    per = _per_call(M)
    for s in range(0, len(wfs), per):
        otf = torch.fft.fftshift(torch.fft.fft2(torch.stack([q.psf for q in psfs[s:s + per]])), dim=(-2, -1)).abs()
        for w, o in zip(wfs[s:s + per], otf):
            dc = o[c, c]
            tan = torch.clamp(o[c, c:][:c] / dc, 0.0, 1.0)
            sag = torch.clamp(o[c:, c][:c] / dc, 0.0, 1.0)
            j = np.arange(c)
            out.append(MTF(j * w.freq_step[0], j * w.freq_step[1], tan, sag))
        del otf
    return out


@dataclass
class PolyPSF:
    psf: torch.Tensor        # (M, M) [ix, iy]: sum_l v_l PSF_l(x - shift_l); peak 1 with no aberration or lateral color
    dx: float                # um, the sampling on both axes
    x: np.ndarray            # um, global image coordinates (the chief-ray point of the reference wavelength at M // 2)
    y: np.ndarray
    weights: np.ndarray      # v_l, the weights of the monochromatic PSFs (Strehl units); sum 1
    shift: np.ndarray        # (W, 2) um: the chief-ray point of each wavelength minus that of the reference
    mono: torch.Tensor       # (W, M, M): the monochromatic PSFs PSF_l(x - shift_l) on the common grid (Strehl units)


def _spectral(wfs: Sequence[PupilWavefront], weights) -> np.ndarray:
    w = np.ones(len(wfs)) if weights is None else np.asarray(weights, float).ravel()
    if w.size != len(wfs) or not np.all(np.isfinite(w)) or np.any(w < 0) or not w.sum() > 0:
        raise ValueError(f"weights {weights!r}: give one finite weight >= 0 for each wavelength, with a sum above 0")
    return w


def polychromatic_psf(wfs: Sequence[PupilWavefront], weights=None, grid_size: int | None = None,
                      dx: float | None = None, reference: int = 0) -> PolyPSF:
    """The polychromatic PSF of one field point (and one plane) from the wavefronts at W wavelengths (the same N),
    on one common image grid (item 9 of the module docstring): grid_size M (None: 2 N) samples of dx um (None: the
    largest step that keeps the window inside half the period of each wavelength), centered on the chief-ray point
    of wfs[reference]. weights: the spectral power of each wavelength (None: equal)."""
    N, dev = _same_grid(wfs)
    w = _spectral(wfs, weights)
    if not 0 <= reference < len(wfs):
        raise ValueError(f"reference {reference}: use an index of the wavelengths")
    M = 2 * N if grid_size is None else int(grid_size)
    if M < 1:
        raise ValueError(f"grid_size {M}: use at least 1")
    lam = np.array([q.wavelength for q in wfs])
    fno = np.array([q.fno for q in wfs])
    period = lam * fno * (N - 1)                                     # um: the period of each monochromatic PSF
    ref = wfs[reference].center
    shift = np.array([[(q.center[0] - ref[0]) * 1e3, (q.center[1] - ref[1]) * 1e3] for q in wfs])
    s_max = np.abs(shift).max(axis=1)
    h = M // 2                                                        # the largest |index - M // 2| of the grid
    if dx is None:
        dx = float(np.min((period / 2 - s_max) / max(h, 1)))
        if not dx > 0:
            raise ValueError("the lateral shift between the wavelengths is above half the period of a PSF: use a "
                             "larger n_pupil")
    dx = float(dx)
    over = h * dx + s_max - period / 2
    if not dx > 0 or np.any(over > 1e-12 * period):
        raise ValueError(f"the window of {M} samples of {dx:.6g} um (with the lateral shifts) is outside half the "
                         f"period lambda F (N - 1) of a monochromatic PSF ({np.round(period, 6)} um): use a smaller "
                         f"grid_size or dx, or a larger n_pupil")
    P = torch.stack([pupil_function(q) for q in wfs])
    counts = (P.abs() > 0).sum(dim=(1, 2)).tolist()
    if min(counts) == 0:
        raise ValueError("no ray reaches the image at a wavelength: its pupil function is zero")
    f64 = {"dtype": torch.float64, "device": dev}
    poff = torch.arange(N, **f64) - N // 2
    xs = torch.as_tensor((np.arange(M) - h) * dx, **f64)
    per = torch.as_tensor(period, **f64)[:, None, None]
    sh = torch.as_tensor(shift, **f64)

    def kernel(axis: int) -> torch.Tensor:                           # (W, M, N): exp(-2 pi i p (x - shift) / T)
        ph = (-2 * math.pi) * poff[None, None, :] * (xs[None, :, None] - sh[:, axis, None, None]) / per
        return torch.polar(torch.ones_like(ph), ph)

    a = kernel(0) @ P @ kernel(1).transpose(-1, -2)                    # (W, M, M) [ix, iy]
    k2 = torch.tensor([float(c) ** 2 for c in counts], **f64)
    mono = (a.real**2 + a.imag**2) / k2[:, None, None]
    v = w * np.asarray(counts, float) ** 2 / (lam * fno) ** 2
    v = v / v.sum()
    psf = (torch.as_tensor(v, **f64)[:, None, None] * mono).sum(dim=0)
    k = xs.cpu().numpy()
    return PolyPSF(psf, dx, ref[0] * 1e3 + k, ref[1] * 1e3 + k, v, shift, mono)


def polychromatic_mtf(wfs: Sequence[PupilWavefront], psfs: Sequence[FFTPSF], weights=None, freq=None,
                      reference: int = 0) -> MTF:
    """The MTF of the polychromatic PSF of one field point (and one plane), tangential (along y) and sagittal
    (along x), from the wavefronts at W wavelengths and their FFT PSFs (fft_psfs; the same grid M >= 2 N - 1)
    (item 9 of the module docstring). freq: the frequencies in cycles/mm on both axes (None: the axes of
    fft_mtf of wfs[reference]). weights: the spectral power of each wavelength (None: equal)."""
    N, dev = _same_grid(wfs)
    w = _spectral(wfs, weights)
    if len(psfs) != len(wfs):
        raise ValueError(f"{len(wfs)} wavefronts and {len(psfs)} PSFs: give one PSF for each wavefront")
    if not 0 <= reference < len(wfs):
        raise ValueError(f"reference {reference}: use an index of the wavelengths")
    M = psfs[0].psf.shape[0]
    if any(q.psf.shape != (M, M) for q in psfs) or M < 2 * N - 1:
        raise ValueError(f"all PSFs need the same grid of at least 2 N - 1 = {2 * N - 1} points")
    c = M // 2
    # the complex OTF of each PSF about its chief-ray point (sample M // 2 to index 0), normalized at frequency 0
    otf = torch.fft.fft2(torch.fft.ifftshift(torch.stack([q.psf for q in psfs]), dim=(-2, -1)))
    otf = otf / otf[:, :1, :1]
    power = np.array([float(q.intensity[q.inside].sum().item()) for q in wfs])
    u = w * power / (w * power).sum()
    ref = wfs[reference].center
    f64 = {"dtype": torch.float64, "device": dev}
    out = []
    for axis, line in ((0, lambda o: o[:, 0, :c]), (1, lambda o: o[:, :c, 0])):    # tangential (y), sagittal (x)
        nu = np.arange(c) * wfs[reference].freq_step[axis] if freq is None else np.asarray(freq, float).ravel()
        if np.any(nu < 0) or not np.all(np.isfinite(nu)):
            raise ValueError("freq: give finite frequencies >= 0 (cycles/mm)")
        zero = torch.zeros((len(wfs), 1), dtype=otf.dtype, device=dev)
        vals = torch.cat([line(otf), zero], dim=1)                     # (W, c + 1): the lags 0 .. c - 1, then 0
        total = torch.zeros(nu.size, dtype=torch.complex128, device=dev)
        for i, q in enumerate(wfs):
            lag = torch.as_tensor(nu / q.freq_step[axis], **f64)       # in pupil samples of lag of this wavelength
            i0 = torch.clamp(torch.floor(lag).long(), 0, c)
            fr = lag - i0
            val = (1 - fr) * vals[i, i0] + fr * vals[i, torch.clamp(i0 + 1, max=c)]
            d_mm = (q.center[1 - axis] - ref[1 - axis])                  # tangential: the shift in y
            ph = torch.as_tensor(-2 * math.pi * nu * d_mm, **f64)
            total += u[i] * val * torch.polar(torch.ones_like(ph), ph)
        out.append((nu, torch.clamp(total.abs(), 0.0, 1.0)))
    return MTF(out[0][0], out[1][0], out[0][1], out[1][1])


@dataclass
class ImageQuality:
    """The results of image_quality: [i_field][i_wavelength][i_plane] for the monochromatic results, [i_field][i_plane]
    for the polychromatic results (None without weights)."""
    fields: list[tuple[float, float]]
    wavelengths: list[float]
    dz: np.ndarray                                   # um: the planes after the image surface
    wavefronts: list[list[list[PupilWavefront]]]
    psfs: list[list[list[FFTPSF]]]
    mtfs: list[list[list[MTF]]] | None
    poly_psfs: list[list[PolyPSF]] | None
    poly_mtfs: list[list[MTF]] | None


def image_quality(optic, fields: Sequence[tuple[float, float]] = ((0.0, 0.0),),
                  wavelengths: Sequence[float] | None = None, dz=(0.0,), weights=None, n_pupil: int = 128,
                  grid_size: int | None = None, device: str = "cuda", trace_chunk: int | None = None,
                  mtf: bool = True) -> ImageQuality:
    """FFT PSF and MTF of all fields, wavelengths and planes dz (um from the image surface along +z: defocus) in one
    batch (item 10 of the module docstring): one trace for each wavelength (pupil_wavefronts), the planes by
    defocus (the conditions of defocus apply; one plane dz = 0 alone needs none of them), the FFTs of
    all planes in one call (fft_psfs, fft_mtfs), and with weights (one spectral weight for each wavelength) the
    polychromatic PSF and MTF of each field and plane (polychromatic_psf with the grid size M of the PSFs and its
    default step, centered on the chief-ray point of the first wavelength; polychromatic_mtf)."""
    dzs = np.atleast_1d(np.asarray(dz, float)).ravel()
    if dzs.size == 0:
        raise ValueError("give at least one plane dz")
    wfs0 = pupil_wavefronts(optic, fields, wavelengths, n_pupil, device, trace_chunk)
    nf, nw, nz = len(wfs0), len(wfs0[0]), dzs.size
    wfs = [[[defocus(w, d) for d in dzs] for w in row] for row in wfs0]
    flat = [w for row in wfs for col in row for w in col]
    psf_flat = fft_psfs(flat, grid_size)
    mtf_flat = fft_mtfs(flat, psf_flat) if mtf else None

    def nest(v):
        return [[[v[(f * nw + i) * nz + z] for z in range(nz)] for i in range(nw)] for f in range(nf)]

    psfs = nest(psf_flat)
    poly_psfs = poly_mtfs = None
    if weights is not None:
        poly_psfs = [[polychromatic_psf([wfs[f][i][z] for i in range(nw)], weights, psf_flat[0].psf.shape[0])
                      for z in range(nz)] for f in range(nf)]
        if mtf:
            poly_mtfs = [[polychromatic_mtf([wfs[f][i][z] for i in range(nw)], [psfs[f][i][z] for i in range(nw)],
                                            weights) for z in range(nz)] for f in range(nf)]
    return ImageQuality([w[0].field for w in wfs0], [w.wavelength for w in wfs0[0]], dzs, wfs, psfs,
                        nest(mtf_flat) if mtf else None, poly_psfs, poly_mtfs)
