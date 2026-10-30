"""Physical optics through a lens system: angular spectrum + real ray tracing (Optiland) + a Debye-type sum.

For a small (isoplanatic) source at the object point, such as a waveguide facet mode or a fiber mode:

1. Angular spectrum of the source: A(fx, fy) = FFT(E_s); a direction (L, M) = lambda (fx, fy) per sample.
2. One real ray per direction from the object point, traced through the system (Optiland) to the image surface:
   image-side directions (L', M'), positions (x_r, y_r) and optical path W.
3. Image field, a sum over rays (Debye-Wolf type, in image space):

       E(x, y) = sum_r a_r exp(i k (W_r + L'_r (x - x_r) + M'_r (y - y_r))),
       a_r = A_r df sqrt(J_r),   J_r = |d(L', M') / d(L, M)|,   df = dfx dfy = dL dM / lambda^2

   Power conservation per ray tube, |U'|^2 df' = |A|^2 df, gives |U'| = |A| / sqrt(J) for the image-side spectrum;
   written as a sum over the object-side samples (df' = J df) the weight is A sqrt(J) df. For a perfect imager
   with magnification m, J = 1/m^2 is a constant (tests/test_pop.py checks the power). The sum is separable in x and y, so it
   is one (Nx x R)(R x Ny) matrix product on the GPU.
4. Coupling efficiency into a mode F: eta = |<F, E>|^2 / (<E, E> <F, F>).

Limits: scalar fields (no polarization), no obliquity factor (paraxial to moderate NA), the source must be small
compared with the isoplanatic patch of the system, rays that do not reach the image are dropped. Object and image
spaces are air (n = 1): the direction cosines are lambda0 times the spatial frequencies, and k = 2 pi / lambda0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

MM = 1e3          # um per mm (Optiland lens units are mm; wavelengths are um)


def gaussian(x: np.ndarray, y: np.ndarray, w: float) -> np.ndarray:
    """A Gaussian field exp(-(x^2 + y^2) / w^2) (w: 1/e^2 intensity radius; MFD = 2 w)."""
    X, Y = np.meshgrid(x, y, indexing="ij")
    return np.exp(-(X**2 + Y**2) / w**2).astype(complex)


def overlap_efficiency(e: np.ndarray, f: np.ndarray) -> float:
    """|<f, e>|^2 / (<e, e> <f, f>) on the same grid."""
    return float(abs(np.vdot(f, e)) ** 2 / (np.vdot(e, e).real * np.vdot(f, f).real))


@dataclass
class ImageField:
    x: np.ndarray
    y: np.ndarray
    E: np.ndarray
    n_rays: int
    device: str


def _device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def angular_spectrum(e_src: np.ndarray, dx: float, wavelength: float, n_pad: int = 2):
    """Directions (L, M) and amplitudes A dL dM of the propagating plane waves of the source (zero padded n_pad x
    for a finer angular sampling). dx and wavelength in the same unit."""
    nx, ny = e_src.shape
    pad = np.zeros((nx * n_pad, ny * n_pad), complex)
    pad[:nx, :ny] = e_src
    pad = np.roll(pad, (-(nx // 2), -(ny // 2)), axis=(0, 1))       # the source center at index 0
    A = np.fft.fft2(pad) * dx * dx
    fx = np.fft.fftfreq(nx * n_pad, d=dx)
    fy = np.fft.fftfreq(ny * n_pad, d=dx)
    L, M = np.meshgrid(wavelength * fx, wavelength * fy, indexing="ij")
    dLdM = (wavelength / (nx * n_pad * dx)) * (wavelength / (ny * n_pad * dx))
    return np.fft.fftshift(L), np.fft.fftshift(M), np.fft.fftshift(A) * dLdM / wavelength**2


def propagate_through(optic, e_src: np.ndarray, dx: float, wavelength: float, x_img: np.ndarray,
                      y_img: np.ndarray, na_cut: float = 0.95, amp_cut: float = 1e-6, n_pad: int = 2) -> ImageField:
    """The image field of the source e_src (centered on the axis, sample step dx in um) through an Optiland
    optic, on the image grid (x_img, y_img) in um, at the image surface of the optic."""
    from optiland.rays import RealRays

    L, M, A = angular_spectrum(e_src, dx, wavelength, n_pad)
    keep = (L**2 + M**2 < na_cut**2) & (np.abs(A) > amp_cut * np.abs(A).max())
    # the Jacobian needs neighbors: trace the full grid of kept directions (a bounding box) and mask later
    ii, jj = np.nonzero(keep)
    sl = (slice(ii.min(), ii.max() + 1), slice(jj.min(), jj.max() + 1))
    L, M, A, keep = L[sl], M[sl], A[sl], keep[sl]
    N = np.sqrt(np.clip(1 - L**2 - M**2, 0, None))
    z0 = float(np.ravel(np.asarray(optic.surface_group.positions))[0])     # the object surface
    n = L.size
    rays = RealRays(np.zeros(n), np.zeros(n), np.full(n, z0), L.ravel(), M.ravel(), N.ravel(),
                    np.ones(n), np.full(n, wavelength))
    optic.surface_group.trace(rays, skip=1, record=False)
    to = lambda a: np.asarray(a.detach().cpu() if hasattr(a, "detach") else a).reshape(L.shape)  # noqa: E731
    xr, yr, Lr, Mr, W, inten = (to(v) for v in (rays.x, rays.y, rays.L, rays.M, rays.opd, rays.i))
    xr, yr, W = xr * MM, yr * MM, W * MM                  # Optiland lengths are in mm; this module uses um
    ok = keep & np.isfinite(W) & (inten > 0)
    # |d(L', M') / d(L, M)| by central differences on the direction grid
    dLp_dL = np.gradient(Lr, L[:, 0], axis=0)
    dLp_dM = np.gradient(Lr, M[0, :], axis=1)
    dMp_dL = np.gradient(Mr, L[:, 0], axis=0)
    dMp_dM = np.gradient(Mr, M[0, :], axis=1)
    J = np.abs(dLp_dL * dMp_dM - dLp_dM * dMp_dL)
    ok &= np.isfinite(J) & (J > 0)
    k = 2 * np.pi / wavelength
    a = A[ok] * np.sqrt(J[ok]) * np.exp(1j * k * (W[ok] - Lr[ok] * xr[ok] - Mr[ok] * yr[ok]))
    dev = _device()
    t = lambda v, dt=torch.float64: torch.as_tensor(np.asarray(v), dtype=dt, device=dev)  # noqa: E731
    ex = torch.exp(1j * k * t(x_img)[:, None] * t(Lr[ok])[None, :])          # (Nx, R)
    ey = torch.exp(1j * k * t(y_img)[:, None] * t(Mr[ok])[None, :])          # (Ny, R)
    E = (ex * t(a, torch.complex128)[None, :]) @ ey.T                        # (Nx, Ny)
    return ImageField(np.asarray(x_img), np.asarray(y_img), E.cpu().numpy(), int(ok.sum()), str(dev))
