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
   with magnification m, J = 1/m^2 is a constant (tests/test_pop.py checks the power). The sum is separable in x
   and y, so it is one (Nx x R)(R x Ny) matrix product on the GPU (in chunks of rays, for large R).
4. Coupling efficiency into a mode F: eta = |<F, E>|^2 / (<E, E> <F, F>).

Limits: scalar fields (no polarization), no obliquity factor (paraxial to moderate NA), the source must be small
compared with the isoplanatic patch of the system, rays that do not reach the image are dropped. Object and image
spaces are air (n = 1): the direction cosines are lambda0 times the spatial frequencies, and k = 2 pi / lambda0.

Ray trace backends (trace_rays): Optiland's active backend (NumPy unless the caller changed it), or Optiland's
PyTorch backend in float64 on "cpu" or "cuda" (PyTorch on ROCm reports an AMD GPU as "cuda") for large ray counts.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import torch

MM = 1e3          # um per mm (Optiland lens units are mm; wavelengths are um)
SUM_ELEMENTS = 1 << 25   # (Nx + Ny) x chunk: the complex128 elements of ex and ey of one ray chunk (512 MB)
COPY_TOL = 1e-9          # mm and direction cosines: the PyTorch copy of an optic against the original (check rays)


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
    device: str                  # the device of the image sum
    trace_device: str = "numpy"  # the device of the ray trace ("numpy", or the PyTorch device, e.g. "cuda:0")


def _device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@contextlib.contextmanager
def optiland_torch(device: str) -> Iterator[None]:
    """Optiland's PyTorch backend in float64 on device ("cpu" or "cuda") inside the block. After the block, the
    earlier Optiland backend, and the device and precision of its PyTorch backend, are set again.

    Optiland keeps its backend as global state (not thread-safe). Its PyTorch default is float32, which is not
    sufficient for the optical path (the phase k W, with W of tens of mm), so this sets float64. Apple "mps" has no
    float64 and is refused."""
    import optiland.backend as be

    if device not in ("cpu", "cuda"):
        raise ValueError(f"trace device {device!r}: use 'cpu' or 'cuda' (the trace needs float64)")
    if "torch" not in be.list_available_backends():
        raise RuntimeError("Optiland has no PyTorch backend in this environment")
    old_backend = be.get_backend()
    be.set_backend("torch")
    old_device, old_bits = be.get_device(), be.get_precision()
    try:
        be.set_device(device)            # ValueError if CUDA is not available
        be.set_precision("float64")
        yield
    finally:
        be.set_precision("float32" if old_bits == 32 else "float64")
        be.set_device(old_device)
        be.set_backend(old_backend)


def _launch(optic, L: np.ndarray, M: np.ndarray, wavelength: float, z0: float):
    """Trace one ray per direction (L, M) from the axis point of the object surface (z0) to the image surface."""
    from optiland.rays import RealRays

    n = L.size
    N = np.sqrt(np.clip(1 - L**2 - M**2, 0, None))
    rays = RealRays(np.zeros(n), np.zeros(n), np.full(n, z0), L, M, N, np.ones(n), np.full(n, wavelength))
    optic.surfaces.trace(rays, skip=1, record=False)
    return rays


def _max_difference(a, b) -> float:
    """The largest difference of the ray data of a and b (positions, directions, OPL, intensity); inf if the NaN
    patterns differ."""
    from optiland.backend.utils import to_numpy

    worst = 0.0
    for name in ("x", "y", "z", "L", "M", "N", "opd", "i"):
        u, v = to_numpy(getattr(a, name)), to_numpy(getattr(b, name))
        if not np.array_equal(np.isnan(u), np.isnan(v)):
            return float("inf")
        fin = np.isfinite(u)
        if fin.any():
            worst = max(worst, float(np.max(np.abs(u[fin] - v[fin]))))
    return worst


def trace_rays(optic, L: np.ndarray, M: np.ndarray, wavelength: float, device: str | None = None):
    """One real ray per direction cosine pair (L, M) from the object point on the axis, traced by Optiland to the
    image surface of optic. Returns Optiland RealRays (lengths in mm) with flat arrays in the order of L.ravel().

    device None: Optiland's active backend (NumPy unless the caller changed it), on optic itself.
    device "cpu" or "cuda": Optiland's PyTorch backend in float64 on that device (optiland_torch), on a copy of the
    optic made under that backend (Optic.to_dict, then Optic.from_dict). The arrays of the result are torch tensors
    on the device. A check traces up to 25 of the rays through the original and through the copy; if they differ
    by more than COPY_TOL (for example, a surface that the dictionary does not keep), a RuntimeError stops it.
    The check traces the original on the NumPy backend if the optic keeps NumPy arrays (else on the active
    backend); if that reference trace is not float64 (an optic made under Optiland's PyTorch backend in float32),
    a ValueError stops it: make the optic under the NumPy backend or in float64."""
    import optiland.backend as be
    from optiland.backend.utils import is_torch_tensor, to_numpy
    from optiland.optic import Optic

    L, M = np.ravel(np.asarray(L, float)), np.ravel(np.asarray(M, float))
    object_z = lambda o: float(np.ravel(to_numpy(o.surfaces.positions))[0])  # noqa: E731
    if device is None:
        return _launch(optic, L, M, wavelength, object_z(optic))
    check = np.unique(np.linspace(0, L.size - 1, min(L.size, 25)).astype(int))
    active = be.get_backend()
    if not any(is_torch_tensor(s.geometry.cs.z) for s in optic.surfaces.surfaces):   # made under NumPy
        be.set_backend("numpy")
    try:
        z0 = object_z(optic)
        data = optic.to_dict()
        ref = _launch(optic, L[check], M[check], wavelength, z0)
    finally:
        be.set_backend(active)
    if to_numpy(ref.x).dtype != np.float64:
        raise ValueError(f"the check rays of the original optic are {to_numpy(ref.x).dtype}, not float64 (an optic "
                         f"made under Optiland's PyTorch backend in float32): make the optic under the NumPy "
                         f"backend or in float64")
    with optiland_torch(device), torch.no_grad():
        copy = Optic.from_dict(data)
        diff = _max_difference(ref, _launch(copy, L[check], M[check], wavelength, z0))
        if not diff <= COPY_TOL:
            raise RuntimeError(f"the PyTorch copy of the optic traces differently from the original (largest "
                               f"difference {diff:.3g}): the optic does not survive Optic.to_dict / from_dict")
        return _launch(copy, L, M, wavelength, z0)


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
                      y_img: np.ndarray, na_cut: float = 0.95, amp_cut: float = 1e-6, n_pad: int = 2,
                      trace_device: str | None = None, ray_chunk: int | None = None) -> ImageField:
    """The image field of the source e_src (centered on the axis, sample step dx in um) through an Optiland
    optic, on the image grid (x_img, y_img) in um, at the image surface of the optic.

    trace_device: None traces with Optiland's active backend (NumPy by default); "cpu" or "cuda" traces with
    Optiland's PyTorch backend in float64 on that device (trace_rays). ray_chunk: rays per chunk of the image sum
    on the GPU, at least 1 (None: (Nx + Ny) x ray_chunk = SUM_ELEMENTS). The sum keeps one chunk of ex and ey,
    (Nx + Ny) x ray_chunk complex128 values (512 MB for SUM_ELEMENTS), so that large ray counts fit in memory."""
    from optiland.backend.utils import is_torch_tensor, to_numpy

    if ray_chunk is not None and ray_chunk < 1:
        raise ValueError(f"ray_chunk {ray_chunk}: use an integer of at least 1, or None")

    L, M, A = angular_spectrum(e_src, dx, wavelength, n_pad)
    keep = (L**2 + M**2 < na_cut**2) & (np.abs(A) > amp_cut * np.abs(A).max())
    # the Jacobian needs neighbors: trace the full grid of kept directions (a bounding box) and mask later
    ii, jj = np.nonzero(keep)
    sl = (slice(ii.min(), ii.max() + 1), slice(jj.min(), jj.max() + 1))
    L, M, A, keep = L[sl], M[sl], A[sl], keep[sl]
    rays = trace_rays(optic, L, M, wavelength, trace_device)
    traced_on = str(rays.x.device) if is_torch_tensor(rays.x) else "numpy"
    to = lambda a: to_numpy(a).reshape(L.shape)  # noqa: E731
    xr, yr, Lr, Mr, W, inten = (to(v) for v in (rays.x, rays.y, rays.L, rays.M, rays.opd, rays.i))
    del rays
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
    xi, yi, Lo, Mo, ao = t(x_img), t(y_img), t(Lr[ok]), t(Mr[ok]), t(a, torch.complex128)
    step = ray_chunk if ray_chunk is not None else max(1, SUM_ELEMENTS // (xi.numel() + yi.numel()))
    E = torch.zeros((xi.numel(), yi.numel()), dtype=torch.complex128, device=dev)
    for s in range(0, ao.numel(), step):
        # in place (exp_, mul_) and freed at the end of each step: only one ex and one ey are in memory
        ex = 1j * k * xi[:, None] * Lo[None, s:s + step]                     # (Nx, chunk), the phase
        ex.exp_().mul_(ao[None, s:s + step])                                 # the weighted x factor
        ey = 1j * k * yi[:, None] * Mo[None, s:s + step]                     # (Ny, chunk)
        ey.exp_()
        E += ex @ ey.T                                                       # (Nx, Ny)
        del ex, ey
    return ImageField(np.asarray(x_img), np.asarray(y_img), E.cpu().numpy(), int(ok.sum()), str(dev), traced_on)
