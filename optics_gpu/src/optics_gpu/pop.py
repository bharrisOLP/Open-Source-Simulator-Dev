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

Two pipelines (propagate_through):
- trace_device None, the NumPy reference: angular spectrum, trace, Jacobian and weights in NumPy; the image sum on
  the GPU if there is one.
- trace_device "cpu" or "cuda", the device pipeline (trace_bundle, image_sum): angular spectrum (torch.fft), trace
  (Optiland's PyTorch backend), Jacobian, weights and image sum stay on that device in float64. The trace,
  Jacobian and weights run in blocks of rows of the direction grid (a halo of one row on each side gives the same
  central differences as one full grid), so the temporaries of the trace are bounded by trace_chunk rays. Only the
  kept rays stay in memory (RayBundle: L', M', N' and a, 40 bytes per ray). A RayBundle gives fields at more planes
  (dz) and on more grids without a new trace: focus and lateral scans on the device.

The image sum of the device pipeline (image_planes; image_sum is one plane of it) computes all planes of a focus
scan in one pass over the rays: per chunk of rays, the x factors of all planes are one (Z Nx x chunk) matrix, and
its product with the y factors sums groups of GROUP_RAYS rays (one batched product for a small output); the groups
are added in complex128.
The groups give the GPU parallel work also for a small image (one 128 x 128 plane), and they keep the float32 sums
short for the complex64 option.

image_planes(..., method="nufft") computes the same sum on a uniform image grid by a type-1 non-uniform FFT
(_image_planes_nufft): each ray spreads its weight with an "exponential of semicircle" kernel onto w x w points of
a fine grid (2x oversampled), one FFT per plane, then a division by the transform of the kernel. The error is at
most tol sum_r |a_r| in the tests (nufft_width sets w by tol). With dtype complex64 the spread adds in float32, in
flush periods that keep at most NUFFT_TERMS32 terms in a float32 value (nufft32_bound gives the error bound).
method "auto" selects the non-uniform FFT (with the spread in complex128) or the direct sum by a rule from timing
rows (_auto_nufft). The default stays the direct sum.
"""

from __future__ import annotations

import contextlib
import functools
import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

import numpy as np
import torch

MM = 1e3          # um per mm (Optiland lens units are mm; wavelengths are um)
SUM_ELEMENTS = 1 << 25   # (Nx + Ny) x chunk: the complex128 elements of ex and ey of one ray chunk (512 MB)
GROUP_RAYS = 1024        # rays per group of image_planes (one matrix of the batched product; see _add_grouped)
GROUP_OUTPUTS = 1 << 20  # image_planes: below Z Nx Ny = GROUP_OUTPUTS, one batched product for the groups of a chunk
ALLOC_SLACK = 6 << 16    # complex128 values (6 MB): allocator rounding, up to 1 MB for each of 6 large temporaries
PHASE32_MAX = 256.0      # rad: the largest grid phase k L' (x - xc) of the complex64 image sum (see image_planes)
TRACE_RAYS = 1 << 21     # rays per block of the device trace, Jacobian and weights (see trace_bundle)
COPY_TOL = 1e-9          # mm and direction cosines: the PyTorch copy of an optic against the original (check rays)
N_CHECK = 25             # check rays of the PyTorch copy of an optic
NUFFT_SIGMA = 2          # the non-uniform FFT: the fine grid has at least NUFFT_SIGMA x the image points per axis
NUFFT_BETA = 2.30        # beta / w of its "exponential of semicircle" kernel for NUFFT_SIGMA 2 (Barnett et al. 2019)
NUFFT_QUAD = 128         # Gauss-Legendre nodes of the Fourier transform of the kernel (the deconvolution)
NUFFT_COPIES = 16        # at most this many copies of the fine grid in the spread (fewer atomic adds to one value)
NUFFT_MIN_CHUNK = 4096   # rays per chunk of the spread, at least, when one plane of the fine grid fills the budget
NUFFT_TOL = 1e-12        # the default tol of the non-uniform FFT in complex128
NUFFT_TOL32 = 1e-6       # the default tol of the non-uniform FFT in complex64 (the float32 spread)
NUFFT_TERMS32 = 1024     # the float32 spread: at most this many terms go into one fine-grid value between two flushes
U32 = 2.0**-24           # the unit roundoff of float32
# method "auto" (_auto_nufft): the non-uniform FFT from AUTO_POINTS image points (4 times that for fewer than
# AUTO_PLANES planes), AUTO_RAYS rays x planes and AUTO_WORK rays x planes x points (per dtype of the direct sum;
# from the timing rows, README "The non-uniform FFT")
AUTO_POINTS = {torch.complex128: 64 * 64, torch.complex64: 128 * 128}
AUTO_PLANES = 5
AUTO_RAYS = {torch.complex128: 2000, torch.complex64: 20000}
AUTO_WORK = {torch.complex128: 8e7, torch.complex64: 2e9}
UNIFORM_TOL = 64 * float(np.finfo(float).eps)   # a uniform image axis: |x_i - (x_0 + i h)| <= UNIFORM_TOL max|x|


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


def _launch(optic, L, M, wavelength: float, z0: float):
    """Trace one ray per direction (L, M) from the axis point of the object surface (z0) to the image surface.
    L and M: NumPy arrays, or float64 tensors on the device of Optiland's PyTorch backend (the start values of the
    rays are then made on that device, without a copy from the host). Optiland can change L and M in place: give
    arrays that the caller does not use again."""
    from optiland.rays import RealRays

    if isinstance(L, torch.Tensor):
        n, o = L.numel(), {"dtype": L.dtype, "device": L.device}
        N = torch.sqrt(torch.clamp(1 - L**2 - M**2, min=0))
        rays = RealRays(torch.zeros(n, **o), torch.zeros(n, **o), torch.full((n,), z0, **o), L, M, N,
                        torch.ones(n, **o), torch.full((n,), wavelength, **o))
    else:
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
    L, M = np.ravel(np.asarray(L, float)), np.ravel(np.asarray(M, float))
    if device is None:
        return _launch(optic, L, M, wavelength, _object_z(optic))
    check = _check_indices(L.size)
    with torch_tracer(optic, wavelength, device, L[check], M[check]) as launch:
        return launch(L, M)


def _object_z(optic) -> float:
    from optiland.backend.utils import to_numpy

    return float(np.ravel(to_numpy(optic.surfaces.positions))[0])


def _check_indices(n: int) -> np.ndarray:
    return np.unique(np.linspace(0, n - 1, min(n, N_CHECK)).astype(int))


@contextlib.contextmanager
def torch_tracer(optic, wavelength: float, device: str, L_check: np.ndarray,
                 M_check: np.ndarray) -> Iterator[Callable]:
    """A ray launcher on Optiland's PyTorch backend in float64 on device (optiland_torch, torch.no_grad), for a copy
    of optic (Optic.to_dict, then Optic.from_dict). Inside the block, launch(L, M) traces one ray per direction from
    the object point to the image surface (see _launch); the copy is made and checked one time for all launches.

    The check traces the rays (L_check, M_check) through the original and through the copy; if they differ by more
    than COPY_TOL (for example, a surface that the dictionary does not keep), a RuntimeError stops it. The check
    traces the original on the NumPy backend if the optic keeps NumPy arrays (else on the active backend); if that
    reference trace is not float64 (an optic made under Optiland's PyTorch backend in float32), a ValueError stops
    it: make the optic under the NumPy backend or in float64."""
    import optiland.backend as be
    from optiland.backend.utils import is_torch_tensor, to_numpy
    from optiland.optic import Optic

    active = be.get_backend()
    if not any(is_torch_tensor(s.geometry.cs.z) for s in optic.surfaces.surfaces):   # made under NumPy
        be.set_backend("numpy")
    try:
        z0 = _object_z(optic)
        data = optic.to_dict()
        ref = _launch(optic, L_check, M_check, wavelength, z0)
    finally:
        be.set_backend(active)
    if to_numpy(ref.x).dtype != np.float64:
        raise ValueError(f"the check rays of the original optic are {to_numpy(ref.x).dtype}, not float64 (an optic "
                         f"made under Optiland's PyTorch backend in float32): make the optic under the NumPy "
                         f"backend or in float64")
    with optiland_torch(device), torch.no_grad():
        copy = Optic.from_dict(data)
        diff = _max_difference(ref, _launch(copy, L_check, M_check, wavelength, z0))
        if not diff <= COPY_TOL:
            raise RuntimeError(f"the PyTorch copy of the optic traces differently from the original (largest "
                               f"difference {diff:.3g}): the optic does not survive Optic.to_dict / from_dict")
        yield lambda L, M: _launch(copy, L, M, wavelength, z0)


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
                      trace_device: str | None = None, ray_chunk: int | None = None,
                      trace_chunk: int | None = None) -> ImageField:
    """The image field of the source e_src (centered on the axis, sample step dx in um) through an Optiland
    optic, on the image grid (x_img, y_img) in um, at the image surface of the optic.

    trace_device None: the NumPy reference pipeline. The trace uses Optiland's active backend (NumPy by default);
    the Jacobian and the weights use NumPy; the image sum runs on the GPU if there is one.
    trace_device "cpu" or "cuda": the device pipeline (trace_bundle, then image_sum). All steps run on that
    device in float64 (Optiland's PyTorch backend for the trace); only the image field comes back to the host.
    trace_chunk: rays per block of the device trace (see trace_bundle).
    ray_chunk: rays per chunk of the image sum, at least 1. None: for the NumPy pipeline, (Nx + Ny) x ray_chunk =
    SUM_ELEMENTS (the sum keeps one chunk of ex and ey, (Nx + Ny) x ray_chunk complex128 values, 512 MB for
    SUM_ELEMENTS, so that large ray counts fit in memory); for the device pipeline, the chunk of image_planes."""
    from optiland.backend.utils import is_torch_tensor, to_numpy

    if ray_chunk is not None and ray_chunk < 1:
        raise ValueError(f"ray_chunk {ray_chunk}: use an integer of at least 1, or None")
    if trace_device is not None:
        bundle = trace_bundle(optic, e_src, dx, wavelength, na_cut, amp_cut, n_pad, trace_device, trace_chunk)
        E = image_sum(bundle, x_img, y_img, ray_chunk=ray_chunk)
        return ImageField(np.asarray(x_img), np.asarray(y_img), E.cpu().numpy(), bundle.n_rays, str(E.device),
                          str(E.device))

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
    step = _sum_step(ray_chunk, xi, yi)
    E = torch.zeros((xi.numel(), yi.numel()), dtype=torch.complex128, device=dev)
    _add_rays(E, xi, yi, Lo, Mo, ao, k, step)
    return ImageField(np.asarray(x_img), np.asarray(y_img), E.cpu().numpy(), int(ok.sum()), str(dev), traced_on)


def _sum_step(ray_chunk: int | None, xi: torch.Tensor, yi: torch.Tensor) -> int:
    if ray_chunk is not None and ray_chunk < 1:
        raise ValueError(f"ray_chunk {ray_chunk}: use an integer of at least 1, or None")
    return ray_chunk if ray_chunk is not None else max(1, SUM_ELEMENTS // (xi.numel() + yi.numel()))


def _add_rays(E: torch.Tensor, xi: torch.Tensor, yi: torch.Tensor, Lo: torch.Tensor, Mo: torch.Tensor,
              ao: torch.Tensor, k: float, step: int) -> None:
    """E += sum_r ao_r exp(i k (Lo_r x + Mo_r y)) on the grid (xi, yi), in chunks of step rays."""
    for s in range(0, ao.numel(), step):
        # in place (exp_, mul_) and freed at the end of each step: only one ex and one ey are in memory
        ex = 1j * k * xi[:, None] * Lo[None, s:s + step]                     # (Nx, chunk), the phase
        ex.exp_().mul_(ao[None, s:s + step])                                 # the weighted x factor
        ey = 1j * k * yi[:, None] * Mo[None, s:s + step]                     # (Ny, chunk)
        ey.exp_()
        E += ex @ ey.T                                                       # (Nx, Ny)
        del ex, ey


# ---------------------------------------------------------------------------------------------------------------
# The device pipeline
# ---------------------------------------------------------------------------------------------------------------

@dataclass
class RayBundle:
    """The kept rays of the device pipeline, in blocks of tensors on one device (float64 and complex128): image-side
    direction cosines L', M', N' and the weights a = A sqrt(J) df exp(i k (W - L' x_r - M' y_r)) (lengths in um).
    The field at the image surface, moved by dz along z, is E(x, y) = sum_r a_r exp(i k (L' x + M' y + N' dz))
    (image_sum)."""
    blocks: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = field(repr=False)
    wavelength: float
    n_rays: int          # the kept rays (the sum of the block lengths)
    n_traced: int        # the traced rays (the bounding box of the kept directions)
    device: torch.device
    # the power of the kept rays, sum_r |A_r df|^2 / df = the integral of |E|^2 over the image plane in um^2 (a 0-d
    # tensor; Parseval, with the power conservation of the sqrt(J) weights). It does not depend on an image grid
    # or on dz: for the overlap with the field on a grid of pitch dx x dy, use power=bundle.power / (dx dy).
    power: torch.Tensor = field(repr=False, default=None)

    @property
    def k(self) -> float:
        return 2 * np.pi / self.wavelength


def _np_gradient0(f: torch.Tensor, x: torch.Tensor, step: float | None) -> torch.Tensor:
    """d f / d x along dim 0 of f (edge_order 1), with the formulas of np.gradient(f, x, axis=0): step is the
    spacing if NumPy takes the axis as uniform (all steps of the full axis equal), else None (the three-point
    formula for unequal steps, with the steps of x)."""
    if f.shape[0] < 2:
        raise ValueError("the Jacobian needs at least 2 directions on each axis of the direction grid")
    out = torch.empty_like(f)
    if step is not None:
        out[1:-1] = (f[2:] - f[:-2]) / (2.0 * step)
        out[0] = (f[1] - f[0]) / step
        out[-1] = (f[-1] - f[-2]) / step
        return out
    d = torch.diff(x)[:, None]
    d1, d2 = d[:-1], d[1:]
    a = -(d2) / (d1 * (d1 + d2))
    b = (d2 - d1) / (d1 * d2)
    c = d1 / (d2 * (d1 + d2))
    out[1:-1] = a * f[:-2] + b * f[1:-1] + c * f[2:]
    out[0] = (f[1] - f[0]) / d[0]
    out[-1] = (f[-1] - f[-2]) / d[-1]
    return out


def _uniform_step(x: np.ndarray) -> float | None:
    """The step of x if np.gradient takes x as uniform (all np.diff(x) equal), else None."""
    d = np.diff(x)
    return float(d[0]) if d.size and (d == d[0]).all() else None


def _angular_spectrum_device(e: torch.Tensor, dx: float, wavelength: float, n_pad: int):
    """angular_spectrum on the device: the 1-D direction axes L (rows) and M (columns) as NumPy arrays (the same
    values as the NumPy grids) and the amplitudes A dL dM as a complex128 tensor on the device of e."""
    nx, ny = e.shape
    pad = torch.zeros((nx * n_pad, ny * n_pad), dtype=torch.complex128, device=e.device)
    pad[:nx, :ny] = e
    pad = torch.roll(pad, (-(nx // 2), -(ny // 2)), dims=(0, 1))     # the source center at index 0
    A = torch.fft.fft2(pad)
    del pad
    A.mul_(dx).mul_(dx)
    L = np.fft.fftshift(wavelength * np.fft.fftfreq(nx * n_pad, d=dx))
    M = np.fft.fftshift(wavelength * np.fft.fftfreq(ny * n_pad, d=dx))
    dLdM = (wavelength / (nx * n_pad * dx)) * (wavelength / (ny * n_pad * dx))
    A = torch.fft.fftshift(A)
    return L, M, A.mul_(dLdM).div_(wavelength**2)


def trace_bundle(optic, e_src, dx: float, wavelength: float, na_cut: float = 0.95, amp_cut: float = 1e-6,
                 n_pad: int = 2, device: str = "cuda", trace_chunk: int | None = None) -> RayBundle:
    """Steps 1 and 2 of propagate_through on one PyTorch device in float64, with the same formulas as the NumPy
    reference pipeline: the angular spectrum of e_src (torch.fft), the trace of one ray per direction in the
    bounding box of the kept directions (Optiland's PyTorch backend, torch_tracer), the Jacobian
    J = |d(L', M') / d(L, M)| (the formulas of np.gradient), and the weights of the rays that reach the image.

    The trace, the Jacobian and the weights run in blocks of rows of the direction grid, about trace_chunk rays
    each (None: TRACE_RAYS). Each block traces one more row on each side (a halo), so that its central differences
    are those of the full grid. Thus the memory of the trace is bounded by one block; the bundle keeps 40 bytes per
    kept ray. The only arrays that come to the host are those of the N_CHECK check rays of torch_tracer; the rest
    are scalars (the bounding box, and the tests in Optiland's own trace: about 50 per block)."""
    if device not in ("cpu", "cuda"):
        raise ValueError(f"trace device {device!r}: use 'cpu' or 'cuda' (the trace needs float64)")
    if trace_chunk is not None and trace_chunk < 1:
        raise ValueError(f"trace_chunk {trace_chunk}: use an integer of at least 1, or None")
    dev = torch.device(device)
    if dev.type == "cuda":
        dev = torch.device("cuda", torch.cuda.current_device())    # an explicit index, as the tensors report it
    e = torch.as_tensor(np.asarray(e_src), dtype=torch.complex128, device=dev)
    L1, M1, A = _angular_spectrum_device(e, dx, wavelength, n_pad)
    del e
    Lt, Mt = (torch.as_tensor(v, dtype=torch.float64, device=dev) for v in (L1, M1))
    keep = Lt[:, None] ** 2 + Mt[None, :] ** 2 < na_cut**2
    absA = A.abs()
    keep &= absA > amp_cut * absA.max()
    del absA
    rows, cols = torch.nonzero(keep.any(1)).flatten(), torch.nonzero(keep.any(0)).flatten()
    if rows.numel() == 0:
        raise ValueError("no direction is kept: increase na_cut or decrease amp_cut")
    i0, i1, j0, j1 = int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
    # the bounding box (a copy, so that the full spectrum is freed)
    A, keep = A[i0:i1, j0:j1].clone(), keep[i0:i1, j0:j1].clone()
    L1, M1, Lt, Mt = L1[i0:i1], M1[j0:j1], Lt[i0:i1], Mt[j0:j1]
    nL, nM = L1.size, M1.size
    step_L, step_M = _uniform_step(L1), _uniform_step(M1)
    k = 2 * np.pi / wavelength
    check = _check_indices(nL * nM)
    rows_per_block = max(1, (trace_chunk if trace_chunk is not None else TRACE_RAYS) // nM)
    blocks = []
    df = 1.0 / ((e_src.shape[0] * n_pad * dx) * (e_src.shape[1] * n_pad * dx))    # dfx dfy (1/um^2)
    power = torch.zeros((), dtype=torch.float64, device=dev)
    with torch_tracer(optic, wavelength, device, L1[check // nM], M1[check % nM]) as launch:
        for r0 in range(0, nL, rows_per_block):
            r1 = min(r0 + rows_per_block, nL)
            h0, h1 = max(r0 - 1, 0), min(r1 + 1, nL)                     # the block with a halo of one row
            # new tensors (repeat copies): Optiland can change the ray arrays in place
            rays = launch(Lt[h0:h1].repeat_interleave(nM), Mt.repeat(h1 - h0))
            shape = (h1 - h0, nM)
            Lr, Mr = rays.L.reshape(shape), rays.M.reshape(shape)
            # |d(L', M') / d(L, M)| on the block with its halo, then the rows of the block
            dLp_dL = _np_gradient0(Lr, Lt[h0:h1], step_L)
            dLp_dM = _np_gradient0(Lr.T, Mt, step_M).T
            dMp_dL = _np_gradient0(Mr, Lt[h0:h1], step_L)
            dMp_dM = _np_gradient0(Mr.T, Mt, step_M).T
            s = slice(r0 - h0, r1 - h0)
            J = torch.abs(dLp_dL * dMp_dM - dLp_dM * dMp_dL)[s]
            del dLp_dL, dLp_dM, dMp_dL, dMp_dM
            xr, yr, W = (v.reshape(shape)[s] * MM for v in (rays.x, rays.y, rays.opd))   # um
            Lr, Mr, Nr, inten = Lr[s], Mr[s], rays.N.reshape(shape)[s], rays.i.reshape(shape)[s]
            del rays
            ok = keep[r0:r1] & torch.isfinite(W) & (inten > 0)
            ok &= torch.isfinite(J) & (J > 0)
            A_ok = A[r0:r1][ok]
            power += (A_ok.real**2 + A_ok.imag**2).sum()
            a = A_ok * torch.sqrt(J[ok]) * torch.exp(1j * k * (W[ok] - Lr[ok] * xr[ok] - Mr[ok] * yr[ok]))
            blocks.append((Lr[ok], Mr[ok], Nr[ok], a))
            del xr, yr, W, Lr, Mr, Nr, inten, J, ok, A_ok
    n_rays = sum(b[3].numel() for b in blocks)
    return RayBundle(blocks, wavelength, n_rays, nL * nM, dev, power / df)


def _grid(v, dev: torch.device, name: str) -> torch.Tensor:
    """A grid axis or a list of planes (a number, a 1-D NumPy array or tensor) as a 1-D float64 tensor on dev;
    a ValueError for more than one dimension (for example a meshgrid array)."""
    t = torch.as_tensor(np.asarray(v) if not isinstance(v, torch.Tensor) else v, dtype=torch.float64, device=dev)
    if t.dim() > 1:
        raise ValueError(f"{name} has the shape {tuple(t.shape)}: give a 1-D axis (not a meshgrid array)")
    return t.reshape(-1)


def image_sum(bundle: RayBundle, x_img, y_img, dz: float = 0.0, ray_chunk: int | None = None,
              dtype: torch.dtype = torch.complex128, method: str = "direct", tol: float | None = None) -> torch.Tensor:
    """The field E (Nx, Ny), complex128 on the device of the bundle, on the grid (x_img, y_img) in um (NumPy
    arrays or tensors) in the plane at dz um after the image surface (dz < 0: before it):
    E(x, y) = sum_r a_r exp(i k (L'_r x + M'_r y + N'_r dz)). A plane wave per ray: this is the angular-spectrum
    propagation of the field in image space (air). One plane of image_planes (ray_chunk, dtype, method and tol:
    see there)."""
    one = dz.reshape(1) if isinstance(dz, torch.Tensor) else [float(dz)]
    return image_planes(bundle, x_img, y_img, one, ray_chunk=ray_chunk, dtype=dtype, method=method, tol=tol)[0]


def image_planes(bundle: RayBundle, x_img, y_img, dz, ray_chunk: int | None = None,
                 dtype: torch.dtype = torch.complex128, method: str = "direct",
                 tol: float | None = None) -> torch.Tensor:
    """The fields E (Z, Nx, Ny), complex128 on the device of the bundle, on the grid (x_img, y_img) in um in the Z
    planes dz (um after the image surface; a number or a 1-D array or tensor):
    E_z(x, y) = sum_r a_r exp(i k (L'_r x + M'_r y + N'_r dz_z)), the same sum as image_sum for each plane.

    method "direct" (the default): the sum below, 8 Nx Ny floating-point operations per ray and plane.
    method "nufft": a type-1 non-uniform FFT (_image_planes_nufft), w^2 complex additions per ray and plane (w = 15
    for tol 1e-12) plus one FFT per plane, with an error of at most tol sum_r |a_r| in the tested cases (tol from
    1e-12 to 1e-2 sets the kernel width w; see nufft_width; None: NUFFT_TOL = 1e-12 for complex128, NUFFT_TOL32 =
    1e-6 for complex64). It needs a uniform grid on each axis (x_i = x_0 + i h within UNIFORM_TOL max|x|; for
    example np.arange or np.linspace); for another grid it warns and uses the direct sum. With dtype complex64 the
    spread adds in float32 (see _image_planes_nufft and nufft32_bound); the FFT stays in complex128.
    method "auto": the non-uniform FFT where _auto_nufft expects it to be faster (a uniform grid; a rule from the
    logged timing rows, README "The non-uniform FFT"), else the direct sum in dtype (no warning). "auto" uses the
    spread in complex128 also for dtype complex64 (tol None: 1e-6), because the float32 spread was slower than the
    complex128 spread in 122 of 144 logged rows on the RX 7900 XT (README); its error, at most tol sum_r |a_r| in
    the tests, is far below the bound of the complex64 direct sum.

    The direct sum: one pass over the rays for all planes. For each chunk of rays:
    1. the weights of each plane, w_zr = a_r exp(i (k L'_r xc + k M'_r yc + k N'_r dz_z)), in float64 (the phase
       reference: xc and yc are the centers of the grid axes; the large phases k N' dz stay in float64);
    2. the grid factors exp(i k L'_r (x - xc)) (Nx, chunk) and exp(i k M'_r (y - yc)) (Ny, chunk) in dtype;
    3. the x factors times the weights of all planes, one (Z Nx, chunk) matrix, and its product with the y factors
       (_add_grouped): each group of GROUP_RAYS rays is summed in dtype, the groups in complex128 (one batched
       product for Z Nx Ny < GROUP_OUTPUTS; else addmm_ into E for complex128, one product per group for complex64).

    dtype: torch.complex128 (default; the reference precision) or torch.complex64 (steps 2 and 3 in float32). For
    complex64 the largest grid phase theta = k max|L'| max|x - xc| (and the same in y) must be at most PHASE32_MAX
    rad, else a ValueError stops it (this check reads one scalar from the device). The guard limits the phase, not
    the error. By the usual rounding analysis (u = 2^-24, to first order), the x and y factors of a term have a
    phase error of at most 3 theta u each (the rounding of k L', of x - xc and of their product), and the float32
    sum of a group of n rays adds at most about n u of the sum of |terms|: |E64 - E128| <= about
    (GROUP_RAYS + 6 PHASE32_MAX) u sum_r |a_r| = 1.5e-4 sum_r |a_r| (to a factor of order 1). Rounding errors of
    random sign stay far below that bound; one plane wave (all rays the same) comes nearer to it.
    tests/test_image_planes.py gives the observed values for the tested fields.

    ray_chunk: rays per chunk, at least 1. None: _planes_step, the chunk whose temporaries (weights, factors,
    products) are at most SUM_ELEMENTS complex128 values, a multiple of GROUP_RAYS when it is larger than
    GROUP_RAYS. The result E (Z Nx Ny complex128 values) is outside that budget. For method "nufft", ray_chunk
    is the rays per chunk of the spread (None: _nufft_steps)."""
    if dtype not in (torch.complex128, torch.complex64):
        raise ValueError(f"dtype {dtype}: use torch.complex128 or torch.complex64")
    if method not in ("direct", "nufft", "auto"):
        raise ValueError(f"method {method!r}: use 'direct', 'nufft' or 'auto'")
    dev = bundle.device
    xi, yi, dzs = _grid(x_img, dev, "x_img"), _grid(y_img, dev, "y_img"), _grid(dz, dev, "dz")
    nz, nx, ny = dzs.numel(), xi.numel(), yi.numel()
    if method != "direct":
        if tol is None:
            tol = NUFFT_TOL32 if dtype == torch.complex64 else NUFFT_TOL
        w, beta = nufft_width(tol)
        if ray_chunk is not None and ray_chunk < 1:
            raise ValueError(f"ray_chunk {ray_chunk}: use an integer of at least 1, or None")
        if nz * nx * ny == 0 or bundle.n_rays == 0:
            return torch.zeros((nz, nx, ny), dtype=torch.complex128, device=dev)
        if method == "nufft" or _auto_nufft(bundle.n_rays, nz, nx, ny, dtype):
            ax, ay = _uniform_axis(xi), _uniform_axis(yi)
            if ax is not None and ay is not None:
                spread = dtype if method == "nufft" else torch.complex128
                return _image_planes_nufft(bundle, ax, ay, nx, ny, dzs, w, beta, ray_chunk, spread)
            if method == "nufft":
                warnings.warn("image_planes: the image grid is not uniform on each axis, so the non-uniform FFT "
                              "cannot be used; the direct sum computes the field", stacklevel=2)
    step = _planes_step(ray_chunk, nz, nx, ny, dtype)
    E = torch.zeros((nz * nx, ny), dtype=torch.complex128, device=dev)
    if E.numel() == 0 or bundle.n_rays == 0:
        return E.reshape(nz, nx, ny)
    real = torch.float64 if dtype == torch.complex128 else torch.float32
    k = bundle.k
    xc, yc = 0.5 * (xi.amin() + xi.amax()), 0.5 * (yi.amin() + yi.amax())      # 0-d tensors (no host copy)
    ix, iy = 1j * (xi - xc).to(real), 1j * (yi - yc).to(real)                  # i (x - xc), in dtype
    if dtype == torch.complex64:
        _check_phase32(bundle, xi - xc, yi - yc)
    kdz = k * dzs[:, None]
    for Lo, Mo, No, ao in bundle.blocks:
        for s in range(0, ao.numel(), step):
            q, p = k * Lo[s:s + step], k * Mo[s:s + step]                     # k L', k M' (float64)
            ph = kdz * No[None, s:s + step]                                    # (Z, chunk) float64
            ph += q * xc + p * yc
            w = 1j * ph
            del ph
            w = w.exp_().mul_(ao[None, s:s + step]).to(dtype)                 # (Z, chunk), the weights
            ex = (ix[:, None] * q.to(real)[None, :]).exp_()                    # (Nx, chunk)
            exz = ex.mul_(w) if nz == 1 else (w[:, None, :] * ex[None]).reshape(nz * nx, -1)   # (Z Nx, chunk)
            del ex, w
            ey = (iy[:, None] * p.to(real)[None, :]).exp_()                    # (Ny, chunk)
            _add_grouped(E, exz, ey)
            del exz, ey
    return E.reshape(nz, nx, ny)


def _planes_memory(nz: int, nx: int, ny: int, dtype: torch.dtype) -> tuple[float, float]:
    """The temporaries of one chunk of image_planes, in complex128 values: (per ray, fixed). Per ray, the largest of
    the three steps of a chunk (each with the vectors k L', k M' and their float32 copies, 2.5 values):
    - the weights: the phase (Z, float64) and exp(i phase) (Z, complex128), then the weights in dtype: 1.5 Z;
    - the x factors (Nx), the weights (Z) and the x factors of all planes (Z Nx), in dtype;
    - the products: the x factors of all planes and the y factors (Z Nx + Ny) in dtype; with groups in one batched
      product (Z Nx Ny < GROUP_OUTPUTS), also the products of the groups (Z Nx Ny / GROUP_RAYS per ray; in dtype,
      and for complex64 their complex128 copy, made by sum with dtype).
    Fixed: with groups in one batched product, the sum of the groups (Z Nx Ny, complex128; the product of a last
    short group is smaller and comes after it); else, for complex64, the product of one group (Z Nx Ny,
    complex64); complex128 without groups adds into E in place (addmm_). Also ALLOC_SLACK for the rounding of the
    PyTorch caching allocator (a block can be up to 1 MB larger than the tensor)."""
    size = 1.0 if dtype == torch.complex128 else 0.5
    out = nz * nx * ny
    weights = 1.5 * nz
    factors = size * (nx + nz + nz * nx) if nz > 1 else size * (nx + 1)
    products = size * (nz * nx + ny)
    if out < GROUP_OUTPUTS:
        products += (size + (dtype == torch.complex64)) * out / GROUP_RAYS
        fixed = float(out)
    else:
        fixed = size * out if dtype == torch.complex64 else 0.0
    return max(weights, factors, products) + 2.5, fixed + ALLOC_SLACK


def _planes_step(ray_chunk: int | None, nz: int, nx: int, ny: int, dtype: torch.dtype) -> int:
    """The rays per chunk of image_planes: ray_chunk, or (None) the most rays whose temporaries (_planes_memory) are
    at most SUM_ELEMENTS complex128 values, a multiple of GROUP_RAYS when it is larger. If the fixed part alone is
    above SUM_ELEMENTS (complex64 with E above 2 SUM_ELEMENTS values, 1 GB), the chunk is GROUP_RAYS rays and the
    temporaries are above the budget."""
    if ray_chunk is not None:
        if ray_chunk < 1:
            raise ValueError(f"ray_chunk {ray_chunk}: use an integer of at least 1, or None")
        return ray_chunk
    per_ray, fixed = _planes_memory(nz, nx, ny, dtype)
    if fixed >= SUM_ELEMENTS:
        return GROUP_RAYS
    step = max(1, int((SUM_ELEMENTS - fixed) / per_ray))
    return step - step % GROUP_RAYS if step > GROUP_RAYS else step


def _add_grouped(E: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> None:
    """E (P, Q, complex128) += u @ v.T for u (P, c) and v (Q, c) of one dtype, the c rays in groups of GROUP_RAYS
    (a last group can have fewer rays): each group summed in the dtype of u and v, the groups added in complex128.
    - P Q < GROUP_OUTPUTS: one batched product of all full groups (more parallel work for the GPU), then the sum
      of the groups in complex128, and one more product for a last short group;
    - else: complex128 adds the whole chunk into E in place (addmm_; the matrix product has enough parallel work,
      and float64 needs no groups); complex64 adds one product per group."""
    (P, c), Q = u.shape, v.shape[0]
    if P * Q >= GROUP_OUTPUTS:
        if u.dtype == torch.complex128:
            E.addmm_(u, v.T)
        else:
            for g in range(0, c, GROUP_RAYS):
                E += u[:, g:g + GROUP_RAYS] @ v[:, g:g + GROUP_RAYS].T
        return
    nb = c // GROUP_RAYS
    n = nb * GROUP_RAYS
    if nb:
        ub = u[:, :n].reshape(P, nb, GROUP_RAYS).transpose(0, 1)                # (nb, P, G): views, no copy
        vb = v[:, :n].reshape(Q, nb, GROUP_RAYS).permute(1, 2, 0)               # (nb, G, Q)
        E += torch.bmm(ub, vb).sum(0, dtype=torch.complex128)
    if n < c:
        if u.dtype == torch.complex128:
            E.addmm_(u[:, n:], v[:, n:].T)
        else:
            E += u[:, n:] @ v[:, n:].T


def _check_phase32(bundle: RayBundle, xr: torch.Tensor, yr: torch.Tensor) -> None:
    """ValueError if a grid phase k L' (x - xc) or k M' (y - yc) of the complex64 image sum is larger than
    PHASE32_MAX rad (one scalar comes to the host)."""
    lm = torch.stack([torch.stack([b[0].abs().amax(), b[1].abs().amax()]) for b in bundle.blocks if b[0].numel()])
    phase = bundle.k * torch.maximum(lm[:, 0].amax() * xr.abs().amax(), lm[:, 1].amax() * yr.abs().amax())
    worst = float(phase)
    if not worst <= PHASE32_MAX:
        raise ValueError(f"the largest grid phase of the complex64 image sum is {worst:.0f} rad (limit "
                         f"{PHASE32_MAX:.0f}): use dtype=torch.complex128, or a smaller image grid")


# ---------------------------------------------------------------------------------------------------------------
# The non-uniform FFT of the image sum (image_planes, method "nufft")
# ---------------------------------------------------------------------------------------------------------------

def nufft_width(tol: float) -> tuple[int, float]:
    """The kernel width w (fine-grid points per axis) and the shape beta of the "exponential of semicircle" (ES)
    kernel phi(z) = exp(beta (sqrt(1 - z^2) - 1)), |z| <= 1, for the tolerance tol (1e-12 to 1e-2) with a fine grid
    of NUFFT_SIGMA = 2 x the image points: w = ceil(log10(1 / tol)) + 3 and beta = NUFFT_BETA w. The kernel and
    beta are those of A. H. Barnett, J. Magland and L. af Klinteberg, "A parallel nonuniform fast Fourier transform
    library based on an 'exponential of semicircle' kernel", SIAM J. Sci. Comput. 41 (2019) C479. Their width,
    ceil(log10(1 / tol)) + 1, is for a relative l2 error of about tol. Here tol is a bound of the largest error
    relative to sum_r |a_r|: the worst case is one ray at the worst position on both axes of a grid with the fine
    grid exactly 2 Nx (the x and y errors add). tests/test_nufft.py (test_nufft_of_one_ray_at_the_worst_position)
    shows that the width of the paper and that width plus 1 are above tol there, and that plus 2 (this width) is
    below tol. The range stops at 1e-12, near the float64 rounding of the sum."""
    if not 1e-12 <= tol <= 1e-2:
        raise ValueError(f"tol {tol}: use a tolerance from 1e-12 to 1e-2")
    w = int(np.ceil(-np.log10(tol) - 1e-9)) + 3
    return w, NUFFT_BETA * w


def _uniform_axis(v: torch.Tensor) -> tuple[float, float] | None:
    """(x_0, h) of a uniform grid axis v (a 1-D tensor; Nx values come to the host): v_i = x_0 + i h within
    UNIFORM_TOL max|v|, h = (v_last - v_0) / (N - 1) not 0. One point: (v_0, 1.0) (any step gives its one value).
    None for a non-uniform axis."""
    a = v.detach().cpu().numpy().astype(float)
    if a.size == 1:
        return float(a[0]), 1.0
    h = (a[-1] - a[0]) / (a.size - 1)
    if not (np.isfinite(h) and h != 0):
        return None
    worst = np.max(np.abs(a - (a[0] + np.arange(a.size) * h)))
    return (float(a[0]), float(h)) if worst <= UNIFORM_TOL * np.max(np.abs(a)) else None


def _fine_size(n: int, w: int) -> int:
    """The points of the fine grid on one axis: the smallest even 2^a 3^b 5^c at least NUFFT_SIGMA n and 2 w."""
    m = max(NUFFT_SIGMA * n, 2 * w)
    while True:
        r = m
        for p in (2, 3, 5):
            while r % p == 0:
                r //= p
        if r == 1 and m % 2 == 0:
            return m
        m += 1


@functools.lru_cache(maxsize=4)
def _gauss_legendre(n: int) -> tuple[np.ndarray, np.ndarray]:
    """The n Gauss-Legendre nodes and weights on [-1, 1] (np.polynomial.legendre.leggauss), read-only arrays."""
    u, wq = np.polynomial.legendre.leggauss(n)
    u.flags.writeable = wq.flags.writeable = False
    return u, wq


def _es_transform(m: np.ndarray, n_fine: int, w: int, beta: float) -> np.ndarray:
    """psi(m) = integral of phi(2 s / w) exp(i 2 pi m s / n_fine) ds (s in fine-grid cells), the Fourier transform
    of the ES kernel at the modes m (phi is even and real): (w / 2) integral over z in [-1, 1] of
    phi(z) cos(xi w z / 2) dz, xi = 2 pi m / n_fine. With z = sin(t), t in [-pi/2, pi/2], phi(sin t) =
    exp(beta (cos t - 1)) and the integrand exp(beta (cos t - 1)) cos(xi w sin(t) / 2) cos(t) are smooth (in z, phi
    has square-root ends, so a Gauss-Legendre rule in z converges slowly); NUFFT_QUAD Gauss-Legendre nodes in t
    (_gauss_legendre: computed one time; NumPy takes about 1 ms for 128 nodes)."""
    u, wq = _gauss_legendre(NUFFT_QUAD)
    t, wt = 0.5 * np.pi * u, 0.5 * np.pi * wq
    weight = wt * np.exp(beta * (np.cos(t) - 1.0)) * np.cos(t)
    xi = 2 * np.pi * np.asarray(m, float) / n_fine
    return 0.5 * w * (np.cos(np.outer(xi, 0.5 * w * np.sin(t))) @ weight)


def _es_start(f: torch.Tensor, n_fine: int, w: int) -> tuple[torch.Tensor, torch.Tensor]:
    """u = (f - round(f)) n_fine, the position of each ray in fine-grid cells, and l0 = ceil(u - w / 2), the first
    of the w fine-grid points of its kernel (float64; the index is l0 mod n_fine)."""
    u = (f - torch.round(f)) * n_fine
    return u, torch.ceil(u - 0.5 * w)


def _es_spread_1d(f: torch.Tensor, n_fine: int, w: int, beta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """The ES kernel values (c, w) and their fine-grid indices (c, w) on one axis for the c rays with f = cycles
    per image grid step (L' h / lambda). Only f mod 1 matters (the field on the grid is periodic in f), so u =
    (f - round(f)) n_fine is in [-n_fine / 2, n_fine / 2]; the kernel covers the w points l0 .. l0 + w - 1,
    l0 = ceil(u - w / 2) (indices mod n_fine), at z = (l - u) 2 / w in [-1, 1]."""
    u, l0 = _es_start(f, n_fine, w)
    j = torch.arange(w, dtype=torch.float64, device=f.device)
    z = (l0[:, None] - u[:, None] + j) * (2.0 / w)
    phi = torch.exp(beta * (torch.sqrt(torch.clamp(1.0 - z * z, min=0.0)) - 1.0))
    idx = torch.remainder(l0.to(torch.int64)[:, None] + j.to(torch.int64), n_fine)
    return phi, idx


def _nufft_steps(ray_chunk: int | None, nz: int, n1: int, n2: int, w: int) -> tuple[int, int, int]:
    """(planes per batch zb, copies of the fine grid K, rays per chunk) of _image_planes_nufft, in complex128
    values. A batch of zb planes keeps the fine grids (n1 n2 zb); the FFT step (the fine grids, the output of the
    FFT and its work arrays) needs a few times that: zb is the most planes with 6 n1 n2 zb at most SUM_ELEMENTS (at
    least 1; tests/test_nufft.py checks the peak against the budget). The spread adds into K copies of the fine
    grids (the rays in turn): K n1 n2 zb is at most SUM_ELEMENTS / 8, K from 1 to NUFFT_COPIES. A chunk of the
    spread keeps per ray the values (w^2 zb), the indices and the products of the kernels (2 w^2) and small vectors
    (6 w + 3 zb): the chunk is the most rays whose temporaries, with the K copies, are at most SUM_ELEMENTS (at
    least 1), or ray_chunk.
    Exception: if 6 n1 n2 is above SUM_ELEMENTS (with the default, an image grid above about 1150 x 1150 points),
    a batch is one plane, the chunk is at least NUFFT_MIN_CHUNK rays (from about 2800 x 2800 points the fine grid
    alone fills the budget), and the peak is above the budget: the fine grid, its FFT step, and NUFFT_MIN_CHUNK
    times the values per ray."""
    zb = max(1, min(nz, SUM_ELEMENTS // (6 * n1 * n2)))
    copies = max(1, min(NUFFT_COPIES, SUM_ELEMENTS // (8 * n1 * n2 * zb)))
    if ray_chunk is not None:
        return zb, copies, ray_chunk
    per_ray = w * w * (zb + 2) + 6 * w + 3 * zb
    step = (SUM_ELEMENTS - copies * n1 * n2 * zb - ALLOC_SLACK) // per_ray
    return zb, copies, max(NUFFT_MIN_CHUNK if 6 * n1 * n2 > SUM_ELEMENTS else 1, step)


def _golden_stride(n: int) -> int:
    """A step coprime with n near 0.618 n (1 for n <= 2): the rays r * step mod n (r = 0 .. n - 1) are all rays,
    and rays next to each other in that order are far apart in the order of the direction grid."""
    if n <= 2:
        return 1
    s = max(1, round(0.6180339887498949 * n))
    while np.gcd(s, n) != 1:
        s += 1
    return s


def _spread32_segments(bundle: RayBundle, fx: float, fy: float, n1: int, n2: int, w: int, copies: int,
                       step: int) -> list[tuple[int, int, int] | None]:
    """The chunks of rays of the float32 spread and the flushes, in the order of the spread: (block, s, e) for the
    rays s .. e - 1 of a block (in the golden order of _image_planes_nufft; ray i of a chunk goes to the copy
    i mod copies), and None for a flush (the float32 copies are added into the complex128 fine grid and set to 0).
    Between two flushes, every value of every copy gets at most NUFFT_TERMS32 terms.

    The count: the first kernel point l0 of a ray (_es_start; fx and fy: cycles per grid step per unit of L' and M')
    falls into one of n1 // w blocks of the fine grid on each axis (the last block has w to 2 w - 1 points). A point
    l gets the terms of the rays with l0 in l - w + 1 .. l (mod n1), and these w points are in at most two adjacent
    blocks. So the ray count of 2 x 2 adjacent blocks of one copy, summed since the last flush, is an upper bound of
    the terms of each point of that copy. A chunk that would take this bound above NUFFT_TERMS32 starts after a
    flush; a chunk that is above it alone is cut in half (a chunk of copies x NUFFT_TERMS32 rays always passes:
    each copy gets at most NUFFT_TERMS32 of its rays). After a chunk that takes the bound above NUFFT_TERMS32 / 2,
    a flush follows (the next chunk of that size would not pass). Each try reads one number from the device."""
    dev = bundle.device
    bx, by = n1 // w, n2 // w
    count = torch.zeros((copies, bx, by), dtype=torch.int64, device=dev)
    segs: list[tuple[int, int, int] | None] = []
    open_ = False                                          # rays since the last flush
    for i, (Lo, Mo, _, ao) in enumerate(bundle.blocks):
        n = ao.numel()
        stride = _golden_stride(n)
        s, c = 0, step
        while s < n:
            e = min(s + c, n)
            sel = torch.arange(s, e, device=dev).mul_(stride).remainder_(n)
            jx = torch.remainder(_es_start(Lo[sel] * fx, n1, w)[1].to(torch.int64), n1).div_(w, rounding_mode="floor")
            jy = torch.remainder(_es_start(Mo[sel] * fy, n2, w)[1].to(torch.int64), n2).div_(w, rounding_mode="floor")
            key = (torch.arange(e - s, device=dev) % copies) * (bx * by) + jx.clamp_(max=bx - 1) * by
            key += jy.clamp_(max=by - 1)
            new = count + torch.bincount(key, minlength=copies * bx * by).view(copies, bx, by)
            win = new + new.roll(-1, 1)
            most = int((win + win.roll(-1, 2)).amax())
            if most <= NUFFT_TERMS32:
                segs.append((i, s, e))
                count, s, open_ = new, e, True
                if 2 * most > NUFFT_TERMS32:
                    segs.append(None)
                    count.zero_()
                    open_ = False
                elif 4 * most <= NUFFT_TERMS32:
                    c = min(step, 2 * c)
            elif open_:
                segs.append(None)
                count.zero_()
                open_ = False
            else:
                c = max(1, (e - s) // 2)
    return segs


def nufft32_bound(nx: int, ny: int, tol: float = NUFFT_TOL32) -> np.ndarray:
    """The bound of the float32 spread on an Nx x Ny grid: an (Nx, Ny) array B with |E32 - E64| <= B sum_r |a_r| at
    each grid point, to first order in u = U32 = 2^-24. E32 is image_planes(method="nufft", dtype=torch.complex64,
    tol=tol), E64 the same non-uniform FFT with the spread in complex128 (the float64 rounding of both is far
    smaller). The error of E64 itself (at most tol sum_r |a_r| in the tests) comes in addition.

    Derivation (_image_planes_nufft, float32 spread). A term t = c_r phi_x phi_y gets five float32 roundings (c_r,
    phi_x and phi_y from float64 to float32, the product phi_x phi_y, and its product with c_r): at most 5 u |t|.
    A fine-grid value of one copy gets at most n = NUFFT_TERMS32 terms between two flushes (_spread32_segments),
    added in float32 from 0 in any order: at most (n - 1) u times the sum of their |t| (the recursive summation
    bound, Higham, "Accuracy and Stability of Numerical Algorithms", 2nd ed., eq. (4.4), for the real and the
    imaginary part; the complex error by the triangle inequality of the 2-norm). At a flush, the K <= NUFFT_COPIES
    copies of a value are added in complex64 (K - 1 more roundings), then into the complex128 fine grid; the FFT
    and the deconvolution are in complex128. So each fine-grid value b_l has an error of at most
    (n + K + 3) u sum_{t in l} |t|, and the mode (m, n) of E, corr(m, n) sum_l b_l exp(...), has an error of at most
    (n + K + 3) u corr(m, n) sum_r |a_r| Phi_x Phi_y, with Phi = sum_j phi_j of one ray on one axis (|c_r| = |a_r|).
    corr = 1 / (psi_x(m) psi_y(n)) (_es_transform) and Phi <= Phi_max (the largest value over 4096 positions in a
    cell, times 1 + 1e-9: Phi changes by less than 1e-4 of its value over a cell for w >= 5, so the sampling error
    is below about 1e-10), so B = (NUFFT_TERMS32 + NUFFT_COPIES + 3) u rho_x(m) rho_y(n), rho(m) = Phi_max / psi(m).
    rho is about 1 at the grid center and largest at the edges of the grid (the largest modes |m| = N / 2).
    Weights near the float32 range limits (|a| below about 1e-30 or above 1e30) are outside this bound."""
    w, beta = nufft_width(tol)
    return (NUFFT_TERMS32 + NUFFT_COPIES + 3) * U32 * np.outer(_es_rho(nx, w, beta), _es_rho(ny, w, beta))


def _es_rho(n: int, w: int, beta: float) -> np.ndarray:
    """rho(m) = Phi_max / psi(m) for the modes m = i - n // 2 of an axis of n points (nufft32_bound)."""
    n_fine = _fine_size(n, w)
    d = np.arange(4096) / 4096.0                                            # the position in the cell
    z = (2 * d[:, None] + 2 * np.arange(w)[None, :] - w) / w                 # the kernel points (_es_spread_1d)
    phi = np.exp(beta * (np.sqrt(np.clip(1.0 - z * z, 0.0, None)) - 1.0))
    return (1 + 1e-9) * phi.sum(1).max() / _es_transform(np.arange(n) - n // 2, n_fine, w, beta)


def _auto_nufft(n_rays: int, nz: int, nx: int, ny: int, dtype: torch.dtype) -> bool:
    """method "auto" of image_planes: True where the non-uniform FFT is expected to be faster than the direct sum
    (the grid must also be uniform). The rule comes from the logged timing rows (README, "The non-uniform FFT"):
    the image grid has at least AUTO_POINTS[dtype] points, 4 times that for fewer than AUTO_PLANES planes (on smaller
    grids the direct sum costs less per ray than the w^2 adds of the spread; with fewer planes, the spread adds
    fewer values per fine-grid index, and the atomic adds of many rays into the same values take longer), the rays
    times the planes are at least AUTO_RAYS[dtype], and the rays times the planes times the points are at least
    AUTO_WORK[dtype] (below these, the fixed cost of the non-uniform FFT per call and per plane, the zeroed fine
    grids, the FFT and the deconvolution, is not paid back). dtype: the dtype of the direct sum; the non-uniform FFT
    of "auto" spreads in complex128."""
    rays, points = n_rays * nz, AUTO_POINTS[dtype] * (1 if nz >= AUTO_PLANES else 4)
    return nx * ny >= points and rays >= AUTO_RAYS[dtype] and rays * nx * ny >= AUTO_WORK[dtype]


def _image_planes_nufft(bundle: RayBundle, ax: tuple[float, float], ay: tuple[float, float], nx: int, ny: int,
                        dzs: torch.Tensor, w: int, beta: float, ray_chunk: int | None,
                        dtype: torch.dtype = torch.complex128) -> torch.Tensor:
    """image_planes by a type-1 non-uniform FFT on the uniform grid x_i = x0 + i hx (i < nx), y_j = y0 + j hy
    (j < ny) (ax = (x0, hx), ay = (y0, hy)): (Z, Nx, Ny) complex128 on the device of the bundle.

    With the modes m = i - Nx // 2 and n = j - Ny // 2 and the grid point (x_ref, y_ref) of m = n = 0,
    E_z(m, n) = sum_r c_zr exp(i 2 pi (m f_r + n g_r)), c_zr = a_r exp(i k (L'_r x_ref + M'_r y_ref + N'_r dz_z))
    (in float64, as the phase reference of the direct sum), f_r = L'_r hx / lambda and g_r = M'_r hy / lambda.
    1. Spread: each ray adds c_zr phi_x phi_y to w x w points of a periodic fine grid of n1 x n2 points (n1 at
       least 2 Nx; index_add_, complex128), at its position (f_r mod 1) n1, (g_r mod 1) n2 in fine-grid cells.
    2. One inverse FFT per plane (unnormalized): sum_l b_l exp(+i 2 pi m l / n1) (and the same in y).
    3. Deconvolution: the modes m, n of the FFT divided by psi_x(m) psi_y(n) (_es_transform). Errors: the kernel
       is cut at |z| = 1 and the fine grid aliases; both fall with w (nufft_width).
    The planes go in batches of zb planes, the rays in chunks (_nufft_steps); each batch computes the kernel values
    again (2 w per ray), and the spread of a batch adds w^2 zb values per ray.

    dtype complex64, the float32 spread: c_zr and the kernel values are computed in float64 as above, then the
    terms in float32 (c_zr in complex64, phi_x phi_y and the product), and index_add_ adds them as float32 pairs
    (the real and imaginary parts as two float32 values, not as complex64 values; bench/bench_atomics.py times the
    atomic adds of index_add_ for float32, float64, complex64 and complex128). The chunks of _spread32_segments
    keep at most NUFFT_TERMS32 terms in each float32 value: at each flush, the copies are added in complex64, that
    sum is added into a complex128 fine grid, and the copies are set to 0. Steps 2 and 3 are in complex128. The
    bound: nufft32_bound."""
    dev, k = bundle.device, bundle.k
    nz = dzs.numel()
    (x0, hx), (y0, hy) = ax, ay
    fx, fy = hx / bundle.wavelength, hy / bundle.wavelength
    n1, n2 = _fine_size(nx, w), _fine_size(ny, w)
    mx, my = np.arange(nx) - nx // 2, np.arange(ny) - ny // 2
    x_ref, y_ref = x0 + (nx // 2) * hx, y0 + (ny // 2) * hy
    corr = 1.0 / np.outer(_es_transform(mx, n1, w, beta), _es_transform(my, n2, w, beta))
    corr = torch.as_tensor(corr, dtype=torch.float64, device=dev)[:, :, None]
    rows = torch.as_tensor(np.remainder(mx, n1), device=dev)
    cols = torch.as_tensor(np.remainder(my, n2), device=dev)
    zb, copies, step = _nufft_steps(ray_chunk, nz, n1, n2, w)
    single = dtype == torch.complex64
    strides = [_golden_stride(b[3].numel()) for b in bundle.blocks]
    if single:
        segs = _spread32_segments(bundle, fx, fy, n1, n2, w, copies, step)
        if segs and segs[-1] is not None:
            segs.append(None)
    else:
        segs = [(i, s, min(s + step, b[3].numel())) for i, b in enumerate(bundle.blocks)
                for s in range(0, b[3].numel(), step)]

    def spread(b: torch.Tensor, i: int, s: int, e: int, kdz: torch.Tensor) -> None:
        """Adds the terms of the rays s .. e - 1 (golden order) of block i into the fine grids b."""
        Lo, Mo, No, ao = bundle.blocks[i]
        sel = torch.arange(s, e, device=dev).mul_(strides[i]).remainder_(ao.numel())
        L, M = Lo[sel], Mo[sel]
        px, ix = _es_spread_1d(L * fx, n1, w, beta)
        py, iy = _es_spread_1d(M * fy, n2, w, beta)
        ph = No[sel, None] * kdz[None, :]                                      # (c, zb) float64
        ph += ((k * x_ref) * L + (k * y_ref) * M)[:, None]
        c = (1j * ph).exp_().mul_(ao[sel, None])                               # (c, zb), the weights
        del sel, ph
        ix += (torch.arange(ix.shape[0], device=dev) % copies)[:, None] * n1   # the copy of each ray
        idx = (ix[:, :, None] * n2 + iy[:, None, :]).reshape(-1)               # (c w^2)
        del ix, iy
        if single:
            px, py, c = px.to(torch.float32), py.to(torch.float32), c.to(torch.complex64)
        pxy = px[:, :, None] * py[:, None, :]                                   # (c, w, w)
        del px, py
        vals = pxy[:, :, :, None] * c[:, None, None, :]                         # (c, w, w, zb)
        del pxy, c
        if single:
            b.index_add_(0, idx, torch.view_as_real(vals).reshape(idx.numel(), -1, 2))
        else:
            b.index_add_(0, idx, vals.reshape(idx.numel(), -1))

    E = torch.empty((nz, nx, ny), dtype=torch.complex128, device=dev)
    for z0 in range(0, nz, zb):
        kdz = k * dzs[z0:z0 + zb]
        nb = kdz.numel()
        if single:
            b = torch.zeros((n1 * n2, nb), dtype=torch.complex128, device=dev)
            b32 = torch.zeros((copies * n1 * n2, nb, 2), dtype=torch.float32, device=dev)
            for seg in segs:
                if seg is None:                     # a flush: the copies added in complex64, then into complex128
                    b += torch.view_as_complex(b32).reshape(copies, n1 * n2, nb).sum(0)
                    b32.zero_()
                else:
                    spread(b32, *seg, kdz)
            del b32
        else:
            b = torch.zeros((copies * n1 * n2, nb), dtype=torch.complex128, device=dev)
            for seg in segs:
                spread(b, *seg, kdz)
            if copies > 1:
                b = b.reshape(copies, n1 * n2, nb).sum(0)
        F = torch.fft.ifft2(b.reshape(n1, n2, nb), dim=(0, 1), norm="forward")
        del b
        E[z0:z0 + nb] = (F.index_select(0, rows).index_select(1, cols) * corr).permute(2, 0, 1)
        del F
    return E


def _image_sum_direct(bundle: RayBundle, x_img, y_img, dz: float = 0.0,
                      ray_chunk: int | None = None) -> torch.Tensor:
    """The reference of image_planes, one plane: the image sum of the NumPy pipeline (_add_rays: one matrix product
    per chunk of rays, complex128, the phases k L' x without a reference) with the weights a exp(i k N' dz)."""
    dev = bundle.device
    xi, yi = _grid(x_img, dev, "x_img"), _grid(y_img, dev, "y_img")
    step = _sum_step(ray_chunk, xi, yi)
    k = bundle.k
    E = torch.zeros((xi.numel(), yi.numel()), dtype=torch.complex128, device=dev)
    for Lo, Mo, No, ao in bundle.blocks:
        if dz:
            ao = ao * torch.exp(1j * (k * dz) * No)
        _add_rays(E, xi, yi, Lo, Mo, ao, k, step)
    return E


def overlap_efficiency_device(e: torch.Tensor, f: torch.Tensor, power=None) -> torch.Tensor:
    """overlap_efficiency on the device: |<f, e>|^2 / (P <f, f>), a 0-d float64 tensor. P: power (a sum of |E|^2
    over grid samples), or None for <e, e> (the power on the grid only). The power on a grid is lower than the
    total power by the part that is off the grid; that part changes with dz (wide angles leave the grid). For
    eta against the total power, give power=bundle.power / (dx dy) (RayBundle.power, grid pitch dx x dy).
    e can also be a stack of fields (Z, Nx, Ny) on the grid of f (image_planes): then the result is a (Z,) tensor,
    one eta per plane (power: None for the power of each plane, or one power for all planes)."""
    if e.dim() == f.dim() + 1:
        ez, fv = e.reshape(e.shape[0], -1), f.reshape(-1)
        p = (ez.real**2 + ez.imag**2).sum(1) if power is None else power
        return (ez @ fv.conj()).abs() ** 2 / (p * torch.vdot(fv, fv).real)
    p = torch.vdot(e.ravel(), e.ravel()).real if power is None else power
    return torch.vdot(f.ravel(), e.ravel()).abs() ** 2 / (p * torch.vdot(f.ravel(), f.ravel()).real)


def lateral_scan(E: torch.Tensor, x_img, y_img, w: float, sx, sy, power=None) -> torch.Tensor:
    """The coupling efficiency of the field E (on the grid x_img, y_img, um) into a Gaussian mode of radius w (um)
    moved laterally to every (sx_i, sy_j): a (len(sx), len(sy)) float64 tensor on the device of E. The Gaussian is
    separable, so all overlaps are two matrix products: Gx^H E Gy*, with Gx[i, :] = exp(-(x - sx_i)^2 / w^2). The
    grid must hold the moved modes (the norm of each moved mode is that of its samples on the grid). power: as in
    overlap_efficiency_device."""
    t = lambda v: torch.as_tensor(np.asarray(v, float), dtype=torch.float64, device=E.device)  # noqa: E731
    x, y, sx, sy = t(x_img), t(y_img), t(np.atleast_1d(sx)), t(np.atleast_1d(sy))
    gx = torch.exp(-((x[None, :] - sx[:, None]) ** 2) / w**2).to(torch.complex128)     # (Sx, Nx)
    gy = torch.exp(-((y[None, :] - sy[:, None]) ** 2) / w**2).to(torch.complex128)     # (Sy, Ny)
    ov = gx @ E @ gy.T                                                                  # <F_s, E> (real F)
    nf = (gx.abs() ** 2).sum(1)[:, None] * (gy.abs() ** 2).sum(1)[None, :]
    p = torch.vdot(E.ravel(), E.ravel()).real if power is None else power
    return ov.abs() ** 2 / (p * nf)
