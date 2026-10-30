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
"""

from __future__ import annotations

import contextlib
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
              dtype: torch.dtype = torch.complex128) -> torch.Tensor:
    """The field E (Nx, Ny), complex128 on the device of the bundle, on the grid (x_img, y_img) in um (NumPy
    arrays or tensors) in the plane at dz um after the image surface (dz < 0: before it):
    E(x, y) = sum_r a_r exp(i k (L'_r x + M'_r y + N'_r dz)). A plane wave per ray: this is the angular-spectrum
    propagation of the field in image space (air). One plane of image_planes (ray_chunk and dtype: see there)."""
    one = dz.reshape(1) if isinstance(dz, torch.Tensor) else [float(dz)]
    return image_planes(bundle, x_img, y_img, one, ray_chunk=ray_chunk, dtype=dtype)[0]


def image_planes(bundle: RayBundle, x_img, y_img, dz, ray_chunk: int | None = None,
                 dtype: torch.dtype = torch.complex128) -> torch.Tensor:
    """The fields E (Z, Nx, Ny), complex128 on the device of the bundle, on the grid (x_img, y_img) in um in the Z
    planes dz (um after the image surface; a number or a 1-D array or tensor):
    E_z(x, y) = sum_r a_r exp(i k (L'_r x + M'_r y + N'_r dz_z)), the same sum as image_sum for each plane.

    One pass over the rays for all planes. For each chunk of rays:
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
    GROUP_RAYS. The result E (Z Nx Ny complex128 values) is outside that budget."""
    if dtype not in (torch.complex128, torch.complex64):
        raise ValueError(f"dtype {dtype}: use torch.complex128 or torch.complex64")
    dev = bundle.device
    xi, yi, dzs = _grid(x_img, dev, "x_img"), _grid(y_img, dev, "y_img"), _grid(dz, dev, "dz")
    nz, nx, ny = dzs.numel(), xi.numel(), yi.numel()
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
