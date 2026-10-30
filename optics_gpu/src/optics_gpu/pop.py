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
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

import numpy as np
import torch

MM = 1e3          # um per mm (Optiland lens units are mm; wavelengths are um)
SUM_ELEMENTS = 1 << 25   # (Nx + Ny) x chunk: the complex128 elements of ex and ey of one ray chunk (512 MB)
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
    ray_chunk: rays per chunk of the image sum, at least 1 (None: (Nx + Ny) x ray_chunk = SUM_ELEMENTS). The sum
    keeps one chunk of ex and ey, (Nx + Ny) x ray_chunk complex128 values (512 MB for SUM_ELEMENTS), so that large
    ray counts fit in memory."""
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


def image_sum(bundle: RayBundle, x_img, y_img, dz: float = 0.0, ray_chunk: int | None = None) -> torch.Tensor:
    """The field E (Nx, Ny), complex128 on the device of the bundle, on the grid (x_img, y_img) in um (NumPy
    arrays or tensors) in the plane at dz um after the image surface (dz < 0: before it):
    E(x, y) = sum_r a_r exp(i k (L'_r x + M'_r y + N'_r dz)). A plane wave per ray: this is the angular-spectrum
    propagation of the field in image space (air). ray_chunk: as in propagate_through."""
    dev = bundle.device
    xi, yi = (torch.as_tensor(np.asarray(v) if not isinstance(v, torch.Tensor) else v, dtype=torch.float64,
                              device=dev) for v in (x_img, y_img))
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
    eta against the total power, give power=bundle.power / (dx dy) (RayBundle.power, grid pitch dx x dy)."""
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
