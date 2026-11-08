"""The eps sampling and the anisotropic subpixel averaging of ``Cell.eps_inverse`` in JAX, on the GPU.

    eps_inv = eps_inverse(cell, sub=8, sampling="auto")     # (3, 3, nx, ny, nz), as Cell.eps_inverse

The method, the samples and the formulas are those of the NumPy path (``Cell.eps_inverse`` in geometry.py), which
is the reference: the samples of each voxel, the periodic images of each object (``Cell.image_shifts``), the count
of the object boundaries that cross each voxel, the analytic normal where one boundary crosses, the fitted gradient
of eps elsewhere, and the tensor <1/eps> n n^T + (1 / <eps>)(I - n n^T) at an interface. See geometry.py.

The test inside or outside an object must give the same result as the reference at each sample, also for a sample
on a face (a fill fraction changes by one sample there, much more than rounding). This module therefore computes the
Cartesian sample coordinates with the same floating-point operations in the same order as ``Cell.points``: the
products t_i e_ic of the lattice coordinates are made with NumPy on the host (1-D arrays), and the device only adds
them, in the order of the reference. The tests of the objects use the operations of ``inside`` in the same order,
with each square rounded before the sum (no fused multiply-add; ``_sum_of_squares``).
The sums over the samples of a voxel (the means and the first moments) have another order of summation, so these
values differ from the reference by rounding only (tests/test_sampling.py: 1e-12 or better on the test cells).
A difference of the order of rounding can change a decision only at an exact tie: the side or the cap of a cylinder
as the nearest surface (``hypot`` of JAX and of NumPy can differ in the last bit), and a fitted gradient that is zero
in exact arithmetic (a thin feature centered in a voxel; one summation order can give zero, the other a rounding).

The work is split into slabs of voxels along the first used axis, so that 3-D cells at sub 8 fit into the memory of
the GPU: one slab holds the eps samples (float64) and one hit mask. A slab has at most ``max_samples`` samples, or
one plane of voxels when a plane has more (a plane is not split: a 3-D cell with a 256 x 256 cross-section at sub 8
has 3.4e7 samples per plane). The consecutive objects of one kind (a run) are a ``lax.scan`` with an inner
``lax.scan`` over their periodic images (padded to the longest image list): one compile per run, not per object.
"""

from __future__ import annotations

import functools
import itertools

import jax
import jax.numpy as jnp
import numpy as np

from bands_gpu.geometry import Block, Cell, Cylinder, Sphere

_KINDS = {Cylinder: "cylinder", Sphere: "sphere", Block: "block"}
# the members of Cell that set the samples, the images, the averages or the tensors: a subclass that overrides one of
# them gets the NumPy path ("auto") or an error ("jax")
_SAMPLING = ("eps_inverse", "voxel_averages", "_sample", "eps_samples", "points", "axes", "image_shifts",
             "unit_basis", "unit_reciprocal", "lattice", "reciprocal", "moment_bound")

# "auto" uses JAX from this work up (per run of objects of one kind): samples (nx s * ny s * nz s) times the periodic
# images of all objects. The NumPy path costs 3.8 to 6.8 ns per sample and image on the GPU computer
# (benchmarks/2026-11-07/bands-sampling.jsonl, sub 8), so 1e8 is 0.38 to 0.68 s: about the first call of the JAX
# path, with its compile (0.386 to 0.574 s for a new cell shape with one run, about 0.53 s more per further run).
# Below, NumPy is faster.
AUTO_MIN_WORK = 1e8


def supported(cell: Cell) -> bool:
    """True if the JAX path can sample this cell: the objects are Cylinder, Sphere or Block (not subclasses), the
    cell uses the sampling of Cell (no override of a method or property in ``_SAMPLING``), and JAX has float64
    (jax_enable_x64)."""
    own = all(getattr(type(cell), m) is getattr(Cell, m) for m in _SAMPLING)
    return all(type(o) in _KINDS for o in cell.objects) and own and bool(jax.config.jax_enable_x64)


def _params(obj) -> np.ndarray:
    """The scalars of the object, made with the same host operations as ``inside`` and ``normal`` in geometry.py."""
    cx, cy, cz = (float(c) for c in obj.center)
    if type(obj) is Cylinder:
        return np.array([cx, cy, cz, obj.radius**2, obj.height / 2, obj.radius], float)
    if type(obj) is Sphere:
        return np.array([cx, cy, cz, obj.radius**2], float)
    return np.array([cx, cy, cz, obj.size[0] / 2, obj.size[1] / 2, obj.size[2] / 2], float)


def _sum_of_squares(*d):
    """d_0^2 + d_1^2 (+ d_2^2) as NumPy computes it: each square rounded, then the sums from the left. The barrier
    keeps the compiler from fusing a product and a sum into one fused multiply-add (one rounding): XLA does that on
    the CPU (other decisions at points on a cylinder surface; tests/test_sampling.py), not on ROCm."""
    sq = jax.lax.optimization_barrier(tuple(v * v for v in d))
    out = sq[0]
    for v in sq[1:]:
        out = out + v
    return out


def _inside(kind: str, p, x, y, z):
    """``inside`` of geometry.py (the same operations in the same order)."""
    if kind == "cylinder":
        return (_sum_of_squares(x - p[0], y - p[1]) <= p[3]) & (jnp.abs(z - p[2]) <= p[4])
    if kind == "sphere":
        return _sum_of_squares(x - p[0], y - p[1], z - p[2]) <= p[3]
    return (jnp.abs(x - p[0]) <= p[3]) & (jnp.abs(y - p[1]) <= p[4]) & (jnp.abs(z - p[2]) <= p[5])


def _normal(kind: str, p, x, y, z):
    """``normal`` of geometry.py: the outward normal (not normalized) of the nearest surface."""
    px, py, pz = x - p[0], y - p[1], z - p[2]
    if kind == "cylinder":
        side = jnp.abs(jnp.hypot(px, py) - p[5])
        cap = jnp.abs(jnp.abs(pz) - p[4])
        on_side = side <= cap
        return jnp.where(on_side, px, 0.0), jnp.where(on_side, py, 0.0), jnp.where(on_side, 0.0, jnp.sign(pz))
    if kind == "sphere":
        return px, py, pz
    d = jnp.stack(jnp.broadcast_arrays(*[jnp.abs(jnp.abs(pc) - h) for pc, h in zip((px, py, pz), p[3:6])]))
    face = jnp.argmin(d, axis=0)                                    # the first face at a tie, as np.argmin
    return tuple(jnp.where(face == c, jnp.sign(pc), 0.0) for c, pc in enumerate((px, py, pz)))


def _coordinate(terms, axes):
    """A Cartesian coordinate as the sum of the terms t_i e_ic (1-D, along axis i), in the order of Cell.points.
    The result has length 1 on the axes that do not change it (it broadcasts, as the sparse arrays of NumPy)."""
    out = None
    for i, v in zip(axes, terms):
        b = v.reshape(tuple(-1 if j == i else 1 for j in range(3)))
        out = b if out is None else out + b
    return jnp.zeros((1, 1, 1)) if out is None else out


@functools.partial(jax.jit, static_argnames=("kinds", "axes", "s", "chunk", "fit_only"))
def _average(terms, cterms, eps_background, runs, coef, bound, *, kinds, axes, s, chunk, fit_only):
    """The voxel averages of one slab of voxels (shape ``chunk``): the same steps as Cell._sample and
    Cell.voxel_averages. ``runs``: per run of consecutive objects of one kind (``kinds``), the arrays (eps, params,
    image shifts, valid) of its objects, the image lists padded to the longest (valid False: no effect)."""
    S = tuple(c * k for c, k in zip(chunk, s))
    vox = (chunk[0], s[0], chunk[1], s[1], chunk[2], s[2])
    X, Y, Z = (_coordinate(terms[c], axes[c]) for c in range(3))
    XC, YC, ZC = (_coordinate(cterms[c], axes[c]) for c in range(3))
    eps = jnp.full(S, eps_background)
    count = jnp.zeros(chunk, jnp.int32)
    analytic = jnp.zeros((3, *chunk))
    for kind, run in zip(kinds, runs):

        def one_object(carry, obj, kind=kind):
            eps, count, analytic = carry
            obj_eps, p, shifts, valid = obj

            def image(carry, tv):
                hit, count, analytic = carry
                t, ok = tv
                h = jnp.broadcast_to(_inside(kind, p, X + t[0], Y + t[1], Z + t[2]), S) & ok
                hv = h.reshape(vox)
                cross = hv.any(axis=(1, 3, 5)) & ~hv.all(axis=(1, 3, 5))
                nrm = jnp.stack([jnp.broadcast_to(v, chunk)
                                 for v in _normal(kind, p, XC + t[0], YC + t[1], ZC + t[2])])
                return (hit | h, count + cross, jnp.where(cross[None], nrm, analytic)), None

            (hit, count, analytic), _ = jax.lax.scan(image, (jnp.zeros(S, bool), count, analytic), (shifts, valid))
            return (jnp.where(hit, obj_eps, eps), count, analytic), None

        (eps, count, analytic), _ = jax.lax.scan(one_object, (eps, count, analytic), run)
    e = eps.reshape(vox)
    mean_eps = e.mean(axis=(1, 3, 5))
    mean_inv = (1.0 / e).mean(axis=(1, 3, 5))
    offs = [(np.arange(k) + 0.5) / k - 0.5 for k in s]
    mom = [(e * offs[i].reshape(tuple(-1 if j == 2 * i + 1 else 1 for j in range(6)))).mean(axis=(1, 3, 5))
           for i in range(3)]
    mom = [jnp.where(jnp.abs(m) <= bound, 0.0, m) for m in mom]           # rounding residues (Cell.moment_bound)
    grad = jnp.stack([mom[0] * coef[0, c] + mom[1] * coef[1, c] + mom[2] * coef[2, c] for c in range(3)])
    if not fit_only:
        one = (count == 1) & ((analytic**2).sum(axis=0) > 0)
        grad = jnp.where(one[None], analytic, grad)
    norm = jnp.sqrt((grad**2).sum(axis=0))
    interface = (mean_inv * mean_eps > 1 + 1e-12) & (norm > 0)
    nvec = jnp.where(interface[None], grad / jnp.where(norm > 0, norm, 1.0)[None], 0.0)
    eye = jnp.eye(3)[:, :, None, None, None]
    nn = nvec[:, None] * nvec[None, :]
    tensor = mean_inv[None, None] * nn + (1.0 / mean_eps)[None, None] * (eye - nn)
    iso = (1.0 / mean_eps)[None, None] * eye
    return {"eps_inv": jnp.where(interface[None, None], tensor, iso), "mean_eps": mean_eps, "count": count,
            "analytic": analytic, "normal": nvec}


def voxel_averages(cell: Cell, sub: int = 8, normal: str = "object", device=None,
                   max_samples: int = 1 << 24) -> dict[str, np.ndarray]:
    """The averages of each voxel on the JAX device ``device`` (default: the default device of JAX): "eps_inv"
    (3, 3, nx, ny, nz), as Cell.eps_inverse; "mean_eps" (the fill fractions: <eps>); "count" (the object boundaries
    that cross the voxel) and "analytic" (3, ...) (the analytic normal of the last of them), as Cell._sample;
    "normal" (3, ...) (the unit normal of the averaging; 0 outside interfaces)."""
    if normal not in ("object", "fit"):
        raise ValueError("normal: 'object' or 'fit'")
    if not supported(cell):
        raise ValueError("sampling='jax' needs objects of the types Cylinder, Sphere, Block, the sampling of Cell "
                         "(no override) and jax_enable_x64")
    s = tuple(sub if n > 1 else 1 for n in cell.shape)
    e = cell.unit_basis
    axes = tuple(tuple(i for i in range(3) if e[i, c] != 0.0) for c in range(3))
    t, tc = cell.axes(sub), cell.axes(1)
    terms = [[t[i] * e[i, c] for i in axes[c]] for c in range(3)]         # the products of Cell.points, on the host
    cterms = [[tc[i] * e[i, c] for i in axes[c]] for c in range(3)]
    w = [n / a if n > 1 else 0.0 for a, n in zip(cell.size, cell.shape)]
    f = cell.unit_reciprocal
    coef = np.array([[w[i] * f[i, c] for c in range(3)] for i in range(3)])
    put = (lambda a: jax.device_put(a, device)) if device is not None else jnp.asarray
    kinds, runs = [], []
    for kind, group in itertools.groupby(cell.objects, key=lambda o: _KINDS[type(o)]):
        group = list(group)
        shifts = [np.asarray(cell.image_shifts(o), float) for o in group]
        pad = np.zeros((len(group), max(len(x) for x in shifts), 3))
        valid = np.zeros(pad.shape[:2], bool)
        for j, x in enumerate(shifts):
            pad[j, :len(x)], valid[j, :len(x)] = x, True
        kinds.append(kind)
        runs.append(tuple(put(a) for a in (np.array([float(o.eps) for o in group]),
                                           np.stack([_params(o) for o in group]), pad, valid)))
    # slabs of voxels along the first used axis (the axis 0 when no axis is used)
    ax = next((i for i, n in enumerate(cell.shape) if n > 1), 0)
    plane = int(np.prod([n * k for i, (n, k) in enumerate(zip(cell.shape, s)) if i != ax])) * s[ax]
    step = max(1, min(cell.shape[ax], max_samples // plane))     # at least one plane (a plane is not split)
    outs = []
    for j0 in range(0, cell.shape[ax], step):
        j1 = min(j0 + step, cell.shape[ax])
        chunk = tuple(j1 - j0 if i == ax else n for i, n in enumerate(cell.shape))
        cut = lambda v, i, k: v[j0 * k:j1 * k] if i == ax else v  # noqa: E731
        tt = tuple(tuple(put(cut(v, i, s[i])) for v, i in zip(terms[c], axes[c])) for c in range(3))
        ct = tuple(tuple(put(cut(v, i, 1)) for v, i in zip(cterms[c], axes[c])) for c in range(3))
        outs.append(_average(tt, ct, put(np.float64(cell.eps_background)), tuple(runs), put(coef),
                             put(np.float64(cell.moment_bound(sub))), kinds=tuple(kinds), axes=axes, s=s, chunk=chunk,
                             fit_only=normal == "fit"))
    vector = {"eps_inv": 2, "analytic": 1, "normal": 1}
    return {key: np.concatenate([np.asarray(o[key]) for o in outs], axis=vector.get(key, 0) + ax)
            for key in outs[0]}


def use_jax(cell: Cell, sub: int, sampling: str) -> bool:
    """The decision of ``sampling``: "numpy" (the reference, Cell.eps_inverse), "jax" (an error if the cell is not
    supported), "auto" (JAX for a supported cell with a work of at least AUTO_MIN_WORK times the runs of objects)."""
    if sampling not in ("auto", "jax", "numpy"):
        raise ValueError("sampling: 'auto', 'jax' or 'numpy'")
    if sampling == "numpy":
        return False
    if sampling == "jax":
        return True
    return supported(cell) and work(cell, sub) >= AUTO_MIN_WORK * max(1, runs(cell))


def runs(cell: Cell) -> int:
    """The number of runs of consecutive objects of one kind (one compile of the JAX path each)."""
    return sum(1 for _ in itertools.groupby(type(o) for o in cell.objects))


def work(cell: Cell, sub: int) -> float:
    """The number of inside tests: samples times the periodic images of all objects."""
    samples = float(np.prod([n * (sub if n > 1 else 1) for n in cell.shape]))
    return samples * sum(len(cell.image_shifts(o)) for o in cell.objects)


def eps_inverse(cell: Cell, sub: int = 8, normal: str = "object", sampling: str = "auto", device=None) -> np.ndarray:
    """The averaged inverse permittivity tensor (3, 3, nx, ny, nz), on the host: Cell.eps_inverse (NumPy) or the
    JAX path on ``device`` (``use_jax`` decides)."""
    if use_jax(cell, sub, sampling):
        return voxel_averages(cell, sub, normal, device)["eps_inv"]
    return cell.eps_inverse(sub) if normal == "object" else cell.eps_inverse(sub, normal=normal)
