"""Subpixel averaging of the permittivity at the location of each Yee E component (`eps_averaging=True`).

References: A. Farjadpour et al., Opt. Lett. 31, 2972 (2006); A. Kottke, A. Farjadpour and S. G. Johnson,
Phys. Rev. E 77, 036611 (2008).

Locations. fdtdx keeps the Yee layout of Taflove: E_x at (i + 1/2, j, k), E_y at (i, j + 1/2, k), E_z at
(i, j, k + 1/2). This package puts the point (i, j, k) at the center of cell (i, j, k), so the component E_a of a
cell is at the cell center plus dx / 2 along the axis a. On an axis of size 0 (2-D and 1-D cells) the coordinate
is 0 and nothing is sampled. In 2-D, E_z (normal to the plane) is at the cell center; in 1-D along z, E_x and E_y
are at the cell centers: there the staircase also samples the material at the component.

Method. The voxel of a component is the cube of side dx centered at the component. A voxel keeps its staircase
material if the staircase materials of the cell centers around it (2 cells along a, 3 along each other axis of the
cell) are all equal. A feature thinner than one cell between cell centers is not seen (as by the staircase). In
each other voxel the material is sampled (the last object that contains the point wins, as in `rasterize`):
- <eps> and <1/eps> on a rank-1 lattice of n points (`lattice`), made symmetric about the voxel center. Each 1-D
  projection of the lattice is the midpoint rule of n points, so the fill fraction of a plane normal to an axis is
  exact to 1 / (2 n) (n = LATTICE_N: 1024, 987, 1021 points in 1-D, 2-D, 3-D);
- the normal n: the first moment of eps with the weight max(0, 1 - r^2 / R^2) over a regular grid (NORMAL_N points
  per axis) in the circumscribed ball of the voxel (radius R = sqrt(d) dx / 2). By the mirror symmetry of the grid,
  the moment of a plane normal to an axis is along that axis; for a slanted plane it is close to the normal.
The component a takes the diagonal of the inverse of the anisotropic average of Kottke et al. (the tensor
eps_eff^-1 = P <1/eps> + (1 - P) / <eps>, P = n n^T):
    1 / eps_aa = (eps_eff^-1)_aa = n_a^2 <1/eps> + (1 - n_a^2) / <eps>.
The off-diagonal terms are dropped, so fdtdx keeps its elementwise update. For a plane normal to an axis this is
the full tensor; for a slanted or curved interface the off-diagonal terms are lost (the MIT package keeps them), and
the error with E in the plane of a curved interface is first order. DIAGONAL = "forward" keeps the diagonal of the
forward tensor in its place, eps_aa = n_a^2 / <1/eps> + (1 - n_a^2) <eps> (as the smoothing of fdtdx itself); the
two are equal when n is along an axis or normal to a.
Periodic axes: the staircase index wraps, the sample coordinates go back into the cell, and the objects repeat
across the faces (`geometry.periodic_shifts`), as in the MIT package.
Small problems are sampled with NumPy; larger ones with one jit-compiled JAX function on the default device (the
GPU), in chunks of about CHUNK_POINTS sample points.
"""

from __future__ import annotations

import functools
import itertools
import math

import numpy as np

LATTICE_N = {1: 1024, 2: 987, 3: 1021}      # lattice points before the symmetric copy (2-D, 3-D: x 2)
FIBONACCI = {987: 610}                       # 2-D: n = F_16, generator F_15
NORMAL_N = {1: 16, 2: 24, 3: 12}             # points per axis of the regular grid for the normal
CHUNK_POINTS = 1 << 22
NUMPY_MAX = 10_000_000                       # sampled points x object copies up to which NumPy is used
DIAGONAL = "inverse"                         # "inverse" (Kottke et al.) or "forward" (see the module text)


@functools.lru_cache(maxsize=None)
def _korobov_3d(n: int) -> int:
    """The Korobov generator a (vector (1, a, a^2 mod n)) with the lowest P_2 figure of merit (Sloan and Joe,
    Lattice Methods for Multiple Integration, 1994): a deterministic choice of a good 3-D rank-1 lattice."""
    k = np.arange(n)
    best, best_p = 2, np.inf
    for a in range(2, n // 2 + 1):
        z = np.array([1, a % n, a * a % n])
        if math.gcd(int(z[1]), n) != 1 or math.gcd(int(z[2]), n) != 1:
            continue
        x = (k[:, None] * z[None, :] % n) / n
        b2 = x * x - x + 1.0 / 6.0
        p = float(np.prod(1 + 2 * np.pi**2 * b2, axis=1).mean() - 1)
        if p < best_p:
            best, best_p = a, p
    return best


@functools.lru_cache(maxsize=None)
def lattice(d: int) -> np.ndarray:
    """Sample offsets (S, d) in units of the cell size, in [-1/2, 1/2]^d, centrally symmetric (S = 2n for d > 1).
    Each coordinate takes the midpoint values (m + 1/2) / n - 1/2 (twice for d > 1)."""
    n = LATTICE_N[d]
    if d == 1:
        return ((np.arange(n) + 0.5) / n - 0.5)[:, None]
    if d == 2:
        z = np.array([1, FIBONACCI[n]])
    else:
        a = _korobov_3d(n)
        z = np.array([1, a, a * a % n])
    k = np.arange(n)
    u = ((k[:, None] * z[None, :]) % n + 0.5) / n - 0.5
    return np.concatenate([u, -u])


@functools.lru_cache(maxsize=None)
def normal_grid(d: int) -> tuple[np.ndarray, np.ndarray]:
    """Offsets (N^d, d) of the regular grid of the normal, in units of the cell size: N points per axis over the
    cube that holds the circumscribed ball of the voxel (radius R = sqrt(d) / 2, so every interface that cuts the
    voxel cuts the ball); and the weights max(0, 1 - r^2 / R^2) (zero at the edge of the ball)."""
    rad = math.sqrt(d) / 2
    g = ((np.arange(NORMAL_N[d]) + 0.5) / NORMAL_N[d] - 0.5) * 2 * rad
    pts = np.array(list(itertools.product(g, repeat=d)))
    return pts, np.maximum(0.0, 1.0 - (pts**2).sum(axis=1) / rad**2)


def interface_voxels(idx: np.ndarray, a: int, trivial, periodic=(False, False, False)) -> np.ndarray:
    """True where the voxel of E_a may hold more than one material: the staircase index differs among the cells
    i and i + 1 along a and i - 1, i, i + 1 along each other axis of size > 0 (at a face the edge cell repeats; at
    a periodic face the cell of the other side)."""
    p = idx
    for b in range(3):
        if trivial[b]:
            continue
        width = [(0, 0)] * 3
        width[b] = (0, 1) if b == a else (1, 1)
        p = np.pad(p, width, mode="wrap" if periodic[b] else "edge")
    ranges = [[0] if trivial[b] else ([0, 1] if b == a else [0, 1, 2]) for b in range(3)]
    n = idx.shape
    mx = mn = None
    for o in itertools.product(*ranges):
        s = p[o[0]:o[0] + n[0], o[1]:o[1] + n[1], o[2]:o[2] + n[2]]
        mx = s if mx is None else np.maximum(mx, s)
        mn = s if mn is None else np.minimum(mn, s)
    return mx != mn


def _inv_eps_chunk(pos, onehot, *, xp, geometry, obj_mat, shifts, eps_tab, trivial, periods, offsets, n_lat, nrm,
                   ball, forward):
    """1 / eps_aa of the voxels centered at pos (C, 3). onehot: (d,) selects the component axis among the axes of
    size > 0 (all zero for a component along an axis of size 0). offsets: (S, d) sample offsets in um (the lattice,
    then the normal grid); nrm: (N, d) the normal grid in cell units; ball: (N,) its weights (`normal_grid`).
    periods: the cell length (um) of each periodic axis, else None: there the sample coordinates go back into the
    cell, and shifts (one list per object, `geometry.periodic_shifts`) give the copies of the objects.
    forward: keep the diagonal of the forward tensor (see DIAGONAL) in place of the inverse tensor."""
    nt = [b for b in range(3) if not trivial[b]]
    coords = []
    for b in range(3):
        if trivial[b]:
            coords.append(0.0)
            continue
        c = pos[:, b:b + 1] + offsets[None, :, nt.index(b)]
        if periods[b] is not None:
            c = xp.mod(c + periods[b] / 2, periods[b]) - periods[b] / 2
        coords.append(c)
    mat = xp.zeros(coords[nt[0]].shape, xp.int32)
    for obj, k, obj_shifts in zip(geometry, obj_mat, shifts):
        inside = None
        for s in obj_shifts:
            q = obj.contains(*[c if trivial[b] or s[b] == 0 else c - s[b] for b, c in enumerate(coords)])
            inside = q if inside is None else inside | q
        if inside is not None:
            mat = xp.where(inside, k, mat)
    eps = eps_tab[mat]
    el, en = eps[:, :n_lat], eps[:, n_lat:]
    m_eps = el.mean(axis=1)
    m_inv = (1.0 / el).mean(axis=1)
    dev = en - en.mean(axis=1, keepdims=True)
    # sums, not matrix products: a GEMM makes the ROCm compiler autotune (seconds) for a few flops
    mom = xp.stack([(dev * (ball * nrm[:, k])[None, :]).sum(axis=1) for k in range(nrm.shape[1])], axis=1)
    nm2 = (mom**2).sum(axis=1)
    na2 = xp.where(nm2 > 0, (mom * onehot[None, :]).sum(axis=1) ** 2 / xp.where(nm2 > 0, nm2, 1.0), 0.0)
    if forward:
        return 1.0 / (na2 / m_inv + (1.0 - na2) * m_eps)
    return na2 * m_inv + (1.0 - na2) / m_eps


def averaged_permittivity(geometry, materials, idx: np.ndarray, dx: float, trivial, periodic=(False, False, False),
                          backend: str = "auto") -> tuple[np.ndarray, int]:
    """The permittivity of each E component, (3, *idx.shape) float32, and the number of sampled voxels.

    geometry, materials, idx: the objects, the material list and the staircase index of `geometry.rasterize` (with
    the same periodic axes). periodic: True for an axis with periodic faces (the objects repeat, as in the MIT
    package). backend: "numpy", "jax" or "auto" (NumPy up to NUMPY_MAX sampled points x objects)."""
    from fdtd_gpu.geometry import periodic_shifts

    shape = idx.shape
    per = tuple(bool(p) and not t for p, t in zip(periodic, trivial))
    mats, obj_mat = list(materials), []
    for o in geometry:                     # an object that holds no cell center is not in the list yet
        j = next((j for j, m in enumerate(mats) if m is o.material), None)
        if j is None:
            mats.append(o.material)
            j = len(mats) - 1
        obj_mat.append(j)
    eps_tab = np.array([m.epsilon for m in mats], np.float64)
    out = np.broadcast_to(eps_tab[idx].astype(np.float32), (3, *shape)).copy()
    nt = [b for b in range(3) if not trivial[b]]
    if not nt:
        return out, 0
    d = len(nt)
    cells = [np.argwhere(interface_voxels(idx, a, trivial, per)) for a in range(3)]
    n_sampled = sum(len(c) for c in cells)
    if n_sampled == 0:
        return out, 0
    lat, (nrm, ball) = lattice(d), normal_grid(d)
    offsets = np.concatenate([lat, nrm]) * dx
    lo = np.array([-n * dx / 2 for n in shape])
    shifts = [periodic_shifts(o, per, shape, dx) for o in geometry]
    use_jax = backend == "jax" or (backend == "auto" and
                                   n_sampled * len(offsets) * sum(len(s) for s in shifts) > NUMPY_MAX)
    if DIAGONAL not in ("inverse", "forward"):
        raise ValueError(f"DIAGONAL must be 'inverse' or 'forward', not {DIAGONAL!r}")
    kw = dict(geometry=geometry, obj_mat=obj_mat, shifts=shifts, trivial=tuple(trivial), n_lat=len(lat),
              periods=tuple(shape[b] * dx if per[b] else None for b in range(3)), forward=DIAGONAL == "forward")
    if use_jax:
        import jax
        import jax.numpy as jnp

        f32 = jnp.float32
        fn = jax.jit(functools.partial(_inv_eps_chunk, xp=jnp, eps_tab=jnp.asarray(eps_tab, f32),
                                       offsets=jnp.asarray(offsets, f32), nrm=jnp.asarray(nrm, f32),
                                       ball=jnp.asarray(ball, f32), **kw))
        chunk = max(1, CHUNK_POINTS // len(offsets))
        chunk = min(chunk, 1 << int(math.ceil(math.log2(max(len(c) for c in cells)))))
    else:
        fn = functools.partial(_inv_eps_chunk, xp=np, eps_tab=eps_tab, offsets=offsets, nrm=nrm, ball=ball, **kw)
        chunk = max(1, CHUNK_POINTS // len(offsets))
    for a in range(3):
        if len(cells[a]) == 0:
            continue
        pos = lo + (cells[a] + 0.5) * dx
        if not trivial[a]:
            pos[:, a] += 0.5 * dx
        pos[:, list(trivial)] = 0.0
        onehot = np.array([1.0 if b == a else 0.0 for b in nt])
        res = []
        for c0 in range(0, len(pos), chunk):
            p = pos[c0:c0 + chunk]
            if use_jax:
                m = len(p)
                if m < chunk:
                    p = np.concatenate([p, np.repeat(p[:1], chunk - m, axis=0)])
                res.append(np.asarray(fn(jnp.asarray(p, f32), jnp.asarray(onehot, f32)))[:m])
            else:
                res.append(fn(p, onehot))
        out[a][tuple(cells[a].T)] = 1.0 / np.concatenate(res)
    return out, n_sampled
