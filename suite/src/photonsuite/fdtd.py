"""3-D FDTD of a device made of cross-section segments along z, with fdtdx (MIT; JAX on ROCm or CUDA).

A device is a list of Segment(z0, z1, CrossSection) (um). The adapter samples each shape (Rect or Polygon) at the
cell centers of the grid, with the edge rule of CrossSection, and gives fdtdx objects whose voxels are exactly these
cells: a box for a full rectangle of cells, else a staircase ExtrudedPolygon along z with its vertices on grid edges
(fdtdx rounds the position and the size of an object separately; on grid edges both are exact). Segment ends go to
the nearest grid edge, so neighbor segments share it (no gap cells). There is no sub-pixel smoothing (fdtdx 0.6.3
smooths only GDS layers extruded along its axis 2). fdtdx paints the objects in list order, so a later shape
replaces an earlier one, as in CrossSection. The first and the last segment extend through the PML, so that the
guide continues into the absorber. A mode source (fundamental TE-like mode) launches at the input; mode-overlap and
Poynting-flux detectors after the source and before the output give the transmission.

Axes: fdtdx axis 0 = z (propagation), axis 1 = x, axis 2 = y (the suite's x horizontal, y vertical).

fdtdx runs in float32. ``import photonsuite`` enables float64 for the other solvers, so the FDTD runs inside
``jax.enable_x64(False)`` (with float64 on, fdtdx 0.6.3 mixes float64 into its float32 time loop and fails), and
inside ``jax.default_device`` of the first GPU. Without it, an FDTD run after an EME run in the same process was
9 times slower, with high CPU use (sloped taper, 40 nm: 149 s against 16 s); a probe did not find the JAX state
that caused it (the default device stayed the GPU). ``FDTDResult.device`` is the device of the field arrays.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from photonsuite.geometry import CrossSection

C0 = 299792458.0


@dataclass(frozen=True)
class Segment:
    z0: float
    z1: float
    xs: CrossSection


def check_segments(segments, first_min: float = 0.0, last_min: float = 0.0):
    """The adapters need contiguous segments (z1 of one = z0 of the next, each longer than 0), one background
    material for all segments (they fill the whole volume with it), and first and last segments of at least
    first_min and last_min um (the ports are there). Raises ValueError."""
    if not segments:
        raise ValueError("no segments")
    for s in segments:
        if not s.z1 > s.z0:
            raise ValueError(f"segment z0={s.z0}, z1={s.z1}: z1 must be larger than z0")
    for a, b in zip(segments, segments[1:]):
        if abs(a.z1 - b.z0) > 1e-9:
            raise ValueError(f"segments must be contiguous: z1={a.z1} is not the next z0={b.z0}")
    if len({s.xs.background for s in segments}) > 1:
        raise ValueError("all segments must have the same background material")
    if segments[0].z1 - segments[0].z0 < first_min - 1e-9:
        raise ValueError(f"the first segment must be at least {first_min} um long (the input port)")
    if segments[-1].z1 - segments[-1].z0 < last_min - 1e-9:
        raise ValueError(f"the last segment must be at least {last_min} um long (the output port)")


@dataclass
class FDTDResult:
    transmission: float      # |a_out|^2 / |a_in|^2 (mode overlaps; biased when the input and output modes differ)
    a_in: complex
    a_out: complex
    cells: int
    seconds: float
    device: str
    p_in: float = float("nan")    # net Poynting flux through the input and output planes (W, arbitrary scale)
    p_out: float = float("nan")

    @property
    def flux_transmission(self) -> float:
        """Total power transmission P_out / P_in (all modes and radiation that crosses the output plane)."""
        return self.p_out / self.p_in

    @property
    def out_mode_fraction(self) -> float:
        """|a_out|^2 / P_out: the fundamental-mode share of the output power, times the overlap bias of the output
        mode. Divide by the same ratio of a pure output mode (a straight-guide calibration) to remove the bias."""
        return abs(self.a_out) ** 2 / self.p_out


def transmission(segments: list[Segment], x_range: tuple[float, float], y_range: tuple[float, float],
                 wavelength: float, step: float = 0.04, pml_cells: int = 10, time_fs: float | None = None,
                 filter_pol: str = "te", seed: int = 0) -> FDTDResult:
    """Fundamental-mode transmission (um units in, um wavelength). The window x_range x y_range must hold the mode."""
    _check(segments, step)                                   # before the import of fdtdx
    import jax

    with jax.enable_x64(False), jax.default_device(_device()):
        return _transmission(segments, x_range, y_range, wavelength, step, pml_cells, time_fs, filter_pol, seed)


def _cell_mask(r, origin, n_xy, step):
    """The cells (x, y) of one shape: True where the cell center is inside the shape, by the edge rule of the shape
    (CrossSection samples the same way). Returns (mask, ix0, iy0) on the cell range of the bounding box, clipped to
    the volume; mask is None when no cell is inside."""
    if hasattr(r, "x0"):
        bx0, by0, bx1, by1 = r.x0, r.y0, r.x1, r.y1
    else:
        xs, ys = zip(*r.vertices)
        bx0, by0, bx1, by1 = min(xs), min(ys), max(xs), max(ys)
    lo = [max(int(np.floor((b - o) / step)) - 1, 0) for b, o in ((bx0, origin[0]), (by0, origin[1]))]
    hi = [min(int(np.ceil((b - o) / step)) + 1, n)
          for b, o, n in ((bx1, origin[0], n_xy[0]), (by1, origin[1], n_xy[1]))]
    if hi[0] <= lo[0] or hi[1] <= lo[1]:
        return None, 0, 0
    xc = origin[0] + (np.arange(lo[0], hi[0]) + 0.5) * step
    yc = origin[1] + (np.arange(lo[1], hi[1]) + 0.5) * step
    X, Y = np.meshgrid(xc, yc, indexing="ij")
    mask = np.asarray(r.contains(X, Y), bool)
    if not mask.any():
        return None, 0, 0
    return mask, lo[0], lo[1]


def _shape_objects(fdtdx, r, name, mat, nz_cells, origin, n_xy, step, res):
    """(object, (ix0, iy0)) pairs for one shape: fdtdx objects whose voxels are exactly the cells of _cell_mask, and
    the cell of the lower corner of each object. A full rectangle of cells becomes one box. Other shapes become
    staircase polygons with their vertices on grid edges (fdtdx places an ExtrudedPolygon by the lower corner of its
    bounding box and takes the size from the box; on grid edges both are exact, and the fdtdx mask at the cell
    centers is the staircase), or boxes per row run where a part has holes."""
    import shapely

    mask, ox, oy = _cell_mask(r, origin, n_xy, step)
    if mask is None:
        return []
    ii, jj = np.nonzero(mask)
    i0, i1, j0, j1 = ii.min(), ii.max() + 1, jj.min(), jj.max() + 1
    if mask[i0:i1, j0:j1].all():
        box = fdtdx.UniformMaterialObject(name=name, material=mat,
                                          partial_real_shape=(nz_cells * res, (i1 - i0) * res, (j1 - j0) * res))
        return [(box, (ox + i0, oy + j0))]
    runs = []                                  # (i0, i1, j): runs of cells along x in each row j
    for j in range(j0, j1):
        col = np.concatenate([[False], mask[:, j], [False]])
        d = np.diff(col.astype(int))
        runs += [(a, b, j) for a, b in zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0])]
    union = shapely.union_all([shapely.box(a, j, b, j + 1) for a, b, j in runs])
    out = []
    for k, part in enumerate(getattr(union, "geoms", [union])):
        px0, py0, px1, py1 = (int(round(v)) for v in part.bounds)
        if part.interiors:                     # a hole: one box per row run of this part
            for m, (a, b, j) in enumerate(t for t in runs if part.contains(shapely.Point(t[0] + 0.5, t[2] + 0.5))):
                box = fdtdx.UniformMaterialObject(name=f"{name}_{k}_{m}", material=mat,
                                                  partial_real_shape=(nz_cells * res, (b - a) * res, res))
                out.append((box, (ox + a, oy + j)))
            continue
        verts = (np.asarray(part.exterior.coords)[:-1] - [(px0 + px1) / 2, (py0 + py1) / 2]) * res
        obj = fdtdx.ExtrudedPolygon(name=f"{name}_{k}", axis=0, vertices=verts, material_name=r.material.name,
                                    materials={r.material.name: mat}, partial_real_shape=(nz_cells * res, None, None))
        out.append((obj, (ox + px0, oy + py0)))
    return out


def _device():
    """The first GPU (ROCm or CUDA), else the CPU."""
    import jax
    try:
        return jax.devices("gpu")[0]
    except RuntimeError:
        return jax.devices("cpu")[0]


def _check(segments, step):
    check_segments(segments, first_min=1.0, last_min=0.5)    # source 0.5 um, input detector 1.0 um into the device
    short = [s for s in segments if s.z1 - s.z0 < step - 1e-9]
    if short:
        raise ValueError(f"{len(short)} segments are shorter than the grid step {step} um (fdtdx cannot place an "
                         f"object of no cells): use fewer segments or a smaller step")


def _transmission(segments, x_range, y_range, wavelength, step, pml_cells, time_fs, filter_pol, seed) -> FDTDResult:
    import time

    import fdtdx

    objs, arrays, config, key, backend, cells = _setup(segments, x_range, y_range, wavelength, step, pml_cells,
                                                       time_fs, filter_pol, seed)
    t0 = time.perf_counter()
    _, arrays = fdtdx.run_fdtd(arrays, objs, config, key)
    a_in = complex(objs["in"].compute_overlap(arrays.detector_states["in"])[0])
    a_out = complex(objs["out"].compute_overlap(arrays.detector_states["out"])[0])
    p_in = float(objs["flux_in"].compute_poynting_flux(arrays.detector_states["flux_in"])[0])
    p_out = float(objs["flux_out"].compute_poynting_flux(arrays.detector_states["flux_out"])[0])
    seconds = time.perf_counter() - t0
    device = str(next(iter(arrays.inv_permittivities.devices())))
    return FDTDResult(abs(a_out) ** 2 / abs(a_in) ** 2, a_in, a_out, cells, seconds, device, p_in, p_out)


def _setup(segments, x_range, y_range, wavelength, step, pml_cells=10, time_fs=None, filter_pol="te", seed=0):
    """The placed fdtdx simulation (objects, arrays, config, key, backend, cells), not run. Call it inside
    jax.enable_x64(False)."""
    _check(segments, step)
    import fdtdx
    import jax

    um = 1e-6
    res = step * um
    pml = pml_cells * step                                     # um
    z_len = segments[-1].z1 - segments[0].z0
    wx, wy = x_range[1] - x_range[0], y_range[1] - y_range[0]
    # a whole number of cells on each axis (fdtdx rounds the volume and the grid separately: 65.5 cells failed)
    shape = tuple(round(length / step) * res for length in (z_len + 2 * pml, wx + 2 * pml, wy + 2 * pml))
    n_max = max(max([s.material.n for s in seg.xs.shapes] + [seg.xs.background.n]) for seg in segments)
    if time_fs is None:
        time_fs = 3.0 * z_len * n_max * um / C0 * 1e15 + 300.0   # 3 x the transit at the largest index, plus the pulse
    backend = "cpu" if _device().platform == "cpu" else "gpu"
    config = fdtdx.SimulationConfig(grid=fdtdx.UniformGrid(spacing=res), time=time_fs * 1e-15, courant_factor=0.99,
                                    backend=backend)
    background_mat = segments[0].xs.background
    volume = fdtdx.SimulationVolume(partial_real_shape=shape,
                                    material=fdtdx.Material(permittivity=background_mat.n**2))
    objects, constraints = [volume], []
    mats: dict[str, object] = {}
    n_cells = [round(sh / res) for sh in shape]
    origin = (segments[0].z0 - pml, x_range[0] - pml, y_range[0] - pml)

    def cell(v, axis):
        """The grid edge nearest to the coordinate v (um) on an axis (0 z, 1 x, 2 y), in [0, cells of the axis]."""
        return min(max(int(np.floor((v - origin[axis]) / step + 0.5)), 0), n_cells[axis])

    for k, seg in enumerate(segments):
        iz0 = 0 if k == 0 else cell(seg.z0, 0)                       # the first and last segments fill the PML
        iz1 = n_cells[0] if k == len(segments) - 1 else cell(seg.z1, 0)
        if iz1 <= iz0:
            raise ValueError(f"segment {k} (z {seg.z0} to {seg.z1} um) has no cells on the {step} um grid")
        for j, r in enumerate(seg.xs.shapes):
            mat = mats.setdefault(r.material.name, fdtdx.Material(permittivity=r.material.n**2))
            for obj, (ix0, iy0) in _shape_objects(fdtdx, r, f"seg{k}_shape{j}", mat, iz1 - iz0, origin[1:],
                                                  n_cells[1:], step, res):
                objects.append(obj)
                constraints.append(obj.place_relative_to(
                    volume, axes=(0, 1, 2), other_positions=(-1, -1, -1), own_positions=(-1, -1, -1),
                    margins=(iz0 * res, ix0 * res, iy0 * res)))
    freq = C0 / (wavelength * um)
    plane = (res, wx * um, wy * um)
    source = fdtdx.ModePlaneSource(
        name="source", mode_index=0, filter_pol=filter_pol, partial_real_shape=plane, direction="+",
        wave_character=fdtdx.WaveCharacter(wavelength=wavelength * um),
        temporal_profile=fdtdx.GaussianPulseProfile(center_wave=fdtdx.WaveCharacter(frequency=freq),
                                                     spectral_width=fdtdx.WaveCharacter(frequency=freq / 10)))
    wc = (fdtdx.WaveCharacter(wavelength=wavelength * um),)
    det_in = fdtdx.ModeOverlapDetector(name="in", mode_index=0, filter_pol=filter_pol, partial_real_shape=plane,
                                       direction="+", wave_characters=wc)
    det_out = fdtdx.ModeOverlapDetector(name="out", mode_index=0, filter_pol=filter_pol, partial_real_shape=plane,
                                        direction="+", wave_characters=wc)
    flux_in = fdtdx.PhasorPoyntingFluxDetector(name="flux_in", partial_real_shape=plane, direction="+",
                                               wave_characters=wc)
    flux_out = fdtdx.PhasorPoyntingFluxDetector(name="flux_out", partial_real_shape=plane, direction="+",
                                                wave_characters=wc)
    objects += [source, det_in, det_out, flux_in, flux_out]
    for obj, z in ((source, pml + 0.5), (det_in, pml + 1.0), (det_out, pml + z_len - 0.5), (flux_in, pml + 1.0),
                   (flux_out, pml + z_len - 0.5)):
        constraints.append(obj.place_relative_to(volume, axes=(0, 1, 2), other_positions=(-1, -1, -1),
                                                 own_positions=(-1, -1, -1), margins=(z * um, pml * um, pml * um)))
    bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(thickness=pml_cells, boundary_type="pml")
    boundary_dict, boundary_constraints = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    objects += list(boundary_dict.values())
    constraints += boundary_constraints
    key = jax.random.PRNGKey(seed)
    objs, arrays, params, config, _ = fdtdx.place_objects(object_list=objects, config=config,
                                                         constraints=constraints, key=key)
    arrays, objs, _ = fdtdx.apply_params(arrays, objs, params, key)
    cells = int(np.prod([round(s / res) for s in shape]))
    return objs, arrays, config, key, backend, cells
