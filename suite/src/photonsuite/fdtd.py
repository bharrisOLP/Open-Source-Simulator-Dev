"""3-D FDTD of a device made of cross-section segments along z, with fdtdx (MIT; JAX on ROCm or CUDA).

A device is a list of Segment(z0, z1, CrossSection) (um). Each rectangle of a segment becomes an fdtdx box from z0 to
z1. The first and the last segment extend through the PML, so that the guide continues into the absorber. A mode
source (fundamental TE-like mode) launches at the input; two mode-overlap detectors (one after the source, one
before the output) give the transmission |a_out|^2 / |a_in|^2 into the fundamental mode, normalized by the
spectrum of the pulse.

Axes: fdtdx axis 0 = z (propagation), axis 1 = x, axis 2 = y (the suite's x horizontal, y vertical).

fdtdx runs in float32. ``import photonsuite`` enables float64 for the other solvers, so the FDTD runs inside
``jax.enable_x64(False)`` (with float64 on, fdtdx 0.6.3 mixes float64 into its float32 time loop and fails).
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
    import jax

    with jax.enable_x64(False):
        return _transmission(segments, x_range, y_range, wavelength, step, pml_cells, time_fs, filter_pol, seed)


def _transmission(segments, x_range, y_range, wavelength, step, pml_cells, time_fs, filter_pol, seed) -> FDTDResult:
    import time

    import fdtdx
    import jax

    check_segments(segments, first_min=1.0, last_min=0.5)    # source 0.5 um, input detector 1.0 um into the device
    um = 1e-6
    res = step * um
    pml = pml_cells * step                                     # um
    z_len = segments[-1].z1 - segments[0].z0
    wx, wy = x_range[1] - x_range[0], y_range[1] - y_range[0]
    shape = ((z_len + 2 * pml) * um, (wx + 2 * pml) * um, (wy + 2 * pml) * um)
    n_max = max(max([s.material.n for s in seg.xs.shapes] + [seg.xs.background.n]) for seg in segments)
    if time_fs is None:
        time_fs = 3.0 * z_len * n_max * um / C0 * 1e15 + 300.0  # 3 x the transit time at the largest index, plus the pulse
    backend = "gpu" if any(d.platform != "cpu" for d in jax.devices()) else "cpu"
    config = fdtdx.SimulationConfig(grid=fdtdx.UniformGrid(spacing=res), time=time_fs * 1e-15, courant_factor=0.99,
                                    backend=backend)
    background_mat = segments[0].xs.background
    volume = fdtdx.SimulationVolume(partial_real_shape=shape,
                                    material=fdtdx.Material(permittivity=background_mat.n**2))
    objects, constraints = [volume], []
    mats: dict[str, object] = {}
    for k, seg in enumerate(segments):
        z0 = seg.z0 - (pml if k == 0 else 0.0)
        z1 = seg.z1 + (pml if k == len(segments) - 1 else 0.0)
        for j, r in enumerate(seg.xs.shapes):
            if not hasattr(r, "x0"):
                raise NotImplementedError("the FDTD adapter takes Rect shapes only (polygons: not yet)")
            x0, x1 = max(r.x0, x_range[0] - pml), min(r.x1, x_range[1] + pml)
            y0, y1 = max(r.y0, y_range[0] - pml), min(r.y1, y_range[1] + pml)
            if x1 <= x0 or y1 <= y0:
                continue
            mat = mats.setdefault(r.material.name, fdtdx.Material(permittivity=r.material.n**2))
            box = fdtdx.UniformMaterialObject(name=f"seg{k}_shape{j}",
                                              partial_real_shape=((z1 - z0) * um, (x1 - x0) * um, (y1 - y0) * um),
                                              material=mat)
            objects.append(box)
            constraints.append(box.place_relative_to(
                volume, axes=(0, 1, 2), other_positions=(-1, -1, -1), own_positions=(-1, -1, -1),
                margins=((z0 - segments[0].z0 + pml) * um, (x0 - x_range[0] + pml) * um,
                         (y0 - y_range[0] + pml) * um)))
    freq = C0 / (wavelength * um)
    plane = (res, wx * um, wy * um)
    source = fdtdx.ModePlaneSource(
        name="source", mode_index=0, filter_pol=filter_pol, partial_real_shape=plane, direction="+",
        wave_character=fdtdx.WaveCharacter(wavelength=wavelength * um),
        temporal_profile=fdtdx.GaussianPulseProfile(center_wave=fdtdx.WaveCharacter(frequency=freq),
                                                     spectral_width=fdtdx.WaveCharacter(frequency=freq / 10)))
    det_in = fdtdx.ModeOverlapDetector(name="in", mode_index=0, filter_pol=filter_pol, partial_real_shape=plane,
                                       direction="+", wave_characters=(fdtdx.WaveCharacter(wavelength=wavelength * um),))
    det_out = fdtdx.ModeOverlapDetector(name="out", mode_index=0, filter_pol=filter_pol, partial_real_shape=plane,
                                        direction="+", wave_characters=(fdtdx.WaveCharacter(wavelength=wavelength * um),))
    wc = (fdtdx.WaveCharacter(wavelength=wavelength * um),)
    flux_in = fdtdx.PhasorPoyntingFluxDetector(name="flux_in", partial_real_shape=plane, direction="+", wave_characters=wc)
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
    t0 = time.perf_counter()
    _, arrays = fdtdx.run_fdtd(arrays, objs, config, key)
    a_in = complex(objs["in"].compute_overlap(arrays.detector_states["in"])[0])
    a_out = complex(objs["out"].compute_overlap(arrays.detector_states["out"])[0])
    p_in = float(objs["flux_in"].compute_poynting_flux(arrays.detector_states["flux_in"])[0])
    p_out = float(objs["flux_out"].compute_poynting_flux(arrays.detector_states["flux_out"])[0])
    seconds = time.perf_counter() - t0
    cells = int(np.prod([round(s / res) for s in shape]))
    return FDTDResult(abs(a_out) ** 2 / abs(a_in) ** 2, a_in, a_out, cells, seconds, backend, p_in, p_out)
