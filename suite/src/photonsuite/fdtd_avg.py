"""3-D FDTD of a segment device with fdtd_gpu: subpixel averaging and the fundamental-mode transmission over a band.

The same device description as photonsuite.fdtd and photonsuite.eme (a list of Segment(z0, z1, CrossSection), um).
fdtd_gpu (this repository, on fdtdx) runs it with the subpixel average of Kottke et al. at each Yee E component
(``eps_averaging=True``, the default), so the result converges at about second order in the cell size also when the
faces of the shapes are not on grid lines (fdtd_gpu README, checks 6 and 7). photonsuite.fdtd (fdtdx directly) has
the staircase only.

Axes: fdtd_gpu x = the suite's z (propagation), fdtd_gpu y = the suite's x (horizontal), fdtd_gpu z = the suite's y
(vertical). So "te" (E mostly horizontal) is the parity EVEN_Z of fdtd_gpu and "tm" is ODD_Z. The cell is the window
x_range x y_range by the device length, with a PML of thickness ``pml`` (um) outside it on each face; the first and
the last segment continue through the PML. ``side_pml`` (um; default ``pml``) sets the PML on the four side faces
only. Each Rect becomes a Block over the z range of its segment, each Polygon a Prism along the propagation axis; a
later shape replaces an earlier one, as in CrossSection.

Side walls (``walls``): "pml" (the default) puts the PML outside the window on all six faces, so a field that reaches
the window edge leaves the cell. "pec" puts the PML only at the two ends along the propagation axis and metal walls
(perfect electric conductors) at the window edges, as the closed window of the EME (photonsuite.eme, meow_gpu: E_t
= 0 on x = x_range and y = y_range). fdtdx sets the tangential E of the first and the last cell layer of a PEC face
to zero. fdtd_gpu puts each E component of a cell at the cell center plus dx / 2 along the component (fdtd_gpu
``_averaging``), so the tangential components of a cell are in the plane through its center, and the walls are the
center planes of these two layers. The cell then has round(W / dx) + 1 cells across a window of width W, and the
walls are at the window center +- round(W / dx) dx / 2: on the window edges when W / dx is a whole number, else up to
dx / 2 from them (``wall_positions``; tests/test_fdtd_avg_walls.py checks the position with the exact Yee dispersion
of a metal box mode). The mode planes cover the whole cross-section of the cell; the mode solver of fdtdx puts its own
PEC walls on the plane edges, dx / 2 outside the FDTD walls (a guided mode that is negligible at the walls does not
see the difference). In the closed window the modes near their cutoff frequency ring for a long time (their group
velocity goes to zero), so the field at the stop point does not decay by ``decay_by`` (a development run: |E|^2
fell to about 3 % of its value at the end of the source in 300 um/c, at the cutoff frequency of a higher mode), and
the run stops at ``max_time`` with ``decayed=False``. These modes are almost orthogonal to the fundamental mode of
the monitors (whose walls are dx / 2 farther out), so T00 converges with the run time, but slowly at a wavelength
near a cutoff: check T00 against a run with a longer ``max_time`` in place of the decay. (The stop rule compares the
field with its largest value after the end of the source. In a short device the pulse has passed the stop point by
then, so that value is small: in development runs the straight guides of the calibration also went to ``max_time``
with the PML.)

Ports (the positions of photonsuite.fdtd): a mode source (the fundamental mode of the parity, solved at the center
frequency; a Gaussian pulse) 0.5 um into the first segment; mode monitors and flux planes 1.0 um into the first
segment (input) and 0.5 um before the end of the last segment (output). Each plane must be at least 0.5 um from the
other end of its segment, so the first segment must be at least 1.5 um long, the last at least 1.0 um and the
device at least 2.0 um (else ValueError): a plane on a segment end gives the mode of a mixed cross-section (with a
1.0 um first segment, the input plane was on the start of a taper, and the mode solver there gave the TE0 index
2.50 against 2.445 of the 0.5 um strip). The run stops when the field at the center of the output plane has
decayed (``decay_by``, of |E|^2).

Mode coefficients: fdtd_gpu gives alpha = (int E x Hm* + int Em* x H) / (2 sqrt(Re int Em x Hm*)) with the mode
(Em, Hm) of the fdtdx mode solver at each frequency, so |alpha|^2 is the power in the mode. On the Yee grid the
fields are co-located by averages of neighbor points, so |alpha|^2 of a pure mode differs from its flux by a few
percent at 20 cells per um (fdtd_gpu README, check 2), and this factor depends on the mode. With ``calibrate=True``
two more runs measure it: straight guides of the input and of the output cross-section (the same length and ports),
b = |alpha|^2 / P at their output plane, where only the guided mode is left. Then

    T00 = (|a_out|^2 / b_out) / (|a_in|^2 / b_in)

is the power in the fundamental mode at the output over that at the input. ``T00_raw`` = |a_out|^2 / |a_in|^2.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from photonsuite.fdtd import Segment, check_segments

SOURCE_AT, INPUT_AT, OUTPUT_BEFORE = 0.5, 1.0, 0.5      # um: from the start of the device, and before its end
PLANE_MARGIN = 0.5                                       # um: from a plane to the other end of its segment


@dataclass
class PortRun:
    """One FDTD run: per frequency the forward amplitude (a_in, a_out) and the backward amplitude (r_in) of the
    fundamental mode, and the net Poynting flux (p_in, p_out) of the input and output planes."""
    a_in: np.ndarray
    r_in: np.ndarray
    a_out: np.ndarray
    p_in: np.ndarray
    p_out: np.ndarray
    neff_in: np.ndarray
    neff_out: np.ndarray
    cells: int
    steps: int
    seconds: float           # the FDTD loop (with the compile)
    setup_seconds: float     # the rasterization and the subpixel average
    device: str
    distance: float = float("nan")   # um between the cells of the input and the output plane
    shape: tuple = ()                # cells along the fdtd_gpu axes (the suite's z, x, y)
    decayed: bool = True             # False: the run stopped at max_time before the field decayed by decay_by

    def phase_index(self, wavelengths, guess) -> np.ndarray:
        """The effective index of the FDTD propagation in a straight guide: arg(a_out / a_in) = 2 pi neff d / lambda
        (the DFT has exp(+i omega t), so a forward wave is exp(+i beta x)), with the multiple of 2 pi nearest to
        the index `guess` (it must be within lambda / (2 d) of the answer)."""
        k_d = 2 * np.pi * self.distance / np.asarray(wavelengths, float)
        phi = np.angle(self.a_out / self.a_in)
        return (phi + 2 * np.pi * np.round((np.asarray(guess) * k_d - phi) / (2 * np.pi))) / k_d


@dataclass
class ModeSpectrum:
    wavelengths: np.ndarray  # um
    resolution: float        # cells per um
    T00: np.ndarray          # power in the fundamental mode, output over input (calibrated; = T00_raw without)
    T00_raw: np.ndarray      # |a_out|^2 / |a_in|^2
    T_flux: np.ndarray       # P_out / P_in: all power that crosses the output plane
    R00: np.ndarray          # |r_in|^2 / |a_in|^2 at the input plane (with the small false backward part of the grid)
    b_in: np.ndarray         # |alpha|^2 / P of a pure input mode (ones without calibration)
    b_out: np.ndarray
    src_rel: np.ndarray      # the source spectrum at each wavelength over its peak
    run: PortRun
    calibration: tuple = ()  # the PortRun of the straight input and output guides

    @property
    def input_mode_share(self) -> np.ndarray:
        """The power of the fundamental mode at the input plane over the net input flux. It is near 1, not a bound:
        the net flux is the forward minus the reflected power, so a reflection R adds about R, and the source
        launches some power outside the mode at the band edges. Logged values (subpixel average,
        benchmarks/2026-11-08/): the taper 1.00006 to 1.00018 at 1.525 to 1.575 um and 0.99977 to 0.99991 at 1.50
        and 1.60 um (suite-eme-fdtd-taper.jsonl); the double step 1.0033 to 1.0045, about 1 + R
        (suite-eme-fdtd-double-step.jsonl)."""
        return np.abs(self.run.a_in) ** 2 / self.b_in / self.run.p_in


WALLS = ("pml", "pec")


def _check_walls(walls: str) -> None:
    if walls not in WALLS:
        raise ValueError(f"walls must be one of {WALLS}, not {walls!r}")


def _cells_between_walls(width: float, resolution: float) -> int:
    """The number of cells between the two PEC walls of one axis: the whole number nearest width x resolution."""
    n = int(round(width * resolution))
    if n < 1:
        raise ValueError(f"the window ({width} um) must be at least one cell ({1 / resolution} um) wide")
    return n


def wall_positions(x_range, y_range, resolution: float) -> tuple[tuple[float, float], tuple[float, float]]:
    """walls="pec": the positions (um, suite axes) of the metal walls of the FDTD cell, ((x0, x1), (y0, y1)): the
    window center +- round(W resolution) / (2 resolution) on each axis (module text)."""
    out = []
    for lo, hi in (x_range, y_range):
        half = 0.5 * _cells_between_walls(hi - lo, resolution) / resolution
        out.append((0.5 * (lo + hi) - half, 0.5 * (lo + hi) + half))
    return tuple(out)


def geometry(segments, x_range, y_range, pml: float, walls: str = "pml", resolution: float | None = None,
             side_pml: float | None = None):
    """(objects, cell size, default material) of fdtd_gpu for the device, in the fdtd_gpu axes (see the module text):
    the cell is centered on the window and on the device length. walls="pec" needs the resolution (the cell has
    one cell more than the window on each transverse axis, so that the walls are on the window edges)."""
    import fdtd_gpu as fg

    _check_walls(walls)
    if walls == "pec" and resolution is None:
        raise ValueError("walls='pec' needs the resolution")

    z_lo, z_hi = segments[0].z0, segments[-1].z1
    zc, xc, yc = 0.5 * (z_lo + z_hi), 0.5 * (x_range[0] + x_range[1]), 0.5 * (y_range[0] + y_range[1])
    media: dict[str, object] = {}

    def medium(m):
        return media.setdefault(m.name, fg.Medium(index=m.n))

    background = medium(segments[0].xs.background)
    objects = []
    beyond = pml + 1.0                                   # the first and last segments continue through the PML
    for k, seg in enumerate(segments):
        za = seg.z0 - (beyond if k == 0 else 0.0)
        zb = seg.z1 + (beyond if k == len(segments) - 1 else 0.0)
        for r in seg.xs.shapes:
            if hasattr(r, "vertices"):
                verts = [fg.Vector3(za - zc, x - xc, y - yc) for x, y in r.vertices]
                objects.append(fg.Prism(verts, zb - za, axis=fg.Vector3(1, 0, 0), material=medium(r.material)))
            else:
                objects.append(fg.Block(fg.Vector3(zb - za, r.x1 - r.x0, r.y1 - r.y0),
                                        center=fg.Vector3(0.5 * (za + zb) - zc, 0.5 * (r.x0 + r.x1) - xc,
                                                          0.5 * (r.y0 + r.y1) - yc),
                                        material=medium(r.material)))
    if walls == "pec":
        across = [(_cells_between_walls(hi - lo, resolution) + 1) / resolution for lo, hi in (x_range, y_range)]
    else:
        side = pml if side_pml is None else side_pml
        across = [hi - lo + 2 * side for lo, hi in (x_range, y_range)]
    size = fg.Vector3(z_hi - z_lo + 2 * pml, *across)
    return objects, size, background


def simulation(segments, x_range, y_range, freqs, resolution: float, pml: float = 0.4, parity: str = "te",
               eps_averaging: bool = True, fwidth: float | None = None, walls: str = "pml",
               side_pml: float | None = None):
    """The fdtd_gpu Simulation of the device with its source, and the monitors (m_in, m_out, f_in, f_out), not
    run. freqs in 1/um (c = 1). walls: "pml" or "pec" (module text)."""
    import fdtd_gpu as fg

    if parity not in ("te", "tm"):
        raise ValueError(f"parity must be 'te' or 'tm', not {parity!r}")
    if side_pml is not None and walls == "pec":
        raise ValueError("side_pml is the PML of the side faces: not with walls='pec'")
    objects, size, background = geometry(segments, x_range, y_range, pml, walls, resolution, side_pml)
    length = segments[-1].z1 - segments[0].z0
    plane = fg.Vector3(0.0, x_range[1] - x_range[0], y_range[1] - y_range[0])
    src = fg.EigenModeSource(pulse(freqs, fwidth), center=fg.Vector3(-length / 2 + SOURCE_AT), size=plane,
                             eig_band=1, eig_parity=_parity(parity))
    if walls == "pec":
        layers = [fg.PML(pml, direction=fg.X)]
    elif side_pml is None:
        layers = [fg.PML(pml)]
    else:
        layers = [fg.PML(pml, direction=fg.X), fg.PML(side_pml, direction=fg.Y), fg.PML(side_pml, direction=fg.Z)]
    sim = fg.Simulation(cell_size=size, resolution=resolution, geometry=objects, sources=[src],
                        boundary_layers=layers, default_material=background, eps_averaging=eps_averaging)
    x_in, x_out = fg.Vector3(-length / 2 + INPUT_AT), fg.Vector3(length / 2 - OUTPUT_BEFORE)
    mons = (sim.add_mode_monitor(freqs, fg.ModeRegion(center=x_in, size=plane)),
            sim.add_mode_monitor(freqs, fg.ModeRegion(center=x_out, size=plane)),
            sim.add_flux(freqs, fg.FluxRegion(center=x_in, size=plane)),
            sim.add_flux(freqs, fg.FluxRegion(center=x_out, size=plane)))
    return sim, mons


def pulse(freqs, fwidth: float | None = None):
    """The Gaussian pulse of the source: centered on the band; by default the band edges are within one standard
    deviation fwidth / (2 pi) of its spectrum, and fwidth is at least 0.1 of the center frequency."""
    import fdtd_gpu as fg

    freqs = np.asarray(freqs, float)
    fcen = 0.5 * (freqs.min() + freqs.max())
    if fwidth is None:
        fwidth = max(0.1 * fcen, 2 * np.pi * (freqs.max() - freqs.min()))
    return fg.GaussianSource(fcen, fwidth=fwidth)


def _parity(parity):
    import fdtd_gpu as fg
    return fg.EVEN_Z if parity == "te" else fg.ODD_Z


def run_ports(segments, x_range, y_range, freqs, resolution: float, pml: float = 0.4, parity: str = "te",
              eps_averaging: bool = True, fwidth: float | None = None, decay_by: float = 1e-7,
              max_time: float | None = None, walls: str = "pml", side_pml: float | None = None) -> PortRun:
    """One FDTD run of the device: the mode amplitudes and fluxes of the input and output planes."""
    import fdtd_gpu as fg

    check_segments(segments, first_min=INPUT_AT + PLANE_MARGIN, last_min=OUTPUT_BEFORE + PLANE_MARGIN)
    length = segments[-1].z1 - segments[0].z0
    if length < INPUT_AT + OUTPUT_BEFORE + PLANE_MARGIN - 1e-9:
        raise ValueError(f"the device must be at least {INPUT_AT + OUTPUT_BEFORE + PLANE_MARGIN} um long (the input "
                         f"and the output plane at least {PLANE_MARGIN} um apart)")
    sim, (m_in, m_out, f_in, f_out) = simulation(segments, x_range, y_range, freqs, resolution, pml, parity,
                                                 eps_averaging, fwidth, walls, side_pml)
    n_max = max(max([s.material.n for s in seg.xs.shapes] + [seg.xs.background.n]) for seg in segments)
    if max_time is None:                          # um/c after the source: 20 transits at the largest index
        max_time = 20.0 * length * n_max + 100.0
    comp = fg.Ey if parity == "te" else fg.Ez
    stop = fg.stop_when_fields_decayed(20.0, comp, fg.Vector3(length / 2 - OUTPUT_BEFORE), decay_by,
                                       max_time=max_time)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sim.run(until_after_sources=stop)
    for w in caught:                                   # pass them on, and note a run that stopped at max_time
        warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
    ci = sim.get_eigenmode_coefficients(m_in, [1], eig_parity=_parity(parity))
    co = sim.get_eigenmode_coefficients(m_out, [1], eig_parity=_parity(parity))
    planes = [sim._range(m.regions[0].center, m.regions[0].size, 0)[0] for m in (m_in, m_out)]
    return PortRun(a_in=ci.alpha[0, :, 0], r_in=ci.alpha[0, :, 1], a_out=co.alpha[0, :, 0],
                   p_in=np.asarray(sim.get_fluxes(f_in)), p_out=np.asarray(sim.get_fluxes(f_out)),
                   neff_in=ci.neff[0], neff_out=co.neff[0], cells=int(np.prod(sim.shape)), steps=int(sim.n_steps),
                   seconds=float(sim.run_seconds), setup_seconds=float(sim.setup_seconds), device=str(sim.device),
                   distance=(planes[1] - planes[0]) * sim.dx, shape=tuple(int(n) for n in sim.shape),
                   decayed=not any("did not decay" in str(w.message) for w in caught))


def mode_transmission(segments, x_range, y_range, wavelengths, resolution: float, pml: float = 0.4,
                      parity: str = "te", eps_averaging: bool = True, calibrate: bool = True,
                      fwidth: float | None = None, decay_by: float = 1e-7, walls: str = "pml",
                      max_time: float | None = None, side_pml: float | None = None) -> ModeSpectrum:
    """The fundamental-mode transmission of the device at each wavelength (um) by 3-D FDTD (fdtd_gpu, float32, on
    the first GPU). resolution: cells per um. calibrate: two more runs (straight input and output guides, with the
    same walls) for the mode factor b (module text). walls: "pml" (absorbing, the default) or "pec" (metal walls at
    the window edges, the closed window of the EME; module text). max_time: the run time after the source (um/c;
    None: 20 transits of the device at the largest index, plus 100), for each run. side_pml: the PML of the side
    faces (um; default pml; walls="pml" only)."""
    _check_walls(walls)
    wl = np.atleast_1d(np.asarray(wavelengths, float))
    freqs = 1.0 / wl
    kw = dict(resolution=resolution, pml=pml, parity=parity, eps_averaging=eps_averaging, fwidth=fwidth,
              decay_by=decay_by, walls=walls, max_time=max_time, side_pml=side_pml)
    run = run_ports(segments, x_range, y_range, freqs, **kw)
    b_in = b_out = np.ones(len(wl))
    cal = ()
    if calibrate:
        z0, z1 = segments[0].z0, segments[-1].z1
        cal = tuple(run_ports([Segment(z0, z1, xs)], x_range, y_range, freqs, **kw)
                    for xs in (segments[0].xs, segments[-1].xs))
        b_in, b_out = (np.abs(c.a_out) ** 2 / c.p_out for c in cal)
    pin, pout = np.abs(run.a_in) ** 2, np.abs(run.a_out) ** 2
    src = pulse(freqs, fwidth)
    spec = np.abs(src.fourier_transform(freqs)) / abs(src.fourier_transform(src.frequency))
    return ModeSpectrum(wavelengths=wl, resolution=float(resolution), T00=(pout / b_out) / (pin / b_in),
                        T00_raw=pout / pin, T_flux=run.p_out / run.p_in, R00=np.abs(run.r_in) ** 2 / pin,
                        b_in=b_in, b_out=b_out, src_rel=spec, run=run, calibration=cal)
