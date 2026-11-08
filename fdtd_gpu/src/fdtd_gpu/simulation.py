"""Simulation: the familiar FDTD script style (MIT package) on fdtdx (MIT, JAX; AMD ROCm or CUDA GPUs).

Units: um; frequency in 1/um (c = 1); time in um/c (1 um/c = 3.336 fs). The cell is centered at the origin. A cell
size of 0 on an axis makes the simulation 2-D (or 1-D): the fields do not change along that axis.

Mapping to fdtdx (SI units inside): axes x, y, z = fdtdx axes 0, 1, 2. The geometry becomes one voxel object:
with eps_averaging=True (the default) one diagonal permittivity per Yee E component, from the subpixel average of
Kottke et al. (`fdtd_gpu._averaging`); with eps_averaging=False the material of each cell center (a staircase).
Dispersion (Lorentz and Drude terms) and D_conductivity: the ADE arrays of fdtdx, set per cell by this package
(`fdtd_gpu._dispersion`): the staircase of the cell centers, or with eps_averaging the rule FACE_RULE (default:
the pole strength times the fill fraction of each E component).
Boundaries: PML layers inside the cell (as in the MIT package); other faces are perfect electric conductors, or
periodic with k_point=Vector3(); trivial axes are periodic. Fields run in float32. The DFT monitors use the
convention of the MIT package, sum f(t) exp(i 2 pi f t) dt / sqrt(2 pi), so fluxes and mode coefficients have the
same normalization for a given source current. add_near2far keeps the DFT fields of closed surfaces for the
near-to-far-field transform (`fdtd_gpu.near2far`: surface equivalence with the exact 3-D or 2-D Green's function).
"""

from __future__ import annotations

import math
import time as _time
import warnings
from dataclasses import dataclass, field

import numpy as np

import fdtd_gpu._dispersion as _disp
from fdtd_gpu.geometry import Medium, Vector3, _v3, rasterize
from fdtd_gpu.harminv import harminv
from fdtd_gpu.near2far import Volume, farfield_grid, radiate
from fdtd_gpu.sources import (ALL, AUTOMATIC, EVEN_Z, NO_PARITY, ODD_Y, ODD_Z, ContinuousSource, EigenModeSource,
                              GaussianSource, Source)

C0 = 299792458.0
UM = 1e-6
High, Low = 0, 1
_COMPONENTS = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
TRIVIAL_CELLS = 3            # cells on an axis of size 0 (periodic; fdtdx planes need one axis of 1 cell)


class PML:
    """A perfectly matched layer inside the cell: PML(thickness, direction=ALL, side=ALL, R_asymptotic=None). fdtdx
    grades its own conductivity profile (cubic; not the profile of the MIT package). R_asymptotic: the round-trip
    amplitude of the continuous layer in vacuum, R = exp(-2 eta0 integral(sigma)); fdtdx sets its conductivity for
    R = 1e-6 (None, the default here; the MIT package has 1e-15). In a medium of index n the round trip is
    R^Re(n) times the loss of the medium in the layer, so a smaller R helps where Re(n) is small (check 14)."""

    def __init__(self, thickness: float, direction: int = ALL, side: int = ALL, R_asymptotic: float | None = None,
                 **unsupported):
        if unsupported:
            raise NotImplementedError(f"PML: unsupported arguments {sorted(unsupported)}")
        if R_asymptotic is not None and not 0 < R_asymptotic < 1:
            raise ValueError(f"R_asymptotic must be between 0 and 1, got {R_asymptotic}")
        self.thickness, self.direction, self.side = float(thickness), direction, side
        self.R_asymptotic = R_asymptotic

    @property
    def R(self) -> float:
        """The round trip of the layer in vacuum (1e-6, the default of fdtdx, if R_asymptotic is None)."""
        return _disp.PML_R_DEFAULT if self.R_asymptotic is None else float(self.R_asymptotic)


class FluxRegion:
    """A plane for flux or mode monitors: FluxRegion(center, size, direction=AUTOMATIC, weight=1)."""

    def __init__(self, center=Vector3(), size=Vector3(), direction: int = AUTOMATIC, weight: float = 1.0):
        self.center, self.size, self.direction, self.weight = _v3(center), _v3(size), direction, float(weight)


ModeRegion = FluxRegion


class Near2FarRegion(FluxRegion):
    """A plane of a near-to-far-field surface: Near2FarRegion(center, size, direction=AUTOMATIC, weight=1). As for
    flux: the normal is +direction; weight -1 on the faces of a closed box whose outward normal is -direction."""


class stop_when_fields_decayed:
    """Stop when max |c(pt)|^2 over the last dt is at most decay_by x its maximum so far (checked every dt, as in
    the MIT package). max_time (um/c) caps the run."""

    def __init__(self, dt: float, c: str, pt, decay_by: float, max_time: float = 5000.0):
        self.dt, self.c, self.pt, self.decay_by, self.max_time = float(dt), c, _v3(pt), float(decay_by), max_time


class Harminv:
    """Record component c at pt; after the run, .modes holds the resonances in [fcen - df/2, fcen + df/2]
    (fdtd_gpu.harminv: a matrix pencil fit, not filter diagonalization). Only the signal after the sources counts."""

    def __init__(self, c: str, pt, fcen: float, df: float, mxbands: int | None = None):
        self.c, self.pt, self.fcen, self.df = c, _v3(pt), float(fcen), float(df)
        self.mxbands = mxbands or 30
        self.modes = []
        self.data = None
        self.data_dt = None


def after_sources(step):
    """The analysis of Harminv uses only the signal after the sources; this wrapper is accepted for compatibility."""
    if not isinstance(step, Harminv):
        raise NotImplementedError("after_sources supports only Harminv")
    return step


@dataclass
class _Monitor:
    freqs: np.ndarray
    regions: list
    names: list = field(default_factory=list)
    axes: list = field(default_factory=list)
    minus: list | None = None
    mode: bool = False


@dataclass
class DftNear2Far(_Monitor):
    """The DFT fields of the planes of add_near2far. After the run: `eps` (the medium of the far fields: a float, or
    for a dispersive or conductive medium an array of the complex eps(f) at each frequency) and
    `flux(direction, where, resolution)`."""
    sim: object = None
    eps: float | np.ndarray = float("nan")

    def flux(self, direction: int, where, resolution: float) -> list:
        """The flux sum Re(E x conj H) . e_direction of the far fields over the points of get_farfields(self,
        resolution, where) (trapezoidal rule), at each frequency."""
        return self.sim._near2far_flux(self, direction, where, resolution)


@dataclass
class FluxData:
    E: list
    H: list


@dataclass
class EigCoeffs:
    alpha: np.ndarray      # (bands, freqs, 2): forward and backward amplitudes; |alpha|^2 is a flux
    neff: np.ndarray       # (bands, freqs) effective index of each mode (from the mode solver)


def _freqs(args):
    if len(args) >= 3 and np.ndim(args[0]) == 0:
        fcen, df, nfreq = float(args[0]), float(args[1]), int(args[2])
        freqs = np.array([fcen]) if nfreq == 1 else np.linspace(fcen - df / 2, fcen + df / 2, nfreq)
        return freqs, list(args[3:])
    return np.atleast_1d(np.asarray(args[0], float)), list(args[1:])


def _device():
    import jax
    try:
        return jax.devices("gpu")[0]
    except RuntimeError:
        return jax.devices("cpu")[0]


class Simulation:
    def __init__(self, cell_size, resolution: float, geometry=(), sources=(), boundary_layers=(),
                 default_material: Medium = Medium(), k_point=None, dimensions: int | None = None,
                 symmetries=(), eps_averaging: bool = True, Courant: float | None = None,
                 force_complex_fields: bool = False, geometry_center=None, **unsupported):
        if unsupported:
            raise NotImplementedError(f"Simulation: unsupported arguments {sorted(unsupported)}")
        if symmetries:
            raise NotImplementedError("symmetries are not supported")
        if force_complex_fields:
            raise NotImplementedError("complex fields are not supported (real float32 fields)")
        if geometry_center is not None and tuple(_v3(geometry_center)) != (0, 0, 0):
            raise NotImplementedError("geometry_center is not supported")
        if Courant is not None:
            warnings.warn("Courant is ignored: fdtdx uses dt = 0.99 dx / (c sqrt 3)")
        if k_point not in (None, False) and tuple(_v3(k_point)) != (0, 0, 0):
            raise NotImplementedError("only k_point=Vector3() (periodic) or None (metal walls) is supported")
        self.cell_size = _v3(cell_size)
        self.resolution = float(resolution)
        self.dx = 1.0 / self.resolution
        self.trivial = tuple(s == 0 for s in self.cell_size)
        if dimensions is not None and dimensions != 3 - sum(self.trivial):
            raise NotImplementedError("dimensions must agree with the zero sizes of cell_size")
        self.shape = tuple(TRIVIAL_CELLS if t else max(1, int(round(s * self.resolution)))
                           for s, t in zip(self.cell_size, self.trivial))
        self.dims = 3 - sum(self.trivial)
        self.eps_averaging = bool(eps_averaging)
        self.geometry, self.sources = list(geometry), list(sources)
        self.boundary_layers, self.default_material = list(boundary_layers), default_material
        self.periodic = k_point not in (None, False)
        for a in range(3):                   # fdtdx wraps the whole axis when one face of it is periodic
            sides = [s for s in (High, Low) if any(p.direction in (ALL, a) and p.side in (ALL, s)
                                                   for p in self.boundary_layers)]
            if self.periodic and not self.trivial[a] and len(sides) == 1:
                raise NotImplementedError(f"k_point with a PML on one side of the {'xyz'[a]} axis only is not "
                                          f"supported (the other face would be periodic): put a PML on both sides")
        self._monitors: list[_Monitor] = []
        self._probes: list = []
        self._done = False
        self.dt = 0.99 * self.dx / math.sqrt(3)          # um/c
        self.meep_time = 0.0
        self.run_seconds = float("nan")
        self.setup_seconds = float("nan")    # the rasterization and the average: set by get_epsilon() and run()
        self.n_averaged = 0                  # the number of sampled voxels: set by get_epsilon() and run()
        self.device = None
        for s in self.sources:
            if not isinstance(s, (Source, EigenModeSource)):
                raise NotImplementedError(f"unsupported source {type(s).__name__}")
            if not isinstance(s.src, (GaussianSource, ContinuousSource)):
                raise NotImplementedError(f"unsupported source time {type(s.src).__name__}")
            if isinstance(s, EigenModeSource):
                self._filter_pol(s.eig_parity, self._normal(s.size, s.direction))     # raises early

    # ---------- grid helpers ----------
    def _lo(self, axis):
        return -self.shape[axis] * self.dx / 2

    def _cell(self, p, axis):
        """The cell that holds the coordinate p (um) on an axis, clipped to the grid."""
        k = int(np.floor((p - self._lo(axis)) / self.dx + 1e-9))
        return min(max(k, 0), self.shape[axis] - 1)

    def _range(self, center, size, axis):
        """(i0, i1) cells of [center - size/2, center + size/2] on an axis: from the cell of the low end to the cell
        of the high end, both included, so that six planes of a box close it (one cell for size 0)."""
        if self.trivial[axis]:
            return 0, self.shape[axis]
        c, s = center[axis], size[axis]
        return self._cell(c - s / 2, axis), self._cell(c + s / 2, axis) + 1

    def _normal(self, size, direction=AUTOMATIC):
        if direction not in (AUTOMATIC, None):
            return int(direction)
        zero = [a for a in range(3) if not self.trivial[a] and size[a] == 0]
        if len(zero) != 1:
            raise ValueError(f"a monitor or mode-source plane needs exactly one zero size on a non-trivial axis "
                             f"(size {tuple(size)})")
        return zero[0]

    def _area(self, normal):
        """Area weight of one cell of a plane normal to an axis (um^(dims - 1); per unit length in 2-D). A trivial
        axis has TRIVIAL_CELLS equal cells: each counts 1 / TRIVIAL_CELLS."""
        tang = [a for a in range(3) if a != normal]
        return (self.dx ** sum(1 for a in tang if not self.trivial[a])
                / TRIVIAL_CELLS ** sum(1 for a in tang if self.trivial[a]))

    def _trapezoid(self, plane_shape, normal):
        """Weights of the cells of a plane: 1/2 at the two end cells of each non-trivial tangential axis (the
        trapezoidal rule). The faces of a box then share their edge cells, and the box is closed: with full
        weights, the edge cells count twice and the flux of a box has an error of order dx / L."""
        w = np.ones(plane_shape)
        for ax in range(3):
            if ax == normal or self.trivial[ax] or plane_shape[ax] < 2:
                continue
            idx = [slice(None)] * 3
            for end in (0, -1):
                idx[ax] = end
                w[tuple(idx)] *= 0.5
        return w

    def source_end_time(self) -> float:
        return max((s.src.end_time for s in self.sources), default=0.0)

    def _periodic_axes(self) -> tuple[bool, bool, bool]:
        """True for an axis of size > 0 with periodic faces: k_point given and no PML on the axis (as in _build; a
        PML on one side only raises in __init__). There the objects repeat. Unlike the MIT package (its default
        ensure_periodicity), the objects do not repeat along an axis with a PML."""
        return tuple(not self.trivial[a] and self.periodic
                     and not any(p.direction in (ALL, a) for p in self.boundary_layers) for a in range(3))

    def _permittivity(self):
        """(staircase index, materials, eps3): eps3 is the (3, *shape) permittivity of the E components (subpixel
        averaged), or None for the staircase or a uniform cell. Sets setup_seconds and n_averaged."""
        t0 = _time.perf_counter()
        per = self._periodic_axes()
        idx, mats = rasterize(self.geometry, self.default_material, self.shape, self.dx, self.trivial, per)
        eps3, self.n_averaged, self._fills = None, 0, None
        if self.eps_averaging and len(mats) > 1:
            import jax

            from fdtd_gpu._averaging import averaged_permittivity
            lossy = _disp.lossy_media(self.default_material, self.geometry)
            track = lossy if lossy and _disp.FACE_RULE == "fill" else None      # fill fractions for the face rule
            with jax.enable_x64(False), jax.default_device(_device()):
                res = averaged_permittivity(self.geometry, mats, idx, self.dx, self.trivial, per, track=track)
            eps3, self.n_averaged = res[0], res[1]
            if track is not None:
                self._fills = res[2]
            if lossy and _disp.FACE_RULE == "staircase":
                eps3 = _disp.staircase_at_lossy(eps3, idx, mats, tuple(t or p for t, p in zip(self.trivial, per)))
        self.setup_seconds = _time.perf_counter() - t0
        return idx, mats, eps3

    def get_epsilon(self) -> np.ndarray:
        """The relative permittivity of each cell, with the trivial axes removed. Staircase: the material of the
        cell center. With eps_averaging: 3 / (1/eps_xx + 1/eps_yy + 1/eps_zz) of the averaged components of the
        cell (each component is at its own Yee point: the cell center + dx / 2 along its axis)."""
        idx, mats, eps3 = self._permittivity()
        eps = np.array([m.epsilon for m in mats])[idx] if eps3 is None else 3.0 / (1.0 / eps3).sum(axis=0)
        return eps[tuple(0 if t else slice(None) for t in self.trivial)]      # a trivial axis has TRIVIAL_CELLS

    # ---------- monitors ----------
    def add_flux(self, *args):
        """add_flux(fcen, df, nfreq, FluxRegion, ...) or add_flux(freqs, FluxRegion, ...)."""
        freqs, regions = _freqs(args)
        return self._add_monitor(freqs, regions, mode=False)

    def add_mode_monitor(self, *args):
        """add_mode_monitor(fcen, df, nfreq, ModeRegion) or add_mode_monitor(freqs, ModeRegion); one plane."""
        freqs, regions = _freqs(args)
        if len(regions) != 1:
            raise NotImplementedError("a mode monitor has exactly one region")
        return self._add_monitor(freqs, regions, mode=True)

    def _add_monitor(self, freqs, regions, mode, cls=_Monitor):
        if self._done:
            raise RuntimeError("add monitors before run()")
        if not regions:
            raise ValueError("no regions")
        mon = cls(freqs=np.asarray(freqs, float), regions=regions, mode=mode)
        k = len(self._monitors)
        for j, r in enumerate(regions):
            mon.names.append(f"mon{k}_{j}")
            mon.axes.append(self._normal(r.size, r.direction))
            if mode:
                self._filter_pol(NO_PARITY, mon.axes[-1])               # 1-D raises
        self._monitors.append(mon)
        return mon

    def get_flux_freqs(self, mon) -> list:
        return list(mon.freqs)

    def _phasors(self, mon, j):
        """E and H phasors (freqs, 3, *plane) of region j, with the DFT scale dt / sqrt(2 pi)."""
        st = self._arrays.detector_states[mon.names[j]]["phasor"]
        ph = np.asarray(st[0], np.complex128) * (self.dt / math.sqrt(2 * math.pi))
        # No time shift: the fdtdx detectors (exact_interpolation, the default) get H as the mean of the H before and
        # after the step, which is at the time of E (tests/test_validation.py checks arg(H / E) of a plane wave).
        return ph[:, :3], ph[:, 3:]

    def get_dft_array(self, mon, component: str, num_freq: int) -> np.ndarray:
        """The DFT field of one component on the first region of a monitor at frequency index num_freq (the cells
        of the plane, co-located on the E_z Yee point by fdtdx; the trivial axes are kept)."""
        self._need_run()
        if component not in _COMPONENTS:
            raise ValueError(f"unknown component {component!r}")
        E, H = self._phasors(mon, 0)
        k = _COMPONENTS.index(component)
        return (E if k < 3 else H)[num_freq, k % 3]

    def get_flux_data(self, mon) -> FluxData:
        E, H = zip(*(self._phasors(mon, j) for j in range(len(mon.regions))))
        return FluxData(list(E), list(H))

    def load_minus_flux_data(self, mon, data: FluxData):
        """Subtract the phasors of another run (the incident fields of a normalization run) before the flux."""
        mon.minus = data

    def get_fluxes(self, mon) -> list:
        """Net flux sum_regions weight x int Re(E x conj H) . n dA at each frequency (MIT package convention)."""
        self._need_run()
        total = np.zeros(len(mon.freqs))
        for j, r in enumerate(mon.regions):
            E, H = self._phasors(mon, j)
            if mon.minus is not None:
                E, H = E - mon.minus.E[j], H - mon.minus.H[j]
            a = mon.axes[j]
            b, c = (a + 1) % 3, (a + 2) % 3
            s = np.real(E[:, b] * np.conj(H[:, c]) - E[:, c] * np.conj(H[:, b]))
            s = s * self._trapezoid(s.shape[1:], a)
            total += r.weight * self._area(a) * s.reshape(len(mon.freqs), -1).sum(axis=1)
        return list(total)

    def get_eigenmode_coefficients(self, mon, bands=(1,), eig_parity: int = NO_PARITY) -> EigCoeffs:
        """Forward and backward amplitudes of the waveguide modes (bands start at 1) at each monitor frequency.
        |alpha|^2 is the flux of the mode (the same scale as get_fluxes)."""
        import jax

        if not mon.mode:
            raise ValueError("use add_mode_monitor for mode coefficients")
        a = mon.axes[0]
        fp = self._filter_pol(eig_parity, a)                    # before the run check: a bad parity raises
        self._need_run()
        det0 = self._objs[mon.names[0]]
        b, c = (a + 1) % 3, (a + 2) % 3
        E, H = self._phasors(mon, 0)
        w = self._area(a)
        alpha = np.zeros((len(bands), len(mon.freqs), 2), complex)
        neff = np.zeros((len(bands), len(mon.freqs)))
        with jax.enable_x64(False), jax.default_device(_device()):
            for ib, band in enumerate(bands):
                det = det0
                if band != 1 or fp != det0.filter_pol:
                    det = det0.aset("mode_index", int(band) - 1).aset("filter_pol", fp)
                    det = det.apply(key=jax.random.PRNGKey(0), inv_permittivities=self._arrays.inv_permittivities,
                                    inv_permeabilities=self._arrays.inv_permeabilities)
                Em = np.asarray(det._mode_E, np.complex128)
                Hm = np.asarray(det._mode_H, np.complex128)
                if Em.ndim == E.ndim - 1:
                    Em, Hm = Em[None], Hm[None]

                def cross_n(P, Q):
                    prod = (P[:, b] * Q[:, c] - P[:, c] * Q[:, b]) * self._trapezoid(P.shape[2:], a)
                    return prod.reshape(len(mon.freqs), -1).sum(axis=1) * w

                N = np.real(cross_n(Em, np.conj(Hm)))
                p1 = cross_n(E, np.conj(Hm))
                p2 = cross_n(np.conj(Em), H)
                alpha[ib, :, 0] = (p1 + p2) / (2 * np.sqrt(N))
                alpha[ib, :, 1] = (p1 - p2) / (2 * np.sqrt(N))
                neff[ib] = np.real(np.asarray(det._mode_neff, np.complex128)).reshape(-1)[: len(mon.freqs)]
        return EigCoeffs(alpha, neff)

    # ---------- near-to-far-field transform ----------
    def add_near2far(self, *args, nperiods: int = 1, **unsupported):
        """add_near2far(fcen, df, nfreq, Near2FarRegion, ...) or add_near2far(freqs, Near2FarRegion, ...): the DFT
        fields on the planes of the regions (as add_flux) for get_farfield and get_farfields. 2-D and 3-D cells."""
        if unsupported:
            raise NotImplementedError(f"add_near2far: unsupported arguments {sorted(unsupported)}")
        if nperiods != 1:
            raise NotImplementedError("nperiods (the sum over periodic copies) is not supported")
        if self.dims == 1:
            raise NotImplementedError("the near-to-far-field transform needs a 2-D or 3-D cell")
        freqs, regions = _freqs(args)
        for r in regions:                   # an explicit direction is not checked by _normal
            a = self._normal(r.size, r.direction)
            if a not in (0, 1, 2) or self.trivial[a]:
                raise ValueError(f"a Near2FarRegion needs a normal along an axis of size > 0 (direction {a})")
        mon = self._add_monitor(freqs, regions, mode=False, cls=DftNear2Far)
        mon.sim = self
        return mon

    def get_near2far_freqs(self, near2far) -> list:
        return list(near2far.freqs)

    def get_near2far_data(self, near2far) -> FluxData:
        """The DFT fields of the planes (for load_minus_near2far_data in a second run)."""
        return self.get_flux_data(near2far)

    def load_minus_near2far_data(self, near2far, data: FluxData):
        """Subtract the DFT fields of another run (for example the incident fields) before the transform."""
        self.load_minus_flux_data(near2far, data)

    def _colocated_points(self, rng) -> np.ndarray:
        """Coordinates (P, 3) of the DFT fields of the cells in the index ranges rng ((i0, i1) per axis), in the C order
        of the cells. fdtdx co-locates the six components on the E_z Yee point: the cell center, + dx / 2 along z. An
        axis of size 0 has the coordinate 0."""
        axes = [np.zeros(r[1] - r[0]) if self.trivial[a] else
                self._lo(a) + (np.arange(r[0], r[1]) + (1.0 if a == 2 else 0.5)) * self.dx for a, r in enumerate(rng)]
        grids = np.meshgrid(*axes, indexing="ij")
        return np.stack([g.reshape(-1) for g in grids], axis=-1)

    def _near2far_currents(self, n2f):
        """Positions (N, 3) and current elements J = n x H, M = -n x E (nfreq, N, 3; times the area of the cell, the
        trapezoidal weight of the plane and the region weight) of all surface cells. In 2-D the fields do not change
        along the axis of size 0: E and H are averaged over its TRIVIAL_CELLS cells, and each cell of the plane gives
        one line element of length dx (times the trapezoidal weight). Sets n2f.eps from the staircase material of
        the surface cells (a warning if they have more than one material: the far fields assume one homogeneous
        medium): the float eps of a lossless, non-dispersive medium, or for a dispersive or conductive medium its
        complex eps(f) at each frequency (Medium.permittivity; the Green's function of the lossy medium)."""
        self._need_run()
        pos, Js, Ms, eps, mids = [], [], [], [], []
        nf = len(n2f.freqs)
        triv = tuple(2 + q for q in range(3) if self.trivial[q])         # axes of E (nfreq, 3, nx, ny, nz)
        for j, r in enumerate(n2f.regions):
            E, H = self._phasors(n2f, j)
            if n2f.minus is not None:
                E, H = E - n2f.minus.E[j], H - n2f.minus.H[j]
            a = n2f.axes[j]
            b, c = (a + 1) % 3, (a + 2) % 3
            rng = [self._range(r.center, r.size, ax) for ax in range(3)]
            if triv:                                # the normal a is not an axis of size 0 (add_near2far checks it)
                E, H = E.mean(axis=triv, keepdims=True), H.mean(axis=triv, keepdims=True)
                rng = [(0, 1) if self.trivial[q] else rng[q] for q in range(3)]
            area = self._area(a) * TRIVIAL_CELLS ** len(triv)               # dx^(dims - 1)
            w = (r.weight * area * self._trapezoid(E.shape[2:], a)).reshape(-1)
            E, H = E.reshape(nf, 3, -1), H.reshape(nf, 3, -1)
            J, M = np.zeros((nf, E.shape[2], 3), complex), np.zeros((nf, E.shape[2], 3), complex)
            J[:, :, b], J[:, :, c] = -H[:, c] * w, H[:, b] * w             # e_a x H
            M[:, :, b], M[:, :, c] = E[:, c] * w, -E[:, b] * w             # -e_a x E
            pos.append(self._colocated_points(rng))
            Js.append(J)
            Ms.append(M)
            sl = tuple(slice(*q) for q in rng)
            eps.append(self.eps[sl].reshape(-1))
            mids.append(self._idx[sl].reshape(-1))
        eps, mids = np.concatenate(eps), np.concatenate(mids)
        lossy = [k for k in np.unique(mids) if self._mats[k].dispersive or self._mats[k].conductive]
        if not lossy:                                       # lossless, non-dispersive: as before
            vals, counts = np.unique(eps, return_counts=True)
            n2f.eps = float(vals[np.argmax(counts)])
            if len(vals) > 1:
                warnings.warn(f"the near2far surface crosses {len(vals)} materials (eps {vals.tolist()}); the far "
                              f"fields use the homogeneous medium eps = {n2f.eps:g} (the material of most surface "
                              f"cells)")
        else:                                               # the complex eps(f) of the medium at each frequency
            ks, counts = np.unique(mids, return_counts=True)
            med = self._mats[ks[np.argmax(counts)]]
            n2f.eps = (np.asarray(med.permittivity(n2f.freqs), complex) if med.dispersive or med.conductive
                       else med.epsilon)
            if len(ks) > 1:
                warnings.warn(f"the near2far surface crosses {len(ks)} materials ({[self._mats[k] for k in ks]}); "
                              f"the far fields use the homogeneous medium {med} (the material of most surface "
                              f"cells)")
        return np.concatenate(pos), np.concatenate(Js, axis=1), np.concatenate(Ms, axis=1)

    def get_farfields_at(self, near2far, points) -> dict:
        """The fields at each point of `points` (P, 3), for all frequencies: {"Ex": (P, nfreq), ..., "Hz": ...}. Not
        in the MIT package (it gives the points of get_farfields in one call). The points must be outside the
        near2far surface (inside a closed surface the result is close to zero)."""
        if not isinstance(near2far, DftNear2Far):
            raise ValueError("use the object of add_near2far")
        src, J, M = self._near2far_currents(near2far)
        triv = next(a for a in range(3) if self.trivial[a]) if self.dims == 2 else None
        E, H = radiate(points, src, J, M, near2far.freqs, eps=near2far.eps, mu=1.0, trivial_axis=triv,
                       device=_device() if triv is None else None)
        return {c: (E if k < 3 else H)[:, :, k % 3].T for k, c in enumerate(_COMPONENTS)}

    def get_farfield(self, near2far, x) -> list:
        """The fields at the point x (outside the near2far surface; it can be outside the cell): a list of 6 nfreq
        complex numbers, Ex, Ey, Ez, Hx, Hy, Hz of the first frequency, then of the second, and so on."""
        ff = self.get_farfields_at(near2far, np.array([tuple(_v3(x))]))
        return list(np.stack([ff[c][0] for c in _COMPONENTS], axis=1).reshape(-1))

    def _where(self, where, center, size) -> Volume:
        if where is not None:
            if center is not None or size is not None:
                raise ValueError("give where or center and size, not both")
            return where
        return Volume(center=Vector3() if center is None else center, size=Vector3() if size is None else size)

    def get_farfields(self, near2far, resolution: float, where=None, center=None, size=None) -> dict:
        """The fields on a grid of points in the volume `where` (or center and size): {"Ex": array, ..., "Hz": array}.
        Each axis of the cell with a size above 0 gets floor(size x resolution) points from the low end to the high end
        of the volume (both included; one point at the center if this is below 2). The arrays have the shape
        (points of each axis with more than one point, nfreq)."""
        vol = self._where(where, center, size)
        pts, shape, _ = farfield_grid(vol.center, vol.size, resolution, [a for a in range(3) if not self.trivial[a]])
        ff = self.get_farfields_at(near2far, pts)
        return {c: v.reshape(*shape, len(near2far.freqs)) for c, v in ff.items()}

    def _near2far_flux(self, n2f, direction, where, resolution):
        vol = self._where(where, None, None)
        a = int(direction)
        if a not in (0, 1, 2) or self.trivial[a]:
            raise ValueError(f"direction must be X, Y or Z, along an axis of size > 0 (got {direction})")
        pts, _, w = farfield_grid(vol.center, vol.size, resolution, [q for q in range(3) if not self.trivial[q]])
        ff = self.get_farfields_at(n2f, pts)
        b, c = "xyz"[(a + 1) % 3], "xyz"[(a + 2) % 3]
        s = np.real(ff["E" + b] * np.conj(ff["H" + c]) - ff["E" + c] * np.conj(ff["H" + b]))
        return list((w[:, None] * s).sum(axis=0))

    def _filter_pol(self, parity, normal):
        """fdtdx filter_pol ("te": E mostly along the first transverse axis, "tm": the second) for a mode plane.

        3-D: NO_PARITY (None: the mode of the largest index), EVEN_Z (E mostly in the plane normal to z) or ODD_Z
        (E mostly along z); a y parity is ignored. 2-D: only the mode with E along the trivial axis (Ez in an x-y
        cell: NO_PARITY or ODD_Z); an in-plane (Hz) mode raises NotImplementedError, because the fdtdx mode solver
        gives NaN fields or the wrong mode for it on the three periodic cells of the trivial axis. 1-D: raises."""
        t = [i for i in range(3) if i != normal]               # ascending transverse axes
        if self.dims == 1:
            raise NotImplementedError("mode sources and mode monitors need a 2-D or 3-D cell")
        if self.dims == 2:
            triv = next(i for i in range(3) if self.trivial[i])
            odd = {2: ODD_Z, 1: ODD_Y}.get(triv)
            if parity not in (None, NO_PARITY) and parity != odd:
                raise NotImplementedError(
                    f"2-D cell: only the mode with E along the trivial axis {'xyz'[triv]} is supported (NO_PARITY"
                    f"{' or ODD_' + 'xyz'[triv].upper() if odd else ''}); in-plane (Hz-like) modes give NaN or the "
                    f"wrong mode in the fdtdx mode solver")
            return "te" if triv == t[0] else "tm"
        if parity in (None, NO_PARITY) or not parity & (EVEN_Z | ODD_Z):
            return None
        if normal == 2:
            raise NotImplementedError("a z parity needs a plane that is not normal to z")
        target = 2 if parity & ODD_Z else next(i for i in t if i != 2)
        return "te" if target == t[0] else "tm"

    def _need_run(self):
        if not self._done:
            raise RuntimeError("call run() first")

    # ---------- run ----------
    def run(self, *step_funcs, until=None, until_after_sources=None):
        """run(until=T), run(until_after_sources=T) or with stop_when_fields_decayed(...) in place of T; Harminv
        objects as step functions. One run per Simulation."""
        if self._done:
            raise NotImplementedError("one run per Simulation (create a new Simulation)")
        if (until is None) == (until_after_sources is None):
            raise ValueError("give until or until_after_sources")
        for f in step_funcs:
            if not isinstance(f, Harminv):
                raise NotImplementedError(f"unsupported step function {f!r}: only Harminv")
        cond = until if until is not None else until_after_sources
        t_start = 0.0 if until is not None else self.source_end_time()
        if isinstance(cond, stop_when_fields_decayed):
            if math.isinf(t_start):
                raise ValueError("a continuous source never ends")
            t_max = t_start + cond.max_time
        else:
            t_max = t_start + float(cond)
        if math.isinf(t_max):
            raise ValueError("the run time is infinite")
        import jax

        with jax.enable_x64(False), jax.default_device(_device()):
            self._run(list(step_funcs), cond, t_start, t_max)
        return self

    def _build(self, n_steps, probes):
        import fdtdx
        import jax

        from fdtd_gpu._voxels import VoxelMaterial

        f_si = C0 / UM                     # 1/um -> Hz
        config = fdtdx.SimulationConfig(grid=fdtdx.UniformGrid(spacing=self.dx * UM), time=n_steps * self.dt * UM / C0,
                                        courant_factor=0.99, backend="cpu" if _device().platform == "cpu" else "gpu")
        volume = fdtdx.SimulationVolume(partial_grid_shape=self.shape,
                                        material=fdtdx.Material(permittivity=self.default_material.epsilon))
        objects, constraints = [volume], []

        def place(obj, lo):
            objects.append(obj)
            constraints.append(obj.set_grid_coordinates(axes=(0, 1, 2), sides=("-", "-", "-"), coordinates=tuple(lo)))

        idx, mats, eps3 = self._permittivity()
        self.eps = np.array([m.epsilon for m in mats])[idx]
        self.eps3 = eps3
        self._idx, self._mats = idx, mats          # the staircase material of each cell (near2far media)
        face, self._plain = None, np.ones(idx.shape, bool)     # _plain: cells without poles or conductivity
        lossy = _disp.lossy_media(self.default_material, self.geometry)
        if lossy:                                   # nothing of this for runs without dispersion or conductivity
            face = _disp.face_assignment(idx, mats, eps3, self._fills, lossy)
            self._plain = face.plain
            _disp.check_stability(face, self.dt, self.dx, self.dims)
        planes =[(s.center, s.size, "an EigenModeSource") for s in self.sources if isinstance(s, EigenModeSource)]
        planes += [(r.center, r.size, "a mode monitor") for m in self._monitors if m.mode for r in m.regions]
        for center, size, what in planes:
            if not self._plain[tuple(slice(*self._range(center, size, a)) for a in range(3))].all():
                raise NotImplementedError(f"{what} plane crosses a dispersive or conductive material (not "
                                          f"validated: the mode solver would see eps_inf only); make the plane "
                                          f"smaller")
        if eps3 is not None:
            place(VoxelMaterial.from_permittivity(eps3, self.default_material.epsilon), (0, 0, 0))
        elif len(mats) > 1:
            fmats = {f"m{k}": fdtdx.Material(permittivity=m.epsilon) for k, m in enumerate(mats)}
            place(VoxelMaterial.from_index(idx, fmats), (0, 0, 0))

        # boundaries
        kinds, thick, sigma_end, layers = {}, {}, {}, {}
        for a, ax in enumerate("xyz"):
            for side, sname in ((Low, "min"), (High, "max")):
                key = f"{sname}_{ax}"
                if self.trivial[a]:
                    kinds[key], thick[key] = "periodic", 1
                    continue
                pml = [p for p in self.boundary_layers if p.direction in (ALL, a) and p.side in (ALL, side)]
                if pml:
                    kinds[key], thick[key] = "pml", max(1, int(round(pml[-1].thickness * self.resolution)))
                    layers[a, side] = (pml[-1], thick[key])
                    if pml[-1].R_asymptotic is not None:          # fdtdx: sigma_end = -(m + 1) ln R / (2 eta0 d)
                        sigma_end[key] = (-(_disp.PML_SIGMA_ORDER + 1) * math.log(pml[-1].R_asymptotic)
                                          / (2 * _disp.ETA0 * thick[key] * self.dx * UM))
                else:
                    kinds[key], thick[key] = ("periodic" if self.periodic else "pec"), 1
        bc = fdtdx.BoundaryConfig(**{f"boundary_type_{k.split('_')[0]}{k.split('_')[1]}": v for k, v in kinds.items()},
                                  **{f"thickness_grid_{k.split('_')[0]}{k.split('_')[1]}": v for k, v in thick.items()},
                                  **{f"sigma_end_{k.split('_')[0]}{k.split('_')[1]}": v for k, v in sigma_end.items()})
        if lossy:                                   # a medium with a small Re(n) in the PML: warn (check 14)
            for msg in _disp.pml_warnings(idx, mats, layers, self.dx, self.sources):
                warnings.warn(msg)
        bdict, bcons = fdtdx.boundary_objects_from_config(bc, volume)
        objects += list(bdict.values())
        constraints += bcons

        # sources
        for k, s in enumerate(self.sources):
            src = s.src
            wc = fdtdx.WaveCharacter(frequency=src.frequency * f_si)
            if isinstance(src, GaussianSource):
                prof = fdtdx.GaussianPulseProfile(center_wave=wc, spectral_width=fdtdx.WaveCharacter(
                    frequency=src.fwidth / (2 * math.pi) * f_si))
            else:
                prof = fdtdx.SingleFrequencyProfile(phase_shift=0.0)
            rng = [self._range(s.center, s.size, a) for a in range(3)]
            gshape = tuple(r[1] - r[0] for r in rng)
            if isinstance(s, EigenModeSource):
                normal = self._normal(s.size, s.direction)
                direction = "+"
                if s.eig_kpoint is not None and s.eig_kpoint[normal] < 0:
                    direction = "-"
                obj = fdtdx.ModePlaneSource(name=f"src{k}", partial_grid_shape=gshape, wave_character=wc,
                                            temporal_profile=prof, direction=direction, mode_index=s.eig_band - 1,
                                            filter_pol=self._filter_pol(s.eig_parity, normal),
                                            static_amplitude_factor=s.amplitude)
            else:
                comp = _COMPONENTS.index(s.component)
                zero = sum(1 for a in range(3) if not self.trivial[a] and s.size[a] == 0)
                amp = s.amplitude * self.dx ** (1 - zero)       # E += -S inv_eps a: a = dx J, J = A / dx^zero
                obj = fdtdx.PointDipoleSource(name=f"src{k}", partial_grid_shape=gshape, wave_character=wc,
                                              temporal_profile=prof, polarization=comp % 3,
                                              source_type="electric" if comp < 3 else "magnetic", amplitude=amp)
            place(obj, [r[0] for r in rng])

        # monitors
        for mon in self._monitors:
            wcs = [fdtdx.WaveCharacter(frequency=f * f_si) for f in mon.freqs]
            for name, r, a in zip(mon.names, mon.regions, mon.axes):
                rng = [self._range(r.center, r.size, ax) for ax in range(3)]
                gshape = tuple(q[1] - q[0] for q in rng)
                if mon.mode:
                    det = fdtdx.ModeOverlapDetector(name=name, partial_grid_shape=gshape, wave_characters=wcs,
                                                    direction="+", mode_index=0, scaling_mode="pulse",
                                                    filter_pol=self._filter_pol(NO_PARITY, a))
                else:
                    det = fdtdx.PhasorDetector(name=name, partial_grid_shape=gshape, wave_characters=wcs,
                                               scaling_mode="pulse", plot=False)
                place(det, [q[0] for q in rng])
        for k, (c, pt) in enumerate(probes):
            rng = [self._range(pt, Vector3(), a) for a in range(3)]
            det = fdtdx.FieldDetector(name=f"probe{k}", partial_grid_shape=tuple(q[1] - q[0] for q in rng),
                                      components=(c,), reduce_volume=True, plot=False)
            place(det, [q[0] for q in rng])

        key = jax.random.PRNGKey(0)
        objs, arrays, params, config, _ = fdtdx.place_objects(object_list=objects, config=config,
                                                             constraints=constraints, key=key)
        arrays, objs, _ = fdtdx.apply_params(arrays, objs, params, key)
        if face is not None:              # after the sources took 1 / eps_inf: see fdtd_gpu._dispersion
            arrays = _disp.material_arrays(arrays, config, face)
            objs = _disp.correct_sources(objs, arrays, config)       # J / (1 + b) in conductive cells
        return objs, arrays, config, key

    def _run(self, harminvs, cond, t_start, t_max):
        import jax
        import jax.numpy as jnp
        from fdtdx.fdtd.fdtd import custom_fdtd_forward

        n_max = int(math.ceil(t_max / self.dt))
        probes = [(h.c, h.pt) for h in harminvs]
        decay = isinstance(cond, stop_when_fields_decayed)
        if decay:
            probes.append((cond.c, cond.pt))
        objs, arrays, config, key = self._build(n_max, probes)
        n_max = config.time_steps_total               # fdtdx rounds time / dt (its own float32 step)
        dt_si = config.time_step_duration
        if abs(dt_si * C0 / UM - self.dt) > 1e-4 * self.dt:
            raise RuntimeError(f"time step mismatch: fdtdx {dt_si * C0 / UM} against {self.dt}")
        self.dt = dt_si * C0 / UM          # the step of fdtdx (its spacing is float32: about 1e-6 different)

        @jax.jit
        def advance(arrs, s, e):
            return custom_fdtd_forward(arrs, objs, config, key, reset_container=False, record_detectors=True,
                                       start_time=s, end_time=e, show_progress=False)[1]

        arrays = arrays.reset()
        t0 = _time.perf_counter()
        if not decay:
            arrays = advance(arrays, jnp.int32(0), jnp.int32(n_max))
            n_done = n_max
        else:
            n0 = int(math.ceil(t_start / self.dt))
            win = max(1, int(round(cond.dt / self.dt)))
            if n0 > 0:
                arrays = advance(arrays, jnp.int32(0), jnp.int32(min(n0, n_max)))
            n_done, max_abs = min(n0, n_max), 0.0
            pname = f"probe{len(probes) - 1}"
            while n_done < n_max:
                e = min(n_done + win, n_max)
                arrays = advance(arrays, jnp.int32(n_done), jnp.int32(e))
                f = np.asarray(arrays.detector_states[pname]["fields"][n_done:e]).reshape(e - n_done, -1)[:, 0]
                cur = float(np.max(f.astype(np.float64) ** 2))
                n_done = e
                max_abs = max(max_abs, cur)
                if cur <= cond.decay_by * max_abs:
                    break
            else:
                warnings.warn(f"the fields did not decay by {cond.decay_by} within max_time {cond.max_time}")
        jax.block_until_ready(arrays.fields.E)
        self.run_seconds = _time.perf_counter() - t0
        self.meep_time = n_done * self.dt
        self.n_steps = n_done
        self._arrays, self._objs = arrays, {o.name: o for o in objs.objects}
        self.device = str(next(iter(arrays.fields.E.devices())))
        self._done = True
        n_src = int(math.ceil(self.source_end_time() / self.dt)) if not math.isinf(self.source_end_time()) else 0
        for k, h in enumerate(harminvs):
            sig = np.asarray(arrays.detector_states[f"probe{k}"]["fields"][:n_done], np.float64).reshape(n_done, -1)
            h.data, h.data_dt = sig[n_src:, 0], self.dt
            h.modes = harminv(h.data, self.dt, h.fcen - h.df / 2, h.fcen + h.df / 2, max_modes=h.mxbands)
