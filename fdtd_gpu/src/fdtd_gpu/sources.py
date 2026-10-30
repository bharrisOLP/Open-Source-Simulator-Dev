"""Sources in the style of the MIT FDTD package. Units: frequency in 1/um (c = 1), time in um/c."""

from __future__ import annotations

import math

import numpy as np

from fdtd_gpu.geometry import Vector3, _v3

# field components (as in the MIT package: Ex, Ey, Ez, Hx, Hy, Hz)
Ex, Ey, Ez, Hx, Hy, Hz = "Ex", "Ey", "Ez", "Hx", "Hy", "Hz"
# directions and parities (the values of the MIT package)
X, Y, Z, ALL, AUTOMATIC = 0, 1, 2, -1, -1
NO_PARITY, EVEN_Z, ODD_Z, EVEN_Y, ODD_Y = 0, 1, 2, 4, 8
TE, TM = EVEN_Z, ODD_Z


class GaussianSource:
    """A Gaussian pulse: frequency (center, 1/um) and fwidth (1/um).

    The current is exp(-(t - t0)^2 / (2 w^2)) cos(2 pi f t) (fdtdx) with w = 1 / fwidth, as in the MIT package
    (its spectrum has the standard deviation fwidth / (2 pi)). Difference: the peak is at t0 = 6 w (fdtdx), not
    at 5 w; the source ends at 12 w.
    """

    def __init__(self, frequency: float | None = None, fwidth: float | None = None, wavelength: float | None = None,
                 width: float | None = None, is_integrated: bool = False, **unsupported):
        if unsupported:
            raise NotImplementedError(f"GaussianSource: unsupported arguments {sorted(unsupported)}")
        if is_integrated:
            raise NotImplementedError("is_integrated is not supported")
        if (frequency is None) == (wavelength is None):
            raise ValueError("give frequency or wavelength")
        self.frequency = float(frequency if frequency is not None else 1.0 / wavelength)
        if (fwidth is None) == (width is None):
            raise ValueError("give fwidth or width")
        self.fwidth = float(fwidth if fwidth is not None else 1.0 / width)
        if not 0 < self.fwidth:
            raise ValueError("fwidth must be positive")

    @property
    def width(self) -> float:
        return 1.0 / self.fwidth

    @property
    def peak_time(self) -> float:
        return 6.0 * self.width

    @property
    def end_time(self) -> float:
        return 12.0 * self.width

    def fourier_transform(self, f):
        """The spectrum (1 / sqrt(2 pi)) int s(t) exp(i 2 pi f t) dt of the current s(t), for f > 0; the term of the
        negative frequency is included. The DFT monitors use the same convention, so a flux divided by
        |fourier_transform|^2 is a response to a unit current."""
        f = np.asarray(f, float)
        w, f0, t0 = self.width, self.frequency, self.peak_time
        om, om0 = 2 * np.pi * f, 2 * np.pi * f0
        phase = np.exp(1j * om * t0)                     # carrier cos(om0 t): the two terms have om0 t0 phases
        return 0.5 * w * phase * (np.exp(-1j * om0 * t0 - 0.5 * w**2 * (om - om0) ** 2)
                                  + np.exp(1j * om0 * t0 - 0.5 * w**2 * (om + om0) ** 2))


class ContinuousSource:
    """A continuous wave cos(2 pi f t) from t = 0 (no smooth turn-on, unlike the MIT package)."""

    def __init__(self, frequency: float | None = None, wavelength: float | None = None, **unsupported):
        if unsupported:
            raise NotImplementedError(f"ContinuousSource: unsupported arguments {sorted(unsupported)}")
        if (frequency is None) == (wavelength is None):
            raise ValueError("give frequency or wavelength")
        self.frequency = float(frequency if frequency is not None else 1.0 / wavelength)
        self.fwidth = 0.0
        self.end_time = math.inf


class Source:
    """A current source J (electric) or M (magnetic) of one component, uniform over its region.

    size 0 on an axis is a delta function along that axis (as in the MIT package: a point source has the dipole
    moment current I l = amplitude x the source signal, independent of the resolution). An extended source
    radiates in both directions.
    """

    def __init__(self, src, component: str, center=Vector3(), size=Vector3(), amplitude: complex = 1.0,
                 **unsupported):
        if unsupported:
            raise NotImplementedError(f"Source: unsupported arguments {sorted(unsupported)}")
        if component not in (Ex, Ey, Ez, Hx, Hy, Hz):
            raise ValueError(f"unknown component {component!r}")
        if np.iscomplexobj(amplitude) and np.imag(amplitude) != 0:
            raise NotImplementedError("complex amplitudes are not supported (real fields)")
        self.src, self.component = src, component
        self.center, self.size, self.amplitude = _v3(center), _v3(size), float(np.real(amplitude))


class EigenModeSource:
    """A waveguide-mode source on a plane (size 0 along the normal), one direction (TFSF).

    eig_band starts at 1 (the mode of the largest effective index of the parity). eig_parity: NO_PARITY, EVEN_Z
    (TE-like: E mostly in the plane normal to z) or ODD_Z (TM-like: E mostly along z); a y parity is not used.
    The direction is +normal, or -normal when eig_kpoint points along -normal. The mode is solved at the center
    frequency of src (the tidy3d mode solver inside fdtdx). The amplitude scale differs from the MIT package:
    normalize by an input monitor or a reference run.
    """

    def __init__(self, src, center=Vector3(), size=Vector3(), eig_band: int = 1, eig_parity: int = NO_PARITY,
                 direction: int = AUTOMATIC, eig_kpoint=None, amplitude: float = 1.0, eig_match_freq: bool = True,
                 **unsupported):
        if unsupported:
            raise NotImplementedError(f"EigenModeSource: unsupported arguments {sorted(unsupported)}")
        if not eig_match_freq:
            raise NotImplementedError("eig_match_freq=False is not supported")
        if eig_band < 1:
            raise ValueError("eig_band starts at 1")
        self.src, self.center, self.size = src, _v3(center), _v3(size)
        self.eig_band, self.eig_parity, self.direction = int(eig_band), int(eig_parity), direction
        self.eig_kpoint = None if eig_kpoint is None else _v3(eig_kpoint)
        self.amplitude = float(amplitude)
