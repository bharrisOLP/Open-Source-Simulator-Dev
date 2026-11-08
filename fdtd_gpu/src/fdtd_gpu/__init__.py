"""fdtd_gpu: a small FDTD front end on fdtdx (MIT, JAX; AMD ROCm GPUs first), in the script style of the MIT FDTD
package. Units: um, frequency in 1/um, c = 1. See README.md for the supported subset and the differences."""

from fdtd_gpu.geometry import Block, Cylinder, Medium, Prism, Sphere, Vector3, air, inf, vacuum
from fdtd_gpu.harminv import HarminvMode, harminv
from fdtd_gpu.near2far import Volume
from fdtd_gpu.simulation import (PML, DftNear2Far, EigCoeffs, FluxData, FluxRegion, Harminv, High, Low, ModeRegion,
                                 Near2FarRegion, Simulation, after_sources, stop_when_fields_decayed)
from fdtd_gpu.sources import (ALL, AUTOMATIC, EVEN_Y, EVEN_Z, NO_PARITY, ODD_Y, ODD_Z, TE, TM, X, Y, Z,
                              ContinuousSource, EigenModeSource, Ex, Ey, Ez, GaussianSource, Hx, Hy, Hz, Source)

__all__ = [
    "ALL", "AUTOMATIC", "EVEN_Y", "EVEN_Z", "NO_PARITY", "ODD_Y", "ODD_Z", "PML", "TE", "TM", "X", "Y", "Z",
    "Block", "ContinuousSource", "Cylinder", "DftNear2Far", "EigCoeffs", "EigenModeSource", "Ex", "Ey", "Ez",
    "FluxData", "FluxRegion", "GaussianSource", "Harminv", "HarminvMode", "High", "Hx", "Hy", "Hz", "Low", "Medium",
    "ModeRegion", "Near2FarRegion", "Prism", "Simulation", "Source", "Sphere", "Vector3", "Volume", "after_sources",
    "air", "harminv", "inf", "stop_when_fields_decayed", "vacuum",
]
