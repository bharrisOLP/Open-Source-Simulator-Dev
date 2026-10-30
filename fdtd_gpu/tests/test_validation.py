"""FDTD checks against independent references (the cases are in cases.py; larger grids in bench/bench_fdtd.py).
Each case runs fdtdx on the GPU in a few seconds."""

import numpy as np
import pytest

import cases

pytestmark = pytest.mark.gpu
NEFF_TOL = 1.5e-2   # mode solver of fdtdx (tidy3d) at 0.05 um against the exact slab: +0.0083 to +0.0086 (res 20)
SHEET_TOL20 = 3e-2


def test_bragg_mirror_against_transfer_matrix():
    """1-D quarter-wave stack, 6 pairs: R(f) from flux planes (incident fields subtracted, as in the MIT package)
    against the exact transfer-matrix reflectance; R + T = 1 (lossless)."""
    r = cases.bragg(resolution=80)
    assert r["device"].startswith(("rocm", "cuda", "gpu"))
    assert r["max_dR"] < 2e-2, r["max_dR"]          # the numerical dispersion of the Yee grid (bench: res 200)
    assert r["max_RT"] < 1e-3, r["max_RT"]


@pytest.mark.parametrize("three_d", [False, True])
def test_straight_waveguide_transmits_one(three_d):
    """Mode source, mode monitors and flux planes on a straight strip: T = 1 (mode overlap and flux)."""
    r = cases.waveguide(resolution=20, three_d=three_d)
    tol = 3e-3 if three_d else 1e-3                          # 3-D: 4.4 cells across the core, 2 x 2 um planes
    assert np.max(np.abs(r["T_mode"] - 1)) < tol, r["T_mode"]
    # T_flux: in 3-D the mode source (solved at the center frequency only) also launches some radiation at the band
    # edges, which crosses the input plane but leaves before the output plane (0.983 at 0.57 / um): check the center
    t_flux = r["T_flux"][len(r["T_flux"]) // 2:len(r["T_flux"]) // 2 + 1] if three_d else r["T_flux"]
    assert np.max(np.abs(t_flux - 1)) < 3 * tol, r["T_flux"]
    assert np.max(r["back_in"]) < 3e-3, r["back_in"]          # false reflection of the discrete overlap (Yee)
    # |alpha|^2 has the scale of the flux; in 3-D (4.4 cells across the core) it is 3 to 5 % below the input flux
    assert np.max(np.abs(r["mode_fraction_in"] - 1)) < (6e-2 if three_d else 1e-2), r["mode_fraction_in"]
    if not three_d:                                           # neff of the solver against the exact slab mode
        exact = np.array([cases.slab_te_neff(f, 3.45, 1.45, 0.4) for f in r["freqs"]])
        assert np.max(np.abs(r["neff"] - exact)) < NEFF_TOL, (r["neff"], exact)
    else:
        assert np.all((r["neff"] > 1.45) & (r["neff"] < 3.45)), r["neff"]


def test_ring_resonances_against_bessel_solution():
    """2-D ring, Ez: the three strongest Harminv resonances against the exact resonances of the dielectric ring
    (Bessel and Hankel functions); the free spectral range."""
    r = cases.ring(resolution=60)
    strong = sorted(r["modes"], key=lambda m: -m["amp"])[:3]
    assert len(strong) == 3
    for m in strong:
        assert abs(m["rel_err"]) < 5e-3, m
    ms = sorted(m["m"] for m in strong)
    assert ms == list(range(ms[0], ms[0] + 3))                  # three neighbor azimuthal orders
    rows = {m["m"]: m for m in strong}
    fsr = (rows[ms[2]]["f"] - rows[ms[0]]["f"]) / 2
    fsr_exact = (rows[ms[2]]["f_exact"] - rows[ms[0]]["f_exact"]) / 2
    assert abs(fsr / fsr_exact - 1) < 2e-2, (fsr, fsr_exact)


def test_dipole_power_purcell_factor_one():
    """3-D point dipole in a homogeneous medium: flux through a closed box / analytic radiated power = 1, with an
    error of second order in the cell size (vacuum, res 10 and 20) and below 1 % in n = 1.5 at res 20."""
    e10 = cases.dipole(resolution=10, n=1.0, half=1.0)["purcell"] - 1
    e20 = cases.dipole(resolution=20, n=1.0, half=1.0)["purcell"] - 1
    assert np.max(np.abs(e20)) < 3e-3, e20
    assert np.all(np.log2(np.abs(e10) / np.abs(e20)) > 1.7), (e10, e20)
    r = cases.dipole(resolution=20, n=1.5, half=1.0)
    assert np.max(np.abs(r["purcell"] - 1)) < 1e-2, r["purcell"]


def test_current_sheet_flux_and_phase():
    """1-D current sheet in vacuum: the plane-wave flux against the exact |I|^2 / 4 per side, E and H in phase (the
    DFT fields are at one time), and the flux error of the Yee grid falls at second order (res 20 to 40)."""
    r20, r40 = cases.sheet(resolution=20), cases.sheet(resolution=40)
    for r in (r20, r40):
        assert np.max(np.abs(r["phase_H_E"])) < 1e-3, r["phase_H_E"]
    e20, e40 = r20["flux_ratio"] - 1, r40["flux_ratio"] - 1
    assert np.max(np.abs(e20)) < SHEET_TOL20, e20
    assert np.all(np.abs(e40) < 0.9 * np.abs(e20)), (e20, e40)                # falls everywhere
    assert np.log2(abs(e20[-1]) / abs(e40[-1])) > 1.8, (e20, e40)          # second order at the top frequency
