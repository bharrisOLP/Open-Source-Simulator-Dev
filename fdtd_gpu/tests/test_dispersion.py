"""Dispersive (Lorentz, Drude) and conductive media: the units and the ADE coefficients against the references (no
FDTD run), and FDTD checks 10 to 15 (a dispersive film against the transfer matrix and the exact Yee grid; the
uniform field: free oscillation and conductive decay; a dipole in a conductive medium; a half-space through the
PML; the near-to-far-field transform in a dispersive or conductive medium)."""

import math
import warnings

import numpy as np
import pytest

import cases
import fdtd_gpu as fg
from fdtd_gpu._dispersion import C0, UM, fdtdx_poles
from references import (chi_exact, chi_yee, dipole_flux_3d, dipole_flux_sphere, dipole_flux_surface, eps_conductive,
                        tmm_rt, uniform_root, yee_halfspace, yee_rt)

POLES = [("lorentz", 1.0, 0.05, 0.5), ("drude", 1.0, 0.05, 1.0), ("lorentz", 0.3, 0.0, 2.0), ("drude", 7.3, 0.04, 0.76)]
YEE_TOL = 5e-5    # FDTD against the exact Yee grid, 0.6 to 1.4 / um (logged rows: up to 2.6e-5 at res 100 and 200)
F = np.linspace(0.2, 2.0, 7)


@pytest.mark.parametrize("pole", POLES)
def test_permittivity_units_and_fdtdx_poles(pole):
    """Medium.permittivity (the MIT package's form) against chi_exact, and the fdtdx poles (SI units) of the same
    Medium against it: the mapping of frequency, gamma and sigma."""
    med = cases.dispersive_medium(2.0, pole)
    ref = 2.0 + chi_exact(F, pole)
    assert np.allclose(med.permittivity(F), ref, rtol=1e-13, atol=0)
    from fdtdx import DispersionModel
    model = DispersionModel(poles=fdtdx_poles(med))
    got = np.array([model.permittivity(2 * math.pi * f * C0 / UM, eps_inf=2.0) for f in F])
    assert np.allclose(got, ref, rtol=1e-12, atol=0)
    assert med.dispersive and not med.conductive


def test_conductivity_permittivity():
    med = fg.Medium(index=1.5, D_conductivity=0.3)
    assert np.allclose(med.permittivity(F), 2.25 * (1 + 0.3j / (2 * np.pi * F)), rtol=1e-14)
    assert med.conductive and not med.dispersive and med.epsilon == 2.25


@pytest.mark.parametrize("pole", POLES)
def test_ade_coefficients_give_the_discrete_chi(pole):
    """The recurrence p^(n+1) = c1 p^n + c2 p^(n-1) + c3 E^n with the coefficients of fdtdx gives chi = c3 / (s - c1
    - c2 / s), s = exp(-i w dt): equal to chi_yee (written from the central differences, not from fdtdx); chi_yee
    tends to chi_exact at second order in dt."""
    from fdtdx.dispersion import compute_pole_coefficients_per_axis
    dt = 0.99 * 0.01 / math.sqrt(3)                       # um/c (res 100)
    c1, c2, c3 = (c[0, 0] for c in compute_pole_coefficients_per_axis(fdtdx_poles(cases.dispersive_medium(1.0, pole)),
                                                                       dt * UM / C0))
    s = np.exp(-2j * np.pi * F * dt)
    assert np.allclose(c3 / (s - c1 - c2 / s), chi_yee(F, pole, dt), rtol=1e-9, atol=0)
    from fdtd_gpu._dispersion import float32_coefficients          # the coefficients that fdtd_gpu gives fdtdx
    f1, f2, f3 = (float(q) for q in float32_coefficients(c1, c2, c3))
    assert np.allclose(f3 / (s - f1 - f2 / s), chi_yee(F, pole, dt, float32=True), rtol=1e-6, atol=0)
    e1 = np.abs(chi_yee(F, pole, dt) - chi_exact(F, pole))
    e2 = np.abs(chi_yee(F, pole, dt / 2) - chi_exact(F, pole))
    assert np.all(np.log2(e1 / e2) > 1.9), np.log2(e1 / e2)


def test_yee_reference_converges_to_transfer_matrix():
    """references.yee_rt (the exact 1-D Yee grid) tends to the transfer matrix at second order in dx (lossy Drude film
    and Lorentz slab with the exact chi on the grid); vacuum nodes transmit 1."""
    for kind in ("drude", "lorentz"):
        p = cases.FILM[kind]
        R0, T0 = tmm_rt(F, [lambda f: np.sqrt(p["eps_inf"] + chi_exact(f, p["pole"]))], [p["d"]])
        errs = []
        for res in (200, 400, 800):
            dx = 1.0 / res
            dt = 0.99 * dx / math.sqrt(3)
            R, T = yee_rt(F, lambda f: np.full(int(round(p["d"] * res)), p["eps_inf"] + chi_yee(f, p["pole"], dt)), dx,
                          dt)
            errs.append(max(np.max(np.abs(R - R0)), np.max(np.abs(T - T0))))
        assert errs[-1] < 1e-3 and np.log2(errs[0] / errs[1]) > 1.8 and np.log2(errs[1] / errs[2]) > 1.8, errs
    R, T = yee_rt(F, lambda f: np.ones(7, complex), 0.01, 0.005)
    assert np.allclose(R, 0, atol=1e-20) and np.allclose(T, 1, rtol=1e-12)


@pytest.mark.parametrize("pole", [("drude", 0.5, 0.02, 1.0), ("lorentz", 0.5, 0.02, 0.5)])
def test_uniform_root(pole):
    """The discrete root (chi_yee) tends to the exact root of eps_inf + chi = 0, and the exact root solves it."""
    eps_inf = 1.0 if pole[0] == "drude" else 2.25
    z = uniform_root(pole, eps_inf)
    assert abs(eps_inf + chi_exact(z, pole)) < 1e-12
    e1 = abs(uniform_root(pole, eps_inf, dt=0.02) - z)
    e2 = abs(uniform_root(pole, eps_inf, dt=0.01) - z)
    assert 3.8 < e1 / e2 < 4.2, (e1, e2)


def test_medium_errors():
    lor = fg.LorentzianSusceptibility(frequency=1.0, gamma=0.1, sigma=1.0)
    with pytest.raises(NotImplementedError):
        fg.Medium(epsilon=2.0, E_susceptibilities=[lor], D_conductivity=0.1)
    with pytest.raises(NotImplementedError):
        fg.LorentzianSusceptibility(frequency=1.0, sigma=-1.0)
    with pytest.raises(NotImplementedError):
        fg.Medium(D_conductivity=-0.1)
    with pytest.raises(NotImplementedError):
        fg.DrudeSusceptibility(frequency=1.0, sigma_diag=fg.Vector3(1, 1, 2))
    with pytest.raises(NotImplementedError):
        fg.DrudeSusceptibility(frequency=1.0, sigma=1.0, sigma_offdiag=fg.Vector3(0, 0.1, 0))
    with pytest.raises(NotImplementedError):
        fg.DrudeSusceptibility(frequency=1.0, sigma=1.0, noise_amp=0.1)
    with pytest.raises(ValueError):
        fg.LorentzianSusceptibility(frequency=0.0, sigma=1.0)
    with pytest.raises(NotImplementedError):
        fg.Medium(E_susceptibilities=["not a susceptibility"])
    assert fg.DrudeSusceptibility(frequency=2.0, gamma=0.1, sigma_diag=fg.Vector3(3, 3, 3)).sigma == 3.0
    assert not fg.Medium(E_susceptibilities=[fg.DrudeSusceptibility(frequency=1.0, sigma=0.0)]).dispersive


def test_float32_coefficients_keep_the_sum():
    """float32 ADE coefficients: c1 + c2 keeps its float64 value to 6e-8 (it sets the resonance), and a Drude pole
    (c1 + c2 = 1) keeps the sum at exactly 1 (no growth of a constant polarization) at the resolutions 20 to 1600."""
    from fdtd_gpu._dispersion import float32_coefficients
    for res in (20, 50, 100, 200, 400, 800, 1600):
        dt = 0.99 / res / math.sqrt(3)
        for w0, g in ((0.0, 2 * np.pi * 0.05), (0.0, 2 * np.pi * 1e-3), (2 * np.pi, 2 * np.pi * 0.05),
                      (2 * np.pi, 0.0)):
            d = 1 + g * dt / 2
            c1, c2 = (2 - w0**2 * dt**2) / d, -(1 - g * dt / 2) / d
            f1, f2, _ = float32_coefficients(c1, c2, 1.0)
            s = float(f1) + float(f2)
            assert s <= 1.0 and abs(s - (c1 + c2)) <= 6.0e-8, (res, w0, g, s - (c1 + c2))
            if w0 == 0:
                assert s == 1.0


def test_dipole_flux_references():
    """The flux of the exact dipole fields through a closed surface (references.dipole_flux_surface): through a
    sphere in a lossy medium (n = 1.2 + 0.1i) equal to the closed form dipole_flux_sphere; through the box of
    cases.box_rule (Gauss-Legendre) in a lossless medium equal to dipole_flux_3d (the flux of a closed surface does
    not depend on it); the closed form for a real n equal to dipole_flux_3d at each radius. In the lossy medium a
    larger sphere has less flux (the loss between the two radii)."""
    f = np.array([0.4, 0.6])
    r0 = np.array([0.02, -0.01, 0.03])
    u, w = cases.sphere_rule(24, 48)
    for r in (1.0, 1.3):
        q = dipole_flux_surface(r0 + r * u, u, w * r**2, f, 1.2 + 0.1j, 0.7, r0)
        assert np.allclose(q, dipole_flux_sphere(f, 1.2 + 0.1j, 0.7, r), rtol=1e-10, atol=0), r
    assert np.all(dipole_flux_sphere(f, 1.2 + 0.1j, 0.7, 1.3) < 0.95 * dipole_flux_sphere(f, 1.2 + 0.1j, 0.7, 1.0))
    pts, nrm, wb = cases.box_rule(r0 - [1.0, 0.9, 1.1], r0 + [1.0, 1.2, 0.8])
    assert np.allclose(dipole_flux_surface(pts, nrm, wb, f, 1.3, 0.7, r0), dipole_flux_3d(f, 1.3, 0.7), rtol=1e-9)
    for r in (0.5, 3.0):
        assert np.allclose(dipole_flux_sphere(f, 1.3, 0.7, r), dipole_flux_3d(f, 1.3, 0.7), rtol=1e-12)


def test_yee_halfspace_reference():
    """references.yee_halfspace: a vacuum half-space reflects nothing (theta_m = theta); a lossy half-space equals
    yee_rt of a thick slab of the same nodes (the wave dies in it); R tends to the Fresnel reflectance at second
    order in dx (Lorentz eps with chi_yee); eps_conductive of the semi-implicit update tends to the exact eps at
    second order in dt."""
    dx, dt = 0.02, 0.99 * 0.02 / math.sqrt(3)
    R, tm = yee_halfspace(F, lambda f: 1.0, dx, dt)
    q = 2 * dx / dt * np.sin(np.pi * F * dt)
    assert np.allclose(R, 0, atol=1e-24) and np.allclose(tm, np.arccos(1 - q**2 / 2), rtol=1e-12)
    eps = lambda f: 2.25 + 0.8j                                                        # noqa: E731
    R_slab, _ = yee_rt(F, lambda f: np.full(4000, eps(f)), dx, dt)          # 80 um: exp(-52) at 0.2 / um
    assert np.allclose(yee_halfspace(F, eps, dx, dt)[0], R_slab, rtol=1e-10, atol=0)
    p = cases.HALF["lorentz"]
    n = np.sqrt(cases.medium_eps(p, F))
    fresnel = np.abs((1 - n) / (1 + n)) ** 2
    errs = []
    for res in (100, 200, 400):
        dx = 1.0 / res
        dt = 0.99 * dx / math.sqrt(3)
        errs.append(np.max(np.abs(yee_halfspace(F, lambda f: cases.medium_eps(p, f, dt), dx, dt)[0] - fresnel)))
    assert np.log2(errs[0] / errs[1]) > 1.9 and np.log2(errs[1] / errs[2]) > 1.9, errs
    e1 = np.abs(eps_conductive(F, 2.25, 0.2, 0.02) - eps_conductive(F, 2.25, 0.2))
    e2 = np.abs(eps_conductive(F, 2.25, 0.2, 0.01) - eps_conductive(F, 2.25, 0.2))
    assert np.all(np.abs(np.log2(e1 / e2) - 2) < 0.01), np.log2(e1 / e2)


@pytest.mark.parametrize("rule", ["fill", "average", "staircase"])
def test_face_rules_in_a_face_cell(rule, monkeypatch):
    """The three face rules in the cells of a 1-D cell (res 10, Ex) next to a Lorentz block and a conductive block
    whose faces fill 1.5 % of a cell. "fill": eps_inf averaged, the pole weight and the conductivity times the fill
    fraction; "average": eps_inf averaged, the poles and the conductivity of the cell center; "staircase": the
    staircase eps_inf next to the lossy blocks. A plain dielectric block keeps the average with each rule."""
    import fdtd_gpu._dispersion as disp
    monkeypatch.setattr(disp, "FACE_RULE", rule)
    lor = cases.dispersive_medium(2.25, ("lorentz", 1.0, 0.05, 0.5))
    lossy = fg.Medium(epsilon=4.0, D_conductivity=0.1)
    geo = [fg.Block(size=fg.Vector3(fg.inf, fg.inf, 0.203), center=fg.Vector3(0, 0, -0.5), material=lor),
           fg.Block(size=fg.Vector3(fg.inf, fg.inf, 0.203), center=fg.Vector3(0, 0, 0.0), material=fg.Medium(4.0)),
           fg.Block(size=fg.Vector3(fg.inf, fg.inf, 0.203), center=fg.Vector3(0, 0, 0.5), material=lossy)]
    sim = fg.Simulation(cell_size=fg.Vector3(0, 0, 2), resolution=10, geometry=geo)
    idx, mats, eps3 = sim._permittivity()
    face = disp.face_assignment(idx, mats, eps3, sim._fills, disp.lossy_media(sim.default_material, geo))
    zc = -1 + (np.arange(20) + 0.5) * 0.1
    k = {z: int(np.argmin(np.abs(zc - z))) for z in (-0.65, -0.55, -0.15, 0.35, 0.45)}

    def at(a, z):                                     # Ex of the cell at z (broadcast over the components)
        return np.broadcast_to(a, (3, *idx.shape))[0, 0, 0, k[z]]

    f = 0.015                                         # the faces at +-0.1015 from the block centers
    assert at(face.dom, -0.55) == 0 and at(face.w, -0.55) == 1 and at(face.sigma, 0.45) == pytest.approx(0.4)
    assert at(face.eps, -0.15) == pytest.approx(1 + 3 * f, abs=2e-3)               # the plain block: averaged
    if rule == "fill":
        assert at(face.dom, -0.65) == 0 and at(face.w, -0.65) == pytest.approx(f, abs=1e-3)
        assert at(face.eps, -0.65) == pytest.approx(1 + 1.25 * f, abs=2e-3)
        assert at(face.sigma, 0.35) == pytest.approx(0.4 * f, abs=4e-4)
        assert eps3.shape[0] == 3 and face.dom.shape[0] == 3
    else:
        assert at(face.dom, -0.65) == -1 and at(face.w, -0.65) == 0
        eps_face = 1.0 if rule == "staircase" else 1 + 1.25 * f
        assert at(face.eps, -0.65) == pytest.approx(eps_face, abs=2e-3)
        assert at(face.sigma, 0.35) == 0
        assert face.dom.shape[0] == 1                 # one set of poles per cell: C = 1 in the ADE arrays


def test_stability_check_uses_the_cell_values():
    """check_stability compares each component with its own pole weight w and eps_inf: a face cell with half the
    pole strength but a low averaged eps_inf can break the bound where the bulk does not."""
    import fdtd_gpu._dispersion as disp
    dx = 0.1
    dt = 0.99 * dx / math.sqrt(3)
    s = (dt / dx) ** 2                                # 1-D
    wp = 2 * math.sqrt(3.0) / dt                      # v = (wp dt / 2)^2 = 3
    med = fg.Medium(epsilon=4.0, E_susceptibilities=[fg.DrudeSusceptibility(frequency=wp / (2 * math.pi),
                                                                            sigma=1.0)])
    assert disp.stability_margin(med, dt, dx, 1)[0] == pytest.approx(3.0) and 3.0 < 4.0 - s

    def face(eps_face):
        return disp.Face([med], np.zeros((1, 2, 1, 1), int), np.array([[[[1.0]], [[0.5]]]], np.float32),
                         np.array([[[[4.0]], [[eps_face]]]]), np.zeros((1, 2, 1, 1)), np.zeros((2, 1, 1), bool))

    disp.check_stability(face(2.0), dt, dx, 1)        # 1.5 < 2 - s
    with pytest.raises(ValueError, match="weight w = 0.5"):
        disp.check_stability(face(1.5), dt, dx, 1)    # 1.5 > 1.5 - s


def test_pml_round_trip_and_warning_rule():
    """_dispersion.pml_round_trip is R in vacuum and R^Re(n) exp(-4 pi f Im(n) d) in a medium; source_band ends where
    the spectrum of a GaussianSource is 1e-2 of its peak; pml_warnings warns for the Drude medium of check 14 in a
    PML of 0.5 um (default R) and not with R_asymptotic 1e-30, not for a medium outside the PML, not for a lossless
    medium in the PML."""
    import fdtd_gpu._dispersion as disp
    f = np.array([0.8, 1.0, 1.2])
    assert np.allclose(disp.pml_round_trip(fg.Medium(), f, 1.0), 1e-6, rtol=1e-12)
    assert np.allclose(disp.pml_round_trip(fg.Medium(), f, 1.0, R=1e-15), 1e-15, rtol=1e-12)
    drude = cases.dispersive_medium(1.0, cases.HALF["drude"]["pole"])
    n = np.sqrt(cases.medium_eps(cases.HALF["drude"], f))
    ref = 1e-6 ** n.real * np.exp(-4 * np.pi * f * n.imag * 0.5)
    assert np.allclose(disp.pml_round_trip(drude, f, 0.5), ref, rtol=1e-12)
    gs = fg.GaussianSource(1.0, fwidth=1.6)
    band = disp.source_band([fg.Source(gs, fg.Ex)])
    rel = np.abs(gs.fourier_transform(band[[0, -1]])) / np.abs(gs.fourier_transform(1.0))
    assert np.allclose(rel, 1e-2, rtol=2e-2), rel
    assert np.allclose(disp.source_band([fg.Source(fg.ContinuousSource(0.7), fg.Ex)]), [0.7])
    idx = np.zeros((3, 3, 20), int)
    idx[..., 12:] = 1                                       # the Drude medium fills the last 8 cells
    srcs = [fg.Source(gs, fg.Ex)]
    msgs = disp.pml_warnings(idx, [fg.Medium(), drude], {(2, 0): (fg.PML(0.5), 5)}, 0.1, srcs)
    assert len(msgs) == 1 and "high z side" in msgs[0], msgs
    assert disp.pml_warnings(idx, [fg.Medium(), drude], {(2, 0): (fg.PML(0.5, R_asymptotic=1e-30), 5)}, 0.1,
                             srcs) == []
    assert disp.pml_warnings(idx, [fg.Medium(), drude], {(2, 1): (fg.PML(0.5), 5)}, 0.1, srcs) == []
    assert disp.pml_warnings(idx, [fg.Medium(), fg.Medium(index=0.2)], {(2, 0): (fg.PML(0.5), 5)}, 0.1, srcs) == []


# ---------- FDTD (GPU) ----------
@pytest.mark.gpu
def test_drude_film_against_fresnel_and_the_yee_grid():
    """Check 10: a Drude film (plasma edge at 1 / um, 0.3 um, faces on cell faces) in vacuum. R and T against the
    exact Yee grid (the ADE and the leapfrog: only float32 and the end of the DFT remain) and against the transfer
    matrix with the exact eps(f): second order in the cell size."""
    r1, r2 = cases.film("drude", resolution=100), cases.film("drude", resolution=200)
    assert r2["device"].startswith(("rocm", "cuda", "gpu"))
    for r in (r1, r2):
        assert max(r["max_dRT_yee"], r["max_dRT_yee32"]) < YEE_TOL, (r["max_dRT_yee"], r["max_dRT_yee32"])
    e1, e2 = r1["max_dRT"], r2["max_dRT"]
    assert e2 < 2e-4 and np.log2(e1 / e2) > 1.8, (e1, e2)
    assert r2["R"][0] > 0.6 and np.min(r2["R"]) < 0.15, r2["R"]        # the plasma edge is in the band


@pytest.mark.gpu
@pytest.mark.parametrize("avg", [True, False])
def test_lorentz_slab_against_transfer_matrix(avg):
    """Check 11: a Lorentz slab (eps_inf 2.25, resonance at 1 / um, 0.5 um, faces on cell faces) in vacuum: strong
    absorption near the resonance; R and T against the exact Yee grid with the float32 coefficients, the same grid
    with float64 coefficients (the float32 coefficients move the resonance) and the transfer matrix. With the faces
    on cell faces all face rules and the staircase give the same arrays."""
    r = cases.film("lorentz", resolution=100, eps_averaging=avg)
    assert r["max_dRT_yee32"] < YEE_TOL and r["max_dRT_yee"] < 2 * YEE_TOL, (r["max_dRT_yee32"], r["max_dRT_yee"])
    assert r["max_dRT"] < 3e-3, r["max_dRT"]
    assert r["max_A"] > 0.7 and r["max_dA"] < 3e-3, (r["max_A"], r["max_dA"])


@pytest.mark.gpu
def test_lorentz_film_with_a_face_inside_a_cell(monkeypatch):
    """Check 11 with the thickness 0.504 um: the second face is inside a cell (fill 0.4 at res 100, 0.8 at res 200).
    Each face rule agrees with the exact Yee grid of that rule (`cases.film_nodes`, float32 coefficients), which
    checks the arrays of the rule. The error of the default rule "fill" on that grid (rule_err: no FDTD) falls at
    second order; with FDTD at res 100 the rule "fill" has less than a third of the error of the rule of the MIT
    package ("average"), which has less than the staircase rule."""
    import fdtd_gpu._dispersion as disp
    rows = {}
    for rule, res in (("fill", 100), ("fill", 200), ("average", 100), ("staircase", 100)):
        monkeypatch.setattr(disp, "FACE_RULE", rule)
        rows[rule, res] = r = cases.film("lorentz", resolution=res, thickness=0.504)
        assert r["max_dRT_yee32"] < YEE_TOL, (rule, res, r["max_dRT_yee32"])
        assert (0 < r["fills"][-1] < 1) == (rule != "staircase"), (rule, r["fills"])
    e1, e2 = rows["fill", 100]["rule_err"], rows["fill", 200]["rule_err"]
    assert np.log2(e1 / e2) > 1.8, (e1, e2)
    assert 3 * rows["fill", 100]["max_dRT"] < rows["average", 100]["max_dRT"] < rows["staircase", 100]["max_dRT"], \
        [(k, v["max_dRT"]) for k, v in rows.items()]
    assert np.log2(rows["fill", 100]["max_dRT"] / rows["fill", 200]["max_dRT"]) > 1.5      # FDTD: float32 at res 200


@pytest.mark.gpu
@pytest.mark.parametrize("kind", ["drude", "lorentz"])
def test_uniform_free_oscillation(kind):
    """Check 12: the k = 0 oscillation of a Drude (volume plasmon) or Lorentz medium: Harminv frequency and decay
    against the root of eps(f) = 0 with the discrete chi (tight) and with the exact chi (second order)."""
    r = cases.uniform(kind, resolution=20)
    assert abs(r["f_err_yee"]) < 1e-5 and abs(r["decay_err_yee"]) < 1e-3, r
    assert abs(r["f_err"]) < 1e-3 and abs(r["decay_err"]) < 1e-2, r


@pytest.mark.gpu
def test_conductive_decay():
    """Check 12: a uniform field in a conductive medium decays as exp(-sigma_D t) (the semi-implicit update:
    (1 - b) / (1 + b) per step)."""
    r = cases.uniform("conductive", resolution=20)
    assert r["sign_changes"] == 0
    assert abs(r["rel_err_yee"]) < 1e-4 and abs(r["rel_err"]) < 1e-3, r


@pytest.mark.gpu
def test_source_in_a_conductive_medium():
    """Check 13: a 3-D point dipole in a conductive medium (eps_inf 1, sigma_D 0.5 / um, also in the PML): the flux
    through the box of the monitor planes against the flux of the exact fields with the complex index through the
    same box. With the correction of fdtd_gpu._dispersion.correct_sources the error is second order (res 10 to
    20); without it (the current of fdtdx alone) the flux is (1 + b)^2 times higher, b = sigma_D dt / 2."""
    r10 = cases.dipole(resolution=10, sigma_d=0.5)
    with warnings.catch_warnings(record=True) as caught:                # no PML warning for this medium
        warnings.simplefilter("always")
        r20 = cases.dipole(resolution=20, sigma_d=0.5)
    assert not [w for w in caught if "fills the PML" in str(w.message)]
    old = cases.dipole(resolution=20, sigma_d=0.5, correct_sources=False)
    e10, e20 = r10["purcell"] - 1, r20["purcell"] - 1
    assert np.max(np.abs(e20)) < 1.5e-3 and np.all(np.log2(np.abs(e10 / e20)) > 1.8), (e10, e20)
    assert np.allclose(old["purcell"] / r20["purcell"], (1 + r20["b"]) ** 2, rtol=1e-5, atol=0), old["purcell"]
    assert np.min(old["purcell"]) - 1 > 1.2e-2                       # the error that the correction removes
    assert np.all(r20["lossless_ref_ratio"] < 0.7)                     # the loss is large inside the box


@pytest.mark.gpu
def test_correct_sources_per_cell(monkeypatch):
    """correct_sources in a 1-D cell (res 10) with a conductive block (eps_inf 2.25, sigma_D 0.5) whose face is at
    the Ex point of a cell (fill 1/2, face rule "fill"): the current of an Ex source is 1 / (1 + b) of the current
    without the correction, with b = sigma_D dt / 2 in the bulk (eps_inf other than 1 cancels), and b = f sigma_D
    eps_inf dt / (2 eps_x) in the face cell (eps_x = 1 + f (eps_inf - 1), E parallel to the face); a source in vacuum
    is the same to the last bit. Only the build of the run (no time steps)."""
    import jax

    import fdtd_gpu._dispersion as disp
    from fdtd_gpu.simulation import _device
    med = fg.Medium(epsilon=2.25, D_conductivity=0.5)
    block = fg.Block(size=fg.Vector3(fg.inf, fg.inf, 0.9), center=fg.Vector3(0, 0, 0.5), material=med)
    zs = {"face": 0.05, "bulk": 0.55, "vacuum": -0.55}           # the Ex points are at the cell centers along z

    def currents(correct):
        monkeypatch.setattr(disp, "CORRECT_SOURCES", correct)
        srcs = [fg.Source(fg.GaussianSource(1.0, fwidth=0.5), fg.Ex, center=fg.Vector3(0, 0, z)) for z in zs.values()]
        sim = fg.Simulation(cell_size=fg.Vector3(0, 0, 2), resolution=10, geometry=[block], sources=srcs)
        with jax.enable_x64(False), jax.default_device(_device()):
            objs, _, _, _ = sim._build(10, [])
        out = {o.name: np.asarray(o._inv_eps_oriented) for o in objs.objects if o.name.startswith("src")}
        return sim.dt, {k: out[f"src{j}"] for j, k in enumerate(zs)}

    dt, old = currents(False)
    _, new = currents(True)
    b_bulk = 0.5 * dt / 2
    f = 0.5
    b_face = f * 0.5 * 2.25 * dt / (2 * (1 + f * 1.25))
    assert np.allclose(new["bulk"][0] / old["bulk"][0], 1 / (1 + b_bulk), rtol=2e-6, atol=0)
    assert np.allclose(new["face"][0] / old["face"][0], 1 / (1 + b_face), rtol=2e-5, atol=0)
    assert abs(b_face / b_bulk - 1) > 0.2                          # the face cell has its own b
    assert np.array_equal(new["vacuum"], old["vacuum"])
    assert not np.array_equal(new["bulk"], old["bulk"])


@pytest.mark.gpu
@pytest.mark.parametrize("kind", ["lorentz", "conductive", "drude"])
def test_halfspace_into_the_pml(kind):
    """Check 14: a half-space that runs through the PML (1-D, res 50). R of its face against the exact Yee grid of a
    half-space (no wave comes back from the PML, as in an infinite medium), the flux planes in the medium against
    the continuum, and for the conductive medium the DFT fields of two planes in it against the forward wave of
    the grid. The Drude medium has eps near 0 at its plasma edge, where the stretched-coordinate PML absorbs little
    (its round trip 1e-6^Re(n)): there the PML reflects up to about 1e-2 and R differs more, and run() warns (the
    Lorentz and conductive half-spaces do not)."""
    r = cases.halfspace(kind, resolution=50)
    assert r["device"].startswith(("rocm", "cuda", "gpu"))
    tol = {"lorentz": 2e-5, "conductive": 5e-6, "drude": 1e-3}[kind]
    assert r["max_dR_y32"] < tol, r["max_dR_y32"]
    assert r["max_dT_in"] < 5e-3, r["max_dT_in"]
    assert (len(r["warned"]) == 1) == (kind == "drude"), r["warned"]
    if kind == "conductive":
        assert r["max_ratio_err_yee"] < 2e-5 and np.nanmax(r["r_pml"]) < 5e-5, (r["max_ratio_err_yee"], r["r_pml"])
    if kind == "drude":                       # the measured reflection of the PML follows the stretched coordinate
        rp, th = np.asarray(r["r_pml"]), np.asarray(r["r_pml_theory"])
        sel = np.isfinite(rp) & (np.asarray(r["freqs"]) >= 1.0)
        assert np.all((rp[sel] > 0.3 * th[sel]) & (rp[sel] < 1.5 * th[sel])), (rp[sel], th[sel])
        assert "high z side" in r["warned"][0] and "R_asymptotic" in r["warned"][0]


@pytest.mark.gpu
def test_pml_r_asymptotic_in_a_drude_halfspace():
    """PML(R_asymptotic=1e-15) sets the conductivity of the fdtdx PML for the round trip 1e-15 in vacuum: in the
    Drude half-space of check 14 (res 50) the reflection of the PML at 1.0 / um (Re n = 0.16) falls below 1e-3 (the
    default gives 1.25e-2), R agrees with the Yee half-space to 5e-5 (default 3.8e-4), and in vacuum the stronger
    layer still sends back less than 1e-5."""
    r = cases.halfspace("drude", resolution=50, r_asymptotic=1e-15)
    f, rp = np.asarray(r["freqs"]), np.asarray(r["r_pml"])
    assert r["max_dR_y32"] < 5e-5, r["max_dR_y32"]
    j = int(np.argmin(np.abs(f - 1.0)))
    assert np.isfinite(rp[j]) and rp[j] < 1e-3, rp
    assert np.nanmax(r["r_pml_vacuum"]) < 1e-5, r["r_pml_vacuum"]
    with pytest.raises(ValueError, match="R_asymptotic"):
        fg.PML(1.0, R_asymptotic=0.0)


@pytest.mark.gpu
@pytest.mark.parametrize("kind", ["lorentz", "conductive"])
def test_near2far_in_a_dispersive_medium(kind):
    """Check 15: a 2-D line source in a uniform Lorentz or conductive medium (it fills the PML too): the transform
    with the complex eps(f) of the medium against the exact fields with the complex index on two circles (one global
    phase per frequency); second order (res 20 to 40) for the Lorentz medium; the transform with eps_inf (the
    medium of a lossless, non-dispersive run) is far off."""
    with warnings.catch_warnings(record=True) as caught:                # no PML warning for these media
        warnings.simplefilter("always")
        r = cases.near2far_medium(kind, resolution=20)
    assert not [w for w in caught if "fills the PML" in str(w.message)]
    assert np.max(r["field_err"]) < 3e-2 and np.max(r["pattern_err"]) < 5e-2, r["field_err"]
    assert np.max(np.abs(r["phase"] + r["omega_dt_half"])) < 4e-2, r["phase"]
    assert np.min(r["err_eps_inf"]) > 0.2, r["err_eps_inf"]
    assert np.allclose(r["eps"], cases.medium_eps(cases.HALF[kind], r["freqs"]), rtol=1e-12)
    if kind == "lorentz":
        r40 = cases.near2far_medium(kind, resolution=40)
        assert np.all(np.log2(r["field_err"] / r40["field_err"]) > 1.8), (r["field_err"], r40["field_err"])


@pytest.mark.gpu
def test_guards():
    """A mode source plane or a mode monitor plane in a dispersive cell raises NotImplementedError (the mode solver
    sees eps_inf only); an unstable Lorentz pole raises ValueError; a near2far surface in a Drude metal gives finite
    fields with the complex eps(f) of the metal."""
    metal = cases.dispersive_medium(1.0, ("drude", 2.0, 0.1, 1.0))
    core = fg.Block(size=fg.Vector3(fg.inf, 0.4, fg.inf), material=metal)
    src = fg.EigenModeSource(fg.GaussianSource(0.6, fwidth=0.1), center=fg.Vector3(-0.5), size=fg.Vector3(0, 2))
    sim = fg.Simulation(cell_size=fg.Vector3(2, 2, 0), resolution=10, geometry=[core], sources=[src])
    with pytest.raises(NotImplementedError, match="EigenModeSource"):
        sim.run(until=1)
    sim = fg.Simulation(cell_size=fg.Vector3(2, 2, 0), resolution=10, geometry=[core],
                        sources=[fg.Source(fg.GaussianSource(0.6, fwidth=0.1), fg.Ez, center=fg.Vector3(-0.5, 0.6))])
    sim.add_mode_monitor(0.6, 0.1, 2, fg.ModeRegion(center=fg.Vector3(0.5), size=fg.Vector3(0, 2)))
    with pytest.raises(NotImplementedError, match="mode monitor"):
        sim.run(until=1)
    fast = cases.dispersive_medium(1.0, ("lorentz", 100.0, 0.1, 1.0))
    sim = fg.Simulation(cell_size=fg.Vector3(0, 0, 2), resolution=10, default_material=fast,
                        sources=[fg.Source(fg.GaussianSource(1.0, fwidth=0.5), fg.Ex, center=fg.Vector3())])
    with pytest.raises(ValueError):
        sim.run(until=1)
    sim = fg.Simulation(cell_size=fg.Vector3(3, 3, 0), resolution=10, default_material=metal,
                        boundary_layers=[fg.PML(0.5)],
                        sources=[fg.Source(fg.GaussianSource(1.0, fwidth=0.5), fg.Ez, center=fg.Vector3())])
    n2f = sim.add_near2far(1.0, 0.2, 2, *cases.box_regions(fg.Near2FarRegion, 0.5, axes=(0, 1)))
    sim.run(until=2)
    ff = np.array(sim.get_farfield(n2f, fg.Vector3(2, 0)))
    assert np.all(np.isfinite(ff)) and np.abs(ff).max() > 0
    assert np.allclose(n2f.eps, metal.permittivity([0.9, 1.1]), rtol=1e-12) and np.all(np.real(n2f.eps) < 0)


@pytest.mark.gpu
@pytest.mark.parametrize("ratio", [0.8, 0.95, 1.05, 1.2])
def test_stability_bound(ratio, monkeypatch):
    """The bound of fdtd_gpu._dispersion.stability_margin, bracketed to 5 %: a lossless Drude medium (eps_inf 1) in a
    periodic 1-D cell with a point source stays bounded at lhs / rhs = 0.8 and 0.95 and grows without bound at 1.05
    and 1.2 (the check is off; the point source excites the largest k of the grid)."""
    import fdtd_gpu._dispersion as disp
    res = 10
    dx = 1.0 / res
    dt = 0.99 * dx / math.sqrt(3)
    wp = 2 * math.sqrt(ratio * (1 - (dt / dx) ** 2)) / dt
    med = fg.Medium(E_susceptibilities=[fg.DrudeSusceptibility(frequency=wp / (2 * math.pi), gamma=0.0, sigma=1.0)])
    lhs, rhs = disp.stability_margin(med, dt, dx, 1)
    assert abs(lhs / rhs - ratio) < 1e-12
    monkeypatch.setattr(disp, "CHECK_STABILITY", False)
    sim = fg.Simulation(cell_size=fg.Vector3(0, 0, 2), resolution=res, default_material=med, k_point=fg.Vector3(),
                        sources=[fg.Source(fg.GaussianSource(1.0, fwidth=0.5), fg.Ex, center=fg.Vector3())])
    mon = sim.add_flux(1.0, 0.5, 3, fg.FluxRegion(center=fg.Vector3(0, 0, 0.5)))
    sim.run(until=150)
    flux = np.array(sim.get_fluxes(mon))
    if ratio < 1:
        assert np.all(np.isfinite(flux)) and np.max(np.abs(flux)) < 1.0, flux
    else:
        assert not np.all(np.isfinite(flux)) or np.max(np.abs(flux)) > 1e10, flux
    monkeypatch.setattr(disp, "CHECK_STABILITY", True)
    sim = fg.Simulation(cell_size=fg.Vector3(0, 0, 2), resolution=res, default_material=med, k_point=fg.Vector3(),
                        sources=[fg.Source(fg.GaussianSource(1.0, fwidth=0.5), fg.Ex, center=fg.Vector3())])
    if ratio < 1:
        sim.run(until=1)
    else:
        with pytest.raises(ValueError):
            sim.run(until=1)
