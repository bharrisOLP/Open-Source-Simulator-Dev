"""Dispersive (Lorentz, Drude) and conductive media: the units and the ADE coefficients against the references (no
FDTD run), and FDTD checks 10 to 12 (a dispersive film against the transfer matrix and the exact Yee grid; the
uniform field: free oscillation and conductive decay)."""

import math

import numpy as np
import pytest

import cases
import fdtd_gpu as fg
from fdtd_gpu._dispersion import C0, UM, fdtdx_poles
from references import chi_exact, chi_yee, tmm_rt, uniform_root, yee_rt

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
def test_guards():
    """A mode source plane in a dispersive cell raises NotImplementedError; an unstable Lorentz pole raises
    ValueError; a near2far surface in a dispersive medium raises NotImplementedError."""
    metal = cases.dispersive_medium(1.0, ("drude", 2.0, 0.1, 1.0))
    core = fg.Block(size=fg.Vector3(fg.inf, 0.4, fg.inf), material=metal)
    src = fg.EigenModeSource(fg.GaussianSource(0.6, fwidth=0.1), center=fg.Vector3(-0.5), size=fg.Vector3(0, 2))
    sim = fg.Simulation(cell_size=fg.Vector3(2, 2, 0), resolution=10, geometry=[core], sources=[src])
    with pytest.raises(NotImplementedError):
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
    with pytest.raises(NotImplementedError):
        sim.get_farfield(n2f, fg.Vector3(10, 0))


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
