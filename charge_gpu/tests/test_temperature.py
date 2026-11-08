"""Checks of the temperature models (charge_gpu.temperature) and of the solvers at temperatures other than 300 K: n_i(T)
against the published fit and the band-gap form, the mobility against Arora et al., the 1-D junction (built-in
potential, depletion width, capacitance, series resistance) and the long-base diode (Shockley) at 250, 350 and
400 K, Gummel and the device solvers at 350 K, and the 300 K results of the code before the temperature models."""
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from charge_gpu import drift_diffusion as dd  # noqa: E402
from charge_gpu import temperature as tm  # noqa: E402
from charge_gpu.injection import diode_1d, shockley_j0  # noqa: E402
from charge_gpu.modulator import rib_pn  # noqa: E402
from charge_gpu.poisson import EPS0, Q, solve as poisson_solve, thermal_voltage  # noqa: E402
from charge_gpu.rc import masetti_mobility, small_signal  # noqa: E402
from test_junction import EPS_SI, NA, ND, NI, junction  # noqa: E402

TEMPS = [250.0, 350.0, 400.0]
HEIGHT = 1e-6                    # y extent of the 1-D junction: per length = per area x HEIGHT

# Hand values (bc -l, not this code). n_i in cm^-3 with 1.00e10 cm^-3 at 300 K and the temperature dependence of
# the Misiakos and Tsamakis fit 5.29e19 (T/300)^2.54 exp(-6726/T) (J. Appl. Phys. 74, 3293 (1993); the fit itself
# gives 9.6956e9 cm^-3 at 300 K).
NI_CM3 = {250: 7.1040036872e7, 275: 1.0443496814e9, 325: 6.8752215061e10, 350: 3.6393852376e11,
          375: 1.5614493092e12, 400: 5.6437135407e12}
# Arora, Hauser and Roulston (1982), cm^2/(V s), at 1e16 and 1e18 cm^-3: (electrons, holes) per T
ARORA_CM2 = {250: ((1761.7264829, 277.3007922), (623.1691977, 154.3086189)),
             300: ((1218.2374350, 262.1103978), (437.4523878, 143.2249112)),
             350: ((888.2744216, 244.7989117), (325.0716937, 131.5133639)),
             400: ((675.5050707, 226.9079428), (252.3248039, 120.0910075))}


def test_intrinsic_density_against_the_published_fit():
    """n_i(T) / n_i(300 K) is the ratio of the Misiakos and Tsamakis fit (hand values, 250 to 400 K); exactly the
    input at 300 K. The fit at 300 K is 9.70e9 cm^-3, within the measured (9.7 +- 0.1)e9 cm^-3 of the paper. An
    independent form: n_i ~ T^1.5 exp(-E_g(T) / 2 k T) with the Varshni band gap of Thurmond, J. Electrochem. Soc.
    122, 1133 (1975) (E_g = 1.170 - 4.73e-4 T^2 / (T + 636) eV), constant effective masses: the ratio agrees within
    7 % (hand values: +4.4 % at 250 K, -6.5 % at 400 K; the fit includes the temperature dependence of the effective
    masses, the simple form does not)."""
    assert tm.intrinsic_density(300.0, NI) == NI
    assert tm.intrinsic_density(300, 1.234e16) == 1.234e16
    assert tm.misiakos_ni_cm3(300.0) == pytest.approx(9.6955709274e9, rel=1e-9)
    assert abs(tm.misiakos_ni_cm3(300.0) - 9.7e9) < 0.1e9
    k = 8.617333262e-5
    eg = lambda t: 1.170 - 4.73e-4 * t * t / (t + 636.0)  # noqa: E731
    simple = lambda t: t**1.5 * np.exp(-eg(t) / (2 * k * t))  # noqa: E731
    for t, ref in NI_CM3.items():
        ni = tm.intrinsic_density(t, NI) * 1e-6
        assert ni == pytest.approx(ref, rel=1e-9), t
        assert abs(simple(t) / simple(300.0) / (ni / 1e10) - 1) < 0.07, t
    assert np.all(np.diff([tm.intrinsic_density(t, NI) for t in sorted(NI_CM3)]) > 0)
    for bad in (240.0, 510.0, np.array([300.0, 310.0])):
        with pytest.raises(ValueError):
            tm.intrinsic_density(bad, NI)


def test_mobility_against_arora():
    """mu_Arora(N, T) against hand values of the published formula (250 to 400 K, 1e16 and 1e18 cm^-3; at low
    doping and 300 K the lattice values 1339.8 and 461.3 cm^2/(V s)); mobility(N, T) = Masetti x mu_Arora(N, T) /
    mu_Arora(N, 300 K), exactly Masetti at 300 K. The mobility falls with T at 1e16 cm^-3 (lattice scattering)."""
    assert tm.arora_mobility(1e6, "n", 300.0) * 1e4 == pytest.approx(1339.8122473, abs=1e-6)
    assert tm.arora_mobility(1e6, "p", 300.0) * 1e4 == pytest.approx(461.2663587, abs=1e-6)
    doping = np.array([1e22, 1e24])
    for t, (mu_e, mu_h) in ARORA_CM2.items():
        for carrier, ref, ref300 in (("n", mu_e, ARORA_CM2[300][0]), ("p", mu_h, ARORA_CM2[300][1])):
            assert np.allclose(tm.arora_mobility(doping, carrier, t) * 1e4, ref, rtol=0, atol=1e-6)
            mu = tm.mobility(doping, carrier, t)
            assert np.allclose(mu, masetti_mobility(doping, carrier) * np.array(ref) / np.array(ref300), rtol=1e-9)
    for carrier in ("n", "p"):
        mu300 = tm.mobility(doping, carrier, 300.0)
        assert np.array_equal(mu300, masetti_mobility(doping, carrier))
        assert tm.mobility(1e22, carrier, 250.0) > tm.mobility(1e22, carrier, 350.0) > tm.mobility(1e22, carrier, 400.0)


def test_lifetime_and_mean_temperature():
    """tau(T) = tau(300 K) (T / 300)^-1/2 (exactly tau at 300 K; exponent 0: constant). mean_temperature: the area
    mean and the spread of a field over the semiconductor cells; the wrong shape is refused; Device.T must be a
    scalar."""
    assert tm.lifetime(1e-6, 300.0) == 1e-6
    assert tm.lifetime(1e-6, 400.0) == pytest.approx(1e-6 * (4 / 3) ** -0.5, rel=1e-14)
    assert tm.lifetime(1e-9, 350.0, exponent=0.0) == 1e-9
    dev, xc = junction(n=400)                                   # uniform cells, all silicon
    field = (300.0 + 40.0 * xc / 1e-6)[:, None]                 # 300 to 340 K, linear
    mean, spread = tm.mean_temperature(dev, field)
    assert mean == pytest.approx(320.0, abs=1e-9)
    assert spread == pytest.approx(40.0 * (xc[-1] - xc[0]) / 1e-6, rel=1e-12)
    rib = rib_pn(half_span=1.0e-6, dx=20e-9)
    mean, spread = tm.mean_temperature(rib, np.where(rib.semiconductor, 330.0, 1000.0))    # the oxide does not count
    assert mean == pytest.approx(330.0, rel=1e-13) and spread == 0.0
    with pytest.raises(ValueError, match="shape"):
        tm.mean_temperature(rib, field)
    dev.T = field
    with pytest.raises(ValueError, match="scalar"):
        poisson_solve(dev, 1.0, 0.0)


def test_results_at_300_K_equal_the_code_before_the_temperature_models(monkeypatch):
    """Device.T = 300 K: the values that the code before the temperature models (commit f3f2f39) gave with
    CHARGE_GPU_DEVICE=cpu on the GPU computer (repr values of a script run, twice the same bits). The same script
    after the change gave the same bits (the report of this change); the tolerance 1e-9 allows other BLAS builds."""
    monkeypatch.setenv("CHARGE_GPU_DEVICE", "cpu")
    old_rc = {0.0: (2.1483917025858977e-09, 0.00010724601931794383, 0.00018528290593764914),
              1.0: (1.4781676900058208e-09, 0.00010478224014753524, 0.000181026373465368)}
    dev, _ = junction(n=400)
    dev.T = 300.0
    for r in small_signal(dev, [0.0, 1.0]):
        assert (r.c, r.r_n, r.r_p) == pytest.approx(old_rc[r.bias], rel=1e-9)
    vt = thermal_voltage()
    mu_n, mu_p = masetti_mobility(1e24, "n"), masetti_mobility(1e24, "p")
    l_n, l_p = np.sqrt(mu_n * vt * 1e-4), np.sqrt(mu_p * vt * 1e-4)
    sols = dd.sweep(diode_1d(1e24, 1e24, 6 * l_n, 6 * l_p), [0.6, 0.65], tau_n=1e-4, tau_p=1e-4)
    assert [s.current for s in sols] == pytest.approx([9.103632526373494e-07, 6.281659396286319e-06], rel=1e-9)
    assert sols[0].tau_n == 1e-4
    r = small_signal(rib_pn(half_span=1.0e-6, dx=20e-9), [1.0])[0]
    assert (r.c, r.r) == pytest.approx((2.3997993240037053e-10, 0.012407319361981554), rel=1e-9)


def w_sze(v_r, t):
    """Depletion width with the Debye term (Sze), at T: W^2 = 2 eps (V_bi + V_R - 2 V_T)(N_A + N_D) / (q N_A N_D),
    V_bi = V_T ln(N_A N_D / n_i(T)^2)."""
    vt = thermal_voltage(t)
    ni = tm.intrinsic_density(t, NI)
    vbi = vt * np.log(NA * ND / ni**2)
    return np.sqrt(2 * EPS_SI * EPS0 * (vbi + v_r - 2 * vt) * (NA + ND) / (Q * NA * ND))


@pytest.mark.parametrize("t", TEMPS)
def test_junction_at_temperature(t):
    """The 1-D abrupt junction of test_junction.py (1e18 cm^-3, 0.5 nm cells) at T, 0, 1 and 3 V reverse bias:
    the built-in potential V_T ln(N_A N_D / n_i(T)^2) (1e-4 V), n p = n_i(T)^2 at 0 V, the depletion width from the
    charge against Sze (1 %), charge neutrality, C = dQ_n/dV against eps / W (the tolerance of the 300 K test,
    3e-4), R'_n and R'_p against the neutral lengths with the mobility at T (1e-4)."""
    dev, xc = junction()
    dev.T = t
    vt, ni = thermal_voltage(t), dev.n_i_t
    res = small_signal(dev, [0.0, 1.0, 3.0], dv=0.02)
    s0 = res[0].solution
    vbi = vt * np.log(NA * ND / ni**2)
    assert s0.psi[-1, 0] - s0.psi[0, 0] == pytest.approx(vbi, abs=1e-4)
    assert np.allclose(s0.n * s0.p, ni**2, rtol=1e-6)
    dx = xc[1] - xc[0]
    half = dev.n_net[:, 0] < 0
    mu_n, mu_p = tm.mobility(ND, "n", t), tm.mobility(NA, "p", t)
    for r in res:
        assert r.converged
        s = r.solution
        q_n = np.sum((ND - s.n[~half, 0]) * dx)
        q_p = np.sum((NA - s.p[half, 0]) * dx)
        assert q_n == pytest.approx(q_p, rel=1e-3)
        w = q_n / ND + q_p / NA
        assert w == pytest.approx(w_sze(r.bias, t), rel=1e-2)
        c_ref = EPS_SI * EPS0 / w_sze(r.bias, t)
        print(f"T {t} K, {r.bias} V: W/W_Sze - 1 = {w / w_sze(r.bias, t) - 1:.2e}, "
              f"C/C_dep - 1 = {r.c / HEIGHT / c_ref - 1:.2e}")
        assert r.c / HEIGHT == pytest.approx(c_ref, rel=3e-4)
        assert r.c_p == pytest.approx(r.c, rel=1e-8)
        neutral = 0.5e-6 - w_sze(r.bias, t) / 2
        assert r.r_n == pytest.approx(neutral / (Q * mu_n * ND * HEIGHT), rel=1e-4)
        assert r.r_p == pytest.approx(neutral / (Q * mu_p * NA * HEIGHT), rel=1e-4)


def matched_bias(v300, t):
    """The bias at T with the injection level n_i^2 exp(V / V_T) of v300 at 300 K (the same n_p / N_A)."""
    return thermal_voltage(t) * (v300 / thermal_voltage(300.0) + 2 * np.log(NI / tm.intrinsic_density(t, NI)))


@pytest.mark.parametrize("t", TEMPS)
def test_long_base_diode_against_shockley_at_temperature(t):
    """The long-base diode of test_drift_diffusion.py (1e18 cm^-3, tau 100 us at 300 K, contacts 6 diffusion lengths
    away) at T: J against J0(T) (exp(V/V_T) - 1), J0 = q n_i(T)^2 (D_n / (L_n N_A) + D_p / (L_p N_D)) with
    D = mu(T) V_T, L = sqrt(D tau(T)). The biases have the injection levels of 0.60, 0.65, 0.70 V at 300 K (where
    the 300 K test is within 1 %), so high injection and the depletion recombination have about the same part:
    within 1 %. The contact currents sum to zero (1e-12 A/m)."""
    vt = thermal_voltage(t)
    tau = tm.lifetime(1e-4, t)
    mu_n, mu_p = tm.mobility(1e24, "n", t), tm.mobility(1e24, "p", t)
    l_n, l_p = np.sqrt(mu_n * vt * tau), np.sqrt(mu_p * vt * tau)
    dev = diode_1d(1e24, 1e24, 6 * l_n, 6 * l_p)
    dev.T = t
    j0 = shockley_j0(1e24, 1e24, mu_n, mu_p, tau, tau, n_i=dev.n_i_t, T=t, length_p=6 * l_n, length_n=6 * l_p)
    biases = [matched_bias(v, t) for v in (0.6, 0.65, 0.7)]
    sols = dd.sweep(dev, biases, tau_n=1e-4, tau_p=1e-4)
    for v, s in zip(biases, sols):
        assert s.converged and s.tau_n == tau
        ratio = s.current / dev.y[-1] / (j0 * np.expm1(v / vt))
        print(f"T {t} K, V {v:.4f} V: J / J_Shockley = {ratio:.5f}")
        assert ratio == pytest.approx(1.0, abs=0.01)
        assert abs(np.sum(s.contact_currents)) < 1e-12


def test_gummel_and_device_solvers_at_350_K():
    """The 1-D junction (1e18 cm^-3, 2.5 nm cells) at 350 K: at 0 V the drift-diffusion equilibrium is poisson.solve
    with n p = n_i(350 K)^2; at 0.5 V forward and 1 V reverse, Gummel (SuperLU and device continuity) and Newton
    with the device linear solver give the solution of Newton with SuperLU."""
    dev, _ = junction(n=400)
    dev.T = 350.0
    ni = dev.n_i_t
    eq = dd.solve(dev, 0.0, 0.0)
    ref0 = poisson_solve(dev, 0.0, 0.0)
    assert eq.converged and np.max(np.abs(eq.psi - ref0.psi)) < 1e-11
    assert np.allclose(eq.n * eq.p, ni**2, rtol=1e-9)
    phi_n, phi_p = eq.quasi_fermi()
    assert np.max(np.abs(phi_n)) < 1e-11 and np.max(np.abs(phi_p)) < 1e-11
    for v_n, v_p in ((0.0, 0.5), (1.0, 0.0)):
        a = dd.solve(dev, v_n, v_p)
        for kw in (dict(method="gummel"), dict(method="gummel", linear="device"), dict(linear="device")):
            b = dd.solve(dev, v_n, v_p, **kw)
            print(f"350 K, {v_p - v_n:+.1f} V, {kw}: psi {np.max(np.abs(a.psi - b.psi)):.1e} V, "
                  f"n {np.max(np.abs(a.n / b.n - 1)):.1e}, p {np.max(np.abs(a.p / b.p - 1)):.1e}, "
                  f"I {b.current - a.current:.1e} A/m (I = {a.current:.3e} A/m)")
            assert a.converged and b.converged and b.linear_fallbacks == 0, kw
            assert np.max(np.abs(a.psi - b.psi)) < 1e-11, kw
            assert np.max(np.abs(a.n / b.n - 1)) < 1e-8 and np.max(np.abs(a.p / b.p - 1)) < 1e-8, kw
            if v_p > 0:                                         # the device floor of test_dd_device.py: 5e-9 A/m
                assert a.current > 1e-5 and b.current == pytest.approx(a.current, abs=5e-9), kw
