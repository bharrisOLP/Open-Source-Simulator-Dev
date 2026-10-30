"""Checks of the PN-junction solver on a 1-D abrupt Si junction (N_A = N_D = 1e24 m^-3 = 1e18 cm^-3)."""
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from charge_gpu.poisson import EPS0, Q, Contact, Device, solve, thermal_voltage  # noqa: E402

NI, NA, ND, EPS_SI = 1.0e16, 1.0e24, 1.0e24, 11.7


def junction(length=1.0e-6, n=2000):
    x = np.linspace(0, length, n + 1)
    xc = 0.5 * (x[1:] + x[:-1])
    n_net = np.where(xc < length / 2, -NA, ND)[:, None]
    return Device(x=x, y=np.array([0.0, 1e-6]), eps_r=np.full((n, 1), EPS_SI), n_net=n_net,
                  semiconductor=np.ones((n, 1), bool),
                  contacts=[Contact("xmin", np.array([True]), 0.0), Contact("xmax", np.array([True]), 0.0)],
                  n_i=NI), xc


def test_built_in_potential_at_zero_bias():
    """A consistency check: at zero bias the potential step equals the difference of the contact potentials, and
    n p = n_i^2 holds by construction (phi_n = phi_p). The physics checks are the next tests."""
    dev, _ = junction()
    sol = solve(dev, 0.0, 0.0)
    assert sol.converged
    vt = thermal_voltage()
    vbi = vt * (np.arcsinh(ND / (2 * NI)) + np.arcsinh(NA / (2 * NI)))      # = vt ln(NA ND / ni^2) to 1e-16
    assert sol.psi[-1, 0] - sol.psi[0, 0] == pytest.approx(vbi, abs=1e-4)
    assert np.allclose(sol.n * sol.p, NI**2, rtol=1e-6)                     # equilibrium: n p = ni^2


@pytest.mark.parametrize("v_r", [0.0, 1.0, 3.0])
def test_depletion_width_against_the_depletion_approximation(v_r):
    dev, xc = junction()
    sol = solve(dev, v_r, 0.0)                                               # reverse bias on the n contact
    assert sol.converged
    dx = xc[1] - xc[0]
    half = dev.n_net[:, 0] < 0                                               # the p side
    q_n = np.sum((ND - sol.n[~half, 0]) * dx)                                # depleted donors per area (m^-2)
    q_p = np.sum((NA - sol.p[half, 0]) * dx)
    w = q_n / ND + q_p / NA
    vt = thermal_voltage()
    vbi = vt * np.log(NA * ND / NI**2)
    eps = EPS_SI * EPS0
    w_dep = np.sqrt(2 * eps * (vbi + v_r - 2 * vt) * (NA + ND) / (Q * NA * ND))   # Sze, with the Debye-tail term
    assert q_n == pytest.approx(q_p, rel=1e-3)                               # charge neutrality
    assert w == pytest.approx(w_dep, rel=1e-2)


def test_plasma_dispersion_values():
    """Soref and Bennett (1987) at 1.55 um: 1e18 cm^-3 electrons give dn = -8.8e-4; holes from 0 to 1e18 cm^-3 give
    -8.5e-18 (1e18)^0.8; from a doped baseline the hole term is P1^0.8 - P0^0.8 (not (P1 - P0)^0.8)."""
    from charge_gpu.modulator import plasma_dispersion
    z = np.zeros(2)
    dn, da = plasma_dispersion(z, z, np.array([1e24, 0.0]), np.array([0.0, 1e24]))
    assert dn[0] == pytest.approx(-8.8e-4, rel=1e-9)
    assert dn[1] == pytest.approx(-8.5e-18 * 1e18**0.8, rel=1e-9)
    assert da[0] == pytest.approx(8.5 * 100, rel=1e-9) and da[1] == pytest.approx(6.0 * 100, rel=1e-9)
    dn, da = plasma_dispersion(0.0, 3e23, 0.0, 1e23)            # holes 3e17 -> 1e17 cm^-3 (depletion)
    assert float(dn) == pytest.approx(-8.5e-18 * (1e17**0.8 - 3e17**0.8), rel=1e-12)
    assert float(dn) > 0 and float(da) < 0                       # fewer carriers: higher index, lower loss


def test_matches_an_independent_sparse_newton_solve():
    """The same equations with scipy (assembled sparse Jacobian, direct solves, plain Newton) at 2 V reverse bias."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spl
    from charge_gpu.poisson import _operators

    dev, _ = junction(n=400)
    sol = solve(dev, 2.0, 0.0)
    o = _operators(dev, 2.0, 0.0, np.float64)
    gx, diag, bc = (np.asarray(a)[:, 0] if np.asarray(a).ndim == 2 else np.asarray(a) for a in (o.gx, o.diag, o.bc))
    vol = np.asarray(o.vol)[:, 0]
    n_net = dev.n_net[:, 0]
    a_mat = sp.diags([diag, -gx, -gx], [0, 1, -1], format="csc")
    vt = thermal_voltage()
    psi = np.asarray(sol.psi)[:, 0] + 0.01                       # start near, but not at, the GPU solution
    for _ in range(50):
        n = NI * np.exp((psi - 2.0) / vt)
        p = NI * np.exp((0.0 - psi) / vt)
        f = a_mat @ psi - bc - Q * (p - n + n_net) * vol
        jac = a_mat + sp.diags(Q * (n + p) * vol / vt)
        d = spl.spsolve(jac.tocsc(), -f)
        psi = psi + d
        if np.max(np.abs(d)) < 1e-13:
            break
    assert np.max(np.abs(psi - np.asarray(sol.psi)[:, 0])) < 1e-8
