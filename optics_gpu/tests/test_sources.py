"""Checks of sources.polarized_mode (the source polarization of the coupling examples) with synthetic fields: two
pure polarizations with different shapes, mixed by a unitary or a non-unitary 2 x 2 matrix, must give back the pure
polarization when the two modes are near-degenerate; non-degenerate modes are not mixed.
"""
import numpy as np
import pytest

from optics_gpu.sources import COND_TOL, DEGENERATE_TOL, polarized_mode

X, Y = np.meshgrid(np.linspace(-5, 5, 41), np.linspace(-6, 6, 49), indexing="ij")
EX = np.stack([np.exp(-(X**2 / 4.0 + Y**2 / 6.0)), 0 * X, 0.1j * X * np.exp(-(X**2 + Y**2) / 5.0)])   # x-polarized
EY = np.stack([0 * X, 2.0 * np.exp(-(X**2 / 6.0 + Y**2 / 4.0)), 0.1j * Y * np.exp(-(X**2 + Y**2) / 5.0)])


def unit(E):
    """E with a transverse power of 1 and the largest |value| of its major component real and positive."""
    c = 0 if np.vdot(E[0], E[0]).real > np.vdot(E[1], E[1]).real else 1
    peak = E[c].flat[int(np.argmax(np.abs(E[c])))]
    E = E * np.conj(peak) / abs(peak)
    return E / np.sqrt(np.vdot(E[0], E[0]).real + np.vdot(E[1], E[1]).real)


@pytest.mark.parametrize("dneff", [1e-7, 2.1e-4], ids=["1e-7", "split-of-the-example"])
@pytest.mark.parametrize("mix", ["unitary", "diagonal", "non-unitary"])
def test_a_mixed_degenerate_pair_gives_back_the_pure_polarizations(mix, dneff):
    """Modes 0 and 1 are two mixes of EX and EY (|dneff| 1e-7, and 2.1e-4 as in the example; below DEGENERATE_TOL):
    pol "x" must give EX and pol "y" EY, to rounding (fraction 1, field equal after normalization and phase).
    "diagonal": the 45-degree rotation of the pair, as the solver gives the modes of the example."""
    t, phi = (0.6, 0.9) if mix == "unitary" else (np.pi / 4, 0.0)
    if mix in ("unitary", "diagonal"):
        m = np.array([[np.cos(t), np.sin(t) * np.exp(1j * phi)], [-np.sin(t) * np.exp(-1j * phi), np.cos(t)]])
    else:
        m = np.array([[1.0, 0.7j], [0.4, -1.3 + 0.2j]])
    E = np.stack([m[0, 0] * EX + m[0, 1] * EY, m[1, 0] * EX + m[1, 1] * EY])
    neff = np.array([1.4473, 1.4473 - dneff])
    for pol, ref in (("x", EX), ("y", EY)):
        field, info = polarized_mode(neff, E, pol)
        assert info["degenerate"] and info["source"] == "0+1" and info["dneff"] == pytest.approx(dneff, rel=1e-6)
        assert info["fraction"] == pytest.approx(1.0, abs=1e-12)
        assert 0.0 < min(info["fractions"]) and max(info["fractions"]) < 1.0           # the input modes are mixed
        assert np.max(np.abs(field - unit(ref))) <= 1e-12
        assert 1.4473 - dneff <= info["neff"].real <= 1.4473


def test_non_degenerate_modes_are_not_mixed():
    """|dneff| above DEGENERATE_TOL: the mode with the larger fraction in pol, without a change (here mode 1 for "x"
    when mode 0 is mostly y), also when that mode is not a pure polarization."""
    E = np.stack([EY + 0.2 * EX, EX + 0.1 * EY])
    neff = np.array([1.46, 1.46 - 10 * DEGENERATE_TOL])
    field, info = polarized_mode(neff, E, "x")
    assert not info["degenerate"] and info["source"] == "1" and info["neff"] == neff[1]
    assert np.max(np.abs(field - unit(E[1]))) <= 1e-12 and 0.9 < info["fraction"] < 1.0
    field, info = polarized_mode(neff, E, "y")
    assert info["source"] == "0" and np.max(np.abs(field - unit(E[0]))) <= 1e-12


def test_equal_fractions_without_degeneracy_are_refused():
    """Two diagonal polarizations of the same power in x (the modes of the example) with |dneff| above the
    tolerance: the choice would depend on rounding, so a ValueError stops it; a larger tol combines them."""
    a, b = EX / np.linalg.norm(EX[0]), EY / np.linalg.norm(EY[1])
    E = np.stack([a + b, a - b])
    with pytest.raises(ValueError, match="same power fraction"):
        polarized_mode(np.array([1.4473, 1.4471]), E, "x", tol=1e-4)
    field, info = polarized_mode(np.array([1.4473, 1.4471]), E, "x", tol=1e-3)
    assert info["degenerate"] and info["fraction"] == pytest.approx(1.0, abs=1e-12)


def test_orthogonal_parts_give_a_power_split_and_the_mean_neff():
    """Two orthogonal diagonal modes of equal power (overlap 0): the x source is their sum with part powers 0.5 and
    0.5 (sum 1), and its neff is the mean of the two."""
    a, b = EX / np.linalg.norm(EX[:2]), EY / np.linalg.norm(EY[:2])
    E = np.stack([(a + b) / np.sqrt(2), (a - b) / np.sqrt(2)])
    field, info = polarized_mode(np.array([1.4473, 1.4471]), E, "x")
    assert info["overlap"] == pytest.approx(0.0, abs=1e-12)
    assert info["part_powers"] == pytest.approx([0.5, 0.5], abs=1e-12)
    assert info["neff"].real == pytest.approx(1.4472, abs=1e-12)
    assert np.max(np.abs(field - unit(EX))) <= 1e-12


def test_two_modes_of_one_polarization_are_refused():
    """The even and odd supermodes of two x-polarized guides (dneff 1e-4): their combinations are all x-like
    (fractions in x 1 and 1), so the pair is not one x-like and one y-like field, and a combination would not be a
    mode: ValueError for x and for y."""
    gl = np.stack([np.exp(-((X + 2) ** 2 + Y**2)), 0 * X, 0 * X])
    gr = np.stack([np.exp(-((X - 2) ** 2 + Y**2)), 0 * X, 0 * X])
    E = np.stack([gl + gr, gl - gr])
    for pol in ("x", "y"):
        with pytest.raises(ValueError, match="not one x-like and one y-like"):
            polarized_mode(np.array([1.46, 1.46 - 1e-4]), E, pol)


@pytest.mark.parametrize("eps", [0.0, 1e-9])
def test_the_same_field_twice_is_refused(eps):
    """Mode 1 equal to mode 0, or to mode 0 plus 1e-9 EY: the power matrix P is near-singular (smallest / largest
    eigenvalue below COND_TOL), so there is no 2-D subspace: ValueError."""
    E = np.stack([EX + 0.3 * EY, EX + 0.3 * EY + eps * EY])
    with pytest.raises(ValueError, match="nearly the same field"):
        polarized_mode(np.array([1.46, 1.46]), E, "x")
    assert COND_TOL < 1e-6


def test_min_fraction_is_enforced():
    """A source with less power in the chosen component than min_fraction stops with a ValueError (here the
    non-degenerate mode EX + 0.1 EY, fraction 0.96)."""
    E = np.stack([EY + 0.2 * EX, EX + 0.1 * EY])
    neff = np.array([1.46, 1.46 - 10 * DEGENERATE_TOL])
    assert polarized_mode(neff, E, "x", min_fraction=0.9)[1]["fraction"] > 0.9
    with pytest.raises(ValueError, match="min_fraction"):
        polarized_mode(neff, E, "x", min_fraction=0.99)


def test_one_mode_and_bad_inputs():
    field, info = polarized_mode(np.array([1.45]), EX[None], "x")
    assert info["source"] == "0" and info["dneff"] == float("inf") and info["fraction"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="pol"):
        polarized_mode(np.array([1.45]), EX[None], "z")
    with pytest.raises(ValueError, match="neff"):
        polarized_mode(np.array([1.45, 1.44]), EX[None], "x")
