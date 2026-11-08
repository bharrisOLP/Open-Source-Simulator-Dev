"""Temperature models of silicon: the intrinsic density n_i(T), the low-field mobility mu(N, T), the SRH lifetimes.

All solvers (poisson.py, rc.py, drift_diffusion.py with both methods and both linear solvers) take the temperature
from ``Device.T`` (one scalar for the device, in K). From it they use:

- V_T = k T / q (poisson.thermal_voltage).
- n_i(T) = n_i(300 K) f(T) / f(300 K), with f(T) = 5.29e19 (T / 300)^2.54 exp(-6726 / T) cm^-3: the fit of Misiakos
  and Tsamakis, J. Appl. Phys. 74, 3293 (1993), to their measurements from 78 to 340 K (9.7e9 cm^-3 at 300 K). The
  scale n_i(300 K) is ``Device.n_i`` (default 1e16 m^-3 = 1.00e10 cm^-3, Sproul and Green, J. Appl. Phys. 70, 846
  (1991)), so the fit gives only the temperature dependence. Above 340 K the fit is an extrapolation (the tests
  compare it with the band-gap form n_i ~ T^1.5 exp(-E_g(T) / 2 k T) up to 400 K).
- Mobility: mu(N, T) = mu_Masetti(N) mu_Arora(N, T) / mu_Arora(N, 300 K). The doping dependence at 300 K is that of
  Masetti, Severi and Solmi (1983) (rc.masetti_mobility); the temperature dependence is that of Arora, Hauser and
  Roulston, IEEE Trans. Electron Devices ED-29, 292 (1982), fitted to measurements from 250 to 500 K:
      electrons: mu = 88 Tn^-0.57 + 7.4e8 T^-2.33 / (1 + (N / (1.26e17 Tn^2.4))^(0.88 Tn^-0.146))
      holes:     mu = 54.3 Tn^-0.57 + 1.36e8 T^-2.23 / (1 + (N / (2.35e17 Tn^2.4))^(0.88 Tn^-0.146))
  (Tn = T / 300, N in cm^-3, mu in cm^2/(V s)). A user mobility (drift_diffusion.solve(mobility=...)) is used as
  given: it must be the value at Device.T.
- SRH lifetimes (drift_diffusion.solve): tau_n and tau_p are the values at 300 K; the solver uses
  tau(T) = tau(300 K) (T / 300)^tau_exponent, default tau_exponent = -1/2. The model: tau = 1 / (sigma v_th N_t)
  with a constant capture cross section sigma and trap density N_t, and the thermal velocity
  v_th = sqrt(3 k T / m*) (Sze and Ng, Physics of Semiconductor Devices, 3rd ed., Ch. 1). The trap stays at midgap:
  n_1 = p_1 = n_i(T). tau_exponent = 0 keeps the lifetimes constant.

At exactly 300 K each function returns its 300 K input unchanged, so the solvers give bit for bit the results of the
code before the temperature models.

Valid range: 250 to 500 K (the range of the Arora fit); other values raise ValueError. Not modeled: a temperature
that changes from cell to cell (a T gradient needs the thermoelectric terms of the current equations; use
mean_temperature for a near-uniform field of thermal_gpu), the temperature dependence of eps_r and of the plasma
dispersion coefficients (Soref and Bennett, 300 K), the thermo-optic effect, band-gap narrowing.
"""

from __future__ import annotations

import numpy as np

T_REF = 300.0                   # K: the temperature of the reference values (n_i, Masetti, the lifetimes)
T_MIN, T_MAX = 250.0, 500.0     # K: the valid range (Arora et al. 1982)
TAU_EXPONENT = -0.5             # tau ~ T^-1/2: constant capture cross section, v_th ~ sqrt(T)

# Arora, Hauser, Roulston (1982): mu_min Tn^-0.57 + A T^-b / (1 + (N / (N_ref Tn^2.4))^(0.88 Tn^-0.146))
ARORA = {"n": dict(mu_min=88.0, a=7.4e8, b=2.33, n_ref=1.26e17),
         "p": dict(mu_min=54.3, a=1.36e8, b=2.23, n_ref=2.35e17)}


def check_temperature(T) -> float:
    """T as a float in K. ValueError for an array (one temperature per device) or a value outside 250 to 500 K."""
    if np.ndim(T) != 0:
        raise ValueError("Device.T must be one scalar temperature (K); for a field of thermal_gpu use "
                         "temperature.mean_temperature")
    t = float(T)
    if not T_MIN <= t <= T_MAX:
        raise ValueError(f"T = {t} K is outside the valid range of the temperature models, {T_MIN} to {T_MAX} K")
    return t


def misiakos_ni_cm3(T) -> float:
    """The fit of Misiakos and Tsamakis (1993): n_i = 5.29e19 (T / 300)^2.54 exp(-6726 / T) in cm^-3."""
    t = np.asarray(T, float)
    return 5.29e19 * (t / 300.0) ** 2.54 * np.exp(-6726.0 / t)


def intrinsic_density(T, n_i_ref: float = 1.0e16) -> float:
    """n_i (m^-3) at T (K): n_i_ref (the value at 300 K) times the ratio of the Misiakos and Tsamakis fit,
    f(T) / f(300 K). At exactly 300 K it returns n_i_ref."""
    t = check_temperature(T)
    if t == T_REF:
        return n_i_ref
    return n_i_ref * float(misiakos_ni_cm3(t) / misiakos_ni_cm3(T_REF))


def arora_mobility(doping_m3, carrier: str, T) -> np.ndarray:
    """Low-field mobility (m^2/(V s)) of Arora, Hauser and Roulston (1982) at T (K), as a function of the dopant
    concentration (m^-3; 1 cm^-3 floor, as rc.masetti_mobility)."""
    c = ARORA[carrier]
    t = float(T)
    tn = t / 300.0
    n_cm = np.maximum(np.abs(np.asarray(doping_m3, float)) * 1e-6, 1.0)
    alpha = 0.88 * tn**-0.146
    mu = c["mu_min"] * tn**-0.57 + c["a"] * t ** -c["b"] / (1 + (n_cm / (c["n_ref"] * tn**2.4)) ** alpha)
    return mu * 1e-4


def mobility(doping_m3, carrier: str, T=T_REF) -> np.ndarray:
    """Low-field mobility (m^2/(V s)) of electrons (carrier "n") or holes ("p") in Si at T (K): the Masetti mobility
    at 300 K (rc.masetti_mobility) times mu_Arora(N, T) / mu_Arora(N, 300 K). At exactly 300 K: the Masetti value."""
    from charge_gpu.rc import masetti_mobility

    t = check_temperature(T)
    mu = masetti_mobility(doping_m3, carrier)
    if t == T_REF:
        return mu
    return mu * arora_mobility(doping_m3, carrier, t) / arora_mobility(doping_m3, carrier, T_REF)


def lifetime(tau_ref: float, T, exponent: float = TAU_EXPONENT) -> float:
    """The SRH lifetime (s) at T (K): tau_ref (the value at 300 K) times (T / 300)^exponent. At exactly 300 K:
    tau_ref."""
    t = check_temperature(T)
    if t == T_REF:
        return tau_ref
    return tau_ref * (t / T_REF) ** exponent


def mean_temperature(dev, t_field) -> tuple[float, float]:
    """(the area mean, max - min) of a temperature field (K) over the semiconductor cells of ``dev``. t_field has the
    shape of the device cells (nx, ny): for example thermal_gpu.solver.solve_steady on the edges dev.x, dev.y. Set
    dev.T to the mean; the spread tells how far the cells are from it (the solvers use one temperature)."""
    t = np.asarray(t_field, float)
    semi = np.asarray(dev.semiconductor, bool)
    if t.shape != semi.shape:
        raise ValueError(f"t_field has the shape {t.shape}; the device cells have {semi.shape}")
    if not semi.any():
        raise ValueError("the device has no semiconductor cells")
    area = (np.diff(np.asarray(dev.x, float))[:, None] * np.diff(np.asarray(dev.y, float))[None, :])[semi]
    ts = t[semi]
    if not np.all(np.isfinite(ts)):
        raise ValueError("t_field is not finite in the semiconductor cells")
    return float(np.sum(area * ts) / np.sum(area)), float(np.max(ts) - np.min(ts))
