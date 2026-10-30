"""Transient check: a slab with T = 0 on both sides, initial T = sin(pi y / L), decays as exp(-t / tau) with
tau = L^2 / (pi^2 alpha), alpha = k / (rho cp). Implicit Euler: the decay per step is 1 / (1 + dt / tau_h), with
tau_h the discrete time constant; checked to 1 % against the continuous tau (fine grid, small step)."""
import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from thermal_gpu.solver import solve_transient  # noqa: E402


def test_slab_decay_time_constant():
    L, k, rho_cp, n = 2e-6, 1.4, 1.6e6, 200          # oxide-like: alpha = 8.75e-7 m^2/s
    tau = L**2 / (np.pi**2 * k / rho_cp)              # about 0.46 us
    y = np.linspace(0, L, n + 1)
    yc = 0.5 * (y[1:] + y[:-1])
    x = np.array([0.0, 1e-6])
    bc = {"ymin": ("dirichlet", 0.0), "ymax": ("dirichlet", 0.0)}
    t0 = np.sin(np.pi * yc / L)[None, :]
    dt = tau / 200
    res = solve_transient(x, y, np.full((1, n), k), np.zeros((1, n)), bc, np.full((1, n), rho_cp),
                          t_end=tau, dt=dt, probes=[(0, n // 2)], t0_field=t0)
    decay = res.probes[-1, 0] / res.probes[0, 0]
    assert decay == pytest.approx(np.exp(-1.0), rel=1e-2)
