"""The validation cases, shared by the tests (small grids) and bench/bench_fdtd.py (larger grids). Each case returns
a dict of results and the reference values. Structures are synthetic, with textbook indices."""

from __future__ import annotations

import numpy as np

import fdtd_gpu as fg
from references import (dipole_flux_3d, disk_resonance, ring_resonance, slab_bloch_frequency,  # noqa: F401
                        slab_te_neff, slab_tm_neff, tmm_reflectance)

SRC_REL_MIN = 1e-3     # waveguide: report T only where the source spectrum is above this fraction of its peak


def bragg(resolution: int = 200, pairs: int = 6, nfreq: int = 41, lam0: float = 1.0, eps_averaging: bool = True):
    """1-D quarter-wave stack (n_H 2.0, n_L 1.25 at lambda0) in air: reflectance against exact transfer matrices.
    lam0 = 1 um: the layer thicknesses (0.125 and 0.2 um) are whole numbers of cells at resolution 40 x k, and the
    layer faces are on cell faces. Another lam0 (for example 1.03 um) puts the faces between grid lines."""
    nH, nL = 2.0, 1.25
    dH, dL = lam0 / (4 * nH), lam0 / (4 * nL)
    stack_len = pairs * (dH + dL)
    pml, gap = 1.0, 1.0
    sz = 2 * pml + 2 * gap + stack_len + 1.0
    z0 = -sz / 2 + pml + 0.5 * gap                  # source
    zr = z0 + 0.25 * gap                             # reflection plane
    zs = z0 + gap                                    # start of the stack
    geometry = []
    z = zs
    for _ in range(pairs):
        geometry.append(fg.Block(size=fg.Vector3(fg.inf, fg.inf, dH), center=fg.Vector3(0, 0, z + dH / 2),
                                 material=fg.Medium(index=nH)))
        geometry.append(fg.Block(size=fg.Vector3(fg.inf, fg.inf, dL), center=fg.Vector3(0, 0, z + dH + dL / 2),
                                 material=fg.Medium(index=nL)))
        z += dH + dL
    zt = z + 0.5                                     # transmission plane
    fcen, df = 1.0, 0.8
    src = [fg.Source(fg.GaussianSource(fcen, fwidth=df), component=fg.Ex, center=fg.Vector3(0, 0, z0))]
    cell = fg.Vector3(0, 0, sz)

    def sim_with(geom):
        sim = fg.Simulation(cell_size=cell, resolution=resolution, geometry=geom, sources=src,
                            boundary_layers=[fg.PML(pml)], eps_averaging=eps_averaging)
        refl = sim.add_flux(fcen, df, nfreq, fg.FluxRegion(center=fg.Vector3(0, 0, zr)))
        tran = sim.add_flux(fcen, df, nfreq, fg.FluxRegion(center=fg.Vector3(0, 0, zt)))
        return sim, refl, tran

    # max_time 300 (the fields decay in about 100 um/c): fdtdx sets up its per-step switches for the whole max_time
    stop = fg.stop_when_fields_decayed(20, fg.Ex, fg.Vector3(0, 0, zt), 1e-7, max_time=300.0)
    sim0, refl0, tran0 = sim_with([])
    sim0.run(until_after_sources=stop)
    inc = np.array(sim0.get_fluxes(tran0))
    data = sim0.get_flux_data(refl0)
    sim1, refl1, tran1 = sim_with(geometry)
    sim1.load_minus_flux_data(refl1, data)
    sim1.run(until_after_sources=stop)
    freqs = np.array(sim1.get_flux_freqs(refl1))
    R = -np.array(sim1.get_fluxes(refl1)) / inc
    T = np.array(sim1.get_fluxes(tran1)) / inc
    layers = [(nH, dH), (nL, dL)] * pairs
    R_exact = tmm_reflectance(freqs, 1.0, layers, 1.0)
    return dict(freqs=freqs, R=R, T=T, R_exact=R_exact, max_dR=float(np.max(np.abs(R - R_exact))),
                max_RT=float(np.max(np.abs(R + T - 1))), seconds=sim0.run_seconds + sim1.run_seconds,
                device=sim1.device, steps=sim1.n_steps, resolution=resolution, cells=int(np.prod(sim1.shape)),
                lam0=lam0, eps_averaging=eps_averaging, setup_seconds=sim1.setup_seconds)


def waveguide(resolution: int = 20, nfreq: int = 5, three_d: bool = False, src_freq: float | None = None,
              eps_averaging: bool = True):
    """Straight strip waveguide (n 3.45 in 1.45, width 0.4 um; 2-D TM (Ez) or 3-D with a 0.22 um core): a mode
    source, mode monitors and flux planes at the input and the output. A lossless straight guide transmits 1.
    src_freq: the center frequency of the source (its mode is solved there); default the center of the monitors.
    src_rel is the amplitude spectrum of the source at each frequency over its peak. Where it is below SRC_REL_MIN,
    T_mode and T_flux are NaN: there both fluxes are float32 noise (for example 2.7e-9 of the peak at 0.72 / um
    with the source at 0.570 / um), and their ratio has no meaning."""
    w, n_core, n_clad = 0.4, 3.45, 1.45
    lx, pml = 8.0, 1.0
    if three_d:
        cell = fg.Vector3(lx, 4.0, 4.0)
        core = fg.Block(size=fg.Vector3(fg.inf, w, 0.22), material=fg.Medium(index=n_core))
        plane = fg.Vector3(0, 4.0 - 2 * pml, 4.0 - 2 * pml)
        parity = fg.EVEN_Z
        comp = fg.Ey
    else:
        cell = fg.Vector3(lx, 4.0, 0)
        core = fg.Block(size=fg.Vector3(fg.inf, w, fg.inf), material=fg.Medium(index=n_core))
        plane = fg.Vector3(0, 4.0 - 2 * pml, 0)
        parity = fg.ODD_Z
        comp = fg.Ez
    fcen, df = 1 / 1.55, 0.15
    xs, x_in, x_out = -lx / 2 + pml + 0.5, -lx / 2 + pml + 1.5, lx / 2 - pml - 0.5
    f_src = fcen if src_freq is None else src_freq
    src = fg.EigenModeSource(fg.GaussianSource(f_src, fwidth=df), center=fg.Vector3(xs), size=plane,
                             eig_band=1, eig_parity=parity)
    sim = fg.Simulation(cell_size=cell, resolution=resolution, geometry=[core], sources=[src],
                        boundary_layers=[fg.PML(pml)], default_material=fg.Medium(index=n_clad),
                        eps_averaging=eps_averaging)
    m_in = sim.add_mode_monitor(fcen, df, nfreq, fg.ModeRegion(center=fg.Vector3(x_in), size=plane))
    m_out = sim.add_mode_monitor(fcen, df, nfreq, fg.ModeRegion(center=fg.Vector3(x_out), size=plane))
    f_in = sim.add_flux(fcen, df, nfreq, fg.FluxRegion(center=fg.Vector3(x_in), size=plane))
    f_out = sim.add_flux(fcen, df, nfreq, fg.FluxRegion(center=fg.Vector3(x_out), size=plane))
    sim.run(until_after_sources=fg.stop_when_fields_decayed(20, comp, fg.Vector3(x_out), 1e-6))
    co_in = sim.get_eigenmode_coefficients(m_in, [1], eig_parity=parity)
    a_in = co_in.alpha[0]
    a_out = sim.get_eigenmode_coefficients(m_out, [1], eig_parity=parity).alpha[0]
    p_in, p_out = np.array(sim.get_fluxes(f_in)), np.array(sim.get_fluxes(f_out))
    freqs = np.array(sim.get_flux_freqs(f_in))
    spec = np.abs(src.src.fourier_transform(freqs))
    src_rel = spec / np.abs(src.src.fourier_transform(f_src))
    keep = np.where(src_rel >= SRC_REL_MIN, 1.0, np.nan)
    return dict(freqs=freqs, T_mode=keep * np.abs(a_out[:, 0]) ** 2 / np.abs(a_in[:, 0]) ** 2,
                T_flux=keep * p_out / p_in, back_in=np.abs(a_in[:, 1]) ** 2 / np.abs(a_in[:, 0]) ** 2,
                mode_fraction_in=np.abs(a_in[:, 0]) ** 2 / p_in, neff=co_in.neff[0], src_freq=f_src,
                src_rel=src_rel, seconds=sim.run_seconds, setup_seconds=sim.setup_seconds,
                device=sim.device, steps=sim.n_steps, resolution=resolution, cells=int(np.prod(sim.shape)),
                eps_averaging=eps_averaging)


def _match_resonances(modes, exact, m_est, spread: int = 2):
    """Each Harminv mode with the exact resonance (azimuthal order m) of the nearest frequency; then the stronger
    mode of each m (a rough staircase circle splits a mode m into a close pair of standing waves)."""
    rows = []
    for m in modes:
        best = None
        for mm in range(max(1, int(round(m_est(m.freq))) - spread), int(round(m_est(m.freq))) + spread + 1):
            for guess in (m.freq - 0.5e-3j, m.freq - 1e-6j, m.freq * 1.002 - 1e-4j, m.freq * 0.998 - 1e-4j):
                try:                         # the Newton iteration can fail from one guess: try the next one
                    fx = exact(mm, guess)
                    break
                except (RuntimeError, ZeroDivisionError):
                    fx = None
            if fx is not None and (best is None or abs(fx - m.freq) < abs(best[1] - m.freq)):
                best = (mm, fx)
        if best is None:
            continue
        rows.append(dict(f=m.freq, Q=m.Q, amp=abs(m.amp), m=best[0], f_exact=best[1].real,
                         Q_exact=best[1].real / (-2 * best[1].imag), rel_err=(m.freq - best[1].real) / best[1].real))
    strongest = {}
    for r in rows:
        if r["m"] not in strongest or r["amp"] > strongest[r["m"]]["amp"]:
            strongest[r["m"]] = r
    return [strongest[k] for k in sorted(strongest)]


def ring(resolution: int = 40, run_after: float = 300.0, eps_averaging: bool = True):
    """2-D ring (n 3.4 in air, inner radius 1.5 um, width 0.22 um; TM: Ez), excited by a short Ez dipole in the
    ring: the resonances by Harminv against the exact Bessel-function resonances, and the free spectral range."""
    n, a, w = 3.4, 1.5, 0.22
    b = a + w
    pml, pad = 1.0, 0.8
    sxy = 2 * (b + pad + pml)
    fcen, df = 0.65, 0.2
    geometry = [fg.Cylinder(radius=b, material=fg.Medium(index=n)), fg.Cylinder(radius=a, material=fg.Medium())]
    pt = fg.Vector3(a + w / 2, 0)
    src = [fg.Source(fg.GaussianSource(fcen, fwidth=df), component=fg.Ez, center=pt)]
    sim = fg.Simulation(cell_size=fg.Vector3(sxy, sxy, 0), resolution=resolution, geometry=geometry, sources=src,
                        boundary_layers=[fg.PML(pml)], eps_averaging=eps_averaging)
    h = fg.Harminv(fg.Ez, fg.Vector3(a + w / 2, 0.3), fcen, df)
    sim.run(fg.after_sources(h), until_after_sources=run_after)
    # |Q|: a mode of Q above about 1e6 can show a small growth (Q < 0) from the noise of the fit
    modes = [m for m in h.modes if abs(m.Q) > 200 and abs(m.amp) > 1e-3 * max(abs(q.amp) for q in h.modes)]
    rows = _match_resonances(modes, lambda mm, g: ring_resonance(mm, g, n, a, b),
                             lambda f: 2 * np.pi * f * slab_te_neff(f, n, 1.0, w) * (a + w / 2))
    fsr = np.diff([r["f"] for r in rows]) / np.diff([r["m"] for r in rows])
    fsr_exact = np.diff([r["f_exact"] for r in rows]) / np.diff([r["m"] for r in rows])
    return dict(modes=rows, fsr=fsr, fsr_exact=fsr_exact, harminv=[[q.freq, q.Q, abs(q.amp)] for q in h.modes],
                seconds=sim.run_seconds, setup_seconds=sim.setup_seconds,
                device=sim.device, steps=sim.n_steps, resolution=resolution, cells=int(np.prod(sim.shape)),
                eps_averaging=eps_averaging)


def disk(resolution: int = 40, te: bool = True, run_after: float = 300.0, eps_averaging: bool = True):
    """2-D disk (n 3.4 in air, radius 1.0 um). te=True: in-plane E (Hz), so the interface normal turns through all
    angles and the anisotropic part of the average acts; te=False: Ez. An Ex (TE) or Ez (TM) dipole 0.1 um inside
    the rim; Harminv on Hz (TE) or Ez. Reference: the exact resonances of the dielectric disk (disk_resonance)."""
    n, R = 3.4, 1.0
    pml, pad = 1.0, 0.8
    sxy = 2 * (R + pad + pml)
    fcen, df = 0.65, 0.2
    geometry = [fg.Cylinder(radius=R, material=fg.Medium(index=n))]
    src = [fg.Source(fg.GaussianSource(fcen, fwidth=df), component=fg.Ex if te else fg.Ez,
                     center=fg.Vector3(R - 0.1, 0))]
    sim = fg.Simulation(cell_size=fg.Vector3(sxy, sxy, 0), resolution=resolution, geometry=geometry, sources=src,
                        boundary_layers=[fg.PML(pml)], eps_averaging=eps_averaging)
    h = fg.Harminv(fg.Hz if te else fg.Ez, fg.Vector3((R - 0.1) * np.cos(0.3), (R - 0.1) * np.sin(0.3)), fcen, df)
    sim.run(fg.after_sources(h), until_after_sources=run_after)
    modes = [m for m in h.modes if abs(m.Q) > 200 and abs(m.amp) > 1e-2 * max(abs(q.amp) for q in h.modes)]
    rows = _match_resonances(modes, lambda mm, g: disk_resonance(mm, g, n, R, te=te),
                             lambda f: 2 * np.pi * f * n * R * 0.75, spread=5)
    return dict(modes=rows, te=te, harminv=[[q.freq, q.Q, abs(q.amp)] for q in h.modes], seconds=sim.run_seconds,
                setup_seconds=sim.setup_seconds, device=sim.device,
                steps=sim.n_steps, resolution=resolution, cells=int(np.prod(sim.shape)), eps_averaging=eps_averaging)


def slab(resolution: int = 20, e_normal: bool = False, width: float = 0.37, period: float = 0.5,
         run_after: float = 200.0, eps_averaging: bool = True):
    """2-D slab waveguide (n 3.45 in 1.45, width 0.37 um: the faces are between grid lines at resolutions
    20 x 2^k), periodic along x with the period 0.5 um, so the guided mode has beta = 2 pi / period. A dipole at the
    slab center (Ez: E parallel to the faces; e_normal: Ey, E normal to the faces, Hz polarization), Harminv after
    the sources. Reference: slab_bloch_frequency, the exact slab mode with the exact Yee dispersion along x and in
    time, so the error is the discretization across the slab only (the geometry). neff = beta / (2 pi f)."""
    n_core, n_clad = 3.45, 1.45
    pml, clad = 1.0, 1.5
    sy = 2 * (pml + clad) + width
    beta = 2 * np.pi / period
    comp = fg.Ey if e_normal else fg.Ez
    core = fg.Block(size=fg.Vector3(fg.inf, width, fg.inf), material=fg.Medium(index=n_core))
    neff_guess = (slab_tm_neff if e_normal else slab_te_neff)(1 / (period * 3.0), n_core, n_clad, width)
    fcen = 1 / (period * neff_guess)
    df = 0.3 * fcen
    src = [fg.Source(fg.GaussianSource(fcen, fwidth=df), component=comp, center=fg.Vector3(0, 0))]
    sim = fg.Simulation(cell_size=fg.Vector3(period, sy, 0), resolution=resolution, geometry=[core], sources=src,
                        boundary_layers=[fg.PML(pml, direction=fg.Y)], k_point=fg.Vector3(),
                        default_material=fg.Medium(index=n_clad), eps_averaging=eps_averaging)
    h = fg.Harminv(comp, fg.Vector3(0.1 * period, 0.02), fcen, df)
    sim.run(fg.after_sources(h), until_after_sources=run_after)
    f_ref = slab_bloch_frequency(beta, n_core, n_clad, width, sim.dx, sim.dt, e_normal=e_normal)
    good = [m for m in h.modes if abs(m.Q) > 1e3]         # lossless: |Q| is large, of either sign (noise)
    m = min(good, key=lambda q: abs(q.freq - f_ref))
    neff = beta / (2 * np.pi * m.freq)
    neff_ref = beta / (2 * np.pi * f_ref)
    return dict(f=m.freq, Q=m.Q, f_ref=f_ref, rel_err=(m.freq - f_ref) / f_ref, neff=neff, neff_ref=neff_ref,
                neff_err=neff - neff_ref, e_normal=e_normal, width=width, seconds=sim.run_seconds,
                setup_seconds=sim.setup_seconds, device=sim.device, steps=sim.n_steps, resolution=resolution,
                cells=int(np.prod(sim.shape)), eps_averaging=eps_averaging)


def dipole(resolution: int = 20, n: float = 1.0, nfreq: int = 5, half: float = 1.0):
    """3-D Ez point dipole in a homogeneous medium: the flux through a closed box (half width "half") against the
    analytic radiated power (Purcell factor 1). The PML is 1 um thick, 0.5 um from the box."""
    pml = 1.0
    s = 2 * (pml + half + 0.5)
    fcen, df = 0.5, 0.2
    gs = fg.GaussianSource(fcen, fwidth=df)
    sim = fg.Simulation(cell_size=fg.Vector3(s, s, s), resolution=resolution, sources=[fg.Source(gs, fg.Ez)],
                        boundary_layers=[fg.PML(pml)], default_material=fg.Medium(index=n))
    regions = []
    for ax in range(3):
        for sign in (-1, 1):
            c = [0.0, 0.0, 0.0]
            c[ax] = sign * half
            sz = [2 * half] * 3
            sz[ax] = 0
            regions.append(fg.FluxRegion(center=fg.Vector3(*c), size=fg.Vector3(*sz), weight=sign))
    box = sim.add_flux(fcen, df, nfreq, *regions)
    sim.run(until_after_sources=20)                  # the pulse leaves the 1 um box in about 2 um/c
    freqs = np.array(sim.get_flux_freqs(box))
    flux = np.array(sim.get_fluxes(box))
    ref = dipole_flux_3d(freqs, n, gs.fourier_transform(freqs))
    return dict(freqs=freqs, purcell=flux / ref, seconds=sim.run_seconds, device=sim.device, steps=sim.n_steps,
                resolution=resolution, n=n, half=half, cells=int(np.prod(sim.shape)))


def sheet(resolution: int = 20, n: float = 1.0):
    """1-D current sheet (Ex) in a homogeneous medium: the flux of the plane wave 3 um away against the exact flux
    |I(f)|^2 / (4 n) per side (E = -I / (2 n), H = n E for each outgoing wave), and the phase of Hy / Ex (0 for a
    wave along +z). Checks the absolute scale and the time alignment of the DFT fields."""
    freqs = np.linspace(0.4, 1.6, 7)
    gs = fg.GaussianSource(1.0, fwidth=1.0)
    sim = fg.Simulation(cell_size=fg.Vector3(0, 0, 8), resolution=resolution, boundary_layers=[fg.PML(1.0)],
                        sources=[fg.Source(gs, fg.Ex, center=fg.Vector3(0, 0, -2))],
                        default_material=fg.Medium(index=n))
    mon = sim.add_flux(freqs, fg.FluxRegion(center=fg.Vector3(0, 0, 1.0)))
    sim.run(until_after_sources=20)
    flux = np.array(sim.get_fluxes(mon))
    ref = np.abs(gs.fourier_transform(freqs)) ** 2 / (4 * n)
    ex = np.array([sim.get_dft_array(mon, fg.Ex, k).mean() for k in range(len(freqs))])
    hy = np.array([sim.get_dft_array(mon, fg.Hy, k).mean() for k in range(len(freqs))])
    return dict(freqs=freqs, flux_ratio=flux / ref, phase_H_E=np.angle(hy / ex), impedance=np.abs(ex / hy) * n,
                seconds=sim.run_seconds, device=sim.device, steps=sim.n_steps, resolution=resolution, n=n,
                cells=int(np.prod(sim.shape)))
