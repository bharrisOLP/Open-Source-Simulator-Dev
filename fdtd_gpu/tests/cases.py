"""The validation cases, shared by the tests (small grids) and bench/bench_fdtd.py (larger grids). Each case returns
a dict of results and the reference values. Structures are synthetic, with textbook indices."""

from __future__ import annotations

import numpy as np

import fdtd_gpu as fg
from references import (chi_exact, chi_yee, dipole_fields_3d, dipole_flux_3d, disk_resonance,  # noqa: F401
                        line_source_fields_2d, line_source_power_2d, ring_resonance, slab_bloch_frequency,
                        slab_te_neff, slab_tm_neff, tmm_reflectance, tmm_rt, uniform_root, yee_rt)

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


# ---------- near-to-far-field transform ----------
_YEE = {"Ex": (1, .5, .5), "Ey": (.5, 1, .5), "Ez": (.5, .5, 1), "Hx": (.5, 1, 1), "Hy": (1, .5, 1), "Hz": (1, 1, .5)}


def yee_point(sim, component: str, center) -> np.ndarray:
    """The position (um) of the Yee component of the cell that holds `center`: the position of a point source. fdtdx
    keeps the layout of Taflove (E_x at (i + 1/2, j, k), ...), and the point (i, j, k) is the center of cell (i, j, k)
    (see fdtd_gpu._averaging). An axis of size 0 has the coordinate 0."""
    return np.array([0.0 if sim.trivial[a] else sim._lo(a) + (sim._cell(center[a], a) + _YEE[component][a]) * sim.dx
                     for a in range(3)])


def box_regions(cls, half: float, axes=(0, 1, 2)):
    """The planes of a closed box of half width `half` about the origin on the given axes (weight -1 on the low
    faces), as FluxRegion or Near2FarRegion."""
    regions = []
    for ax in axes:
        for sign in (-1, 1):
            c = [0.0, 0.0, 0.0]
            c[ax] = sign * half
            sz = [2 * half if a in axes else 0.0 for a in range(3)]
            sz[ax] = 0
            regions.append(cls(center=fg.Vector3(*c), size=fg.Vector3(*sz), weight=sign))
    return regions


def sphere_rule(n_theta: int, n_phi: int):
    """Unit vectors (Q, 3) and weights (Q,) on the unit sphere: Gauss-Legendre in cos(theta), the trapezoidal rule in
    phi. Exact for polynomials of degree 2 n_theta - 1 in cos(theta) times trigonometric polynomials of degree below
    n_phi."""
    ct, wt = np.polynomial.legendre.leggauss(n_theta)
    ph = 2 * np.pi * np.arange(n_phi) / n_phi
    st = np.sqrt(1 - ct**2)
    u = np.stack([np.outer(st, np.cos(ph)), np.outer(st, np.sin(ph)), np.outer(ct, np.ones(n_phi))], axis=-1)
    return u.reshape(-1, 3), np.outer(wt, np.full(n_phi, 2 * np.pi / n_phi)).reshape(-1)


def _ff_arrays(ff):
    """{"Ex": (P, F), ...} -> E, H (F, P, 3)."""
    E = np.stack([ff[c].T for c in ("Ex", "Ey", "Ez")], axis=-1)
    H = np.stack([ff[c].T for c in ("Hx", "Hy", "Hz")], axis=-1)
    return E, H


def _compare(E, E_ref):
    """Per frequency: the global phase arg sum(E conj E_ref), the complex error max |E exp(-i phase) - E_ref| /
    max |E_ref| after that phase, and the pattern error max | |E|^2 - |E_ref|^2 | / max |E_ref|^2."""
    phase = np.angle(np.sum(E * np.conj(E_ref), axis=(1, 2)))
    scale = np.abs(E_ref).max(axis=(1, 2))
    err = np.abs(E * np.exp(-1j * phase)[:, None, None] - E_ref).max(axis=(1, 2)) / scale
    i, i_ref = (np.abs(E) ** 2).sum(-1), (np.abs(E_ref) ** 2).sum(-1)
    pattern = np.abs(i - i_ref).max(axis=1) / i_ref.max(axis=1)
    return phase, err, pattern


def near2far_dipole(resolution: int = 20, n: float = 1.0, nfreq: int = 5, halves=(1.0, 0.5), component: str = "Ez",
                    r_far: float = 1000.0, r_near: float = 3.0, n_theta: int = 24, n_phi: int = 48,
                    fcen: float = 0.5, df: float = 0.2):
    """3-D point dipole in a homogeneous medium: the near-to-far-field transform from closed boxes of the half widths
    `halves` (all in one run) against the analytic dipole fields (references.dipole_fields_3d) on a sphere of radius
    r_far (pattern, complex fields after one global phase, the radiated power) and at r_near (near terms); the power
    of the far field against the flux through the same box and the analytic power; the far fields of the boxes
    against each other. API: get_farfield against get_farfields_at, the shape of get_farfields, and DftNear2Far.flux
    through a cube of half width 0.1 r_far against the power on the sphere. The source and the monitors have the
    center frequency fcen and the width df (1/um)."""
    import time

    pml = 1.0
    s = 2 * (pml + max(halves) + 0.5)
    gs = fg.GaussianSource(fcen, fwidth=df)
    sim = fg.Simulation(cell_size=fg.Vector3(s, s, s), resolution=resolution, sources=[fg.Source(gs, component)],
                        boundary_layers=[fg.PML(pml)], default_material=fg.Medium(index=n))
    n2fs = [sim.add_near2far(fcen, df, nfreq, *box_regions(fg.Near2FarRegion, h)) for h in halves]
    boxes = [sim.add_flux(fcen, df, nfreq, *box_regions(fg.FluxRegion, h)) for h in halves]
    sim.run(until_after_sources=20)
    freqs = np.array(sim.get_near2far_freqs(n2fs[0]))
    current = gs.fourier_transform(freqs)
    r0 = yee_point(sim, component, fg.Vector3())
    axis = "xyz".index(component[1])
    u, w = sphere_rule(n_theta, n_phi)
    p_far = r0 + r_far * u
    E_ref, _ = dipole_fields_3d(p_far, freqs, n, current, r0, axis)
    p_near = r0 + r_near * u[:: max(1, len(u) // 200)]
    E_ref_near, _ = dipole_fields_3d(p_near, freqs, n, current, r0, axis)
    P_exact = dipole_flux_3d(freqs, n, current)
    rows = []
    t0 = time.perf_counter()
    for h, n2f, box in zip(halves, n2fs, boxes):
        E, H = _ff_arrays(sim.get_farfields_at(n2f, p_far))
        P_far = r_far**2 * (w * np.real(np.cross(E, np.conj(H)) * u).sum(-1)).sum(-1)
        flux = np.array(sim.get_fluxes(box))
        phase, err, pattern = _compare(E, E_ref)
        En, _ = _ff_arrays(sim.get_farfields_at(n2f, p_near))
        _, err_near, _ = _compare(En * np.exp(-1j * phase)[:, None, None], E_ref_near)
        rows.append(dict(half=h, E=E, P_far=P_far, flux=flux, phase=phase, err=err, pattern=pattern,
                         err_near=err_near))
    ff_seconds = (time.perf_counter() - t0) / len(halves)
    n2f = n2fs[0]
    one = np.array(sim.get_farfield(n2f, fg.Vector3(*p_far[7]))).reshape(len(freqs), 6)
    E7, H7 = _ff_arrays(sim.get_farfields_at(n2f, p_far[7:8]))
    api_point = np.abs(one - np.concatenate([E7[:, 0], H7[:, 0]], axis=-1)).max() / np.abs(one).max()
    plane = sim.get_farfields(n2f, 5, center=fg.Vector3(0, 0, 20), size=fg.Vector3(4, 2, 0))     # 20 x 10 points
    L = 0.1 * r_far
    flux_cube = np.zeros(len(freqs))
    for reg in box_regions(fg.FluxRegion, L):                                     # 40 x 40 points per face
        a = int(np.argmin(tuple(reg.size)))
        flux_cube += reg.weight * np.array(n2f.flux(a, fg.Volume(center=reg.center, size=reg.size), 20 / L))
    r = rows[0]
    return dict(freqs=freqs, component=component, n=n, resolution=resolution, halves=list(halves),
                far_over_flux=np.array([q["P_far"] / q["flux"] for q in rows]),
                far_over_exact=np.array([q["P_far"] / P_exact for q in rows]),
                flux_over_exact=np.array([q["flux"] / P_exact for q in rows]),
                field_err=np.array([q["err"] for q in rows]), pattern_err=np.array([q["pattern"] for q in rows]),
                near_err=np.array([q["err_near"] for q in rows]), phase=np.array([q["phase"] for q in rows]),
                omega_dt_half=np.pi * freqs * sim.dt,
                box_diff=np.abs(rows[0]["E"] - rows[1]["E"]).max(axis=(1, 2)) / np.abs(r["E"]).max(axis=(1, 2)),
                api_point=api_point, plane_shape=list(plane["Ez"].shape),
                cube_over_far=flux_cube / r["P_far"], r0=r0, eps=n2f.eps,
                seconds=sim.run_seconds, farfield_seconds=ff_seconds, device=sim.device, steps=sim.n_steps,
                cells=int(np.prod(sim.shape)))


def near2far_line(resolution: int = 20, n: float = 1.0, nfreq: int = 5, halves=(1.0, 0.5), component: str = "Ez",
                  r_far: float = 1000.0, r_near: float = 3.0, n_phi: int = 72, run_after: float = 40.0,
                  fcen: float = 0.5, df: float = 0.2):
    """2-D line source (x-y cell) in a homogeneous medium: the near-to-far-field transform (2-D Hankel form) from
    closed squares of the half widths `halves` against the analytic line-source fields (references.
    line_source_fields_2d) on a circle of radius r_far (pattern, complex fields after one global phase, power) and
    at r_near; the power of the far field against the flux through the same square and the analytic power
    (references.line_source_power_2d); the far fields of the squares against each other; DftNear2Far.flux through a
    square of half width 0.1 r_far against the power on the circle. component: "Ez" (E normal to the plane) or
    "Ex" (E in the plane). The source and the monitors have the center frequency fcen and the width df (1/um)."""
    import time

    pml = 1.0
    s = 2 * (pml + max(halves) + 0.5)
    gs = fg.GaussianSource(fcen, fwidth=df)
    sim = fg.Simulation(cell_size=fg.Vector3(s, s, 0), resolution=resolution, sources=[fg.Source(gs, component)],
                        boundary_layers=[fg.PML(pml)], default_material=fg.Medium(index=n))
    n2fs = [sim.add_near2far(fcen, df, nfreq, *box_regions(fg.Near2FarRegion, h, (0, 1))) for h in halves]
    boxes = [sim.add_flux(fcen, df, nfreq, *box_regions(fg.FluxRegion, h, (0, 1))) for h in halves]
    sim.run(until_after_sources=run_after)
    freqs = np.array(sim.get_near2far_freqs(n2fs[0]))
    current = gs.fourier_transform(freqs)
    r0 = yee_point(sim, component, fg.Vector3())
    phi = 2 * np.pi * np.arange(n_phi) / n_phi
    u = np.stack([np.cos(phi), np.sin(phi), np.zeros(n_phi)], axis=-1)
    E_ref, _ = line_source_fields_2d(r0 + r_far * u, freqs, n, current, r0, component)
    E_ref_near, _ = line_source_fields_2d(r0 + r_near * u, freqs, n, current, r0, component)
    P_exact = line_source_power_2d(freqs, current, in_plane=component != "Ez")
    rows = []
    t0 = time.perf_counter()
    for h, n2f, box in zip(halves, n2fs, boxes):
        E, H = _ff_arrays(sim.get_farfields_at(n2f, r0 + r_far * u))
        P_far = r_far * 2 * np.pi / n_phi * np.real(np.cross(E, np.conj(H)) * u).sum(-1).sum(-1)
        flux = np.array(sim.get_fluxes(box))
        phase, err, pattern = _compare(E, E_ref)
        En, _ = _ff_arrays(sim.get_farfields_at(n2f, r0 + r_near * u))
        _, err_near, _ = _compare(En * np.exp(-1j * phase)[:, None, None], E_ref_near)
        rows.append(dict(E=E, P_far=P_far, flux=flux, phase=phase, err=err, pattern=pattern, err_near=err_near))
    ff_seconds = (time.perf_counter() - t0) / len(halves)
    n2f = n2fs[0]
    L = 0.1 * r_far
    flux_sq = np.zeros(len(freqs))
    for reg in box_regions(fg.FluxRegion, L, (0, 1)):                             # 200 points per side
        a = int(np.argmin(tuple(reg.size)[:2]))
        flux_sq += reg.weight * np.array(n2f.flux(a, fg.Volume(center=reg.center, size=reg.size), 100 / L))
    r = rows[0]
    return dict(freqs=freqs, component=component, n=n, resolution=resolution, halves=list(halves),
                far_over_flux=np.array([q["P_far"] / q["flux"] for q in rows]),
                far_over_exact=np.array([q["P_far"] / P_exact for q in rows]),
                flux_over_exact=np.array([q["flux"] / P_exact for q in rows]),
                field_err=np.array([q["err"] for q in rows]), pattern_err=np.array([q["pattern"] for q in rows]),
                near_err=np.array([q["err_near"] for q in rows]), phase=np.array([q["phase"] for q in rows]),
                omega_dt_half=np.pi * freqs * sim.dt,
                box_diff=np.abs(rows[0]["E"] - rows[1]["E"]).max(axis=(1, 2)) / np.abs(r["E"]).max(axis=(1, 2)),
                square_over_far=flux_sq / r["P_far"], r0=r0, eps=n2f.eps,
                seconds=sim.run_seconds, farfield_seconds=ff_seconds, device=sim.device, steps=sim.n_steps,
                cells=int(np.prod(sim.shape)))


def closed_surface(half: float, h: float, axes):
    """Points (N, 3), outward normals (N, 3) and trapezoidal weights (N,) of a closed box (3-D, axes (0, 1, 2)) or
    square (2-D, axes (0, 1); the weights are lengths) of half width `half`, spacing h."""
    m = int(round(2 * half / h)) + 1
    t = np.linspace(-half, half, m)
    wt = np.full(m, 2 * half / (m - 1))
    wt[[0, -1]] *= 0.5
    pts, nrm, wts = [], [], []
    for a in axes:
        tang = [q for q in axes if q != a]
        grids = np.meshgrid(*[t] * len(tang), indexing="ij")
        w = np.ones(1)
        for _ in tang:
            w = np.multiply.outer(w, wt)
        for sign in (-1, 1):
            p = np.zeros((grids[0].size, 3))
            p[:, a] = sign * half
            for q, g in zip(tang, grids):
                p[:, q] = g.reshape(-1)
            nv = np.zeros_like(p)
            nv[:, a] = sign
            pts.append(p)
            nrm.append(nv)
            wts.append(w.reshape(-1))
    return np.concatenate(pts), np.concatenate(nrm), np.concatenate(wts)


def transform_exact(dims: int = 3, component: str = "Ez", h: float = 0.05, n: float = 1.5, half: float = 1.0,
                    freqs=(0.4, 0.6), compare_backends: bool = False):
    """The transform of exact fields (no FDTD): the fields of a point dipole (3-D, references.dipole_fields_3d) or of
    a line current (2-D, references.line_source_fields_2d) at r0 = (0.1, -0.05, 0.07) (z = 0 in 2-D) on a closed
    box or square (half width `half`, spacing h, trapezoidal rule) against the exact fields at 1000 um and 3 um from
    r0 (max |error| / max |exact|), and inside the surface at 0.4 um from r0 (max |E| / min |E_exact|: the
    extinction theorem gives zero). compare_backends (3-D): the JAX and NumPy sums on the far points."""
    import time

    from fdtd_gpu.near2far import radiate

    freqs = np.asarray(freqs, float)
    if dims == 3:
        r0 = np.array([0.1, -0.05, 0.07])
        u, _ = sphere_rule(8, 16)
        axes, triv, inside = (0, 1, 2), None, r0 + 0.4 * u[::8]

        def exact(p):
            return dipole_fields_3d(p, freqs, n, 1.0, r0, "xyz".index(component[1]))
    else:
        r0 = np.array([0.1, -0.05, 0.0])
        phi = 2 * np.pi * np.arange(36) / 36
        u = np.stack([np.cos(phi), np.sin(phi), np.zeros_like(phi)], axis=-1)
        axes, triv, inside = (0, 1), 2, r0 + 0.4 * u[::4]

        def exact(p):
            return line_source_fields_2d(p, freqs, n, 1.0, r0, component)
    far, near = r0 + 1000 * u, r0 + 3 * u
    pts, nrm, w = closed_surface(half, h, axes)
    E, H = exact(pts)
    J, M = np.cross(nrm, H) * w[:, None], -np.cross(nrm, E) * w[:, None]
    t0 = time.perf_counter()
    Et, Ht = radiate(np.concatenate([far, near, inside]), pts, J, M, freqs, eps=n * n, trivial_axis=triv)
    seconds = time.perf_counter() - t0
    nf, nn = len(far), len(near)
    (Ef, Hf), (En, _), (Ei, _) = exact(far), exact(near), exact(inside)

    def rel(a, b):
        return float(np.abs(a - b).max() / np.abs(b).max())
    out = dict(dims=dims, component=component, h=h, n=n, half=half, freqs=freqs, surface_cells=len(pts),
               far_E=rel(Et[:, :nf], Ef), far_H=rel(Ht[:, :nf], Hf), near_E=rel(Et[:, nf:nf + nn], En),
               inside=float(np.linalg.norm(Et[:, nf + nn:], axis=-1).max() / np.linalg.norm(Ei, axis=-1).min()),
               seconds=seconds)
    if compare_backends and dims == 3:
        Enp, _ = radiate(far, pts, J, M, freqs, eps=n * n, backend="numpy")
        Ejx, _ = radiate(far, pts, J, M, freqs, eps=n * n, backend="jax")
        out["jax_vs_numpy"] = rel(Ejx, Enp)
    return out


def _shell(center, half, n_face: int):
    """Points on the 6 faces of the box center +- half (half per axis), n_face x n_face points per face."""
    pts = []
    for a in range(3):
        t = [q for q in range(3) if q != a]
        g = np.meshgrid(*[np.linspace(center[q] - half[q], center[q] + half[q], n_face) for q in t], indexing="ij")
        for sign in (-1, 1):
            p = np.zeros((n_face * n_face, 3))
            p[:, a] = center[a] + sign * half[a]
            p[:, t[0]], p[:, t[1]] = g[0].reshape(-1), g[1].reshape(-1)
            pts.append(p)
    return np.concatenate(pts)


def near2far_distance(resolution: int = 20, cells=(1, 2, 3, 5, 10), nfreq: int = 5, n_face: int = 9):
    """How close to the surface the transform is accurate. 3-D Ez dipole in vacuum (as near2far_dipole; one closed
    box of half width 1 um): the fields of the transform on cube shells m cells (m in `cells`) outside the box of
    the DFT points (n_face x n_face points per face), against the analytic dipole fields after the global phase of
    the far fields (1000 um). The same for the transform of the exact dipole fields on the same surface points
    (trapezoidal rule, spacing dx): the error of the quadrature alone. Errors: max abs(error) / max abs(exact) of E
    and of H on each shell, the worst frequency."""
    from fdtd_gpu.near2far import radiate

    pml, half = 1.0, 1.0
    s = 2 * (pml + half + 0.5)
    fcen, df = 0.5, 0.2
    gs = fg.GaussianSource(fcen, fwidth=df)
    sim = fg.Simulation(cell_size=fg.Vector3(s, s, s), resolution=resolution, sources=[fg.Source(gs, fg.Ez)],
                        boundary_layers=[fg.PML(pml)])
    n2f = sim.add_near2far(fcen, df, nfreq, *box_regions(fg.Near2FarRegion, half))
    sim.run(until_after_sources=20)
    freqs = np.array(sim.get_near2far_freqs(n2f))
    current = gs.fourier_transform(freqs)
    r0 = yee_point(sim, "Ez", fg.Vector3())
    src, _, _ = sim._near2far_currents(n2f)
    lo, hi = src.min(axis=0), src.max(axis=0)                       # the box of the DFT points
    ctr, hw = (lo + hi) / 2, (hi - lo) / 2
    u, _ = sphere_rule(8, 16)
    E_far, _ = _ff_arrays(sim.get_farfields_at(n2f, r0 + 1000 * u))
    phase, far_err, _ = _compare(E_far, dipole_fields_3d(r0 + 1000 * u, freqs, 1.0, current, r0, 2)[0])
    rot = np.exp(-1j * phase)[:, None, None]
    pts, nrm, w = closed_surface(hw[0], sim.dx, (0, 1, 2))           # the same points as the DFT box
    pts = pts + ctr
    Ex_s, Hx_s = dipole_fields_3d(pts, freqs, 1.0, current, r0, 2)
    J, M = np.cross(nrm, Hx_s) * w[:, None], -np.cross(nrm, Ex_s) * w[:, None]

    def err(a, b):
        return float(np.max(np.abs(a - b).max(axis=(1, 2)) / np.abs(b).max(axis=(1, 2))))
    rows = {k: [] for k in ("err_E", "err_H", "exact_E", "exact_H")}
    for m in cells:
        p = _shell(ctr, hw + m * sim.dx, n_face)
        E_ref, H_ref = dipole_fields_3d(p, freqs, 1.0, current, r0, 2)
        E, H = _ff_arrays(sim.get_farfields_at(n2f, p))
        Ee, He = radiate(p, pts, J, M, freqs, eps=1.0)
        rows["err_E"].append(err(E * rot, E_ref))
        rows["err_H"].append(err(H * rot, H_ref))
        rows["exact_E"].append(err(Ee, E_ref))
        rows["exact_H"].append(err(He, H_ref))
    return dict(freqs=freqs, resolution=resolution, cells=list(cells), dx=sim.dx, far_err=float(far_err.max()),
                surface_points_match=bool(np.allclose(np.sort(src, axis=0), np.sort(pts, axis=0))), **rows,
                seconds=sim.run_seconds, device=sim.device, cells_total=int(np.prod(sim.shape)))


def near2far_subtract(resolution: int = 20, nfreq: int = 5, s2=(0.0, -1.6)):
    """load_minus_near2far_data with incident fields that differ from the total fields. 2-D (x-y), Ez, n = 1, the
    cell interior +-2 um, a closed square of half width 1 um. Run A: a line source S2 at s2, outside the square.
    Run B: S2 and a line source S1 at the origin, inside the square; two near2far objects on the same square, one
    with the DFT fields of A subtracted. Results (worst frequency): the far fields (1000 um, 72 points) of B - A and of
    B against the analytic field of S1 (one global phase); far(A) / max abs(field of S2) (zero for a source outside
    a closed surface); and at points 0.5 um from the center, inside the square, transform(A) against -(field of S2)
    (the transform of the fields of an outside source gives minus those fields inside; the fields of an inside
    source give zero there). At the same inner points, max abs(transform) / max abs(field of S2) of B - A
    ("inside_sub": close to 0, only S1 is left) and of B ("inside_plain": close to 1, the -S2 part)."""
    pml, half = 1.0, 1.0
    s = 2 * (pml + 2.0)
    fcen, df = 0.5, 0.2
    gs = fg.GaussianSource(fcen, fwidth=df)
    S1 = fg.Source(gs, fg.Ez)
    S2 = fg.Source(gs, fg.Ez, center=fg.Vector3(*s2))

    def sim_with(sources):
        return fg.Simulation(cell_size=fg.Vector3(s, s, 0), resolution=resolution, sources=sources,
                             boundary_layers=[fg.PML(pml)])
    simA = sim_with([S2])
    n2fA = simA.add_near2far(fcen, df, nfreq, *box_regions(fg.Near2FarRegion, half, (0, 1)))
    simA.run(until_after_sources=40)
    simB = sim_with([S1, S2])
    n2fB = simB.add_near2far(fcen, df, nfreq, *box_regions(fg.Near2FarRegion, half, (0, 1)))
    n2fBA = simB.add_near2far(fcen, df, nfreq, *box_regions(fg.Near2FarRegion, half, (0, 1)))
    simB.load_minus_near2far_data(n2fBA, simA.get_near2far_data(n2fA))
    simB.run(until_after_sources=40)
    freqs = np.array(simB.get_near2far_freqs(n2fB))
    current = gs.fourier_transform(freqs)
    r1, r2 = yee_point(simB, "Ez", S1.center), yee_point(simB, "Ez", S2.center)
    phi = 2 * np.pi * np.arange(72) / 72
    u = np.stack([np.cos(phi), np.sin(phi), np.zeros_like(phi)], axis=-1)
    far, inside = 1000 * u, 0.5 * u[::4]
    E1, _ = line_source_fields_2d(far, freqs, 1.0, current, r1)
    E2, _ = line_source_fields_2d(far, freqs, 1.0, current, r2)
    E2_in, _ = line_source_fields_2d(inside, freqs, 1.0, current, r2)
    EBA, _ = _ff_arrays(simB.get_farfields_at(n2fBA, far))
    EB, _ = _ff_arrays(simB.get_farfields_at(n2fB, far))
    EA, _ = _ff_arrays(simA.get_farfields_at(n2fA, far))
    EA_in, _ = _ff_arrays(simA.get_farfields_at(n2fA, inside))
    EBA_in, _ = _ff_arrays(simB.get_farfields_at(n2fBA, inside))
    EB_in, _ = _ff_arrays(simB.get_farfields_at(n2fB, inside))
    phase, err_sub, _ = _compare(EBA, E1)
    _, err_plain, _ = _compare(EB, E1)
    rot = np.exp(-1j * phase)[:, None, None]
    ext = np.abs(EA).max(axis=(1, 2)) / np.abs(E2).max(axis=(1, 2))
    scale_in = np.abs(E2_in).max(axis=(1, 2))
    inside_err = np.abs(EA_in * rot + E2_in).max(axis=(1, 2)) / scale_in
    return dict(freqs=freqs, resolution=resolution, s2=list(s2), err_sub=err_sub, err_plain=err_plain,
                outside_source_far=ext, inside_err=inside_err, inside_sub=np.abs(EBA_in).max(axis=(1, 2)) / scale_in,
                inside_plain=np.abs(EB_in).max(axis=(1, 2)) / scale_in, phase=phase,
                omega_dt_half=np.pi * freqs * simB.dt, seconds=simA.run_seconds + simB.run_seconds,
                device=simB.device, cells=int(np.prod(simB.shape)))


# ---------- dispersive and conductive media ----------
# Synthetic media with round numbers, in the textbook forms of the Lorentz and Drude models (Jackson, Classical
# Electrodynamics, 3rd ed., Sec. 7.5). A pole is (kind, f_n, gamma, sigma) in the units of the MIT package.
FILM = {"drude": dict(eps_inf=1.0, pole=("drude", 1.0, 0.05, 1.0), d=0.3),        # plasma edge at f = 1 / um
        "lorentz": dict(eps_inf=2.25, pole=("lorentz", 1.0, 0.05, 0.5), d=0.5)}   # resonance at f = 1 / um
UNIFORM = {"drude": dict(eps_inf=1.0, pole=("drude", 0.5, 0.02, 1.0)),
           "lorentz": dict(eps_inf=2.25, pole=("lorentz", 0.5, 0.02, 0.5)),
           "conductive": dict(eps_inf=2.25, sigma_d=0.2)}


def dispersive_medium(eps_inf: float, pole=None, sigma_d: float = 0.0):
    """A Medium with eps_inf, one pole (kind, f_n, gamma, sigma) or none, and D_conductivity sigma_d."""
    sus = []
    if pole is not None:
        kind, fn, gamma, sigma = pole
        cls = fg.DrudeSusceptibility if kind == "drude" else fg.LorentzianSusceptibility
        sus = [cls(frequency=fn, gamma=gamma, sigma=sigma)]
    return fg.Medium(epsilon=eps_inf, E_susceptibilities=sus, D_conductivity=sigma_d)


def film_nodes(zc, dx: float, zs: float, d: float, eps_averaging: bool, rule: str):
    """The weights of the Ex nodes zc (cell centers along z) for the exact Yee reference of a film [zs, zs + d]:
    (first, last, w_eps, w_chi): eps_j = 1 + w_eps (eps_inf - 1) + w_chi chi for the nodes first..last. f is the fill
    fraction of the voxel of the node on the points of `_averaging.lattice(1)` (the points of the average); E is
    parallel to the faces, so the average of eps_inf is the arithmetic mean. Staircase (or the rule "staircase"):
    w = the center is inside; rule "average": w_eps = f, w_chi = inside; rule "fill": w_eps = w_chi = f."""
    from fdtd_gpu._averaging import lattice

    inside = ((zc > zs) & (zc < zs + d)).astype(float)
    pts = zc[:, None] + lattice(1)[None, :, 0] * dx
    f = ((pts > zs) & (pts < zs + d)).mean(axis=1)
    if not eps_averaging or rule == "staircase":
        w_eps, w_chi = inside, inside
    elif rule == "average":
        w_eps, w_chi = f, inside
    else:
        w_eps, w_chi = f, f
    used = np.flatnonzero((w_eps > 0) | (w_chi > 0))
    first, last = int(used[0]), int(used[-1])
    return first, last, w_eps[first:last + 1], w_chi[first:last + 1]


def film(kind: str = "drude", resolution: int = 100, nfreq: int = 41, eps_averaging: bool = True,
         thickness: float | None = None, fwidth: float = 1.6):
    """1-D film of FILM[kind] in vacuum at normal incidence (Ex; cell along z; 41 frequencies from 0.6 to 1.4 / um;
    a Gaussian source of fwidth 1.6 / um, so the amplitude at the band edges is 0.29 of the peak). R from a flux
    plane with the incident fields subtracted, T from a flux plane behind the film (as check 1). thickness: the film
    thickness (default FILM[kind]["d"]); the first face is on a cell face at the resolutions 20 k, so a thickness
    that is not a whole number of cells puts the second face inside a cell.
    References: (a) the transfer matrix with the exact eps(f) (Born and Wolf: the Fresnel coefficients at the two
    faces and the phase in the film); (b) the exact R and T of the 1-D Yee grid (references.yee_rt) with the
    discrete chi of the ADE (chi_yee) and the face rule of the run (`film_nodes`; fdtd_gpu._dispersion.FACE_RULE):
    what remains against (b) is the float32 arithmetic and the end of the DFT; (c) as (b) with the float32 ADE
    coefficients (chi_yee(float32=True)), which removes the part of the coefficients. rule_err: (b) against (a),
    the error of the face rule and of the grid without FDTD."""
    from fdtd_gpu import _dispersion

    p = FILM[kind]
    d = p["d"] if thickness is None else float(thickness)
    pml, gap = 1.0, 1.0
    sz = 2 * pml + 2 * gap + 1.0 + np.ceil(d * 20 - 1e-9) / 20     # a whole number of cells at the resolutions 20 k
    z0 = -sz / 2 + pml + 0.5 * gap                  # source
    zr = z0 + 0.25 * gap                             # reflection plane
    zs = z0 + gap                                    # first face of the film (on a cell face)
    zt = zs + np.ceil(d * 20 - 1e-9) / 20 + 0.5      # transmission plane
    med = dispersive_medium(p["eps_inf"], p["pole"])
    geometry = [fg.Block(size=fg.Vector3(fg.inf, fg.inf, d), center=fg.Vector3(0, 0, zs + d / 2), material=med)]
    fcen, df = 1.0, 0.8
    src = [fg.Source(fg.GaussianSource(fcen, fwidth=fwidth), component=fg.Ex, center=fg.Vector3(0, 0, z0))]
    cell = fg.Vector3(0, 0, sz)

    def sim_with(geom):
        sim = fg.Simulation(cell_size=cell, resolution=resolution, geometry=geom, sources=src,
                            boundary_layers=[fg.PML(pml)], eps_averaging=eps_averaging)
        refl = sim.add_flux(fcen, df, nfreq, fg.FluxRegion(center=fg.Vector3(0, 0, zr)))
        tran = sim.add_flux(fcen, df, nfreq, fg.FluxRegion(center=fg.Vector3(0, 0, zt)))
        return sim, refl, tran

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
    R_exact, T_exact = tmm_rt(freqs, [lambda f: np.sqrt(p["eps_inf"] + chi_exact(f, p["pole"]))], [d])
    zc = sim1._lo(2) + (np.arange(sim1.shape[2]) + 0.5) * sim1.dx           # the Ex points (cell centers in z)
    rule = _dispersion.FACE_RULE
    first, last, w_eps, w_chi = film_nodes(zc, sim1.dx, zs, d, eps_averaging, rule)

    def yee(f32):
        return yee_rt(freqs, lambda f: 1 + w_eps * (p["eps_inf"] - 1) + w_chi * chi_yee(f, p["pole"], sim1.dt, f32),
                      sim1.dx, sim1.dt)

    (R_yee, T_yee), (R_y32, T_y32) = yee(False), yee(True)
    A, A_exact = 1 - R - T, 1 - R_exact - T_exact
    mid = np.abs(freqs - fcen) <= 0.25 + 1e-9

    def worst(Ra, Ta, sel=slice(None)):
        return float(max(np.max(np.abs(R - Ra)[sel]), np.max(np.abs(T - Ta)[sel])))

    return dict(kind=kind, freqs=freqs, R=R, T=T, R_exact=R_exact, T_exact=T_exact, R_yee=R_yee, T_yee=T_yee,
                max_dR=float(np.max(np.abs(R - R_exact))), max_dT=float(np.max(np.abs(T - T_exact))),
                max_dRT=worst(R_exact, T_exact), max_dRT_yee=worst(R_yee, T_yee), max_dRT_yee32=worst(R_y32, T_y32),
                rule_err=float(max(np.max(np.abs(R_yee - R_exact)), np.max(np.abs(T_yee - T_exact)))),
                max_dRT_mid=worst(R_exact, T_exact, mid), max_dRT_yee_mid=worst(R_yee, T_yee, mid),
                max_dA=float(np.max(np.abs(A - A_exact))), max_A=float(np.max(A_exact)),
                max_dR_yee=float(np.max(np.abs(R - R_yee))), max_dT_yee=float(np.max(np.abs(T - T_yee))),
                nodes=last - first + 1, fills=[float(w_eps[0]), float(w_eps[-1])], d=d, face_rule=rule,
                fwidth=fwidth, seconds=sim0.run_seconds + sim1.run_seconds, device=sim1.device, steps=sim1.n_steps,
                resolution=resolution, cells=int(np.prod(sim1.shape)), eps_averaging=eps_averaging)


def uniform(kind: str = "drude", resolution: int = 20, run_after: float = 100.0):
    """A uniform field (k = 0, so curl H = 0) in a periodic 1-D cell (1 um along z) filled with UNIFORM[kind]: Ex
    from a current over the whole cell; Harminv records Ex after the source.
    - drude, lorentz: the free oscillation (the strongest Harminv mode) against the root of eps_inf + chi(f) = 0,
      with the exact chi (references.uniform_root) and with the discrete chi of the ADE (chi_yee).
    - conductive: the decay rate of Ex (a fit of log |Ex| over 25 um/c) against sigma_D (E decays as
      exp(-sigma_D t) = exp(-sigma t / (eps0 eps)) with the SI conductivity sigma) and against the semi-implicit
      update of fdtdx, the factor (1 - b) / (1 + b) per step with b = sigma_D dt / 2."""
    p = UNIFORM[kind]
    L = 1.0
    med = dispersive_medium(p["eps_inf"], p.get("pole"), p.get("sigma_d", 0.0))
    if kind == "conductive":
        gs, fcen, df = fg.GaussianSource(0.1, fwidth=0.5), 1.0, 1.0      # Harminv only records the signal here
    else:
        f_l = uniform_root(p["pole"], p["eps_inf"]).real
        gs, fcen, df = fg.GaussianSource(f_l, fwidth=0.2), f_l, 0.4
    src = [fg.Source(gs, fg.Ex, center=fg.Vector3(), size=fg.Vector3(0, 0, L))]
    sim = fg.Simulation(cell_size=fg.Vector3(0, 0, L), resolution=resolution, sources=src, k_point=fg.Vector3(),
                        default_material=med)
    h = fg.Harminv(fg.Ex, fg.Vector3(0, 0, 0.1), fcen, df)
    sim.run(fg.after_sources(h), until_after_sources=run_after)
    out = dict(kind=kind, seconds=sim.run_seconds, device=sim.device, steps=sim.n_steps, resolution=resolution,
               dt=sim.dt, cells=int(np.prod(sim.shape)))
    if kind == "conductive":
        s = np.asarray(h.data, float)
        t = np.arange(len(s)) * h.data_dt
        keep = t <= 25.0
        slope = np.polyfit(t[keep], np.log(np.abs(s[keep])), 1)[0]
        sd = p["sigma_d"]
        rate_yee = 2 / sim.dt * np.arctanh(sd * sim.dt / 2)
        resid = np.log(np.abs(s[keep])) - np.polyval(np.polyfit(t[keep], np.log(np.abs(s[keep])), 1), t[keep])
        return dict(out, rate=-slope, sigma_d=sd, rate_yee=rate_yee, rel_err=-slope / sd - 1,
                    rel_err_yee=-slope / rate_yee - 1, max_log_resid=float(np.max(np.abs(resid))),
                    sign_changes=int(np.count_nonzero(np.diff(np.sign(s[keep])))))
    m = max(h.modes, key=lambda q: abs(q.amp))
    exact, yee = uniform_root(p["pole"], p["eps_inf"]), uniform_root(p["pole"], p["eps_inf"], dt=sim.dt)
    return dict(out, f=m.freq, decay=m.decay, Q=m.Q, f_exact=exact.real, decay_exact=exact.imag, f_yee=yee.real,
                decay_yee=yee.imag, f_err=m.freq / exact.real - 1, f_err_yee=m.freq / yee.real - 1,
                decay_err=m.decay / exact.imag - 1, decay_err_yee=m.decay / yee.imag - 1, n_modes=len(h.modes))
