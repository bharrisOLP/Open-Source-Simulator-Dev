"""A forward-biased PIN phase shifter with self-heating: the carriers (charge-gpu), the heat of the current
(thermal-gpu) and the mode index with both index changes (meow-gpu), on one CrossSection.

A forward current changes the index of the waveguide in two ways, with opposite signs:
- the injected free carriers lower the index and add loss (plasma dispersion; by default the fit at 1.55 um of
  R. A. Soref and B. R. Bennett, IEEE J. Quantum Electron. 23, 123 (1987): charge_gpu.modulator.plasma_dispersion);
- the heat of the current raises the temperature, and dn/dT of the materials raises the index.

1. Carriers (pin_electrical): the cross-section on the charge window (physics.charge_device: the cell centers, an
   ohmic contact on the left and on the right window side), the drift-diffusion of charge-gpu (drift_diffusion.sweep:
   Newton with SuperLU, continuation from 0 V) at the forward biases V = v_p - v_n. I' (A/m): the current per length.
2. Heat (heat_source): the heat density of the drift-diffusion solution without the thermoelectric terms (the
   Peltier and Thomson heat; G. K. Wachutka, IEEE Trans. Computer-Aided Design 9, 1141 (1990)):

       H = -J_n . grad(phi_n) - J_p . grad(phi_p) + q R (phi_p - phi_n)

   phi_n and phi_p are the quasi-Fermi potentials (J_n = -q mu_n n grad(phi_n), J_p = -q mu_p p grad(phi_p)). The
   first two terms are the Joule heat of the two currents (|J|^2 / sigma: never negative); the last term is the heat
   of the recombination (positive in forward bias). The discrete form uses the Scharfetter-Gummel currents of
   charge-gpu: a face between two semiconductor cells a and b gives -I_n (phi_n,b - phi_n,a) - I_p (phi_p,b -
   phi_p,a), with I_n and I_p the face currents from a to b, one half to each cell; a contact face gives the same
   with the contact voltage at the contact node, all to its cell; each cell gives q R A (phi_p - phi_n). Summation by
   parts with the discrete continuity equations (div J_n = q R, div J_p = -q R in each cell) gives sum H = I' V
   exactly, to the residual of the solve: all of the electrical power becomes heat. This sum is an identity of the
   discrete scheme (it holds for any cell potentials): it checks the continuity residual and the flux code, not where
   the heat goes; the local check is that each term is not negative.
   The electrostatic J . E = -J . grad(psi) is not the heat source: its sum is I' (psi_p - psi_n) at the two contact
   nodes, I' (V - V_c) with V_c = V_T (asinh(N_A / 2 n_i) + asinh(N_D / 2 n_i)) (0.952 V at 1e18 cm^-3 and 300 K), so
   it is negative below V_c. Heat.je gives its sum (a check).
3. Temperature (thermal-gpu, steady; the model of physics.temperature: the bottom of the thermal window at
   t_ambient, convection h = top_h W/m^2/K at the top, adiabatic sides or sides at side_temperature): the heat of
   each charge cell goes to the thermal cells in proportion to the area of overlap (the sum stays the same).
4. Modes (meow-gpu), full mode solves of the fundamental mode with the carrier change, with the thermal change, and
   with both. The carrier change: the plasma dispersion of the carriers at V against the carriers at 0 V (absolute
   concentrations, as physics.modulator_vpil), dk = dalpha lambda / (4 pi), averaged over sub x sub points in the
   optics cell around each Yee point (as charge-gpu's examples/pn_modulator_rc.py). The thermal change: dn_dT dT at
   the Yee points (the interpolation of physics.phase_shifter_ppi). First-order estimates: the weights of
   switching.index_weights (of the mode without a change) times the complex dn, summed.

Outputs per bias (PinPhaseShifter): dneff_carrier, dneff_thermal, dneff (both changes in one solve), the phase per
length, L_pi, P_pi, I_pi, the loss that the current adds, and the bias and current where the two parts cancel
(crossover). The carriers are solved at one temperature (t_ambient): the heat does not change the currents.

Limit: heat_source reads the private helper charge_gpu.drift_diffusion._fluxes and the private fields of its _Mesh
(face and contact indices, the contact node potentials, the cell areas). A refactor of charge-gpu can change them;
the tests of the heat sum find a change of the flux convention. A public accessor of the face currents in
charge-gpu is a later change of that package.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from photonsuite.geometry import CrossSection, Material, Rect
from photonsuite.physics import (ConvergenceError, Temperature, Window, _centers, _require_x64, charge_device,
                                 mode_solve, yee_points)

DRIFT_DIFFUSION = "drift-diffusion"
DB_PER_NEPER = 10 * np.log10(np.e)

# Textbook values, as in the other examples of the suite (n at 1.55 um; dn/dT of Si: G. Cocorullo and I. Rendina,
# Electron. Lett. 28, 83 (1992); eps_r: Sze); kappa in W/m/K.
SI = Material("Si", n=3.476, dn_dT=1.86e-4, kappa=148.0, eps_r=11.7, semiconductor=True)
SIO2 = Material("SiO2", n=1.444, dn_dT=0.95e-5, kappa=1.4, eps_r=3.9)


def pin_rib(width=0.5, height=0.22, slab=0.09, half_span=1.5, gap=0.3, doping=1e18, background=-1e15, box=2.0,
            si: Material = SI, sio2: Material = SIO2, slab_half_width: float | None = None,
            substrate_half_width: float | None = None, substrate: float = 3.0) -> CrossSection:
    """The lateral PIN rib of charge_gpu.injection.rib_pin as a CrossSection (um, cm^-3; synthetic, no process data):
    a rib (width x height) on a slab, the intrinsic zone |x| < width/2 + gap with the background doping (default p
    1e15 cm^-3), p+ (``doping``) on the left and n+ on the right of it, in SiO2, on a buried oxide of ``box`` um over
    a Si substrate ``substrate`` um thick. Tags: "substrate", "slab", "p", "n", "core".

    The slab ends at +-slab_half_width (default half_span, the contacts); the substrate spans +-substrate_half_width
    (default half_span + 3.5). The charge window (-half_span, half_span) x (-0.5, height + 0.5) at 0.01 um samples only
    the cells inside it, so physics.charge_device gives the Device of rib_pin() for every slab_half_width >=
    half_span: a longer slab changes only the thermal model (a heat path along the doped Si)."""
    edge = width / 2 + gap
    ws = half_span if slab_half_width is None else slab_half_width
    wsub = half_span + 3.5 if substrate_half_width is None else substrate_half_width
    if ws < half_span:
        raise ValueError("slab_half_width must be at least half_span (the contacts are at the slab ends)")
    return CrossSection(sio2, (Rect(-wsub, wsub, -box - substrate, -box, si, tag="substrate"),
                               Rect(-ws, ws, 0.0, slab, si, doping=background, tag="slab"),
                               Rect(-ws, -edge, 0.0, slab, si, doping=-doping, tag="p"),
                               Rect(edge, ws, 0.0, slab, si, doping=doping, tag="n"),
                               Rect(-width / 2, width / 2, 0.0, height, si, doping=background, tag="core")))


@dataclass
class PinElectrical:
    device: object          # the charge_gpu Device of the cross-section on the charge window
    window: Window
    biases: np.ndarray      # forward biases V = v_p - v_n (V)
    solutions: list         # drift_diffusion.DDSolution per bias
    reference: object       # the DDSolution at 0 V
    seconds: float          # the time of the sweep


def pin_electrical(xs: CrossSection, charge: Window, biases, tau: float = 1e-9, **kw) -> PinElectrical:
    """Drift-diffusion of the cross-section on the charge window at the forward biases (V > 0, increasing): sweep from
    0 V (Newton with SuperLU unless kw says otherwise), SRH lifetimes tau_n = tau_p = tau (s, at 300 K). The window
    needs a p contact and an n contact (the semiconductor cells on its left and right sides)."""
    from charge_gpu import drift_diffusion as dd

    _require_x64()
    biases = np.asarray(biases, float)
    if biases.ndim != 1 or biases.size == 0 or np.any(biases <= 0) or np.any(np.diff(biases) <= 0):
        raise ValueError("biases: forward biases (V > 0) in increasing order")
    dev = charge_device(xs, charge)
    kinds = set()
    for column, contact in ((dev.n_net[0, :], dev.contacts[0]), (dev.n_net[-1, :], dev.contacts[1])):
        if contact.mask.any():                      # the rule of charge-gpu: n if the mean net doping is positive
            kinds.add("n" if np.mean(column[contact.mask]) > 0 else "p")
    if kinds != {"n", "p"}:
        raise ValueError("a PIN needs a p contact and an n contact: semiconductor cells with p doping on one side of "
                         "the charge window and n doping on the other")
    t0 = time.perf_counter()
    sols = dd.sweep(dev, [0.0, *biases], tau_n=tau, tau_p=tau, **kw)
    seconds = time.perf_counter() - t0
    bad = [s.bias for s in sols if not s.converged]
    if bad:
        raise ConvergenceError(f"drift-diffusion at {bad} V: not converged")
    return PinElectrical(dev, charge, biases, sols[1:], sols[0], seconds)


@dataclass
class Heat:
    joule: np.ndarray           # (nx, ny) W/m per charge cell: the Joule heat of the electron and hole currents
    recombination: np.ndarray   # (nx, ny) W/m per charge cell: q R A (phi_p - phi_n)
    je: float                   # W/m: the sum of the electrostatic J . E (not a heat source; module docstring)

    @property
    def total(self) -> np.ndarray:
        """(nx, ny) W/m per charge cell: the heat source H A."""
        return self.joule + self.recombination


def heat_source(sol) -> Heat:
    """The heat of a drift-diffusion solution (charge_gpu.drift_diffusion.DDSolution), per charge cell (module
    docstring, step 2). The face currents are those of the solver (drift_diffusion._fluxes, a private helper of
    charge-gpu: a change there can change the heat; the tests check the sum against I' V)."""
    from charge_gpu.drift_diffusion import _fluxes, srh
    from charge_gpu.poisson import Q

    m = sol.mesh
    psi = np.asarray(sol.psi, float).ravel()
    n, p = sol.n.ravel()[m.sflat], sol.p.ravel()[m.sflat]
    (_, _, _, jn, jp), (_, _, _, jnc, jpc) = _fluxes(m, psi, n, p)
    ps = psi[m.sflat]
    phi_n = ps - m.vt * np.log(n / m.n_i)
    phi_p = ps + m.vt * np.log(p / m.n_i)
    v_c = np.where(np.asarray(sol.contact_kinds)[m.cid] == "n", sol.v_n, sol.v_p)
    face = -Q * (jn * (phi_n[m.sb] - phi_n[m.sa]) + jp * (phi_p[m.sb] - phi_p[m.sa]))
    cface = -Q * (jnc * (v_c - phi_n[m.cs]) + jpc * (v_c - phi_p[m.cs]))
    ns = m.n_semi
    joule = 0.5 * (np.bincount(m.sa, face, ns) + np.bincount(m.sb, face, ns)) + np.bincount(m.cs, cface, ns)
    rec = Q * srh(n, p, m.n_i, sol.tau_n, sol.tau_p)[0] * m.svol * (phi_p - phi_n)
    je = -Q * (np.sum((jn + jp) * (psi[m.fb] - psi[m.fa])) + np.sum((jnc + jpc) * (m.cpsi - psi[m.cflat])))

    def grid(v):
        out = np.zeros(m.n_cells)
        out[m.sflat] = v
        return out.reshape(m.shape)

    return Heat(grid(joule), grid(rec), float(je))


def cell_overlap(src, dst) -> np.ndarray:
    """(len(src) - 1, len(dst) - 1): the part of each source cell (edges src) inside each destination cell."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    lo = np.maximum(src[:-1, None], dst[None, :-1])
    hi = np.minimum(src[1:, None], dst[None, 1:])
    return np.clip(hi - lo, 0.0, None) / np.diff(src)[:, None]


def to_thermal(power, x, y, win: Window) -> np.ndarray:
    """The heat density (W/m^3) of the thermal cells of win from the power per length of the cells on the edges x, y
    (um): each cell gives its power to the thermal cells in proportion to the area of overlap, so the sum stays the
    same. ValueError if a cell with power is not completely inside the thermal window."""
    power = np.asarray(power, float)
    xt, yt = win.edges()
    ox, oy = cell_overlap(x, xt), cell_overlap(y, yt)
    inside = np.outer(ox.sum(axis=1), oy.sum(axis=1))
    if np.any((power != 0) & (inside < 1 - 1e-9)):
        raise ValueError("a cell of the heat source is not inside the thermal window")
    return (ox.T @ power @ oy) / np.outer(np.diff(xt), np.diff(yt)) * 1e12


def uniform_heat(xs: CrossSection, win: Window, tags, power_per_length: float) -> np.ndarray:
    """The heat density (W/m^3) of power_per_length (W/m) spread uniformly over the thermal cells of the shapes with
    these tags (one tag or a sequence), as physics.thermal_problem does for a heater."""
    tags = (tags,) if isinstance(tags, str) else tuple(tags)
    unknown = [t for t in tags if not any(s.tag == t for s in xs.shapes)]
    if not tags or unknown:
        raise ValueError(f"no shape with the tags {unknown or tags} in the cross-section")
    x, y = win.edges()
    X, Y = np.meshgrid(_centers(x), _centers(y), indexing="ij")
    mask = np.any([xs.mask(t, X, Y) for t in tags], axis=0)
    if not mask.any():
        raise ValueError(f"no thermal cell inside the shapes {tags}: use a smaller step")
    return np.where(mask, power_per_length / (mask.sum() * (win.step * 1e-6) ** 2), 0.0)


def steady_temperature(xs: CrossSection, win: Window, q, t_ambient: float = 300.0, top_h: float = 10.0,
                       side_temperature: float | None = None) -> Temperature:
    """The steady temperature of the thermal window with the heat density q (W/m^3) per cell; the boundary
    conditions of physics.temperature (the bottom at t_ambient, convection top_h at the top), with adiabatic sides,
    or with both sides at side_temperature (K: a heat sink over the whole height of the window sides). Raises
    ConvergenceError when the solve does not converge."""
    from thermal_gpu.solver import solve_steady

    x, y = win.edges()
    xc, yc = _centers(x), _centers(y)
    _, _, prop = xs.sample(*np.meshgrid(xc, yc, indexing="ij"))
    bc = {"ymin": ("dirichlet", t_ambient), "ymax": ("robin", top_h, t_ambient)}
    if side_temperature is not None:
        bc["xmin"] = bc["xmax"] = ("dirichlet", float(side_temperature))
    res = solve_steady(x * 1e-6, y * 1e-6, prop["kappa"], np.asarray(q, float), bc)
    if not res.converged:
        raise ConvergenceError(f"thermal solve: residual {res.residual:.1e} after {res.iterations} iterations")
    rise = RegularGridInterpolator((xc, yc), res.T - t_ambient, bounds_error=False, fill_value=None)
    return Temperature(rise, res.T, xc, yc, res.iterations, res.residual)


def carrier_index(sol, ref, xc, yc, wavelength: float, step: float, sub: int = 8, plasma=None):
    """The complex index change dn + i dk (a function of X, Y in um) of the carriers of sol against those of ref:
    plasma(n0, p0, n1, p1) -> (dn, dalpha in 1/m) on the charge cells (centers xc, yc in um; 0 outside), dk = dalpha
    lambda / (4 pi), averaged over sub x sub points in the square of side ``step`` around each point."""
    if plasma is None:
        from charge_gpu.modulator import plasma_dispersion as plasma
    dn, dalpha = plasma(ref.n, ref.p, sol.n, sol.p)
    f = RegularGridInterpolator((xc, yc), dn + 1j * dalpha * (wavelength * 1e-6) / (4 * np.pi), bounds_error=False,
                                fill_value=0.0)
    off = ((np.arange(sub) + 0.5) / sub - 0.5) * step
    ox, oy = (a.ravel() for a in np.meshgrid(off, off, indexing="ij"))

    def index(X, Y):
        pts = np.stack([(np.ravel(X)[:, None] + ox).ravel(), (np.ravel(Y)[:, None] + oy).ravel()], -1)
        return f(pts).reshape(np.size(X), ox.size).mean(axis=1).reshape(np.shape(X))

    return index


@dataclass
class PinPhaseShifter:
    wavelength: float                # um
    biases: np.ndarray               # V (forward)
    current: np.ndarray              # I' (A/m): the current per length out of the n contact
    heat: np.ndarray                 # W/m: the sum of the heat density on the thermal grid (I' V)
    heat_joule: np.ndarray           # W/m: the Joule part of the drift-diffusion heat
    heat_recombination: np.ndarray   # W/m: the recombination part
    je: np.ndarray                   # W/m: the sum of the electrostatic J . E (a check, not the heat)
    rise_semiconductor: np.ndarray   # K: the area mean temperature rise of the semiconductor cells of the charge window
    rise_core: np.ndarray | None     # K: the area mean rise of the thermal cells of the core (core_tag), else None
    neff0: complex                   # the mode index at 0 V (carriers at 0 V, no heat)
    dneff_carrier: np.ndarray        # complex: the change with the carrier change only (full mode solve)
    dneff_thermal: np.ndarray        # complex: with the thermal change only
    dneff: np.ndarray                # complex: with both changes
    first_carrier: np.ndarray        # complex: the first-order estimates of the two parts
    first_thermal: np.ndarray
    heat_model: str                  # "drift-diffusion", or the tags of a uniform heat
    temperatures: list               # physics.Temperature per bias
    electrical: PinElectrical
    seconds: float                   # the time of the thermal and mode part

    @property
    def power(self) -> np.ndarray:
        """I' V (W/m)."""
        return self.current * self.biases

    @property
    def phase_per_length(self) -> np.ndarray:
        """The net phase change per length (rad/m): 2 pi Re(dneff) / lambda (negative: the carriers win)."""
        return 2 * np.pi * np.real(self.dneff) / (self.wavelength * 1e-6)

    @property
    def l_pi(self) -> np.ndarray:
        """The length (m) for a phase change of pi: lambda / (2 |Re dneff|)."""
        with np.errstate(divide="ignore"):
            return self.wavelength * 1e-6 / (2 * np.abs(np.real(self.dneff)))

    @property
    def p_pi(self) -> np.ndarray:
        """The power (W) for pi: I' V L_pi."""
        return self.power * self.l_pi

    @property
    def i_pi(self) -> np.ndarray:
        """The current (A) for pi: I' L_pi."""
        return self.current * self.l_pi

    @property
    def loss_db_per_cm(self) -> np.ndarray:
        """The loss of the mode that the current adds (dB/cm), from Im(dneff)."""
        return DB_PER_NEPER * 4 * np.pi * np.imag(self.dneff) / (self.wavelength * 1e-4)

    @property
    def loss_in_l_pi_db(self) -> np.ndarray:
        """The added loss over L_pi (dB)."""
        return self.loss_db_per_cm * self.l_pi * 100.0

    def crossover(self) -> dict | None:
        """The first zero of Re(dneff) after a nonzero value: in a bias interval where it changes sign, the bias, the
        current and the power at the zero by linear interpolation in the current (an estimate: use a finer bias list
        near it); or a bias sample where it is exactly 0 (the interval is that one bias). None: no zero."""
        d = np.real(self.dneff)
        for k in range(1, len(d)):
            if d[k - 1] * d[k] < 0:
                f = d[k - 1] / (d[k - 1] - d[k])
                i = self.current[k - 1] + f * (self.current[k] - self.current[k - 1])
                v = float(np.interp(i, self.current[k - 1:k + 1], self.biases[k - 1:k + 1]))
                return {"bias": v, "current": float(i), "power": float(i * v), "interval": (float(self.biases[k - 1]),
                                                                                        float(self.biases[k]))}
            if d[k] == 0 and d[k - 1] != 0:
                v, i = float(self.biases[k]), float(self.current[k])
                return {"bias": v, "current": i, "power": i * v, "interval": (v, v)}
        return None


def pin_optical(xs: CrossSection, optics: Window, thermal: Window, wavelength: float, electrical: PinElectrical,
                heat=DRIFT_DIFFUSION, plasma=None, sub: int = 8, core_tag: str | None = None,
                t_ambient: float = 300.0, top_h: float = 10.0, solver=None,
                side_temperature: float | None = None) -> PinPhaseShifter:
    """The heat, the temperature and the mode index (module docstring, steps 2 to 4) for the solutions of
    pin_electrical. heat: "drift-diffusion" (the heat density of the solution), or one tag or a sequence of tags
    (I' V spread uniformly over the thermal cells of these shapes: a simpler model, and the check against
    physics.phase_shifter_ppi). plasma: (n0, p0, n1, p1) in m^-3 -> (dn, dalpha in 1/m); None: Soref and Bennett at
    1.55 um (then the wavelength must be 1.55 um). The charge window must be inside the thermal window.
    side_temperature: None (adiabatic window sides) or the temperature (K) of both window sides (steady_temperature).
    The thermal environment (the window width, a heat path to the contacts) changes dneff_thermal much: README."""
    from meow_gpu.meow_backend import GPUModeSolver

    from photonsuite.switching import index_weights

    _require_x64()
    if plasma is None and abs(wavelength - 1.55) > 1e-9:
        raise ValueError("the default plasma dispersion (Soref and Bennett) is the fit at 1.55 um: give plasma= for "
                         "another wavelength")
    t0 = time.perf_counter()
    dev = electrical.device
    xe, ye = np.asarray(dev.x) * 1e6, np.asarray(dev.y) * 1e6
    xc, yc = _centers(xe), _centers(ye)
    semi_area = np.where(dev.semiconductor, np.outer(np.diff(xe), np.diff(ye)), 0.0)
    semi_pts = np.stack([a[dev.semiconductor] for a in np.meshgrid(xc, yc, indexing="ij")], -1)
    xt, yt = thermal.edges()
    cell_m2 = np.outer(np.diff(xt), np.diff(yt)) * 1e-12
    core = None
    if core_tag is not None:
        core = xs.mask(core_tag, *np.meshgrid(_centers(xt), _centers(yt), indexing="ij"))
        if not core.any():
            raise ValueError(f"no thermal cell inside the core {core_tag!r}: use a smaller step")

    solver = solver or GPUModeSolver(precision="mixed", extra=4)
    neff, e, h = mode_solve(xs, optics, wavelength, 1, solver=solver)
    points = yee_points(optics)
    props = [xs.sample(X, Y)[2] for X, Y in points]
    w = index_weights(optics, [pp["n"] for pp in props], e[0], h[0])

    def first(f):
        return complex(sum(np.sum(wc * f(X, Y)) for wc, (X, Y) in zip(w, points)))

    def change(dn):
        return complex(mode_solve(xs, optics, wavelength, 1, dn=dn, solver=solver)[0][0] - neff[0])

    rows = {k: [] for k in ("heat", "joule", "rec", "je", "rise_s", "rise_c", "dc", "dt", "db", "fc", "ft")}
    temps = []
    for v, sol in zip(electrical.biases, electrical.solutions):
        hs = heat_source(sol)
        if isinstance(heat, str) and heat == DRIFT_DIFFUSION:
            q = to_thermal(hs.total, xe, ye, thermal)
        else:
            q = uniform_heat(xs, thermal, heat, sol.current * v)
        th = steady_temperature(xs, thermal, q, t_ambient, top_h, side_temperature)
        temps.append(th)
        rows["heat"].append(float(np.sum(q * cell_m2)))
        rows["joule"].append(float(hs.joule.sum()))
        rows["rec"].append(float(hs.recombination.sum()))
        rows["je"].append(hs.je)
        rows["rise_s"].append(float(np.sum(semi_area[dev.semiconductor] * th.rise(semi_pts))
                                    / np.sum(semi_area[dev.semiconductor])))
        if core is not None:
            rows["rise_c"].append(float(np.sum(cell_m2 * core * (th.T - t_ambient)) / np.sum(cell_m2 * core)))

        f_c = carrier_index(sol, electrical.reference, xc, yc, wavelength, optics.step, sub, plasma)

        def f_t(X, Y, th=th):
            return xs.sample(X, Y)[2]["dn_dT"] * th.rise(np.stack([X.ravel(), Y.ravel()], -1)).reshape(X.shape)

        rows["fc"].append(first(f_c))
        rows["ft"].append(first(f_t))
        rows["dc"].append(change(f_c))
        rows["dt"].append(change(f_t))
        rows["db"].append(change(lambda X, Y, f_c=f_c, f_t=f_t: f_c(X, Y) + f_t(X, Y)))

    a = {k: np.array(r) for k, r in rows.items()}
    model = heat if isinstance(heat, str) else ",".join(heat)
    return PinPhaseShifter(wavelength, electrical.biases.copy(),
                           np.array([s.current for s in electrical.solutions]), a["heat"], a["joule"], a["rec"],
                           a["je"], a["rise_s"], a["rise_c"] if core is not None else None, complex(neff[0]), a["dc"],
                           a["dt"], a["db"], a["fc"], a["ft"], model, temps, electrical, time.perf_counter() - t0)


def pin_phase_shifter(xs: CrossSection, optics: Window, charge: Window, thermal: Window, wavelength: float, biases,
                      tau: float = 1e-9, heat=DRIFT_DIFFUSION, plasma=None, sub: int = 8, core_tag: str | None = None,
                      t_ambient: float = 300.0, top_h: float = 10.0, solver=None,
                      side_temperature: float | None = None, **kw) -> PinPhaseShifter:
    """pin_electrical, then pin_optical (module docstring). kw goes to charge_gpu.drift_diffusion.sweep."""
    el = pin_electrical(xs, charge, biases, tau, **kw)
    return pin_optical(xs, optics, thermal, wavelength, el, heat, plasma, sub, core_tag, t_ambient, top_h, solver,
                       side_temperature)
