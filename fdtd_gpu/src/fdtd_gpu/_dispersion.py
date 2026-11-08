"""Dispersive and conductive materials on fdtdx: the poles of each Medium as fdtdx poles (SI units), and the
per-cell arrays of the auxiliary-differential-equation (ADE) update of fdtdx.

fdtdx 0.6.3 (`fdtdx.dispersion`, `fdtdx.fdtd.update.update_E`) keeps for each pole p the polarization
P_p = P / eps_0 and updates it with central differences at the time step n:
    P^(n+1) = c1 P^n + c2 P^(n-1) + c3 E^n,  c1 = (2 - w0^2 dt^2) / D,  c2 = -(1 - g dt / 2) / D,  c3 = K dt^2 / D,
with D = 1 + g dt / 2, K = delta_eps w0^2 (Lorentz) or wp^2 (Drude); then E^(n+1) gets + (1 / eps_inf) sum_p
(P^n - P^(n+1)), so that eps_inf E + sum P changes by dt curl H. The electric conductivity sigma (S/m) has the
semi-implicit update E^(n+1) (1 + b) = E^n (1 - b) + ..., b = sigma dt / (2 eps_0 eps).

This package gives fdtdx only non-dispersive, lossless materials (eps_inf), so that fdtdx allocates no ADE arrays,
and sets the arrays itself after the objects are placed (`material_arrays`). Without eps_averaging each cell gets
the poles and the conductivity of the material of its center (a staircase). With eps_averaging the rule FACE_RULE
sets them per E component (default "fill": the pole strength times the fill fraction of the voxel, see FACE_RULE).
The sources were applied with 1 / eps_inf at that time, so a current source adds J to
dD/dt = curl H - J (fdtdx itself would scale it by 1 / Re eps(f_source) in a dispersive cell).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from fdtd_gpu.geometry import DrudeSusceptibility, LorentzianSusceptibility, Medium

C0 = 299792458.0
UM = 1e-6
EPS0 = 1.0 / (4e-7 * math.pi * C0**2)   # as fdtdx.constants.eps0 (eps0 c eta0 = 1 in its conductivity update)
_W = 2 * math.pi * C0 / UM             # f in 1/um -> omega in rad/s


def fdtdx_poles(medium: Medium) -> tuple:
    """The fdtdx poles (LorentzPole, DrudePole; SI units) of the susceptibilities of a Medium (sigma > 0 only)."""
    from fdtdx import DrudePole, LorentzPole

    poles = []
    for s in medium.E_susceptibilities:
        if s.sigma <= 0:
            continue
        if isinstance(s, DrudeSusceptibility):
            poles.append(DrudePole(plasma_frequency=_W * s.frequency * math.sqrt(s.sigma), damping=_W * s.gamma))
        elif isinstance(s, LorentzianSusceptibility):
            poles.append(LorentzPole(resonance_frequency=_W * s.frequency, damping=_W * s.gamma,
                                     delta_epsilon=s.sigma))
        else:                                                   # pragma: no cover - Medium checks the types
            raise NotImplementedError(type(s).__name__)
    return tuple(poles)


CHECK_STABILITY = True       # tests set False to run an unstable case


def stability_margin(medium: Medium, dt: float, dx: float, dims: int, eps: float | None = None) -> tuple:
    """(lhs, rhs) of the stability bound of the Yee grid with the ADE poles of `medium` (lossless poles, exact):
    sum_p v_p / (1 - u_p) < eps - s, with u_p = (w0_p dt / 2)^2, v_p = K_p dt^2 / 4, s = dims (c dt / dx)^2 (the
    largest (k dt / 2)^2 of the grid; dims = the axes of size > 0, along which the fields can vary) and eps the
    permittivity of the E components of the cells (default eps_inf). Derivation: with s_n = exp(-i w dt) the grid
    gives eps_inf x + sum_p K_p x / (w0_p^2 - x) = k^2 for x = (2 / dt)^2 sin^2(w dt / 2); the scheme is stable if
    the largest root x is at most (2 / dt)^2, and above the largest w0_p^2 the left side increases with x.
    fdtdx runs at 0.99 of the 3-D Courant limit, so s = 0.98 in 3-D, 0.65 in 2-D and 0.33 in 1-D. u_p >= 1 (2 pi f_n
    dt >= 2) is unstable for each eps (fdtdx raises this case too). Damping (gamma > 0) is not in the bound."""
    s = dims * (dt / dx) ** 2
    lhs = 0.0
    for p in medium.E_susceptibilities:
        if p.sigma <= 0:
            continue
        w0 = 2 * math.pi * p.frequency if isinstance(p, LorentzianSusceptibility) else 0.0
        u = (w0 * dt / 2) ** 2
        v = p.sigma * (2 * math.pi * p.frequency) ** 2 * dt**2 / 4
        lhs += math.inf if u >= 1 else v / (1 - u)
    return lhs, (medium.epsilon if eps is None else eps) - s


def check_stability(face, dt: float, dx: float, dims: int):
    """Raise ValueError if a component of a cell violates the bound of `stability_margin` with its own values: the
    pole strength K times the weight w of the face rule, and the eps_inf of the component (`face_assignment`; dt
    and dx in um). A partly filled face cell has a smaller K and can have a smaller eps_inf than the bulk."""
    if not CHECK_STABILITY or not face.media:
        return
    lhs = np.array([stability_margin(m, dt, dx, dims)[0] for m in face.media] + [0.0])     # index -1: no poles
    s = dims * (dt / dx) ** 2
    with np.errstate(invalid="ignore"):
        left = np.where(face.w > 0, lhs[face.dom] * face.w, 0.0)
    bad = (face.w > 0) & ~(left < face.eps - s)
    if bad.any():
        j = tuple(np.argwhere(bad)[0])
        dom, w, left = (np.broadcast_to(a, bad.shape) for a in (face.dom, face.w, left))
        m, l_, r_ = face.media[dom[j]], float(left[j]), float(np.broadcast_to(face.eps, bad.shape)[j] - s)
        est = "" if math.isinf(l_) or r_ <= 0 else f" (about {math.sqrt(l_ / r_):.2g} times the resolution)"
        raise ValueError(f"{m}: the ADE update is unstable at this resolution: w sum K dt^2 / 4 / (1 - (w0 dt / 2)^2) "
                         f"= {l_:.3g} must be below eps - {dims} (c dt / dx)^2 = {r_:.3g} (weight w = "
                         f"{float(w[j]):.3g}); increase the resolution{est}, or eps_inf (fdtdx runs at 0.99 of "
                         f"the 3-D Courant limit)")


def float32_coefficients(c1: float, c2: float, c3: float) -> tuple:
    """The float32 values of the ADE coefficients of one pole. c1 is rounded; c2 is rounded so that c1 + c2 keeps
    its float64 value as well as float32 allows (to 6e-8, half a unit in the last place of c2). 1 - c1 - c2 =
    w0^2 dt^2 / D sets the resonance (cos(w dt) = c1 / (1 - c2) = 1 - w0^2 dt^2 / 2 without damping); a separate
    rounding of c1 and c2 moves w0^2 by up to 1.8e-7 / (w0 dt)^2 relative. The sum never exceeds 1: for a Drude
    pole (c1 + c2 = 1) a float32 sum above 1 gives the constant polarization a growth of (c1 + c2 - 1) / (g dt) per
    step."""
    c1f = np.float32(c1)
    c2f = np.float32((c1 + c2) - float(c1f))
    if float(c1f) + float(c2f) > 1.0:
        c2f = np.float32(1.0 - float(c1f))          # exact (Sterbenz): 1 <= c1f <= 2
    return c1f, c2f, np.float32(c3)


FACE_RULE = "fill"
"""The rule for the E components next to a dispersive or conductive material with eps_averaging=True:
- "fill" (default): eps_inf averaged (Kottke et al., as for any material), the pole strength K of the dispersive
  material scaled by its fill fraction in the voxel of the component, and the conductivity sigma_D eps_inf scaled
  by the fill fraction of each conductive material. For E parallel to the faces this is the arithmetic mean of the
  whole eps(f), which is exact for a plane normal to an axis: second order in the cell size. For E normal to a face
  the exact average is the harmonic mean of eps(f), which no pole of this form gives: first order there. In a
  voxel with two dispersive materials only the one with the larger fill fraction keeps its poles (scaled).
- "average": the rule of the MIT package: eps_inf averaged, the poles and the conductivity of the cell center.
- "staircase": the cells within one cell of a dispersive or conductive cell take the staircase eps_inf too.
With faces on cell faces the three rules give the same arrays."""
FACE_RULES = ("fill", "average", "staircase")


def lossy_media(default_material: Medium, geometry) -> list:
    """The distinct (by identity) dispersive or conductive media of the background and of the objects."""
    out = []
    for m in [default_material] + [o.material for o in geometry]:
        if (m.dispersive or m.conductive) and not any(m is q for q in out):
            out.append(m)
    return out


@dataclass
class Face:
    """The material arrays of each E component (C = 3 with eps_averaging, else C = 1) of each cell.
    media: the dispersive media; dom: (C, *cells) the index in media of the poles of the component (-1: none);
    w: (C, *cells) the factor of their strength K; eps: (C, *cells) eps_inf of the component; sigma: (C, *cells)
    sum of f sigma_D eps_inf over the conductive media (the conductivity in units of eps_0 c / um); plain: (*cells)
    True where no component has poles or conductivity."""
    media: list
    dom: np.ndarray
    w: np.ndarray
    eps: np.ndarray
    sigma: np.ndarray
    plain: np.ndarray


def face_assignment(idx: np.ndarray, mats, eps3, fills, lossy) -> Face:
    """The Face arrays of a staircase index `idx` (materials `mats`), the eps_inf `eps3` (3, *cells) of the E
    components (None without averaging; with the rule "staircase" already from `staircase_at_lossy`), the fill
    fractions `fills` (len(lossy), 3, *cells) of the media `lossy` (needed by the rule "fill"), with the rule
    FACE_RULE."""
    if FACE_RULE not in FACE_RULES:
        raise ValueError(f"FACE_RULE must be one of {FACE_RULES}, not {FACE_RULE!r}")
    media = [m for m in lossy if m.dispersive]
    eps_cells = np.array([m.epsilon for m in mats])[idx]
    stair = np.full(len(mats), -1)                 # material index -> index in media (-1: no poles)
    sig_cell = np.zeros(len(mats))                 # sigma_D eps_inf of each material
    for k, m in enumerate(mats):
        j = next((j for j, q in enumerate(media) if q is m), -1)
        stair[k] = j
        sig_cell[k] = m.D_conductivity * m.epsilon
    if eps3 is None or FACE_RULE != "fill":
        dom = stair[idx][None]
        w = (dom >= 0).astype(np.float32)
        sigma = sig_cell[idx][None]
        eps = eps_cells[None] if eps3 is None else eps3
        if eps3 is not None:                               # the poles stay one per cell (C = 1 in the ADE arrays)
            sigma = sigma / eps_cells[None] * eps         # sigma_D (cell center) eps of the component, as before
    else:
        pos = [t for t, m in enumerate(lossy) if m.dispersive]
        if pos:
            f = fills[pos]                                 # (media, 3, *cells)
            best = np.argmax(f, axis=0)
            w = np.take_along_axis(f, best[None], axis=0)[0].astype(np.float32)
            dom = np.where(w > 0, best, -1)
        else:
            dom, w = np.full(eps3.shape, -1), np.zeros(eps3.shape, np.float32)
        sigma = np.zeros(eps3.shape)
        for t, m in enumerate(lossy):
            if m.conductive:
                sigma = sigma + fills[t] * (m.D_conductivity * m.epsilon)
        eps = eps3
    plain = ~((dom >= 0) | (sigma > 0)).any(axis=0)
    return Face(media, np.asarray(dom), np.asarray(w, np.float32), np.asarray(eps, np.float64),
                np.asarray(sigma, np.float64), plain)


def staircase_at_lossy(eps3: np.ndarray, idx: np.ndarray, mats, wrap) -> np.ndarray:
    """The averaged permittivity eps3 (3, *cells) with the staircase value (eps_inf of the cell center) in each cell
    within one cell (the 3 x 3 x 3 neighborhood) of a dispersive or conductive cell, for all three components (the
    rule "staircase" of FACE_RULE). wrap[a]: the axis is periodic (or of size 0). A dispersive feature that holds no
    cell center is not seen (as by the staircase)."""
    lossy = ~np.array([not (m.dispersive or m.conductive) for m in mats])[idx]
    if not lossy.any():
        return eps3
    near = lossy.copy()
    for a in range(3):
        grown = near.copy()
        for step in (-1, 1):
            moved = np.roll(near, step, axis=a)
            if not wrap[a]:                        # no wrap: the cell that rolled in from the other end is not near
                edge = [slice(None)] * 3
                edge[a] = 0 if step == 1 else -1
                moved[tuple(edge)] = False
            grown |= moved
        near = grown
    out = np.array(eps3, copy=True)
    out[:, near] = np.array([m.epsilon for m in mats])[idx][near]
    return out


def material_arrays(arrays, config, face: Face):
    """Set the ADE arrays (c1, c2, c3 of shape (poles, C, *cells), the polarizations P of shape (poles, 3, *cells))
    and the electric conductivity ((C, *cells)) of the fdtdx ArrayContainer `arrays` from the Face arrays: each
    component gets c1 and c2 of its poles and c3 times w; the conductivity is sigma = eps_0 face.sigma (c / um),
    so a bulk component decays at the rate sigma_D of the MIT package. Arrays without poles or conductivity stay
    None."""
    import jax.numpy as jnp
    from fdtdx.dispersion import compute_pole_coefficients_per_axis

    dt = config.time_step_duration                      # s
    dtype = arrays.inv_permittivities.dtype
    poles = [fdtdx_poles(m) for m in face.media]
    npole = max((len(p) for p in poles), default=0)
    if npole and (face.dom >= 0).any():
        tab = np.zeros((3, len(poles) + 1, npole), np.float32)       # the last row (index -1): no poles
        for k, p in enumerate(poles):
            c = compute_pole_coefficients_per_axis(p, dt)              # three (len(p), 3) arrays, isotropic
            for j in range(len(p)):
                tab[:, k, j] = float32_coefficients(c[0][j, 0], c[1][j, 0], c[2][j, 0])
        for q, name in enumerate(("dispersive_c1", "dispersive_c2", "dispersive_c3")):
            per = tab[q][face.dom]                                      # (C, *cells, poles)
            if q == 2:
                per = per * face.w[..., None]                           # float32: K scaled by the weight
            arrays = arrays.aset(name, jnp.asarray(np.moveaxis(per, -1, 0), dtype))     # (poles, C, *cells)
        zeros = jnp.zeros((npole, 3, *face.dom.shape[1:]), arrays.fields.E.dtype)
        arrays = arrays.aset("fields->dispersive_P_curr", zeros)
        arrays = arrays.aset("fields->dispersive_P_prev", zeros)
    if (face.sigma > 0).any():
        spacing = C0 * dt / config.courant_number           # fdtdx scales sigma (S/m) by this length (m)
        sigma = EPS0 * (C0 / UM) * face.sigma                                      # S/m
        arrays = arrays.aset("electric_conductivity", jnp.asarray((sigma * spacing).astype(np.float32), dtype))
    return arrays
