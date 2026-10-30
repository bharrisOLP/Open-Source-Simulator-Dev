"""Drift-diffusion (the van Roosbroeck system) for forward and reverse bias, on the grid of poisson.py.

    -div(eps grad psi) = q (p - n + N_D - N_A)
     div J_n =  q R,     J_n = q mu_n (-n grad psi + V_T grad n)
     div J_p = -q R,     J_p = q mu_p (-p grad psi - V_T grad p)
     R = (n p - n_i^2) / (tau_p (n + n_i) + tau_n (p + n_i))          (Shockley-Read-Hall, trap at midgap)

Unknowns: the potential psi in all cells, and the carrier densities n and p in the semiconductor cells.

Discretization:
- The cell-centered finite volumes of poisson.py. The Poisson operator, its face coefficients and the ohmic contact
  values come from poisson._operators, so at 0 V this system has the solution of poisson.solve.
- Scharfetter-Gummel fluxes on each face between two semiconductor cells a and b (distance h between the cell
  centers, face length L). The electron current from a to b (J_n / q, integrated over the face) is
        j_n = D_n L / h (B(d) n_b - B(-d) n_a),     d = (psi_b - psi_a) / V_T,     B(x) = x / (exp(x) - 1),
  and the hole current is j_p = D_p L / h (B(d) p_a - B(-d) p_b). D = mu V_T (Einstein relation). The face value of
  D is the mean of the two cells. No carrier flux crosses a face to an insulator cell.
- Ohmic contacts: on each contact face the boundary node (half a cell from the center) has psi_b = V + V_T
  asinh(N / (2 n_i)) (as poisson.py), n_b = n_i exp((psi_b - V) / V_T) and p_b = n_i exp((V - psi_b) / V_T): charge
  neutral and in equilibrium, with both quasi-Fermi levels at the contact voltage V. The same Scharfetter-Gummel
  flux connects the cell to the node, so the contact current is the sum of these face currents.
- The sum of the two continuity equations of a cell is div (J_n + J_p) = 0. So the discrete total current is the
  same through every cross-section, to the tolerance of the solve.

Models:
- Mobility: the Masetti mobility of rc.py (Si, 300 K, from |N_D - N_A|; phosphorus parameters for electrons, boron
  parameters for holes). It is constant in the field: no velocity saturation, no carrier-carrier scattering.
- Recombination: Shockley-Read-Hall only, with constant lifetimes. The default tau_n = tau_p = 1 us is a usual
  value for moderately doped bulk Si; SOI rib waveguides have much shorter effective lifetimes (surface
  recombination), so the use cases give their lifetime. No Auger recombination, no band-gap narrowing, no
  Fermi-Dirac statistics (Boltzmann only), no surface recombination at the oxide.

Solvers:
- "newton": the coupled Newton method on (psi, n, p). The Jacobian is assembled on the host (scipy) and solved with
  the sparse direct LU of SuperLU, on the CPU. The columns of n and p are scaled with n and p (so the linear unknowns
  are the relative changes dn/n and dp/p), the rows are equilibrated. Damping: the step of psi is limited to
  10 V_T, a decrease of a carrier density is applied as n exp(dn/n) (it stays positive).
- "gummel": the decoupled Gummel iteration. The Poisson step is the nonlinear Poisson equation of poisson.py with
  the quasi-Fermi levels of the last carriers per cell: symmetric positive definite, solved on the device (JAX) with
  multigrid-preconditioned CG. The continuity steps (nonsymmetric) are linear systems for n and then for p, with the
  recombination linearized at the last carriers. Their linear solver is the option ``linear``: "superlu" (sparse
  direct LU on the CPU, the reference) or "device" (continuity.py: the Slotboom form of the SG matrix, BiCGSTAB with
  a multigrid preconditioner as one jitted loop on the JAX device, and a convergence flag from the componentwise
  backward error of the true residual; a device solve that is not converged is replaced by SuperLU for that step
  and counted in DDSolution.linear_fallbacks). Robust at low bias; linear convergence (more iterations than Newton).
- sweep(): bias continuation from 0 V in steps of at most ``step``, with a halved step after a failed solve.

SI units throughout (doping and carriers in m^-3, currents in A per m of device length along z).
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

from charge_gpu import poisson
from charge_gpu.poisson import Q, Device, thermal_voltage

TAU_SI = 1.0e-6          # s: the default SRH lifetimes (moderately doped bulk Si)
CARRIER_FLOOR = 1.0e6    # m^-3 (1 cm^-3): relative changes of smaller densities do not count for convergence
CARRIER_MIN = 1.0e-100   # m^-3: the lower limit of n and p in the Newton method (keeps the column scaling finite)


def bernoulli(x):
    """B(x) = x / (exp(x) - 1), without overflow or cancellation."""
    x = np.asarray(x, float)
    small = np.abs(x) < 1e-6
    xs = np.where(small, 1.0, x)
    with np.errstate(over="ignore"):
        big = xs / np.expm1(xs)
    return np.where(small, 1.0 - x / 2 + x * x / 12, big)


def _dbernoulli(x):
    """B'(x) = B(x) (1 - B(-x)) / x."""
    x = np.asarray(x, float)
    small = np.abs(x) < 1e-4
    xs = np.where(small, 1.0, x)
    return np.where(small, -0.5 + x / 6 - x**3 / 180, bernoulli(xs) * (1.0 - bernoulli(-xs)) / xs)


def srh(n, p, n_i, tau_n, tau_p):
    """R (m^-3 s^-1) and its derivatives dR/dn, dR/dp (midgap trap: n_1 = p_1 = n_i)."""
    den = tau_p * (n + n_i) + tau_n * (p + n_i)
    num = n * p - n_i * n_i
    r = num / den
    return r, (p * den - num * tau_p) / den**2, (n * den - num * tau_n) / den**2


def _set_contact_voltages(dev: Device, v_n: float, v_p: float) -> list[str]:
    """The rule of poisson.solve: a contact over n doping takes v_n, else v_p. Returns "n" or "p" per contact."""
    nx, ny = dev.n_net.shape
    kinds = []
    for c in dev.contacts:
        under = {"xmin": dev.n_net[0, :], "xmax": dev.n_net[nx - 1, :], "ymin": dev.n_net[:, 0],
                 "ymax": dev.n_net[:, ny - 1]}[c.side][c.mask]
        kind = "n" if under.size and np.mean(under) > 0 else "p"
        c.voltage = v_n if kind == "n" else v_p
        kinds.append(kind)
    return kinds


@dataclass
class _Mesh:
    shape: tuple
    vt: float
    n_i: float
    sflat: np.ndarray        # flat cell index of each semiconductor cell
    svol: np.ndarray         # area of each semiconductor cell
    vol: np.ndarray          # area of each cell (flat)
    nnet: np.ndarray         # N_D - N_A (flat, 0 outside the semiconductor)
    a_mat: sp.csr_matrix     # the Poisson operator of poisson.py (host copy)
    bc: np.ndarray           # its contact terms
    fa: np.ndarray           # interior faces: flat cell indices a, b (b is the +x or +y neighbor)
    fb: np.ndarray
    sa: np.ndarray           # their semiconductor indices
    sb: np.ndarray
    fcn: np.ndarray          # D_n L / h, D_p L / h
    fcp: np.ndarray
    fcol: np.ndarray         # x faces: column index i (face between i and i + 1); y faces: -1
    cflat: np.ndarray        # contact faces: cell, semiconductor index, boundary values, coefficients, contact
    cs: np.ndarray
    cpsi: np.ndarray
    cn: np.ndarray
    cp: np.ndarray
    ccn: np.ndarray
    ccp: np.ndarray
    cid: np.ndarray
    kinds: list
    ops: object = field(repr=False)          # poisson._Ops on the device (for the Gummel Poisson step)
    host: tuple = field(repr=False)          # host face and contact conductances (multigrid hierarchy)

    @property
    def n_cells(self):
        return self.vol.size

    @property
    def n_semi(self):
        return self.sflat.size


def default_mobility(dev: Device):
    """(mu_n, mu_p) per cell in m^2/(V s): the Masetti mobility of rc.py from |N_D - N_A| (Si, 300 K)."""
    from charge_gpu.rc import masetti_mobility

    if abs(dev.T - 300.0) > 1.0:
        raise ValueError("the Masetti mobility is for 300 K: give the mobility")
    doping = np.abs(dev.n_net)
    return masetti_mobility(doping, "n"), masetti_mobility(doping, "p")


def _mesh(dev: Device, v_n: float, v_p: float, mobility=None) -> _Mesh:
    kinds = _set_contact_voltages(dev, v_n, v_p)
    vt = thermal_voltage(dev.T)
    nx, ny = dev.n_net.shape
    dx, dy = np.diff(np.asarray(dev.x, float)), np.diff(np.asarray(dev.y, float))
    semi = np.asarray(dev.semiconductor, bool)
    flat = np.arange(nx * ny).reshape(nx, ny)
    sidx = np.full((nx, ny), -1)
    sidx[semi] = np.arange(int(semi.sum()))
    mu_n, mu_p = default_mobility(dev) if mobility is None else mobility
    d_n = np.broadcast_to(np.asarray(mu_n, float), (nx, ny)) * vt
    d_p = np.broadcast_to(np.asarray(mu_p, float), (nx, ny)) * vt

    # the Poisson operator of poisson.py, on the host
    ops, host = poisson._operators(dev, v_n, v_p, jnp.float64)
    gx, gy, _ = host
    rows = [flat.ravel(), flat[:-1].ravel(), flat[1:].ravel(), flat[:, :-1].ravel(), flat[:, 1:].ravel()]
    cols = [flat.ravel(), flat[1:].ravel(), flat[:-1].ravel(), flat[:, 1:].ravel(), flat[:, :-1].ravel()]
    vals = [np.asarray(ops.diag).ravel(), -gx.ravel(), -gx.ravel(), -gy.ravel(), -gy.ravel()]
    a_mat = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nx * ny,) * 2)

    # interior faces between two semiconductor cells
    i, j = np.nonzero(semi[:-1] & semi[1:])
    hx = 0.5 * (dx[i] + dx[i + 1])
    fa_x, fb_x = flat[i, j], flat[i + 1, j]
    cn_x = 0.5 * (d_n[i, j] + d_n[i + 1, j]) * dy[j] / hx
    cp_x = 0.5 * (d_p[i, j] + d_p[i + 1, j]) * dy[j] / hx
    col_x = i
    i, j = np.nonzero(semi[:, :-1] & semi[:, 1:])
    hy = 0.5 * (dy[j] + dy[j + 1])
    fa_y, fb_y = flat[i, j], flat[i, j + 1]
    cn_y = 0.5 * (d_n[i, j] + d_n[i, j + 1]) * dx[i] / hy
    cp_y = 0.5 * (d_p[i, j] + d_p[i, j + 1]) * dx[i] / hy
    fa, fb = np.concatenate([fa_x, fa_y]), np.concatenate([fb_x, fb_y])

    # contact faces (semiconductor cells under the contact mask)
    cl = {k: [] for k in ("flat", "psi", "cn", "cp", "id")}
    for k, c in enumerate(dev.contacts):
        if c.side in ("xmin", "xmax"):
            ii = 0 if c.side == "xmin" else nx - 1
            jj = np.flatnonzero(np.asarray(c.mask, bool) & semi[ii, :])
            ii = np.full(jj.size, ii)
            h, length = dx[ii] / 2, dy[jj]
        else:
            jj = 0 if c.side == "ymin" else ny - 1
            ii = np.flatnonzero(np.asarray(c.mask, bool) & semi[:, jj])
            jj = np.full(ii.size, jj)
            h, length = dy[jj] / 2, dx[ii]
        z = np.arcsinh(dev.n_net[ii, jj] / (2 * dev.n_i))
        cl["flat"].append(flat[ii, jj])
        cl["psi"].append(c.voltage + vt * z)
        cl["cn"].append(d_n[ii, jj] * length / h)
        cl["cp"].append(d_p[ii, jj] * length / h)
        cl["id"].append(np.full(ii.size, k))
    cat = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in cl.items()}
    cflat = cat["flat"].astype(int)
    zc = (cat["psi"] - np.array([dev.contacts[k].voltage for k in cat["id"].astype(int)])) / vt
    vol = (dx[:, None] * dy[None, :]).ravel()
    sflat = flat[semi]
    return _Mesh(shape=(nx, ny), vt=vt, n_i=dev.n_i, sflat=sflat, svol=vol[sflat], vol=vol,
                 nnet=(dev.n_net * semi).ravel(), a_mat=a_mat, bc=np.asarray(ops.bc).ravel(),
                 fa=fa, fb=fb, sa=sidx.ravel()[fa], sb=sidx.ravel()[fb],
                 fcn=np.concatenate([cn_x, cn_y]), fcp=np.concatenate([cp_x, cp_y]),
                 fcol=np.concatenate([col_x, np.full(fa_y.size, -1)]),
                 cflat=cflat, cs=sidx.ravel()[cflat], cpsi=cat["psi"], cn=dev.n_i * np.exp(zc),
                 cp=dev.n_i * np.exp(-zc), ccn=cat["cn"], ccp=cat["cp"], cid=cat["id"].astype(int), kinds=kinds,
                 ops=ops, host=host)


def _fluxes(m: _Mesh, psi, n, p):
    """Face currents (in units of q): interior faces a -> b, and contact faces from the cell to the contact."""
    d = (psi[m.fb] - psi[m.fa]) / m.vt
    bp, bm = bernoulli(d), bernoulli(-d)
    jn = m.fcn * (bp * n[m.sb] - bm * n[m.sa])
    jp = m.fcp * (bp * p[m.sa] - bm * p[m.sb])
    dc = (m.cpsi - psi[m.cflat]) / m.vt
    cbp, cbm = bernoulli(dc), bernoulli(-dc)
    jnc = m.ccn * (cbp * m.cn - cbm * n[m.cs])
    jpc = m.ccp * (cbp * p[m.cs] - cbm * m.cp)
    return (d, bp, bm, jn, jp), (dc, cbp, cbm, jnc, jpc)


def _system(m: _Mesh, psi, n, p, tau_n, tau_p):
    """Residual F(psi, n, p) and its Jacobian (CSC). Rows: Poisson (all cells), electron and hole continuity
    (semiconductor cells): F_n = sum_faces j_n(out) - R A_cell, F_p = sum_faces j_p(out) + R A_cell."""
    nc, ns, vt = m.n_cells, m.n_semi, m.vt
    nf, pf = np.zeros(nc), np.zeros(nc)
    nf[m.sflat], pf[m.sflat] = n, p
    f0 = m.a_mat @ psi - m.bc - Q * m.vol * (pf - nf + m.nnet)
    (d, bp, bm, jn, jp), (dc, cbp, cbm, jnc, jpc) = _fluxes(m, psi, n, p)
    r, drn, drp = srh(n, p, m.n_i, tau_n, tau_p)
    fn = (np.bincount(m.sa, jn, ns) - np.bincount(m.sb, jn, ns) + np.bincount(m.cs, jnc, ns) - r * m.svol)
    fp = (np.bincount(m.sa, jp, ns) - np.bincount(m.sb, jp, ns) + np.bincount(m.cs, jpc, ns) + r * m.svol)

    io, ip = nc, nc + ns                                   # offsets of the n and p unknowns and rows
    dbp, dbm = _dbernoulli(d), _dbernoulli(-d)
    gn = m.fcn * (dbp * n[m.sb] + dbm * n[m.sa]) / vt     # d j_n / d psi_b (= - d j_n / d psi_a)
    gp = m.fcp * (dbp * p[m.sa] + dbm * p[m.sb]) / vt
    cdbp, cdbm = _dbernoulli(dc), _dbernoulli(-dc)
    gnc = -m.ccn * (cdbp * m.cn + cdbm * n[m.cs]) / vt   # d j_nc / d psi_cell
    gpc = -m.ccp * (cdbp * p[m.cs] + cdbm * m.cp) / vt
    a = m.a_mat.tocoo()
    s = np.arange(ns)
    sa, sb, fa, fb = m.sa, m.sb, m.fa, m.fb
    rows = [a.row, m.sflat, m.sflat,
            # electron rows: + j_n at a, - j_n at b
            io + sa, io + sa, io + sa, io + sa, io + sb, io + sb, io + sb, io + sb,
            # hole rows
            ip + sa, ip + sa, ip + sa, ip + sa, ip + sb, ip + sb, ip + sb, ip + sb,
            # contact faces
            io + m.cs, io + m.cs, ip + m.cs, ip + m.cs,
            # recombination
            io + s, io + s, ip + s, ip + s]
    cols = [a.col, io + s, ip + s,
            io + sb, io + sa, fb, fa, io + sb, io + sa, fb, fa,
            ip + sa, ip + sb, fb, fa, ip + sa, ip + sb, fb, fa,
            io + m.cs, m.cflat, ip + m.cs, m.cflat,
            io + s, ip + s, io + s, ip + s]
    vals = [a.data, Q * m.svol, -Q * m.svol,
            m.fcn * bp, -m.fcn * bm, gn, -gn, -m.fcn * bp, m.fcn * bm, -gn, gn,
            m.fcp * bp, -m.fcp * bm, gp, -gp, -m.fcp * bp, m.fcp * bm, -gp, gp,
            -m.ccn * cbm, gnc, m.ccp * cbp, gpc,
            -drn * m.svol, -drp * m.svol, drn * m.svol, drp * m.svol]
    size = nc + 2 * ns
    jac = sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(size, size))
    return np.concatenate([f0, fn, fp]), jac


def _converged_metric(m, dpsi, yn, yp, n, p):
    wn = n / (n + CARRIER_FLOOR)
    wp = p / (p + CARRIER_FLOOR)
    return max(float(np.max(np.abs(dpsi))) / m.vt, float(np.max(np.abs(yn) * wn, initial=0.0)),
               float(np.max(np.abs(yp) * wp, initial=0.0)))


def _update(c, y, t, log_max):
    """c (1 + t y) for an increase, c exp(t y) for a decrease (stays positive; the same to first order). The factor
    stays within exp(+-log_max) (the change of a density at a fixed quasi-Fermi level when psi changes by the step
    limit), and the density stays above CARRIER_MIN."""
    lo = np.exp(np.clip(t * y, -log_max, 0.0))
    hi = np.minimum(1.0 + t * np.maximum(y, 0.0), np.exp(log_max))
    return np.maximum(c * np.where(y > 0, hi, lo), CARRIER_MIN)


def _newton(m: _Mesh, psi, n, p, tau_n, tau_p, tol, max_iter, dpsi_max):
    nc, ns = m.n_cells, m.n_semi
    err = np.inf
    for it in range(1, max_iter + 1):
        f, jac = _system(m, psi, n, p, tau_n, tau_p)
        scale = np.concatenate([np.full(nc, m.vt), n, p])
        js = jac @ sp.diags(scale)
        rs = 1.0 / np.asarray(abs(js).max(axis=1).todense()).ravel()
        js = (sp.diags(rs) @ js).tocsc()
        y = spl.splu(js).solve(-rs * f)
        dpsi = m.vt * y[:nc]
        yn, yp = y[nc:nc + ns], y[nc + ns:]
        big = float(np.max(np.abs(dpsi)))
        t = min(1.0, dpsi_max / big) if big > 0 else 1.0
        psi = psi + t * dpsi
        n, p = _update(n, yn, t, dpsi_max / m.vt), _update(p, yp, t, dpsi_max / m.vt)
        err = _converged_metric(m, t * dpsi, t * yn, t * yp, n, p)
        if err < tol and t == 1.0:
            return psi, n, p, it, True, err
    return psi, n, p, max_iter, False, err


def _continuity_matrix(m: _Mesh, psi, carrier: str, other, den, ns):
    """The linear system M c = rhs of one continuity equation at fixed psi, with R = (c other - n_i^2) / den."""
    (d, bp, bm, _, _), (dc, cbp, cbm, _, _) = _fluxes(m, psi, np.ones(ns), np.ones(ns))
    sa, sb, s = m.sa, m.sb, np.arange(ns)
    if carrier == "n":                      # F_n = sum j_n - R A: j_n = c (B(d) n_b - B(-d) n_a)
        c, cc = m.fcn, m.ccn
        rows = [sa, sa, sb, sb, m.cs, s]
        cols = [sb, sa, sb, sa, m.cs, s]
        vals = [c * bp, -c * bm, -c * bp, c * bm, -cc * cbm, -m.svol * other / den]
        rhs = -np.bincount(m.cs, cc * cbp * m.cn, ns) - m.svol * m.n_i**2 / den
    else:                                   # F_p = sum j_p + R A: j_p = c (B(d) p_a - B(-d) p_b)
        c, cc = m.fcp, m.ccp
        rows = [sa, sa, sb, sb, m.cs, s]
        cols = [sa, sb, sa, sb, m.cs, s]
        vals = [c * bp, -c * bm, -c * bp, c * bm, cc * cbp, m.svol * other / den]
        rhs = np.bincount(m.cs, cc * cbm * m.cp, ns) + m.svol * m.n_i**2 / den
    mat = sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(ns, ns))
    return mat, rhs


def _poisson_qf(dev: Device, m: _Mesh, psi, phi_n, phi_p, tol, cg_tol, max_newton=100):
    """The Gummel Poisson step on the device: nonlinear Poisson with fixed quasi-Fermi levels per cell (poisson.py),
    multigrid-preconditioned CG per Newton step, log damping. The hierarchy (rebuilt per Newton step: the Newton
    term changes the diagonal) is that of continuity.hierarchy: the coarsening of thermal_gpu.multigrid down to 256
    cells, so the dense coarsest inverse is cheap (the host cost of the 1024-cell pseudo-inverse of build_hierarchy
    was most of the Gummel time)."""
    from charge_gpu.continuity import hierarchy as build_hierarchy

    d = poisson.solver_device()
    put = lambda a: jax.device_put(jnp.asarray(a, jnp.float64), d)  # noqa: E731
    o = dataclasses.replace(m.ops, phi_n=put(phi_n.reshape(m.shape)), phi_p=put(phi_p.reshape(m.shape)))
    gx_h, gy_h, extra0 = m.host
    vt = m.vt
    x = put(psi.reshape(m.shape))
    cg = 0
    for _ in range(max_newton):
        f, jdiag = poisson._residual_and_jdiag(o, x, dev.n_i, vt)
        levels, cinv = build_hierarchy(gx_h, gy_h, extra0 + np.asarray(jdiag), d, jnp.float64)
        delta, it = poisson._cg_mg(o, jdiag, f, levels, cinv, cg_tol, 2000)
        x = x + vt * jnp.sign(delta) * jnp.log1p(jnp.abs(delta) / vt)
        cg += int(it)
        if float(jnp.max(jnp.abs(delta))) < tol:
            break
    return np.asarray(x).ravel(), cg


def _gummel(dev, m: _Mesh, psi, n, p, tau_n, tau_p, tol, max_iter, linear="superlu"):
    """The Gummel iteration. linear: "superlu" (the continuity steps with the sparse direct LU on the CPU, the
    reference) or "device" (continuity.Solver: BiCGSTAB with the Slotboom multigrid on the JAX device). A device solve
    that is not converged (backward error above continuity.CHECK) or that cannot run (continuity.SpanError) is not
    used: that step is solved with SuperLU and counted in stats["fallbacks"]. Returns psi, n, p, the iterations, the
    convergence flag, the last metric and a dict of the linear-solve statistics (the backward error of the last
    continuity steps, for both linear solvers)."""
    from charge_gpu.continuity import CHECK

    ns, vt, ni = m.n_semi, m.vt, m.n_i
    phi_n, phi_p = np.zeros(m.n_cells), np.zeros(m.n_cells)
    err = np.inf
    stats = {"cg": 0, "krylov": 0, "backward_error": 0.0, "linear_converged": True, "fallbacks": 0,
             "t_poisson": 0.0, "t_continuity": 0.0}
    dev_solver = None
    if linear == "device":
        from charge_gpu.continuity import Solver, SpanError

        dev_solver = Solver(m)
    tiny = 1e-300

    def superlu(carrier, other, den):
        mat, rhs = _continuity_matrix(m, psi_new, carrier, other, den, ns)
        c = spl.spsolve(mat, rhs)
        with np.errstate(invalid="ignore", divide="ignore"):
            omega = float(np.max(np.abs(rhs - mat @ c) / (abs(mat) @ np.abs(c) + np.abs(rhs))))
        return c, omega

    def step(carrier, other, den, c_old):
        res = None
        if dev_solver is not None:
            try:
                res = dev_solver.solve(psi_new, carrier, other, den, c_old)
                stats["krylov"] += res.iterations
            except SpanError:
                res = None
        if res is not None and res.converged:
            c, omega = res.c, res.backward_error
        else:
            if dev_solver is not None:
                stats["fallbacks"] += 1
            c, omega = superlu(carrier, other, den)
        stats["backward_error"] = max(stats["backward_error"], omega)
        return np.maximum(c, tiny), bool(omega <= CHECK)

    for it in range(1, max_iter + 1):
        phi_n[m.sflat] = psi[m.sflat] - vt * np.log(np.maximum(n, tiny) / ni)
        phi_p[m.sflat] = psi[m.sflat] + vt * np.log(np.maximum(p, tiny) / ni)
        t0 = time.perf_counter()
        psi_new, k = _poisson_qf(dev, m, psi, phi_n, phi_p, tol=max(min(1e-3 * tol * vt, 1e-12), 1e-14), cg_tol=1e-12)
        t1 = time.perf_counter()
        stats["cg"] += k
        stats["backward_error"] = 0.0                       # the value of the last iteration
        den = tau_p * (n + ni) + tau_n * (p + ni)
        n_new, ok_n = step("n", p, den, n)
        den = tau_p * (n_new + ni) + tau_n * (p + ni)
        p_new, ok_p = step("p", n_new, den, p)
        stats["linear_converged"] = ok_n and ok_p
        stats["t_poisson"] += t1 - t0
        stats["t_continuity"] += time.perf_counter() - t1
        err = _converged_metric(m, psi_new - psi, n_new / n - 1, p_new / p - 1, n_new, p_new)
        psi, n, p = psi_new, n_new, p_new
        if err < tol and stats["linear_converged"]:
            return psi, n, p, it, True, err, stats
    return psi, n, p, max_iter, False, err, stats


@dataclass
class DDSolution:
    psi: np.ndarray          # (nx, ny) potential (V)
    n: np.ndarray            # (nx, ny) carrier densities (m^-3), 0 outside the semiconductor
    p: np.ndarray
    v_n: float               # contact voltages of the n and p contacts (V)
    v_p: float
    contact_currents: np.ndarray   # conventional current out of the device at each contact (A/m), dev.contacts order
    contact_kinds: list            # "n" or "p" per contact
    contact_sides: list            # "xmin", "xmax", "ymin" or "ymax" per contact
    iterations: int
    converged: bool
    error: float             # the last convergence metric (relative update)
    method: str
    device: str              # where the solves ran
    tau_n: float
    tau_p: float
    mesh: _Mesh = field(repr=False)
    linear: str = "superlu"        # the linear solver of the continuity steps (gummel): "superlu" or "device"
    krylov_iterations: int = 0     # BiCGSTAB iterations of all continuity steps (linear="device")
    linear_error: float | None = None  # gummel: the largest componentwise backward error of the last continuity
    #                                    steps (both linear solvers); newton: None (no continuity steps)
    linear_converged: bool = True  # gummel: the last continuity steps have a backward error of at most 1e-11
    linear_fallbacks: int = 0      # linear="device": continuity steps solved with SuperLU (device solve not converged)
    timings: dict = field(default_factory=dict)   # gummel: the wall time (s) of the Poisson and continuity steps

    @property
    def bias(self) -> float:
        """Forward bias v_p - v_n (V)."""
        return self.v_p - self.v_n

    @property
    def current(self) -> float:
        """The current (A/m) out of the device through the n contacts: positive in forward bias."""
        k = np.array([c == "n" for c in self.contact_kinds])
        return float(np.sum(self.contact_currents[k]))

    def quasi_fermi(self):
        """(phi_n, phi_p) in V in the semiconductor (NaN elsewhere): n = n_i exp((psi - phi_n)/V_T)."""
        m = self.mesh
        semi = np.zeros(m.n_cells, bool)
        semi[m.sflat] = True
        semi = semi.reshape(m.shape)
        with np.errstate(divide="ignore"):
            phi_n = np.where(semi, self.psi - m.vt * np.log(self.n / m.n_i), np.nan)
            phi_p = np.where(semi, self.psi + m.vt * np.log(self.p / m.n_i), np.nan)
        return phi_n, phi_p

    def x_currents(self) -> np.ndarray:
        """Total current (A/m, in +x) through each interior column boundary x_(i+1) (nx - 1 values). Only the x faces
        count, so this is the terminal current of a lateral device: all contacts on the xmin and xmax sides (else
        ValueError)."""
        if any(s not in ("xmin", "xmax") for s in self.contact_sides):
            raise ValueError("x_currents needs all contacts on the xmin and xmax sides")
        m = self.mesh
        (_, _, _, jn, jp), _ = _fluxes(m, self.psi.ravel(), self.n.ravel()[m.sflat], self.p.ravel()[m.sflat])
        xf = m.fcol >= 0
        return Q * np.bincount(m.fcol[xf], (jn + jp)[xf], m.shape[0] - 1)

    def recombination(self) -> np.ndarray:
        """R (m^-3 s^-1) per cell (0 outside the semiconductor)."""
        m = self.mesh
        r = np.zeros(m.n_cells)
        r[m.sflat] = srh(self.n.ravel()[m.sflat], self.p.ravel()[m.sflat], m.n_i, self.tau_n, self.tau_p)[0]
        return r.reshape(m.shape)


def solve(dev: Device, v_n: float, v_p: float, guess=None, method: str = "newton", tau_n: float = TAU_SI,
          tau_p: float = TAU_SI, mobility=None, tol: float = 1e-10, max_iter: int | None = None,
          dpsi_max: float | None = None, linear: str = "superlu") -> DDSolution:
    """Drift-diffusion at one bias: the n contacts at v_n, the p contacts at v_p (forward bias: v_p > v_n). Each
    contact takes v_n when the net doping under it is positive, else v_p (as poisson.solve).

    guess: an earlier solution (DDSolution or poisson.Solution: psi, n, p). None: poisson.solve at this bias (the
    constant quasi-Fermi solution). method: "newton" (coupled, CPU direct LU) or "gummel" (Poisson on the device;
    continuity with the solver ``linear``). linear (gummel only): "superlu" (CPU direct LU, the reference) or
    "device" (BiCGSTAB with the Slotboom multigrid on the JAX device, continuity.py). mobility: (mu_n, mu_p) in
    m^2/(V s), per cell or scalar; None: Masetti. tol: convergence on the relative update (psi / V_T, n / n, p / p).
    """
    if method not in ("newton", "gummel"):
        raise ValueError(f"method must be 'newton' or 'gummel', not {method!r}")
    if linear not in ("superlu", "device"):
        raise ValueError(f"linear must be 'superlu' or 'device', not {linear!r}")
    if method == "newton" and linear != "superlu":
        raise ValueError("method='newton' solves with SuperLU only: use method='gummel' for linear='device'")
    if guess is None:
        guess = poisson.solve(dev, v_n, v_p)
    m = _mesh(dev, v_n, v_p, mobility)
    semi_flat = m.sflat
    psi = np.asarray(guess.psi, float).ravel().copy()
    n = np.maximum(np.asarray(guess.n, float).ravel()[semi_flat], CARRIER_MIN)
    p = np.maximum(np.asarray(guess.p, float).ravel()[semi_flat], CARRIER_MIN)
    stats = {"krylov": 0, "backward_error": None, "linear_converged": True, "fallbacks": 0}
    if method == "newton":
        psi, n, p, it, conv, err = _newton(m, psi, n, p, tau_n, tau_p, tol, max_iter or 60,
                                           dpsi_max or 10 * m.vt)
        where = "cpu (scipy SuperLU)"
    else:
        psi, n, p, it, conv, err, stats = _gummel(dev, m, psi, n, p, tau_n, tau_p, tol, max_iter or 1000, linear)
        cont = "cpu (scipy SuperLU)" if linear == "superlu" else f"{poisson.solver_device()} (BiCGSTAB, multigrid)"
        where = f"poisson: {poisson.solver_device()}; continuity: {cont}"
    _, (_, _, _, jnc, jpc) = _fluxes(m, psi, n, p)
    currents = Q * np.bincount(m.cid, jnc + jpc, len(dev.contacts))
    nf, pf = np.zeros(m.n_cells), np.zeros(m.n_cells)
    nf[semi_flat], pf[semi_flat] = n, p
    return DDSolution(psi=psi.reshape(m.shape), n=nf.reshape(m.shape), p=pf.reshape(m.shape), v_n=v_n, v_p=v_p,
                      contact_currents=currents, contact_kinds=m.kinds,
                      contact_sides=[c.side for c in dev.contacts], iterations=it, converged=conv, error=err,
                      method=method, device=where, tau_n=tau_n, tau_p=tau_p, mesh=m, linear=linear,
                      krylov_iterations=stats["krylov"], linear_error=stats["backward_error"],
                      linear_converged=stats["linear_converged"], linear_fallbacks=stats["fallbacks"],
                      timings={k[2:] + "_s": v for k, v in stats.items() if k.startswith("t_")})


def sweep(dev: Device, biases, step: float = 0.05, min_step: float = 1e-3, method: str = "newton",
          **kw) -> list[DDSolution]:
    """Solutions at the forward biases V = v_p - v_n (v_n = 0; a negative V is a reverse bias), in the given order.
    Continuation from 0 V (started from poisson.solve): steps of at most ``step``, each solve started from the last
    converged solution. A failed solve (not converged, or a RuntimeError or LinAlgError of the linear solver, for
    example a singular LU factor) halves the step, down to min_step, then RuntimeError. kw goes to solve() (not
    ``guess``: sweep sets it)."""
    if "guess" in kw:
        raise ValueError("sweep starts from poisson.solve at 0 V and sets guess itself: do not give guess")
    cur = solve(dev, 0.0, 0.0, method=method, **kw)
    if not cur.converged:
        raise RuntimeError("drift-diffusion did not converge at 0 V")
    v_cur, h, out = 0.0, step, []
    for v in biases:
        while abs(v - v_cur) > 1e-12:
            dv = float(np.clip(v - v_cur, -h, h))
            v_new = v if abs(v - v_cur - dv) < 1e-12 else v_cur + dv
            try:
                trial = solve(dev, 0.0, v_new, guess=cur, method=method, **kw)
            except (RuntimeError, np.linalg.LinAlgError):
                trial = None
            if trial is not None and trial.converged:
                cur, v_cur, h = trial, v_new, min(step, 1.5 * h)
            else:
                h /= 2
                if h < min_step:
                    raise RuntimeError(f"drift-diffusion did not converge from {v_cur} V to {v_new} V")
        out.append(cur)
    return out
