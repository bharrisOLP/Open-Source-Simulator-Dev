"""An independent reference for 2-D TM bands: the scalar equation -lap(Ez) = w^2 eps Ez with Bloch-periodic
second-order finite differences (scipy, CPU), eps averaged by supersampling. Not the method of the solver."""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl


def fd_tm_bands(n, kfrac, radius=0.2, eps_rod=12.0, num=2, sub=8):
    h = 1.0 / n
    x = -0.5 + h * (np.arange(n * sub) + 0.5) / sub
    X, Y = np.meshgrid(x, x, indexing="ij")
    eps_s = np.where(X**2 + Y**2 <= radius**2, eps_rod, 1.0)
    eps = eps_s.reshape(n, sub, n, sub).mean(axis=(1, 3))       # arithmetic mean: E_z is parallel to interfaces

    def d2(kc):
        ph = np.exp(2j * np.pi * kc)                             # Bloch phase across the cell
        main = -2 * np.ones(n)
        m = sp.diags([main, np.ones(n - 1), np.ones(n - 1)], [0, 1, -1], format="lil", dtype=complex)
        m[0, n - 1] = 1 / ph
        m[n - 1, 0] = ph
        return m.tocsr() / h**2

    lap = sp.kron(d2(kfrac[0]), sp.eye(n)) + sp.kron(sp.eye(n), d2(kfrac[1]))
    vals = spl.eigs(-lap, k=num, M=sp.diags(eps.ravel()), sigma=0.01, which="LM", return_eigenvectors=False)
    return np.sort(np.sqrt(np.abs(vals.real))) / (2 * np.pi)
