"""Time each section of the Davidson loop on the current device (development tool)."""
import sys
import time
from collections import defaultdict
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from structures import strip  # noqa: E402

import meow_gpu.fde.eigensolver as E  # noqa: E402
from meow_gpu.fde.operator import make_problem  # noqa: E402

T = defaultdict(float)
N = defaultdict(int)


def timed(name, f):
    def g(*a, **k):
        t0 = time.perf_counter()
        out = f(*a, **k)
        jax.block_until_ready(out)
        T[name] += time.perf_counter() - t0
        N[name] += 1
        return out
    return g


for name in ["_ritz", "precondition", "_project_out", "_apply_block", "_insert", "_gram", "_combine"]:
    setattr(E, name, timed(name, getattr(E, name)))
orig_eig = np.linalg.eig
np.linalg.eig = timed("host eig", orig_eig)
orig_svqb = E._svqb
E._svqb = timed("_svqb (total)", orig_svqb)

dx = float(sys.argv[1]) if len(sys.argv) > 1 else 0.01
precision = sys.argv[2] if len(sys.argv) > 2 else "mixed"
b = int(sys.argv[3]) if len(sys.argv) > 3 else 10
exx, eyy, ezz, x, y = strip(dx=dx, dy=dx, w_dom=4.0, h_dom=1.82)
p = make_problem(exx, eyy, ezz, x, y, 1.55)
t0 = time.perf_counter()
pc = E.make_preconditioner(p)
print(f"make_preconditioner {time.perf_counter() - t0:.3f}s")
target = -(np.sqrt(np.max(np.real(exx))) * (1 + 1e-14)) ** 2
x0 = jnp.asarray(np.random.default_rng(0).normal(size=(b, 2) + exx.shape) + 0j)
op = (p.device(jnp.complex64), p.device(jnp.complex128)) if precision == "mixed" else p.device(jnp.complex64)
tol = 1e-10 if precision == "mixed" else 1e-5
r = E.davidson(op, pc, b - 8, target, x0, tol=tol, precision=precision)      # cold
for rep in range(2):                                                            # warm, from the cold vectors
    T.clear(); N.clear()
    t0 = time.perf_counter()
    r2 = E.davidson(op, pc, b - 8, target, r.vectors, tol=tol, precision=precision)
    total = time.perf_counter() - t0
r = r2
print(f"total {total:.3f}s iterations {r.iterations}")
for k in sorted(T, key=T.get, reverse=True):
    print(f"  {k:18s} {T[k]*1e3:9.1f} ms  calls {N[k]:4d}  {T[k]/N[k]*1e3:7.3f} ms/call")
