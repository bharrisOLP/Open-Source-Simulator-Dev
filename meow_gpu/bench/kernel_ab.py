"""A/B: kernel times for an EME cross-section of meow and for the synthetic strip of the same grid size."""
import sys
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
import meow_gpu.fde.eigensolver as E  # noqa: E402
from bench_eme_taper import build  # noqa: E402
from meow_gpu.fde.operator import make_problem  # noqa: E402
from structures import strip  # noqa: E402


def bench(name, f, *a, n=50):
    jax.block_until_ready(f(*a))
    t0 = time.perf_counter()
    for _ in range(n):
        r = f(*a)
    jax.block_until_ready(r)
    print(f"  {name:12s} {(time.perf_counter() - t0) / n * 1e3:8.3f} ms", flush=True)


cells, css = build(4, 0.02)
cs = css[1]
probs = {"meow cell": make_problem(cs.nx**2, cs.ny**2, cs.nz**2, cs.mesh.x, cs.mesh.y, cs.env.wl),
         "strip": make_problem(*strip(dx=0.02, dy=0.02, w_dom=4.0, h_dom=1.82)[:5], 1.55)}
for name, p in probs.items():
    print(name, p.shape, p.eps_xx.dtype, "imag max", float(np.abs(p.eps_xx.imag).max()),
          "x steps", np.unique(np.round(np.diff(cs.mesh.x if name == "meow cell" else [0, 1]), 9))[:3])
    op = p.device(jnp.complex64)
    pc = E.make_preconditioner(p)
    v = jnp.asarray(np.random.default_rng(0).normal(size=(18, 2) + p.shape), jnp.complex64)
    th = jnp.full(18, -6.0, jnp.complex64)
    bench("apply", E._apply_block, op, v)
    bench("precondition", E.precondition, pc, v, th)
    print("  devices:", {str(a.devices()) for a in jax.tree_util.tree_leaves(op)}, "lam dtype", pc.lam.dtype,
          "denormal-ish entries in op:", int(sum(jnp.sum((jnp.abs(a) > 0) & (jnp.abs(a) < 1e-30)) for a in jax.tree_util.tree_leaves(op))))
