"""A/B of the inner products (long reductions over the n unknowns) and of the host eigenproblems of the Davidson loop,
at the size of the EME taper cell (development tool).

    OMP_NUM_THREADS=1 python bench/kernel_reduce.py    # the taper cell of bench_eme_taper.py (36k unknowns), b = 18
    python bench/kernel_reduce.py                      # the same with the default OpenBLAS threads (host eig part)

A Gram matrix <a_i, b_j> of a few rows has a long contraction (n) and a small result (b x M): a plain GEMM has few
output tiles, so few compute units work. The split variants sum P partial products of n / P unknowns each. Also:
the GMRES kernel of the refinement with the earlier basis layout against ``_correction_krylov``, both under
``jax.vmap`` over 8 members (a batched refinement), and the host eigenproblems of 8 members in one thread and in a
thread pool.
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import meow_gpu.fde.eigensolver as E  # noqa: E402
from bench_eme_taper import build  # noqa: E402
from meow_gpu.device import put  # noqa: E402
from meow_gpu.fde.operator import make_problem  # noqa: E402


def bench(name, f, *a, n=40):
    jax.block_until_ready(f(*a))
    t0 = time.perf_counter()
    for _ in range(n):
        r = f(*a)
    jax.block_until_ready(r)
    print(f"  {name:40s} {(time.perf_counter() - t0) / n * 1e3:8.3f} ms", flush=True)
    return r


def split_gram(p):
    @jax.jit
    def g(a, b):
        fa, fb = a.reshape(a.shape[0], -1), b.reshape(b.shape[0], -1)
        n = fa.shape[1]
        pad = (-n) % p
        if pad:
            fa, fb = jnp.pad(fa, ((0, 0), (0, pad))), jnp.pad(fb, ((0, 0), (0, pad)))
        fa = fa.reshape(fa.shape[0], p, -1)
        fb = fb.reshape(fb.shape[0], p, -1)
        return jnp.einsum("ipk,jpk->ij", fa.conj(), fb)
    return g


def split_gram_t(p):
    """The partial products with the chunk axis leading (a transpose copy of each operand)."""
    @jax.jit
    def g(a, b):
        fa, fb = a.reshape(a.shape[0], -1), b.reshape(b.shape[0], -1)
        n = fa.shape[1]
        pad = (-n) % p
        if pad:
            fa, fb = jnp.pad(fa, ((0, 0), (0, pad))), jnp.pad(fb, ((0, 0), (0, pad)))
        fa = jnp.swapaxes(fa.reshape(fa.shape[0], p, -1), 0, 1)
        fb = jnp.swapaxes(fb.reshape(fb.shape[0], p, -1), 0, 1)
        return jnp.sum(jnp.matmul(fa.conj(), jnp.swapaxes(fb, 1, 2)), axis=0)
    return g


def krylov_before():
    """``_correction_krylov`` before 2026-10-30: the Krylov basis as (steps + 1, b, n) (the block axis second)."""
    from functools import partial

    from meow_gpu.fde.operator import apply

    @partial(jax.jit, static_argnums=(5,))
    def k(op, pc, q, theta, rhs, steps):
        b = rhs.shape[0]
        shape = rhs.shape[1:]
        beta = jnp.sqrt(jnp.sum(jnp.abs(rhs.reshape(b, -1)) ** 2, axis=1))
        v0 = rhs / jnp.where(beta > 0, beta, 1.0).astype(rhs.dtype)[:, None, None, None]
        vb = jnp.zeros((steps + 1, b) + shape, rhs.dtype).at[0].set(v0)
        hb = jnp.zeros((b, steps + 1, steps), rhs.dtype)
        th = theta[:, None, None, None]

        def body(j, carry):
            vb, hb = carry
            z = E._project_out(E.precondition(pc, vb[j], theta), q)
            wv = E._project_out(apply(op, z) - th * z, q)
            f3 = vb.reshape(steps + 1, b, -1)
            for _ in range(2):
                hj = jnp.einsum("jbn,bn->bj", f3, wv.reshape(b, -1).conj()).conj()
                wv = wv - jnp.einsum("bj,jbn->bn", hj, f3).reshape(wv.shape)
                hb = hb.at[:, :, j].add(hj)
            hn = jnp.sqrt(jnp.sum(jnp.abs(wv.reshape(b, -1)) ** 2, axis=1))
            hb = hb.at[:, j + 1, j].set(hn.astype(hb.dtype))
            ok = hn > 1e-30
            scale = jnp.where(ok, hn, 1.0).astype(wv.dtype)
            vb = vb.at[j + 1].set(jnp.where(ok[:, None, None, None], wv / scale[:, None, None, None], 0))
            return vb, hb

        vb, hb = jax.lax.fori_loop(0, steps, body, (vb, hb))
        return vb, hb, beta
    return k


def large_moves(compiled_text: str, min_elements: int) -> dict:
    """The transpose and copy instructions of an optimized HLO text (with the fused computations) whose result has
    at least ``min_elements`` elements: {kind: count}."""
    import re
    out: dict = {}
    for line in compiled_text.splitlines():
        m = re.search(r"= (\w+)\[([\d,]*)\]\{[^}]*\} (transpose|copy)\(", line)
        if m and m.group(2) and np.prod([int(d) for d in m.group(2).split(",")]) >= min_elements:
            out[m.group(3)] = out.get(m.group(3), 0) + 1
    return out


@jax.jit
def fused_gram(a, b):
    fa, fb = a.reshape(a.shape[0], -1), b.reshape(b.shape[0], -1)
    return jnp.sum(fa.conj()[:, None, :] * fb[None, :, :], axis=-1)


def main():
    _, css = build(4, 0.02)
    cs = css[1]
    p = make_problem(cs.nx**2, cs.ny**2, cs.nz**2, cs.mesh.x, cs.mesh.y, cs.env.wl)
    rng = np.random.default_rng(0)
    b, m = 18, 72
    shape = (2,) + p.shape
    print("grid", p.shape, "n", 2 * np.prod(p.shape), "b", b, "M", m, "device", E.solver_device(), flush=True)
    v = put(rng.normal(size=(m,) + shape) + 1j * rng.normal(size=(m,) + shape), jnp.complex64)
    t = put(rng.normal(size=(b,) + shape) + 1j * rng.normal(size=(b,) + shape), jnp.complex64)
    ref = np.asarray(E._gram(v, t), dtype=np.complex128)
    print("Gram <v_i, t_j> (72 x 18), complex64:")
    bench("_gram (current)", E._gram, v, t)
    bench("_gram t,t (18 x 18)", E._gram, t, t)
    for name, f in [("fused multiply-reduce", fused_gram)] + [(f"split P={q}", split_gram(q)) for q in (16, 64, 256)] \
            + [(f"split, chunk axis first P={q}", split_gram_t(q)) for q in (16, 64, 256)]:
        r = bench(name, f, v, t)
        print(f"    max rel diff {np.abs(np.asarray(r) - ref).max() / np.abs(ref).max():.1e}")
        bench(name + " (t,t)", f, t, t)
    print("vmap over B = 8 members:")
    vb, tb = jnp.stack([v] * 8), jnp.stack([t] * 8)
    bench("vmap _gram (current)", jax.jit(jax.vmap(E._gram)), vb, tb)
    for q in (16, 64, 256):
        bench(f"vmap split P={q}", jax.jit(jax.vmap(split_gram(q))), vb, tb)
    bench("vmap _project_out (current)", E._project_out_b, tb, vb)
    bench("_project_out (one member)", E._project_out, t, v)
    bench("_combine (one member)", E._combine, put(rng.normal(size=(m, b)) + 0j, jnp.complex64), v)

    # the GMRES of the refinement: one call of 20 steps, b = 18
    op = p.device(jnp.complex64)
    pc = E.make_preconditioner(p)
    q = E._svqb(t, 1e-4)[0]
    rhs = E._project_out(put(rng.normal(size=(b,) + shape) + 0j, jnp.complex64), q)
    th = put(np.full(b, -8.0), jnp.complex64)
    print("refinement kernels (b = 18, 20 steps):")
    _, hk0, _ = bench("_correction_krylov, (steps+1, b, n) basis", krylov_before(), op, pc, q, th, rhs, 20, n=5)
    _, hk, _ = bench("_correction_krylov, (b, steps+1, n) basis", E._correction_krylov, op, pc, q, th, rhs, 20, n=5)
    print(f"    max |dH| {np.abs(np.asarray(hk) - np.asarray(hk0)).max():.1e}  |H| {np.abs(np.asarray(hk0)).max():.1e}")
    basis = 21 * b * rhs[0].size            # the elements of the Krylov basis (20 steps)
    for name, f in (("(steps+1, b, n)", krylov_before()), ("(b, steps+1, n)", E._correction_krylov)):
        text = f.lower(op, pc, q, th, rhs, 20).compile().as_text()
        print(f"    optimized HLO, {name}: transposes and copies of >= {basis} elements: "
              f"{large_moves(text, basis)}", flush=True)
    print("refinement kernels, vmap over B = 8 members (a batched refinement):")
    from meow_gpu.fde.eigensolver import stack
    opb, pcb = stack([op] * 8), stack([pc] * 8)
    qb, thb, rhsb = jnp.stack([q] * 8), jnp.stack([th] * 8), jnp.stack([rhs] * 8)
    for name, f in (("(steps+1, b, n)", krylov_before()), ("(b, steps+1, n)", E._correction_krylov)):
        kb = jax.jit(jax.vmap(f, in_axes=(0, 0, 0, 0, 0, None)), static_argnums=(5,))
        bench(f"vmap _correction_krylov, {name}", kb, opb, pcb, qb, thb, rhsb, 20, n=3)
    bench("precondition", E.precondition, pc, rhs, th)
    bench("_apply_block", E._apply_block, op, rhs)

    print("host eig of the projected matrix (complex, 72 x 72):")
    hs = [rng.normal(size=(m, m)) + 1j * rng.normal(size=(m, m)) for _ in range(8)]
    t0 = time.perf_counter()
    for _ in range(5):
        for h in hs:
            np.linalg.eig(h)
    print(f"  serial, 8 members  {(time.perf_counter() - t0) / 5 * 1e3:8.3f} ms", flush=True)
    for nt in (2, 4, 8):
        with ThreadPoolExecutor(nt) as ex:
            list(ex.map(np.linalg.eig, hs))
            t0 = time.perf_counter()
            for _ in range(5):
                list(ex.map(np.linalg.eig, hs))
            print(f"  pool of {nt}, 8 members {(time.perf_counter() - t0) / 5 * 1e3:8.3f} ms", flush=True)
    E._host_map(np.linalg.eig, [(h,) for h in hs])
    t0 = time.perf_counter()
    for _ in range(5):
        E._host_map(np.linalg.eig, [(h,) for h in hs])
    limit = "threadpoolctl" if E.ThreadpoolController is not None else "no threadpoolctl"
    print(f"  _host_map ({limit})  {(time.perf_counter() - t0) / 5 * 1e3:8.3f} ms", flush=True)
    t0 = time.perf_counter()
    for _ in range(5):
        np.linalg.eig(np.stack(hs))
    print(f"  one stacked call   {(time.perf_counter() - t0) / 5 * 1e3:8.3f} ms", flush=True)
    import scipy.linalg as sl
    t0 = time.perf_counter()
    for _ in range(5):
        for h in hs:
            sl.eig(h, check_finite=False, overwrite_a=False)
    print(f"  scipy serial       {(time.perf_counter() - t0) / 5 * 1e3:8.3f} ms", flush=True)


if __name__ == "__main__":
    main()
