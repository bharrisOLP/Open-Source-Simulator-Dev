"""Time each stage of compute_modes_gpu for the cells of the EME taper benchmark (development tool)."""
import sys
import time
from collections import defaultdict
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import meow_gpu.meow_backend as B  # noqa: E402
from bench_eme_taper import build  # noqa: E402

T = defaultdict(float)


def timed(name, f):
    def g(*a, **k):
        t0 = time.perf_counter()
        out = f(*a, **k)
        jax.block_until_ready(out) if not isinstance(out, list) else None
        T[name] += time.perf_counter() - t0
        return out
    return g


B.make_problem = timed("make_problem", B.make_problem)
B.make_preconditioner = timed("make_preconditioner", B.make_preconditioner)
B.davidson = timed("davidson", B.davidson)
B.fields = timed("fields", B.fields)
import meow.fde.post_process as PP  # noqa: E402
import meow_gpu.fde.eigensolver as E  # noqa: E402
for _n in ["_ritz", "precondition", "_project_out", "_apply_block", "_insert", "_gram", "_combine", "_svqb"]:
    setattr(E, _n, timed("E." + _n, getattr(E, _n)))
_eig = np.linalg.eig
np.linalg.eig = timed("host eig", _eig)
pp = timed("post_process", PP.post_process_modes)
cells, css = build(10, float(sys.argv[1]) if len(sys.argv) > 1 else 0.02)
t_nx = time.perf_counter()
_ = [cs.nx for cs in css]
print(f"cs.nx (meow index sampling with subpixel smoothing, 10 cells): {time.perf_counter() - t_nx:.2f}s")
solver = B.GPUModeSolver(precision="single")
B.compute_modes_gpu(css[0], 10, solver=solver, post_process=pp)
print('=== WARM CELLS ===', flush=True)
T.clear()
t0 = time.perf_counter()
for cs in css[1:]:
    B.compute_modes_gpu(cs, 10, solver=solver, post_process=pp)
    print("  cell:", solver.last_info, flush=True)
total = time.perf_counter() - t0
print(f"9 warm cells: {total:.2f}s ({total / 9:.3f} s/cell)")
for k in sorted(T, key=T.get, reverse=True):
    print(f"  {k:22s} {T[k] / 9 * 1e3:8.1f} ms/cell")
