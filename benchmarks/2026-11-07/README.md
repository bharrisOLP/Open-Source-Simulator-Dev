# Benchmark logs, 2026-11-07

Raw rows, one per run, as printed by the bench scripts and examples on the GPU computer (Radeon RX 7900 XT, gfx1100;
CPU: Ryzen 7 7800X3D, 16 threads; ROCm 10.1 pip wheels, JAX 0.11.1). Another session shared the computer on this
day, so every run had `OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4` and
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.4`, unless the row or this table says otherwise. Times can be longer than on a
free computer.

| File | Script and command | Claims that cite it |
|---|---|---|
| `fdtd-gpu-avg.jsonl` | `fdtd_gpu/bench/bench_fdtd.py` (`env PYTHONPATH=src`), nine commands in this order: `--case all`; `--case waveguide --three-d`; the same with `--src-freq 0.5701613`; `--case ring --resolution 40,60,80 --both`; `--case slab --resolution 20,40,80 --both`; the same with `--e-normal`; `--case bragg --resolution 40,80,160 --both --lam0 1.03`; `--case disk --resolution 40,60,80 --both`; `--case setup --resolution 20,50` | fdtd_gpu README: the checks of the subpixel averaging. Line 31 (Bragg, staircase, res 160) is a repeat of its command: a scipy RuntimeWarning on stderr cut the row of the logged run (the row has a "note" key). |
| `fdtd-gpu-forward-diagonal.jsonl` | `bench_fdtd.py --case disk --resolution 40,60,80 --forward-diagonal` (rows 1-3) and `--case waveguide --three-d --forward-diagonal` (row 4), stderr dropped | fdtd_gpu README: the comparison of the two diagonals |
| `meow-kernel-reduce.txt` | `meow_gpu/bench/kernel_reduce.py` (text output), first with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`, then with no thread variables. Lines 78-79: a stray message of the remote shell ("(null): No such file or directory") splits one line; no claim cites that line. | meow README: the GMRES layout (the large transposes in the optimized HLO, the times) and `_host_map` |
| `meow-cold-switch.jsonl` | `meow_gpu/bench/bench_refine.py --cases double,mixed-refine --repeat 2` (rows 1-2), the same with `--switch-tol 1e-3` (rows 3-4); `bench_eme_batch.py --precision mixed --batches 8 --repeat 2` (rows 5-6), the same with `--switch-tol 1e-3` (rows 7-8) | meow README: the cost of the cold-start stop at 2e-5 |
| `optics-polish.jsonl` | `optics_gpu/examples/coupling_scans_on_device.py`: `--save-mode` with the default polish and with `--polish davidson` (rows 1-2), then the optics run on each mode file (rows 3-4); the core with the staircase error and the earlier source rule | optics README: the polish paragraph (both polishes give the same mode) |
| `optics-pol-x.jsonl` | the same example: `--save-mode --pol x` (row 1), the optics run (row 2); the core with the staircase error | optics README: the history of the source problem |
| `optics-sym-x.jsonl` | the same example with the symmetric core: `--save-mode --pol x --mode-file ../results/optics_gpu/sin_tip_mode_sym_x.npz` (row 1), the optics run on that file (row 2) | optics README: the example section (eta 0.8866) |
| `chip-to-fiber.jsonl` | `examples/chip_to_fiber_coupling.py`: `--save-mode --pol x` in `.venv-rocm` (row 1), then the optics run in `.venv-optics` (row 2) | STATUS (item 5): eta 0.8866 at focus |
