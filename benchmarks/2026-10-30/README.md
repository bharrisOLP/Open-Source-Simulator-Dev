# Benchmark logs, 2026-10-30

Raw JSON rows, one per run, as printed by the bench scripts on the GPU computer (Radeon RX 7900 XT, gfx1100; CPU:
Ryzen 7 7800X3D, 16 threads; ROCm 10.1 pip wheels, JAX 0.11.1). "device" is the JAX device of the solver.

| File | Script | Claims that cite it |
|---|---|---|
| `bands-before-ap-fix.jsonl` | `bands_gpu/bench/bench_bands.py` (before the A P fix of LOBPCG) | bands README: the table of the solver before the A P fix (4.8x to 15.6x) |
| `run-20261030-0626.jsonl` | `thermal_gpu/bench/bench_steady.py` (Jacobi, the solver default at that time; now `--precond jacobi`), `bands_gpu/bench/bench_bands.py`, `examples/pn_modulator.py` (charge with Jacobi CG, before the plasma-dispersion fix: its V_pi L values are superseded) | thermal Jacobi timings; bands after the A P fix (11.7x); charge with Jacobi CG, 7.4 s against 19.5 s |
| `thermal-mg.jsonl` | `thermal_gpu/bench/bench_steady.py` (multigrid) | 0.10 / 0.12 / 0.21 s on the GPU at 20 / 10 / 5 nm (2.9 million cells) |
| `taper-conv.jsonl` | `meow_gpu/bench/bench_eme_taper.py` (EME, mixed precision, meow's own post-processing), `suite/bench/fdtd_taper.py` (uncalibrated FDTD) | the old EME values of the taper (0.955 at 60 and 120 cells; 20 modes: 0.947), before the neff correction |
| `fdtd.jsonl` | `suite/bench/fdtd_taper.py`, `suite/bench/fdtd_taper_calibrated.py`, `suite/bench/fdtd_straight.py` | the calibrated FDTD taper transmission T00 = 0.9969 / 0.9978 / 0.9945 (25 nm 100 and 200 segments, 20 nm 100 segments), and the straight strip (T = 0.9999995, 10.7 s) |
| `eme-neff-fix.jsonl` | `meow_gpu/bench/bench_eme_taper.py --backend gpu --precision mixed` (with the neff correction of the post-processing) | the EME taper T00 = 0.9867 / 0.9900 / 0.9912 at 30 / 60 / 120 cells, 0.9899 with 20 modes (T00 = S21_00 squared); rows 1 to 4 come from the script before it had the post_process, P_right and P_left keys (all with the corrected post-processing); rows 5 and 6 (60 cells) compare the corrected and the meow post-processing; row 7 is `suite/bench/eme_taper.py` (photonsuite.eme on the FDTD segment list, 0.2 um cells: T00 = 0.9888) |
