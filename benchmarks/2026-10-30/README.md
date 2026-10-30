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
| `fdtd-v3.jsonl` | the FDTD benches again, after the placement and device corrections of the FDTD adapter, and `suite/bench/sloped_taper.py` (80 degree sidewalls, EME and FDTD) | the FDTD values of suite/README.md: 0.9974 / 0.9981 / 0.9961; sloped 0.9985 / 0.9975, EME 0.9878 |
| `fdtd-diagnostics.jsonl` | rows of the FDTD adapter before its corrections, each with a note (40 nm gaps, the slow runs after an EME run) | the numbers of the corrections in suite/README.md: T_flux 0.44, T 0.17, 149 s against 16 s, 1307 s |
| `eme-neff-fix.jsonl` | `meow_gpu/bench/bench_eme_taper.py --backend gpu --precision mixed` (with the neff correction of the post-processing) | the EME taper T00 = 0.9867 / 0.9900 / 0.9912 at 30 / 60 / 120 cells, 0.9899 with 20 modes (T00 = S21_00 squared); rows 1 to 4 come from the script before it had the post_process, P_right and P_left keys (all with the corrected post-processing); rows 5 and 6 (60 cells) compare the corrected and the meow post-processing; row 7 is `suite/bench/eme_taper.py` (photonsuite.eme on the FDTD segment list, 0.2 um cells: T00 = 0.9888) |
| `charge-rc.jsonl` | `charge_gpu/examples/pn_modulator_rc.py` (on the GPU computer, `.venv-rocm`) | charge README: the RC example table and its GPU timings |
| `thermal3d.jsonl` | `thermal_gpu/bench/bench_steady3d.py` (heater 100 / 50 / 25 nm, V and W cycles; cube 64^3 to 192^3; GPU and `THERMAL_GPU_DEVICE=cpu`) | thermal README: the 3-D timing table (8x to 46x) |
| `bands-oblique.jsonl` | `bands_gpu/bench/bench_bands.py --case tri-holes --n 128` (GPU and `BANDS_GPU_DEVICE=cpu`) | bands README: 5.64 s against 89.3 s, 13x per iteration |
| `eme-batch.jsonl` | `meow_gpu/bench/bench_eme_batch.py --precision single` and `mixed`, `--batches 4,8,16 --repeat 2` | meow README: the batched EME table (1.2x single, no gain mixed) |
| `optics-trace.jsonl` | `optics_gpu/bench/bench_trace.py` (singlet 1e4 to 1e7 rays, 4f 1e5 to 1e7 rays; NumPy, PyTorch CPU and GPU; `.venv-optics`) | optics README: 7.3x / 7.4x at 1e7 rays against one-thread NumPy (4.5x / 4.3x with the copy to the host) |
| `charge-dd.jsonl` | `charge_gpu/bench/bench_dd.py --dx 10e-9 5e-9 --no-diode` (GPU and `CHARGE_GPU_DEVICE=cpu`), `charge_gpu/examples/pin_phase_shifter.py --modes --tau 1e-9 1e-8` | charge README: drift-diffusion timings and the PIN phase-shifter table |
| `thermal-transient.jsonl` | `thermal_gpu/bench/bench_transient3d.py`, `thermal_gpu/bench/bench_kappa_t.py` (GPU and `THERMAL_GPU_DEVICE=cpu`), and the `--dt-scan 3` rows at the end | thermal README: the transient and kappa(T) tables, the time-step check |
| `bands-findk.jsonl` | `bands_gpu/bench/bench_find_k.py` (strip, pcw, slab with `--sub`; GPU and `BANDS_GPU_DEVICE=cpu`) | bands README: find_k timings and the slab sub-sample study |
| `meow-refine.jsonl` | `meow_gpu/bench/bench_refine.py --repeat 2` (final code of the refinement) | meow README: mixed 11.72 s against 19.48 s, 10.8x fewer complex128 applies (rerun after the fallback fix, with the `polish_used` field) |
| `meow-bend.jsonl` | `meow_gpu/bench/bench_bend.py` (mixed, single, double; tight bends with PML; the 40 and 20 nm and 4-mode rows of the review) | meow README: bend timings and agreement with tidy3d |
| `optics-pipeline.jsonl` | `optics_gpu/bench/bench_pipeline.py` (singlet and 4f, 1e5 to 1e7 rays) and the last row from `optics_gpu/examples/coupling_scans_on_device.py` | optics README: the device pipeline table and the scan example |
| `charge-dd-device.jsonl` | `charge_gpu/bench/bench_dd.py --dx 10e-9 5e-9 --no-diode --no-sweep` (final code, LU coarsest inverse) | charge README: the device continuity table |
| `fdtd-gpu.jsonl` | `fdtd_gpu/bench/bench_fdtd.py` (Bragg, waveguide 2-D and 3-D, ring, dipole, sheet) | fdtd_gpu README: all check numbers and times |
