# Status

Updated: 2026-11-08.

## In work

**Now (2026-11-08; the fourth to eleventh waves, one agent per package).** Each change gets one review before its
commit. The eleventh wave is done (below); the minor findings of its reviews (for example of review f39279a5164e)
are open. In work (twelfth wave): the EME-FDTD gap of the double step and a rule for complex pairs at the cut
(meow_gpu, suite); the review minors of photonsuite; the non-uniform FFT as the default (optics_gpu).
- Eleventh wave (new features), done: bands_gpu, the hole slab of Johnson et al. (1999) against 3-D FDTD (4847fc7;
  fdtd_gpu has no Bloch phases, so M and K fold to Gamma in supercells of a 7/4 crystal with cos-weighted sources;
  first-order convergence; the extrapolations are 4e-5 to 6e-5 (K) and 2e-5 to 2e-4 (M) below bands_gpu; the earlier
  triangular values carried a face bias (the slab faces on a subpixel sample), corrected gap at n 48: 0.265304 to
  0.344444); optics_gpu, through-focus, multi-field and polychromatic PSF and MTF in one batch (0e9795c; the fixed cost
  of a wavefront call cut from 42 ms to about 9 ms; a batch of 45 planes 4.5x to 11.5x faster than one call per plane
  and 14x to 34x faster than Optiland on the CPU (pupils 64 to 256, without the polychromatic part; 8.4x to 14.1x with
  it); defocus refuses an image surface with different media on its two sides); charge_gpu, a public face-current
  accessor, and photonsuite.pin, a two-way electro-thermal loop on the mean Si temperature (a210ccf; the converged
  current is up to 7.2 % (1 ns) and 11.9 % (10 ns) above one-way; the cross-over moves 12 and 21 mV lower).
- Tenth wave, done: meow_gpu, the modes that its solver "missed" were spurious PEC-wall modes (2d19636; the tidy3d
  operator keeps tangential E on the near walls as decoupled unknowns; tidy3d excludes them too; meow-gpu now deflates
  them, and eps_ref is 0.99 min Re(eps) for every problem, because near-uniform windows failed without it); the
  converged EME rows of 505c4c5 already used the physical mode sets (the rerun agrees within 1.4e-8, except the taper at
  60 modes and 1.60 um, 3.6e-5, and at 80 modes, up to 2.2e-4, probably the choice of a complex-pair member at the cut,
  and the 120-mode double step at 1.575 um, 9.4e-4, whose 505c4c5 solve did not converge); photonsuite, a PIN injection
  phase shifter with self-heating (0753ed0; the carrier and thermal index changes cancel near 1.16 V at a lifetime of 1
  ns (1.21 V at 10 ns) in this thermal model, and a heat path removes the cancel point up to 1.3 V).
- Tenth wave, also done: the add-on route, the GNU timeout in the suite runner and a CuPy cache folder per policy (the
  two modules with 83 and 14 errors pass; the other 58 modules that name the cache give the same counts), and no patch
  0005 (the repair must equal the whole-volume update of the same device bit for bit) (2e48a3c, 49c6849).
- Ninth wave (new features), done: optics_gpu, the image-quality analyses of lens-design tools on the GPU: the OPD map,
  the Zernike fit (Fringe and Noll), the FFT PSF, the MTF and a Huygens PSF (603ae68, 66da5d3; a perfect lens against
  the Airy pattern and the diffraction-limited MTF to 3.5e-4, a pure defocus against the exact form to 1.4e-11 waves, a
  singlet against Optiland to 3.2e-11 (1e-11 on the GPU path); slower than Optiland up to a pupil of 256 (a fixed cost
  of about 42 ms), 1.23x to 12.0x faster from 512 to 2048; the Optiland notice in THIRD_PARTY_NOTICES.md); fdtd_gpu,
  current sources in conductive cells (they were (1 + b) too strong), dispersive media in the PML (a Drude medium near
  its plasma edge gets up to 1.4e-2 back: a run-time warning and `PML(R_asymptotic=...)`), and near2far in lossy media
  (5ab6e9a); meow_gpu, an EME interface that conserves power and a star-product cascade (505c4c5; 5.45 GiB instead of
  28.3 GiB at 80 modes; on the taper 0.0005 below to 0.0090 above the 3-D FDTD at 40 to 80 modes, not converged in the
  mode count; the double step stays 0.004 to 0.010 above the FDTD at 40 to 120 modes, not explained). Found by that
  work: the meow-gpu solver did not give some eigenvalues of the full matrix (about 10 of the 40 nearest); the tenth
  wave showed that they are spurious PEC-wall modes (2d19636). Four of the five reviews of this paragraph failed and
  were fixed (the first optics review passed; the review of its follow-up failed): the README advised to end a Drude
  medium before the PML (that reflects more); the post-processing removed valid modes (complex pairs, the wrong branch);
  the README described the image-space index of Optiland wrongly; the uutils timeout of the GPU computer sent SIGKILL
  only to the direct child (tools/gpu_run.sh now uses GNU timeout, 7202be9).
- Eighth wave (new features), done: fdtd_gpu, Drude and Lorentz dispersion on the ADE of fdtdx 0.6.3 and
  D_conductivity on its semi-implicit conductivity update (0a4c4b0; fdtdx needed no patch; a fill-fraction face
  rule, second order for E parallel to a face on the exact Yee grid (the FDTD follows it to res 200, then the
  float32 ADE coefficients limit it); on a Lorentz film 0.504 um thick the MIT rule and the staircase have 3.8 to
  8.9 times its error; checks against the
  transfer matrix and the exact Yee grid; a derived stability bound that a run enforces); photonsuite, EME against
  3-D FDTD with subpixel averaging on a short SOI taper (f7ecdea; the FDTD T00 converges to about 0.909 at
  1.55 um; meow's EME gives 0.876 to 0.919 and does not converge in the mode count up to 40 modes; the probable
  cause: its interface solve does not conserve power, a gain of 0.8 to 1.2 % without correction); the add-on
  route, all 409 test modules of the add-on on the AMD GPU (698a2a4; 379 clean with four patches, each failure of
  the other 30 sorted; patch 0004: a subtraction with a NaN subtrahend inverts the sign bit of the NaN on this GPU, now
  host-ordered in the source deposit). The fdtd_gpu and add-on reviews failed first and were fixed: an off-grid
  check tested no face inside a cell, and the face rule was worse than the MIT rule there (fdtd_gpu); patch 0004
  missed the second subtraction of the deposit (add-on). The GPU computer had no problem during the 409-module
  runs (kernel 7.0.0-42).
- Seventh wave (new features), done: charge_gpu, temperature dependence from 250 to 500 K (42e1e84; n_i(T) of
  Misiakos and Tsamakis, the Arora mobility(T), tau ~ T^-1/2; bit-identical at 300 K; the long-base diode within
  0.5 % of Shockley(T); at 360 K the depletion modulator has 19.6 to 19.7 % more series resistance and 17.0 to
  20.6 % less intrinsic bandwidth, at 0 to 4 V reverse bias); photonsuite, the switching time of a thermo-optic
  phase shifter (a24b9c6; the phase from the thermal transient and one mode solve, the exact first-order
  derivative; 17.94 us with constant kappa, 17.61 us (on) and 17.87 us (off) with Si and SiO2 kappa(T)); fdtd_gpu,
  a near-to-far-field transform with exact Green's functions (c17cb93; second order against the analytic dipole
  and line-source fields; far-field power equal to the box flux to 0.3 %; fdtdx applies the source current half a
  time step late, documented). Its review follow-ups are done (63b31de, 1e801bc; the two wording notes of review
  b6be8c2f2946 in 5ab6e9a).
- Sixth wave (new features), done: thermal_gpu, transients with kappa(T) and rho c_p(T) (486b1d5; second order
  in dt against solve_ivp; timings a47f265: a second run costs 1.8x to 2.6x the constant-kappa transient with Si
  and SiO2 kappa(T)); optics_gpu, a type-1 non-uniform FFT for the image sum (ab60025; at tol 1e-12, 1.3x to about
  194x faster than the direct complex128 sum, where the upper end has a direct time estimated from a subset of
  the rays; error at most 1.8e-14 of sum|a|; the default stays "direct"); bands_gpu, the eps sampling on the GPU
  (2ce1bdd; 16.5x to 404x faster after the first call, equal to the CPU path; 3-D checks against an exact slab and
  an independent supercell; no published gap value was found without WebFetch). All three reviews failed first
  and were fixed: a label array in the wrong layout was reshaped silently (thermal); the NUFFT tolerance was not a
  bound, and large grids fell to one ray per chunk (optics); in symmetric voxels the normal came from rounding,
  so the CPU and GPU paths gave TE bands 7.8e-4 apart (bands).
- Done: charge_gpu, the coupled Newton step on the GPU (297baf2; 5.8x to 5.9x faster than SuperLU at 5 nm) and
  a follow-up of its review (f937f57); optics_gpu, all focus planes in one pass (0bbad9f; 3.1x per plane in
  complex128, about 27x in complex64); meow_gpu, the profile of the batched path (88d5158; the taper in mixed
  precision 7.9 s instead of 11.7 s per cell, single 1.86x faster at B = 4).
- Done on 2026-11-07: fdtd_gpu, subpixel averaging of the permittivity, now the default (46da80a; ring and
  Bragg converge at second order); charge_gpu, the SuperLU fallback catches only real device failures (0bb5179);
  meow_gpu, a cold start of "mixed" searches to 2e-5 in complex64 (18f2178; before, the SiN tip could give wrong
  modes 3 and 4 with converged=True, decided by the BLAS thread count); optics_gpu, a symmetric core and an
  explicit x-polarized source in the scans example (e15c847; the core had a one-column staircase error, so the
  two lowest modes were diagonal and rounding chose the source component; eta 0.8866 instead of 0.8888), and the
  same fix in the top-level example `examples/chip_to_fiber_coupling.py` (the documents commit of 2026-11-07);
  fdtd_gpu, objects repeat across periodic faces (eade281).
- Done on 2026-11-07, late: the add-on route of item 2 runs on the AMD GPU (a6938b6; time loop 16x to 28x faster
  than the serial CPU solver on a 2.1-million-cell 3-D case; a longer setup); its review follow-up (21bb956: patch
  0002 by parameter name, CuPy pinned to 14.2.0, whole-run times; the patches are installed into the pinned clone).
- Fifth wave (2026-11-07, night; the minor findings of the last reviews): fdtd_gpu, an error for k_point with a
  one-sided PML, a test of the wrap padding and a faster rasterize (0f13a8b); meow_gpu, a warm-start test on the
  SiN tip and a lock for the BLAS limit (f0bf399); charge_gpu, sweep() raises an error of the device solve at once
  (9127c9b); the add-on route, patch 0003 (the bulk read of the permittivity for this build; ebfec13): the setup
  before the first step falls from 21-42 s to 1.9 s (2-D) and 13.0 to 13.2 s (3-D), so whole runs with the kernel
  tables are now faster on the GPU.
- 2026-11-07, the GPU computer stopped at about 16:49:53 EST (the kernel time stamps; the last journal entry is
  16:49:44): the kernel (7.0.0-38-generic) stopped with an Oops in the
  amdgpu/TTM memory manager ("list_del corruption" and "list_add corruption", then a NULL pointer dereference in
  `ttm_resource_move_to_lru_tail`) during a GPU memory allocation (KFD ioctl) of a `python` process of the user,
  about 80 s after the RX 7900 XT resumed from runtime suspend. `panic_on_oops` is 1, so kdump saved a crash dump
  (`/var/crash/202611071650` on that computer). The OOM kills and the "Fatal error during GPU init" of the next
  boot came from the kdump capture kernel (small memory, one CPU; the GPU error is for the integrated GPU), not from
  the crash. The computer stayed down until the user restarted it (18:00; now kernel 7.0.0-42-generic). The other
  session on the computer used no GPU and no Python then (its message), so the process was very probably the
  add-on test run of this repository: a short login session made its log file at 16:49:44.64, about 8 s before
  the Oops, and the crashed python process has a higher process number than that session. Not known: if the
  trigger is a driver bug or the program. Found by the reviews of the documents commits (the records have the log
  details). On kernel 7.0.0-42, the 6 modules of that run ran one at a time (19:24 to 19:38) with no crash (some
  of their tests fail: the defects that the patches fix, and certification digests that the patches put out of
  date; `addon_gpu/README.md`), and the add-on tests, benches and example checks ran until 21:00, also with no
  crash. Still: run the add-on tests in small parts, and one GPU job at a time on the shared computer.

**1. meow-gpu (EME): version 0.1 works.** Folder: `meow_gpu/` (`meow_gpu/README.md` has the results).

Done:
- GPU FDE solver in JAX: the tidy3d mode operator (tested equal to machine precision), block Davidson with a
  fast-diagonalization preconditioner, warm start, single/mixed/double precision, drop-in `compute_modes_gpu`.
- Tests: operator, eigenvalues in each precision, warm start, meow backend against tidy3d, device placement.
- Results on the RX 7900 XT: one wide section (200k unknowns, 20 modes) 3.9x faster cold and 11.4x faster warm
  than tidy3d on one core; EME taper (30 cells) 3.3x faster than one core and 2.1x faster than 8 processes,
  |dS21| about 2e-5.

- Correction of meow's post-processing: each mode keeps its own neff (meow 0.15 averages them in Gram-Schmidt;
  this changed the EME of a taper from 0.955 to 0.991, see item 7).

- Batched solves (`solve_batch`, `compute_modes_gpu_batch`; an option): the same modes as the per-cell path, but
  on the RX 7900 XT only 1.2x in single precision and no gain in mixed precision for the 30-cell taper.

- Mixed-precision refinement (the default of "mixed"): complex128 residuals and Rayleigh-Ritz, complex64 GMRES on
  the correction equation; the 30-cell taper in 11.7 s instead of 19.5 s (10.8x fewer complex128 applies), the
  same 1e-10 tolerance; near-degenerate pairs keep their mode identity (tested against ARPACK). When the
  refinement stalls (a weakly guided SiN tip above a dense band of window modes), a fallback runs the earlier
  polish from the complex64 block.

- Bend modes on the GPU (the bent-waveguide operator of tidy3d; equal to its matrix to 1e-14). These still fall
  back to tidy3d: angled propagation, and a bend that the transform cannot represent (radius 0, or the center of
  the bend inside the cross-section).

Open (later, in this order): profile the batched path (the work per member is its larger cost); angled propagation.

**2. Meep on AMD: two routes work** (the fdtdx front end `fdtd_gpu/`, and the add-on route `addon_gpu/`).
Findings (git-ignored clones in `.local/upstream/`):
- The Erised fork (GPL-2.0-or-later) is a Python add-on: Meep builds the simulation, and the add-on time-steps it
  with an array module (NumPy or CuPy) plus Triton kernels and CUDA C kernels (cupy.RawKernel). Route to AMD: CuPy
  for ROCm (hipRTC compiles RawKernel source) and the AMD backend of Triton. About 250k lines of Python with tests.
- gpmeep: a fork of the Meep C++ core (1.35.0-beta) with about 19k lines of CUDA and C++. Route to AMD: HIPify.
- fdtdx (MIT, JAX): already runs on this card. A front end on it in the style of the MIT package is the third
  route: `fdtd_gpu/` (2026-10-30, version 0.1: Simulation, geometry, sources, flux and mode monitors, Harminv;
  checked against a Bragg transfer matrix, exact ring resonances, the analytic dipole power and a current sheet).
  It needs no third-party GPL code; the mode solver of fdtdx uses tidy3d (LGPL-2.1).
- Probe result (2026-10-30, `tools/probe_cupy_triton.sh`, `tools/probe/`): YES. PyTorch 2.14.1 for ROCm 7.2 runs
  on gfx1100; Triton 3.8.0 (AMD) runs a kernel; CuPy 14.2 (cupy-rocm-7-0) compiles and runs a RawKernel, with
  versioned links to the HIP libraries of the PyTorch wheel and hipcc from the ROCm 10.1 pip SDK.
- **Add-on route: approved and installed.** The user approved it (2026-10-30: "Meep add on route: yes clone it,
  fine"). `tools/env/addon_gpu.sh` (named `tools/env/fdtd_gpu.sh` until 2026-10-30, before the folder `fdtd_gpu/`
  of the fdtdx route) cloned the add-on at commit 8993426 into `.local/upstream/erised-fork` on the GPU computer and
  made `.local/envs/addon-gpu`: the solver 1.35.0 (conda-forge, serial), the add-on 1.0.0, PyTorch 2.14.1 for ROCm
  7.2, Triton 3.8.0 and CuPy 14.2 see the RX 7900 XT. `tools/gpu_run.sh .local/upstream/erised-fork ...` and
  `tools/gpu_run.sh addon_gpu ...` use that environment.
- **Add-on route: runs on the AMD GPU (a6938b6, 2026-11-07; `addon_gpu/README.md`).** Two patches
  (`tools/env/addon_patches/`, GPL-2.0-or-later): hipRTC gets `-ffp-contract=off` for the NVIDIA option
  `--fmad=false`, so the hand-written kernel table compiles; subnormal float32 values are kept on ROCm (CuPy-HIP
  flushed them silently, a wrong-arithmetic defect). Both kernel tables equal the add-on's array path word for
  word, and agree with the CPU solver to 8.3e-6 or better (field, relative L2). Time loop on a 2,097,152-cell 3-D case
  (1,280 steps): 37.7 s on the serial CPU solver (float64), 1.34 s with the hand-written table (28x) and 2.35 s
  with Triton (16x), in float32 (`benchmarks/2026-11-07/addon-bench-full.jsonl`). The setup before the first step
  was longer on the GPU route (39-42 s against 11.5 s), a per-point read of the permittivity; patch 0003 (ebfec13)
  enables the add-on's bulk read for this build (exact: 0 words differ), and the setup is now 1.9 s (2-D) and
  13.0 to 13.2 s (3-D) against 1.25 s and 11.6 s on the CPU solver. Whole runs with the kernel tables, after a
  warm-up run in the same process: 2.19x to 2.34x faster in 2-D and 3.12x to 3.37x in 3-D (the array path: 1.36x
  and 2.61x; small runs stay slower on the GPU) (`benchmarks/2026-11-07/addon-bench-full-0003.jsonl`). The patches
  are installed in the pinned clone (0004 in its final form since the run of 2026-11-08, 09:05). Keep
  `.local/addon-runs/` (the raw logs, the saved fields and the empty log file whose time stamp dates the crash)
  and `.local/upstream/addon-work-3` (the state that made `addon-bench-full.jsonl`) on the GPU computer: they are
  the evidence of committed claims. Nothing is certified on AMD (the kernels run with `MEEP_GPU_HARDWARE=any`).
- **Add-on route: the whole test suite of the add-on (698a2a4, 2026-11-08).** `addon_gpu/suite_runner.py` ran
  all 409 test modules, one at a time, each with a 300 s limit. With the four patches 379 are clean; the failures
  of the other 30 are certification records (expected; 40 tests), the environment (230 tests and 97 errors: the
  solver 1.35.0 refuses `D_conductivity_offdiag`, the `keep` policy needs a CuPy cache folder with `ftz_stripped`
  in its name, the hardware switch), Triton 3.8 (2) and a test-order defect of the add-on (1). Patch 0004: on this
  GPU `a - b` with a NaN `b` inverts the sign bit of the NaN (NumPy on the host keeps it); the two device
  subtractions of the source deposit now take the NaN in the host's order on ROCm builds. Logs:
  `benchmarks/2026-11-08/addon-*`. Open: the CuPy cache folder of `tools/env/addon_gpu_activate.sh`; `timeout -k`
  in `tools/gpu_run.sh`; the float32 `D - P` update of `deposit_repair.py` keeps the GPU's NaN sign.

**3. bands-gpu (the method of MPB): version 0.1 works.** Folder: `bands_gpu/` (`bands_gpu/README.md`).
- Plane-wave operator with FFTs, anisotropic subpixel averaging, block LOBPCG (stabilized), warm start on the k
  path, TE/TM for 2-D cells.
- Checks: exact homogeneous and 1-D Bragg bands; the rod-lattice TM gap agrees with an independent
  finite-difference solver to 2e-4 (both are 0.7 % below the resolution-32 numbers of the MPB tutorial).
- GPU: 11.7x the same code on 16 CPU threads (rods 128 x 128, 16 k points: 5.2 s against 60.3 s;
  `benchmarks/2026-10-30/`).
- Oblique lattices (triangular, hexagonal, fcc) and the analytic interface normal of the objects (TE at second
  order from n = 64 to 128 also in sheared voxels; slower above 128, cause not known); triangular air holes
  128 x 128: 5.6 s on the GPU against 89 s on 16 CPU threads (the CPU run did not converge all k points; 13x per
  iteration).
- find_k (the k of each band at a fixed frequency, as MPB find-k): Newton with the Hellmann-Feynman group
  velocity, parity; a dielectric slab against the exact supercell (1.2e-5 to 4.4e-5 at n 1025; the error depends on the subpixel
  sample count), a photonic-crystal line-defect waveguide against an independent finite-difference solver.
- Next: a comparison with MPB itself.

**4. thermal-gpu: version 0.1 works.** Folder: `thermal_gpu/` (`thermal_gpu/README.md`).
- Cell-centered finite volumes (series face conductance), Jacobi CG as one device loop, steady and implicit
  transient solves.
- Checks: exact parabola, exact two-layer series resistance, sparse direct solve (1e-7 of the rise), slab decay
  time constant (1 %).
- Example `examples/thermo_optic_phase_shifter.py`: thermal-gpu + meow-gpu give P_pi = 19.0 mW for a synthetic SOI
  heater (published devices of this type: typically 20 to 30 mW).
- Multigrid preconditioner (shared, below).
- 3-D steady solver (`solver3d.py`, 3-D aggregation multigrid with semi-coarsening): a 100 um heater with its ends
  (11 M cells, 25 nm) in 1.03 s of CG on the GPU against 34 s on 16 CPU threads; mid-length rise equal to 2-D to
  1e-5 (7e-6 at 100 and 50 nm, 8e-6 at 25 nm), the ends at 0.53 of the 2-D rise, the phase integral 1.0001 of 2-D
  (100 and 50 nm).
- 3-D transients (implicit Euler, BDF2, Crank-Nicolson) and kappa(T) (Picard): the 10-90 % rise time of the 3-D
  heater is 17.94 us (equal to 2-D to 3e-5); kappa(T) of Si and SiO2 lowers the heater rise by 0.9 % at 100 W/m (Si alone: +1.3e-5). 3-D transient
  with 2.7 M cells: 20 s on the GPU against 349 s on 16 CPU threads.
- Next: kappa(T) in the transients; a 3-D heater with vias and pads.

**5. optics-gpu (the coupling use case of ZEMAX POP): version 0.1 works.** Folder: `optics_gpu/`.
- Angular spectrum of the source, real rays through the lens system (Optiland, MIT), image field as a separable
  ray sum on the GPU (PyTorch), coupling by overlap.
- Checks: 4f imaging and analytic Gaussian coupling (1e-3), power conservation (1 %), and an aberrated singlet
  against Optiland's independent Huygens PSF (correlation 0.998).
- Example `examples/chip_to_fiber_coupling.py`: SiN tip mode (meow-gpu) through a 2.5x lens pair into SMF-28:
  eta 0.8866 at focus and 0.6665 at +-60 um (2026-11-07: a symmetric core and an x-polarized source;
  `benchmarks/2026-11-07/chip-to-fiber.jsonl`; 0.889 before, with a core that had a one-column staircase error),
  with a focus scan (14 um mode window; an 8 um window cut the mode tails).
- Optiland's ray trace on the GPU (PyTorch backend, float64; `tools/env/optics_rocm.sh`): equal to the NumPy trace
  (difference 0); at 1e7 rays 7.3x faster than one-thread NumPy (4.5x with the copy to the host); not faster below about 1e5 rays.
- The whole pipeline on the GPU (trace, weights, image sum; focus and lateral scans from one trace): 1.4x to 1.85x
  faster than the NumPy pipeline from 1e6 rays; the image sum is now 93 % to 97 % of the time.
- Next: a faster image sum (for example a non-uniform FFT).

**6. charge-gpu (reverse-biased PN junctions for depletion modulators): version 0.1 works.** Folder:
`charge_gpu/`.
- Nonlinear Poisson with constant quasi-Fermi levels, Newton with device-loop CG; plasma dispersion
  (Soref-Bennett).
- Checks: depletion width (Sze, 1 %), charge neutrality, an independent scipy Newton solve (1e-8 V); the
  built-in potential as a consistency check.
- Example `examples/pn_modulator.py`: V_pi L 1.39, 1.60, 1.96 V cm at 1, 2, 4 V for a synthetic SOI rib (published:
  typically 1.5 to 3 V cm); the loss decreases by 1.0 to 2.8 dB/cm (fewer free carriers).
- Forward bias and currents: see the drift-diffusion solvers below.
- Junction capacitance, series resistance and RC bandwidth (`charge_gpu/src/charge_gpu/rc.py`, example
  `charge_gpu/examples/pn_modulator_rc.py`): C' against eps/W to 1e-4; at 3e17 cm^-3 and 2 V: 0.197 fF/um,
  39.4 Ohm mm, 20.4 GHz intrinsic. No GPU gain at 73k cells (charge part 1.6x at 293k cells, against the Mac CPU).
- Drift-diffusion (Scharfetter-Gummel, SRH; coupled Newton with SuperLU on the CPU, or Gummel with the Poisson step
  on the GPU): the long-base diode within 1 % of Shockley (0.60 to 0.75 V); a PIN phase shifter at 0.9 V with a carrier lifetime of 1 ns:
  L_pi 1.07 mm, P_pi 3.7 mW.
- The Gummel continuity solves on the GPU (BiCGSTAB with a multigrid preconditioner, SuperLU fallback): at 5 nm
  and 0.6 V the device Gummel path is the fastest (1.64 s against 4.76 s for Newton); Newton with SuperLU stays the
  fastest in high injection.
- Next: the coupled Newton step on the device.

**Shared: aggregation multigrid** (`thermal_gpu/src/thermal_gpu/multigrid.py`), used by thermal-gpu (steady and
transient) and charge-gpu (the Poisson Newton steps, the Gummel continuity solves and the preconditioner of the
device Newton). Heater stack: 17 to 27 CG iterations from 50 to 5 nm cells (Jacobi: 809 to 3982 from 50 to
10 nm); cells with a large aspect ratio need more. Heater stack at 5 nm (2.9 million cells):
0.21 s on the GPU, 1.30 s on 16 CPU threads (`benchmarks/2026-10-30/thermal-mg.jsonl`). PN modulator: 226 to 560
CG iterations per bias point instead of 11k to 36k; 2.1 s instead of 17.7 s on the Mac CPU (review record).

**7. photonsuite (one device description for all solvers): version 0.1 works.** Folder: `suite/`.
- `CrossSection` of rectangles with multiphysics materials; `mode_index`, `temperature`, `carriers`,
  `phase_shifter_ppi`, `modulator_vpil`; reproduces both examples within 1e-3. Every solve is checked (a solve
  that does not converge raises `ConvergenceError`); the import enables float64.
- `Polygon` shapes. 3-D FDTD adapter on fdtdx (`suite/src/photonsuite/fdtd.py`) and EME adapter on meow +
  meow-gpu (`suite/src/photonsuite/eme.py`), on the same segment description. FDTD straight strip: T = 0.9999995
  in 10.7 s on the GPU.
- **Finding (resolved): EME against FDTD on a 10 um Si taper (0.5 -> 2.0 um).** meow EME gave T00 = 0.955 and the
  calibrated FDTD 0.997. Cause: meow 0.15 averages the neff of the higher modes in its Gram-Schmidt step
  (`docs/upstream-issues.md`). With the correction in meow-gpu: EME 0.9867 / 0.9900 / 0.9912 at 30 / 60 / 120
  cells; FDTD 0.9961 at 20 nm (0.997 to 0.998 at 25 nm; its flux transmission is 1.0003 to 1.002). Agreement
  within about 0.5 %. Also found: meow drops polygon prisms along z (corrected in the EME adapter).
- `Polygon` in the FDTD adapter (exact cells at the cell centers; staircase fdtdx polygons on grid edges; no
  sub-pixel smoothing). A taper with 80 degree sidewalls: EME 0.9878 (0.10 % below vertical walls), FDTD 0.9975 at
  20 nm (0.14 % above): the sign of this small change is below the accuracy of the comparison.
- FDTD adapter fixes: each shape fills exactly the cells of its centers (segments of 2.5 cells left oxide gaps:
  T_flux 0.44 at 40 nm; sloped polygons moved by up to half a cell), the run on the GPU (after an EME run in the
  same process it was 9 to 13 times slower), a whole number of cells per axis, and clear errors for segments shorter
  than the step. The FDTD rows were run again (`fdtd-v3.jsonl`; old rows in `fdtd-diagnostics.jsonl`).
- Next: layer stacks from a GDS cross-section.

## Queue (in order of work)

The user asked to start with Meep and meow. After those two, the order is by complexity, the simplest first.

| # | Tool | Open-source base | GPU path on AMD | Complexity | Notes |
|---|---|---|---|---|---|
| 1 | meow (EME) | meow (Apache-2) + own FDE solver | JAX on ROCm | medium | v0.1 done; improvements open |
| 2 | Meep (FDTD) | Meep (GPL-2); GPU forks: Erised add-on (Triton/CUDA), gpmeep (CUDA C++) | CuPy-ROCm + Triton-AMD, HIP, or fdtdx | very high | fdtdx route v0.1 done (`fdtd_gpu/`, subpixel averaging); the add-on route runs on AMD (`addon_gpu/`, four patches; all 409 test modules of the add-on ran; not certified) |
| 3 | MPB (band and mode solver) | MPB (GPL-2) | plane-wave basis: FFT + block eigensolver in JAX | medium-high | v0.1 done (own code: `bands_gpu/`) |
| 4 | Thermal | own finite-volume solver (femwell / scikit-fem as references) | CG on GPU (JAX) | medium | v0.1 done (`thermal_gpu/`) |
| 5 | ZEMAX (ray + physical optics) | Optiland (MIT, PyTorch) + own physical optics | PyTorch on ROCm | medium | v0.1 done (`optics_gpu/`) |
| 6 | Charge | own nonlinear Poisson (reverse bias); DEVSIM (Apache-2) for full drift-diffusion | CG on GPU (JAX) | high | v0.1 done: reverse bias, and drift-diffusion for forward bias (`charge_gpu/`) |
| 7 | Tidy3D-like suite | items 1, 3, 4, 6 + fdtdx | JAX on ROCm | very high | v0.1 for cross-sections; 3-D FDTD (fdtdx) and EME adapters (`suite/`) |

"EME" and "FDTD" in the request of the user are items 1 and 2.

## Decisions for the user

These do not block the work. The agent continues with the default in brackets.

- None open.

Decided (2026-10-30):
- License: MIT for the code of this repository (user: "Add a open source MIT license to this"). `LICENSE`; the
  notes on upstream licenses are in `README.md` ("License"), the notices of adapted code in
  `THIRD_PARTY_NOTICES.md` and the license texts in `LICENSES/`.
- Add-on route of item 2: approved (user: "Meep add on route: yes clone it, fine"). Installed on the GPU computer
  (see item 2). The other open route is to HIPify the gpmeep C++ fork.

## Log

- 2026-10-30: repository started. Review gate, remote tool, landscape study (`docs/landscape.md`).
- 2026-10-30: meow-gpu 0.1 (GPU FDE solver for meow) with tests and benchmarks.
- 2026-10-30: process lessons. One adversarial review per commit (then a fix list, no second review). No
  permission dialogs (the hook denies instead).
- 2026-10-30: bands-gpu, thermal-gpu 0.1; then optics-gpu, charge-gpu 0.1 and the three examples.
- 2026-10-30: shared multigrid (thermal, charge), photonsuite 0.1, benchmark logs (`benchmarks/2026-10-30/`).
- 2026-10-30: FDTD (fdtdx) and EME adapters in photonsuite; two meow 0.15 errors found and corrected
  (`docs/upstream-issues.md`).
- 2026-10-30: Polygon in the FDTD adapter and exact cell placement. Then four agents in parallel (user request:
  "fan out some agents"), each on one package, each change with its own single review: charge RC bandwidth,
  thermal 3-D, bands oblique lattices, meow batched solves.
- 2026-10-30: second wave of five agents: optics GPU ray trace, drift-diffusion, thermal transients and kappa(T),
  find_k and the mixed-precision refinement of meow-gpu, each with its single review (one review found a
  blocker: the refinement could return the wrong mode of a near-degenerate pair; fixed and tested). New rule
  (user): every test and bench runs on the Ubuntu GPU computer, never on the Mac (`tools/gpu_run.sh`, a GPU lock).
- 2026-10-30: third wave of four agents: bend modes in meow-gpu, the whole optics pipeline on the GPU, the Gummel
  continuity solves of charge-gpu on the GPU, and `fdtd_gpu/` (an FDTD front end on fdtdx, the third FDTD route).
  The reviews found major findings, among them: a wrong H half-step shift in fdtd_gpu, NaN fields or the wrong
  mode for a 2-D in-plane (EVEN_Z) mode in fdtd_gpu (now an error), no convergence of the charge device at 5 V
  reverse bias and above (a pseudo-inverse on the coarsest level, now LU), warm starts across member kinds in the
  bend batch, no test that detects a wrong sign of dz in the optics image sum (test added). The optics agent
  found a regression of the meow refinement (SiN tip of the coupling example); the fix adds an admission rule and
  a fallback to the earlier polish (one review; it found that the first fallback could return a wrong 4th mode as
  converged; fixed and tested on the grid of the example).
- 2026-10-30: fourth wave: charge-gpu, the coupled Newton step on the GPU (297baf2) and a follow-up (f937f57);
  optics-gpu, all focus planes in one pass (0bbad9f; the review found a memory budget below the real use, fixed
  with a test); meow-gpu, the profile of the batched path (88d5158). The user decided: an MIT license, and the
  add-on route of item 2 (installed on the GPU computer). A review found that `meow_gpu/fde` follows the
  MIT-licensed mode-solver folder of tidy3d: the notices are now in `THIRD_PARTY_NOTICES.md` and the file headers.
- 2026-11-07: a usage limit stopped the work for some days; it continues with fdtd_gpu (subpixel averaging) and
  the add-on port.
- 2026-11-07: the GPU computer stopped with a kernel Oops in the amdgpu/TTM memory manager (see "In work"); it
  ran again from 18:00, shared with another session (4 CPU threads, about 8 GB of VRAM for this repository).
  Commits: fdtd-gpu subpixel averaging (46da80a) and periodic faces (eade281); charge-gpu device-error fallback
  (0bb5179); meow-gpu cold-start stop (18f2178; a review-follow-up test found wrong converged modes on the SiN tip
  with 1 BLAS thread); optics-gpu symmetric core and explicit source (e15c847; its review found the staircase
  error); the documents with the crash cause and the top-level example (d694a86); the add-on route runs on the
  AMD GPU (a6938b6). The reviews also found the cause of the crash and the asymmetric core.
