# fdtd-gpu

A small FDTD front end on [fdtdx](https://github.com/ymahlau/fdtdx) (MIT, JAX) in the script style of the MIT FDTD
package (Python interface): `Simulation(cell_size, resolution, geometry, sources, boundary_layers)`, geometry objects
with materials by index, Gaussian-pulse sources in frequency units, flux and mode monitors, `run(until=...)` or run
until the fields decay. Users of that package can run their usual device scripts on an AMD GPU (tested on a Radeon
RX 7900 XT, gfx1100, ROCm 10.1.0, JAX 0.11.1, fdtdx 0.6.3). No code of the MIT package is used; the interface is
written from its documentation.

```python
import fdtd_gpu as fg                                             # um; frequency in 1/um (c = 1); time in um/c

si, sio2 = fg.Medium(index=3.45), fg.Medium(index=1.45)
core = fg.Block(size=fg.Vector3(fg.inf, 0.4, fg.inf), material=si)
src = fg.EigenModeSource(fg.GaussianSource(1 / 1.55, fwidth=0.15), center=fg.Vector3(-2.5), size=fg.Vector3(0, 2),
                         eig_band=1, eig_parity=fg.ODD_Z)        # 2-D TM (Ez) mode, launched along +x
sim = fg.Simulation(cell_size=fg.Vector3(8, 4, 0), resolution=20, geometry=[core], sources=[src],
                    boundary_layers=[fg.PML(1.0)], default_material=sio2)
mon = sim.add_mode_monitor(1 / 1.55, 0.15, 5, fg.ModeRegion(center=fg.Vector3(2.5), size=fg.Vector3(0, 2)))
flux = sim.add_flux(1 / 1.55, 0.15, 5, fg.FluxRegion(center=fg.Vector3(2.5), size=fg.Vector3(0, 2)))
sim.run(until_after_sources=fg.stop_when_fields_decayed(20, fg.Ez, fg.Vector3(2.5), 1e-6))
alpha = sim.get_eigenmode_coefficients(mon, [1], eig_parity=fg.ODD_Z).alpha  # (bands, freqs, 2): forward, backward
P = sim.get_fluxes(flux)
```

## Supported subset

Supported:
- **Cell:** `cell_size` (a size of 0 makes the cell 2-D or 1-D), `resolution` (pixels per um), centered at the
  origin.
- **Materials:** `Medium(epsilon=...)` or `Medium(index=...)`, isotropic, with optional dispersion or loss in the
  names and units of the MIT package (see "Differences"): `E_susceptibilities=[LorentzianSusceptibility(frequency,
  gamma, sigma), DrudeSusceptibility(frequency, gamma, sigma), ...]` (then `epsilon` is eps_inf), or
  `D_conductivity=sigma_D` (eps(f) = eps_inf (1 + i sigma_D / (2 pi f))). `Medium.permittivity(f)` gives the
  complex eps(f) (the MIT package has `epsilon(f)`; here `epsilon` stays the float eps_inf).
- **Geometry:** `Block` (axis-aligned), `Cylinder` and `Prism` along x, y or z, `Sphere`. Later objects replace
  earlier ones. `eps_averaging=True` (the default, as in the MIT package): subpixel averaging of the permittivity at
  each Yee E component (see "Differences"; next to a dispersive or conductive material: the face rule "fill");
  `eps_averaging=False`: the staircase.
- **Boundaries:** `PML(thickness, direction, side, R_asymptotic)` inside the cell (`R_asymptotic`: see "Media in the
  PML"). Other faces are metal (PEC), or periodic with `k_point=Vector3()` (the objects repeat across periodic
  faces; a PML on both sides or none).
- **Sources:** `Source(GaussianSource or ContinuousSource, component, center, size, amplitude)`: a point, line or
  plane current. `EigenModeSource(src, center, size, eig_band, eig_parity, eig_kpoint)`.
- **Monitors:** `add_flux` (`FluxRegion` with weights; `get_flux_data`, `load_minus_flux_data`), `add_mode_monitor`
  with `get_eigenmode_coefficients(mon, bands, eig_parity)` (alpha and neff), `get_flux_freqs`, `get_dft_array`,
  `get_epsilon`. Flux and DFT-field planes work in any cell, also in a dispersive or conductive medium (check 14).
- **Near-to-far-field transform** (2-D and 3-D cells; see "Differences"): `add_near2far(fcen, df, nfreq,
  Near2FarRegion, ...)`, `get_farfield(n2f, x)` (a list of 6 nfreq complex numbers: Ex, Ey, Ez, Hx, Hy, Hz of the
  first frequency, then of the second, ...), `get_farfields(n2f, resolution, where=Volume(center, size))` (a dict of
  arrays), `n2f.flux(direction, where, resolution)`, `get_near2far_freqs`, `get_near2far_data`,
  `load_minus_near2far_data`; not in the MIT package: `get_farfields_at(n2f, points)` (a dict of (P, nfreq) arrays at
  a list of points).
- **Run:** `run(until=T)`, `run(until_after_sources=T)`, `stop_when_fields_decayed(dt, c, pt, decay_by)`,
  `Harminv(c, pt, fcen, df)` with `after_sources`.

Not supported (the option raises NotImplementedError, or the name is not in the package): `geometry_center`,
`symmetries`, non-uniform grids; anisotropy (`epsilon_diag`, `sigma_diag` with different values, `sigma_offdiag`),
gain (negative `sigma` or `D_conductivity`), `D_conductivity` together with `E_susceptibilities`, other
susceptibility types (noisy, gyrotropic), `mu`, `H_susceptibilities`, `B_conductivity`, nonlinearity; mode sources
and mode monitors whose planes cross a dispersive or conductive material (the mode solver of fdtdx sees eps_inf
only; near2far surfaces in such a medium are supported, check 15); rotated blocks (`e1`, `e2`,
`e3`), tilted axes, sloped prism sidewalls, `MaterialGrid`; Bloch phases (`k_point` other than 0); complex
amplitudes, `amp_func`, custom source times, `eig_match_freq=False`; field arrays (`get_array`), force, energy and
LDOS monitors; `add_near2far(..., nperiods)` (the sum over periodic copies), `output_farfields`, `save_near2far` and
`load_near2far` (files), near2far in 1-D cells; other step functions; a second `run` on the same `Simulation`.

`eig_parity` in 3-D: `ODD_Z` (TM-like: E mostly along z) and `EVEN_Z` (TE-like: E mostly in the plane normal to z)
select the polarization in the mode solver; a y parity is ignored. In 2-D only the mode with E along the trivial
axis is supported (Ez in an x-y cell: `NO_PARITY` or `ODD_Z`). An in-plane (Hz, `EVEN_Z`) 2-D mode raises
NotImplementedError in `EigenModeSource` and `get_eigenmode_coefficients`: on the three periodic cells of the trivial
axis the fdtdx mode solver gives NaN fields (source) or silently the Ez mode (monitor). 1-D cells have no mode
sources or mode monitors.

## Differences from the MIT package (read before you compare numbers)

- **Subpixel averaging** (`eps_averaging=True`, the default; `fdtd_gpu/_averaging.py`). Each E component of each
  cell gets its own permittivity: the average over the cube of side dx centered at the component. fdtdx keeps the
  Yee layout of Taflove; this package puts E_a at the cell center plus dx / 2 along a. The rule is the diagonal of
  the inverse tensor of the anisotropic average of Kottke, Farjadpour and Johnson (Phys. Rev. E 77, 036611, 2008):
  1 / eps_aa = n_a^2 <1/eps> + (1 - n_a^2) / <eps>, with n the interface normal. fdtdx gets three permittivity
  components per cell and keeps its elementwise update: the FDTD loop takes the same time (ring at res 80: 8.83 s
  averaged, 8.79 s staircase). Differences from the MIT package:
  - only the diagonal: the off-diagonal terms (the slanted part of the normal) are dropped. The rule is exact for
    an interface normal to an axis, and second order for E parallel to all interfaces (checks 3, 6, 7). For E in
    the plane of a curved interface (a 2-D disk or ring with Hz; in 3-D the sidewalls of rings and bends) the error
    stays first order, of the size of the staircase error, but the Q is no longer limited by the staircase (check
    8). The full tensor needs the 9-component (anisotropic) update of fdtdx (its documentation: about 3 times the
    cost per step; not measured here). An alternative is the diagonal of the forward tensor,
    eps_aa = n_a^2 / <1/eps> + (1 - n_a^2) <eps> (the smoothing of fdtdx itself; `fdtd_gpu._averaging.DIAGONAL =
    "forward"`, bench option `--forward-diagonal`). It changes only components with a slanted normal (check 8 and
    the corners of the 3-D strip). Rows: `benchmarks/2026-11-07/fdtd-gpu-forward-diagonal.jsonl` (`--case disk
    --resolution 40,60,80 --forward-diagonal`, rows 1 to 3; `--case waveguide --three-d --forward-diagonal`, row 4).
    Check 8, m = 8, 9, 10: -0.052, -0.112, -0.167 % (res 40), +0.022, -0.001, -0.023 % (res 60), +0.037, +0.025,
    +0.012 % (res 80); Q of m = 8 to 10: 7.8e4 to 6.7e5, 2.0e5 to 1.1e6, 2.8e5 to 1.5e6. The fit of check 8 gives
    G = 0.081 to 0.091 um^-1 (inverse diagonal: 0.124 to 0.138): also first order, with a smaller first-order
    term, but larger errors at res 40 for m = 9 and 10. 3-D strip (check 2, res 20): T_mode changes by at most
    1.5e-6, neff by +1.7e-4 to +4.0e-4. The default stays the diagonal of the inverse tensor, which follows Kottke
    et al.;
  - <eps> and <1/eps> come from fixed sample points (a rank-1 lattice of 1024, 1974 and 2042 points per voxel in
    1-D, 2-D and 3-D), not from an adaptive integration with a tolerance (`subpixel_tol` is not accepted). The
    fill fraction of a plane normal to an axis is exact to 1 / (2 n): 4.9e-4 in 1-D and 3-D, 5.07e-4 in 2-D;
  - the normal is the weighted first moment of eps on a regular grid (24 x 24 in 2-D, 12 x 12 x 12 in 3-D) over
    the circumscribed ball of the voxel, not the analytic normal of the object. For a slanted plane the averaged
    components are within 0.7 % (median 0.2 %) of the exact diagonal in 2-D (`test_slanted_plane_kottke_diagonal`,
    45 voxels) and within 2.4 % (median 0.27 %) in 3-D (`test_slanted_plane_3d`, 256 voxels);
  - only voxels next to a change of the staircase material are sampled: a feature thinner than one cell that
    holds no cell center is not seen (as by the staircase);
  - the JAX path (float32) can put a sample point within about 1e-7 um of a surface on the other side: one of
    the about 2000 points of a voxel, up to 1.3e-3 of eps in 1.3 % of the voxels of the test geometry
    (`test_numpy_and_jax_agree`);
  - fdtdx gets the components through the linear blend of one material over the background (`_voxels.py`): the
    float32 rounding of the blend is below 3e-5 of eps for objects below the background (the bound of
    `test_fdtdx_gets_the_components_and_get_epsilon`; measured 1.1e-5 for its geometry).
  `eps_averaging=False` gives the staircase (each cell takes the material of its center), which converges at
  about first order in the cell size. On a periodic axis (`k_point=Vector3()` and no PML on that axis) the objects
  repeat across the faces, for the staircase and for the average (`test_periodic_faces_repeat_objects`): each copy
  that touches the cell. Difference: the MIT package (its default `ensure_periodicity`) also repeats the objects
  along an axis with a PML when k_point is set; this package does not. `k_point` with a PML on one side of an axis
  only raises NotImplementedError (fdtdx would make the whole axis periodic). Setup time (`--case setup`: a 3-D
  strip and ring in an 8 x 4 x 4 um cell): 3.48 s at res 20 (1.0 M cells, 36,688 sampled voxels; with the start of
  JAX and the compile), then 2.92 s at res 50 (16 M cells, 247,880 voxels) in the same process. The 2-D checks take
  0.02 to 0.7 s. Small problems are sampled with NumPy, larger ones on the GPU. `get_epsilon` returns
  3 / (1/eps_xx + 1/eps_yy + 1/eps_zz) of each cell (each component at its own Yee point). `setup_seconds` (NaN
  before) and `n_averaged` (the number of sampled voxels) are set by `get_epsilon()` and `run()`.
- **Dispersion and conductivity** (`fdtd_gpu/_dispersion.py`; checks 10 to 12). fdtdx 0.6.3 has Lorentz and Drude
  poles (`fdtdx.dispersion`) with an auxiliary-differential-equation (ADE) update in its E update: per pole,
  P^(n+1) = c1 P^n + c2 P^(n-1) + c3 E^n (central differences at the step n), then eps_inf E + sum P changes by
  dt curl H. Mapping (time dependence exp(-i omega t), omega = 2 pi f, c = 1):
  - `LorentzianSusceptibility(frequency=f_n, gamma, sigma)`: chi = sigma f_n^2 / (f_n^2 - f^2 - i f gamma), fdtdx
    `LorentzPole(resonance_frequency=2 pi f_n, damping=2 pi gamma, delta_epsilon=sigma)` (rad/s);
    `DrudeSusceptibility`: chi = -sigma f_n^2 / (f^2 + i f gamma), fdtdx `DrudePole(plasma_frequency=2 pi f_n
    sqrt(sigma), damping=2 pi gamma)`. These are the forms of the MIT package (its documentation, "Materials").
  - `D_conductivity=sigma_D`: the electric conductivity sigma = eps_0 eps_inf sigma_D (S/m; in a face cell times the
    fill fraction with the face rule "fill", eps of the component with the other rules), so the fields in the
    medium decay as exp(-sigma_D t), as in the MIT package. fdtdx updates it semi-implicitly (factor (1 - b) /
    (1 + b) per step, b = sigma_D dt / 2). The MIT package multiplies the whole eps(f) by (1 + i sigma_D / omega);
    fdtdx adds i sigma / omega, which is the same only without susceptibilities, so the two together raise
    NotImplementedError.
  - The discrete scheme has the susceptibility chi_yee = K dt^2 / (w0^2 dt^2 - 4 sin^2(omega dt / 2) - i g dt
    sin(omega dt)) (K = sigma w_n^2, w0 = w_n for Lorentz and 0 for Drude, g = 2 pi gamma): second order in dt
    (`tests/references.py`, `chi_yee`; equal to the recurrence with the coefficients of fdtdx to 1e-9 in
    `test_ade_coefficients_give_the_discrete_chi`).
  - This package gives fdtdx only eps_inf (so fdtdx allocates no ADE arrays) and sets the ADE arrays and the
    conductivity itself after fdtdx placed the objects. With `eps_averaging=False` each cell gets the poles and the
    conductivity of the material of its center (a staircase). With `eps_averaging=True` the face rule
    `fdtd_gpu._dispersion.FACE_RULE` sets them for each E component (bench option `--face-rule`):
    - "fill" (the default): eps_inf is averaged as for any material, the pole strength K of the dispersive material
      is multiplied by its fill fraction in the voxel of the component (c3 times the fraction; c1 and c2 stay), and
      the conductivity by the fill fraction of each conductive material. The fill fractions come from the lattice
      points of the average (`_averaging.averaged_permittivity(track=...)`). For E parallel to a face this is the
      arithmetic mean of the whole eps(f), the exact average for a plane normal to an axis: on the exact Yee grid the
      error falls at second order (check 11, a face inside a cell; the FDTD follows it to res 200, above that the
      float32 ADE coefficients limit it). For E normal to a face the exact average is the harmonic mean of eps(f),
      which no pole of this form gives: there the rule is first order. In a voxel with two dispersive materials only
      the one with the larger fill fraction keeps its poles (scaled); the other one is dropped (first order there).
      The ADE arrays then have one set of c1, c2, c3 per component (3 times the memory of one set per cell).
    - "average": the rule of the MIT package (its documentation, "Subpixel Smoothing": only the non-dispersive part
      is averaged, so such interfaces converge at first order): eps_inf averaged, the poles and the conductivity of
      the cell center.
    - "staircase": the cells within one cell (the 3 x 3 x 3 neighborhood) of a dispersive or conductive cell take
      the staircase eps_inf too (`_dispersion.staircase_at_lossy`).
    With faces on cell faces the three rules give the same arrays. Check 11 compares them with a face inside a cell.
    A dispersive feature that holds no cell center is not sampled, so it is not seen. The ADE coefficients are float32
    (`_dispersion.float32_coefficients`): c2 is rounded so that c1 + c2 keeps its value to 6e-8, because 1 - c1 - c2
    = w0^2 dt^2 / D sets the resonance (a relative error of w0^2 of up to 6e-8 / (w0 dt)^2: 7.4e-4 for f_n = 1 / um
    at res 400), and a Drude pole keeps c1 + c2 = 1 exactly (a float32 sum above 1 makes a constant polarization grow
    by (c1 + c2 - 1) / (g dt) per step). The sources took 1 / eps_inf at that time, so a current source adds J to
    dD/dt = curl H - J, as in the MIT package (fdtdx alone scales a dipole in a dispersive cell by
    1 / Re eps(f_source)). In a conductive cell fdtdx adds the source current after the division by (1 + b)
    (`fdtdx.fdtd.update.update_E`), so its current alone is 1 + b times the current of the semi-implicit form
    E^(n+1) (1 + b) = E^n (1 - b) + (dt / eps) (curl H - J) (the loss term averaged over the two time levels, as in
    J. B. Schneider, Understanding the Finite-Difference Time-Domain Method, Sec. 3.12 "Lossy Material", the reference
    of the fdtdx code; J stands next to curl H in Ampere's law), with b = sigma_D dt / 2 (0.0071 at res 20 for
    sigma_D = 0.5). `_dispersion.correct_sources`
    divides the current of each electric point, line or plane source by 1 + b of its cells (b = 0 in a cell without
    conductivity: there the source does not change). Check 13 measures the error and the correction.
  - **Stability.** fdtdx runs at 0.99 of the 3-D Courant limit, and the ADE with E^n is explicit. For lossless
    poles the grid is stable only if sum_p v_p / (1 - u_p) < eps - s, with u_p = (w0_p dt / 2)^2,
    v_p = K_p dt^2 / 4, s = dims (c dt / dx)^2 = 0.327, 0.653, 0.980 in 1-D, 2-D, 3-D (`_dispersion.stability_margin`;
    derived from the dispersion relation of the grid for a uniform medium; damping is not in the bound). `run()`
    checks each E component of each cell with its own values: K times the weight of the face rule and the eps of the
    component (`_dispersion.check_stability`; a face cell has a smaller K, but it can also have a smaller averaged
    eps). For one Drude pole with the plasma frequency f_p = f_n sqrt(sigma) this is res > 1.80 f_p / sqrt(eps - s)
    (res in 1/um): with eps_inf = 1, res > 12.7 f_p in 3-D (76 / um for f_p = 6 / um), 3.05 f_p in 2-D, 2.19 f_p
    in 1-D. `run()` raises ValueError when a material breaks the bound (with an estimate of the factor of resolution
    that is needed). `test_stability_bound` brackets the bound to 5 % (1-D, lossless Drude: bounded at 0.8 and 0.95
    of the limit, growth without bound at 1.05 and 1.2).
    The MIT package uses Courant 0.5 by default, so it has more margin; `Courant` is still ignored here.
  - **Media in the PML.** A dispersive or conductive medium can run into the PML. The PML of fdtdx stretches the
    coordinate in the curl (a CPML); the ADE and the conductivity update stay the same inside it, so in theory it is
    matched to any medium that does not change along its normal. Its conductivity is designed for the round trip
    R = 1e-6 in vacuum (the default `sigma_end` of fdtdx); in a medium the stretch multiplies k = k0 n, so the round
    trip becomes R^Re(n), times the loss of the medium in the PML (`_dispersion.pml_round_trip`). Check 14 measures
    this: R of a Lorentz or a conductive half-space agrees with the Yee grid of an infinite half-space to the
    float32 level (1.6e-5, 5.2e-7), but in a Drude medium just above its plasma frequency (Re n = 0.16 at 1.0 / um)
    the PML reflects up to 1.4e-2, and R changes by up to 4.4e-4. Limits and remedies:
    - A thicker PML does not lower R^Re(n): fdtdx makes its conductivity proportional to 1 / thickness (read from
      its `objects/boundaries/perfectly_matched_layer.py`); only the loss of the medium in the layer grows.
    - An end of the medium before the PML is worse for a wave that propagates in it: the face of the Drude medium of
      check 14 reflects 0.53 of the power at 1.0 / um and 0.082 at 1.2 / um (its Fresnel reflectance, in the rows of
      check 14), against 1.6e-4 (power) from the PML at 1.0 / um. Only a medium that absorbs the wave before its end
      makes the end harmless, and then the PML reflection is small too.
    - `PML(thickness, R_asymptotic=R)` (the name of the MIT package, whose default is 1e-15; here the default is
      1e-6 of fdtdx) sets `sigma_end = -(m + 1) ln R / (2 eta0 d)` (m = 3, d the thickness) of fdtdx, so the round
      trip is R^Re(n). With R = 1e-15 the test of the Drude half-space (res 50) requires the PML reflection at
      1.0 / um below 1e-3, R against the Yee half-space below 5e-5 and the vacuum reflection below 1e-5. A much
      smaller R is expected to need a thicker layer: the discrete layer reflects more when its conductivity changes
      faster from cell to cell (not measured here beyond R = 1e-15).
    - `run()` warns when a dispersive or conductive medium fills cells of a PML and its round trip (above, with the
      R and the thickness of that PML) is above 1e-3 at a frequency where the spectrum of a source is at least 1e-2
      of its peak (`_dispersion.pml_warnings`). The Drude half-space of check 14 warns; the Lorentz and the
      conductive half-spaces, and the media of checks 13 and 15, do not (tests).
  - Not tested: Q of a dispersive resonator.
- **float32 fields**, and the time step of fdtdx: dt = 0.99 dx / (c sqrt 3) in all dimensions (`Courant` is
  ignored).
- **One-way mode sources.** `EigenModeSource` is the fdtdx TFSF mode source: the mode is solved at the center
  frequency only, and its absolute amplitude differs from the MIT package. Normalize by an input monitor or a
  reference run (as usual). A `Source` with a size emits in both directions, as in the MIT package.
- **Gaussian pulse.** Current exp(-(t - t0)^2 / (2 w^2)) cos(2 pi f t) with w = 1 / fwidth (the spectral width of
  the MIT package), but the peak is at t0 = 6 w (not 5 w) and the source ends at 12 w.
  `GaussianSource.fourier_transform(f)` gives its spectrum in the DFT convention of the monitors.
- **DFT and flux.** Phasors sum f(t) exp(i 2 pi f t) dt / sqrt(2 pi) (the convention of the MIT package). The fdtdx
  detectors co-locate all fields on the E_z Yee point and take H as the mean of H before and after the step, so E
  and H are at the same time (check 5: arg(H / E) of a plane wave below 2e-4 rad). The flux of the Yee grid is
  high by O((k dx)^2) (check 5). The cells of a monitor plane have the trapezoidal weight 1/2 at the plane edges,
  so that six planes close a box.
- **Mode coefficients.** alpha+- = (int E x Hm* +- int Em* x H) / (2 sqrt(int Re Em x Hm*)), so |alpha|^2 is a
  flux. On the Yee grid the co-located fields are averages of neighbor points (for a plane normal to x or y, the
  transverse H is an average of two points along the normal); this gives a small false backward coefficient
  (check 2). `EigCoeffs.neff` is the effective index from the mode solver at each frequency.
- **Near-to-far-field transform** (`fdtd_gpu/near2far.py`; check 9). Conventions:
  - Time dependence exp(-i omega t) (the DFT sums exp(+i omega t)), so an outgoing wave is exp(+i k r). Units: um,
    f in 1/um, c = 1, eps0 = mu0 = 1 (vacuum impedance 1); omega = 2 pi f, k = n omega. The far fields have the
    scale of the DFT fields, so sum Re(E x conj H) over a closed surface far away is the flux of `get_fluxes`.
  - Surface equivalence: the DFT fields on the planes give J = n x H and M = -n x E (n = the axis direction times
    the region weight: weight -1 on the low faces of a closed box, as for flux). The currents radiate with the exact
    Green's function of the homogeneous medium: in 3-D exp(i k R) / (4 pi R) with all near terms (the Hertzian
    dipole fields, Jackson, Classical Electrodynamics, 3rd ed., eq. 9.18), in 2-D (i/4) H0^(1)(k rho) (line currents
    per unit length; the Hankel form). So `get_farfield` is valid at each distance outside the surface (it can be
    outside the cell), not only in the far zone. The sum over the cells is a quadrature: keep the points at least 5
    cells away from the surface. At 1, 2, 5 and 10 cells outside the box of check 9 (res 20) the error of E is
    0.13, 8.4e-3, 4.6e-3 and 4.0e-3 (`--case distance`; at 1000 um 1.6e-3). Inside a closed surface the result is
    close to zero.
  - The surface cells are the cells of the planes, with the trapezoidal weight 1/2 at the plane edges (as for flux);
    the field point of each cell is the point where fdtdx co-locates the six components: the cell center, + dx / 2
    along z (in 3-D). A point source is at its own Yee point (`tests/cases.py`, `yee_point`).
  - The medium of the far fields is the staircase material of the surface cells. If the surface cells have more
    than one material, a warning, and the material of most cells is used. The surface must lie in one homogeneous
    medium (the transform assumes that medium everywhere outside it). In a dispersive or conductive medium the
    transform uses its complex eps(f) at each frequency (`Medium.permittivity`; `n2f.eps` is then an array of nfreq
    values), so k = omega sqrt(eps(f)) has Im k > 0 and the fields decay with the distance (check 15).
  - `get_farfields`: each axis of the cell with a size above 0 gets floor(size x resolution) points from the low end
    to the high end of the volume (both included; one point at the center if this is below 2); the arrays have the
    shape (the axes with more than one point, nfreq). `n2f.flux` sums Re(E x conj H) . e_direction over these
    points with the trapezoidal rule.
  - Phase: against the current spectrum of `GaussianSource.fourier_transform`, the far fields of a point source have
    the extra phase -omega dt / 2. fdtdx takes the current of the E update of step n at the time n dt of the new E;
    the leapfrog scheme puts it at the time of the curl of H, half a step earlier. So the effective current is
    J(t + dt / 2), and the same phase is in each DFT output of a point source (the DFT fields, the mode
    coefficients and the far fields), not only in the far fields. Magnetic point sources get the same half-step
    shift (fdtdx `update_H` takes the source at the time step + 1/2, where the curl of E is at the time step; not
    tested). The mode source (TFSF) is not checked. Measured (check 9, n = 1, res 20, both boxes, all frequencies):
    the far fields have the phase -omega dt / 2 within 2.14e-3 rad (3-D Ez), 2.26e-3 (3-D Ex), 2.16e-3 (2-D Ez) and
    2.41e-3 (2-D Ex); at res 40 within 5.3e-4 (3-D Ez, 2-D Ez) and 6.0e-4 (2-D Ex). The rest is the phase error of
    the grid dispersion. It grows with k dx, and in a medium also with the lower Courant number c dt / (n dx) (check
    9: at the same k, 1.08e-2 rad in n = 1.5 against 6.7e-3 in vacuum, 3-D): at n = 1.5 within 1.08e-2 (3-D) and
    1.46e-2 rad (2-D) at res 20, 2.65e-3 (3-D) and 3.60e-3 (2-D) at res 40. Ratios of two runs (normalization)
    cancel the phase.
  - The 3-D sum runs in float64 with JAX on the GPU (chunks of about 2 M point-cell pairs); the 2-D sum in float64
    with NumPy and the Hankel functions of SciPy on the CPU, with one line element per cell (E and H averaged over
    the TRIVIAL_CELLS cells of the axis of size 0). Time of the 3-D case of check 9 at res 20 (1,152 far and 231
    near points, 5 frequencies; 10,086 or 2,646 surface cells; `farfield_seconds`, the mean of the two boxes): 0.53 s
    per box in the rows of `--component Ex` and `--index 1.5` (one process each, with the compile), 0.29 s in the
    res-20 row of `--resolution 10,20,30,40` (the res-10 run of the same process compiled some of the shapes). The
    work is points x surface cells x frequencies; a chunk holds at least one point, so a surface of millions of
    cells makes many small chunks.
  - Only the 2-D cell in the x-y plane is checked (the code takes any axis of size 0).
  - Differences from the MIT package: no `nperiods`, no files (`output_farfields`, `save_near2far`,
    `load_near2far`), the extra `get_farfields_at`; the rule of the medium and the points of `get_farfields` above are
    those of this package. The interface follows the documentation of the MIT package; no code of it is used.
- **Harminv** is a matrix pencil fit of the signal after the sources (`fdtd_gpu.harminv`), not filter
  diagonalization. It uses only the band [fcen - df/2, fcen + df/2]. `freq`, `decay` and `Q` have the meaning of
  the MIT package; `amp` and `err` do not: `amp` is the complex amplitude of the frequency-shifted and low-pass
  filtered signal (about half of the real cosine amplitude, times the gain of the filter), and `err` is one relative
  residual of the whole fit (the same number for all modes).
- **Prism center.** As in the MIT package: with `center`, the centroid of all vertices (the mean of the vertices,
  moved by height / 2 along the axis) goes to `center`.
- **2-D and 1-D** cells have 3 periodic cells on each axis of size 0 (fdtdx needs one axis of one cell to find the
  normal of a plane). Fluxes in 2-D are per unit length, in 1-D per unit area, as in the MIT package.
- Runs use the first GPU (`jax.default_device`) inside `jax.enable_x64(False)`.

## Checks (each against an independent reference)

`tests/cases.py` has the cases and `tests/references.py` the references. `tests/test_validation.py` (checks 1 to 5)
and `tests/test_averaging.py` (checks 6 to 8, and the convergence of check 3) run them on small grids;
`bench/bench_fdtd.py` runs them on larger grids (one JSON line per case, with the resolution and the cells). The
numbers of checks 1 to 3 and 6 to 8, of the res-20 rows of checks 4 and 5, and the setup times are from the 39
rows of `benchmarks/2026-11-07/fdtd-gpu-avg.jsonl` (subpixel averaging on, unless the text says "staircase"; RX
7900 XT, 2026-11-07: `--case all`, the two 3-D waveguide commands and the six commands from `--case ring
--resolution 40,60,80 --both` to `--case setup` under "Tests and bench", in that order; other jobs can share the
computer, so times can change between runs). The numbers of check 9 are from the rows of
`benchmarks/2026-11-08/fdtd-gpu-near2far.jsonl` (the first seven check-9 commands under "Tests and bench") and of
`benchmarks/2026-11-08/fdtd-gpu-near2far-2.jsonl` (the next four) and of
`benchmarks/2026-11-08/fdtd-gpu-near2far-3.jsonl` (the last three). In the logged run, a
scipy RuntimeWarning on stderr cut the row of check 7 (staircase, res 160); line 31 of the file is the row of a
repeat of the same command on the same day (with a "note" key), which replaces it. (`references.disk_resonance`
now hides these warnings.) The other rows of checks 4 and 5 are from version 0.1 (the staircase only) in
`benchmarks/2026-10-30/fdtd-gpu.jsonl`: these checks have no geometry, so the averaging does not change them (the
res-20 rows of the two files are equal). Rounding: Q to two significant digits, orders to three decimals, errors to
the digits shown. "res" is the resolution in cells per um; times are the FDTD loops (with compile).

1. **1-D Bragg mirror** (quarter-wave stack at 1 um, n 2.0 / 1.25, 6 pairs, in air; 41 frequencies from 0.6 to
   1.4 / um). R from a flux plane with the incident fields subtracted (`load_minus_flux_data`), T from a flux plane
   behind the stack. Reference: exact transfer matrices (Born and Wolf).
   - res 200: max abs(R - R_exact) 0.0024, max abs(R + T - 1) 5.3e-5 (6.4 s for both runs). The error is the
     numerical dispersion of the Yee grid (the stop band moves to a lower frequency). The layer faces are on cell
     faces, so the averaged permittivity is the staircase (check 7 puts them between grid lines). The test (res 80)
     requires max abs(R - R_exact) < 0.02 and abs(R + T - 1) < 1e-3.
2. **Straight Si strip** (n 3.45 in 1.45, width 0.4 um; 2-D Ez, and 3-D with a 0.22 um core). Mode source, mode
   monitors and flux planes 1 um after the source and 0.5 um before the output PML; 5 frequencies from 0.570 to
   0.720 / um. Reference: a lossless straight guide transmits 1.
   - 2-D, res 20: T_mode 0.99991 to 1.000000, T_flux 0.99916 to 1.000000. False backward coefficient
     abs(alpha-)^2 / abs(alpha+)^2: 4.4e-4 to 1.3e-3 (the co-located fields of the Yee grid). abs(alpha+)^2 / input
     flux: 1.0001 to 1.0009. neff from the mode solver: 3.117 to 3.215, against 3.108 to 3.207 for the exact slab
     (`references.slab_te_neff`; +0.008 to +0.009 from the discretization of the mode solver at 0.05 um). The faces
     are on grid lines, so these numbers are those of the staircase.
   - 3-D, res 20 (1.0 M cells, 2.8 s; the average 0.69 s): T_mode 0.9979 to 1.00003. T_flux 0.99999 at the center
     frequency, 0.9855 and 0.990 at the band edges. At the input, abs(alpha+)^2 is 0.954 to 0.974 of the flux (4.4
     cells across the core). neff from the mode solver 1.948 to 2.412 (staircase 1.900 to 2.373: the staircase core
     has 4 cells, 0.20 um, along z).
   - Cause of the low T_flux at the band edges: the mode source is solved at its center frequency only, so at
     other frequencies it also launches radiation that crosses the input plane and leaves before the output plane.
     Run with the source centered (and its mode solved) at the edge frequency 0.570 / um (`--src-freq 0.5701613`):
     T_flux at 0.570 / um is 1.000001 (0.9855 before), and it falls at the other frequencies (0.9945 at 0.608 / um,
     0.978 at 0.645 / um). Outside the band of the source the transmission is noise: there the source spectrum is
     1.5e-5 (0.683 / um) and 2.7e-9 (0.720 / um) of its peak, both fluxes are float32 noise, and their ratio has no
     meaning (row 7 of the 0.1 rows gave T_mode 32.8 and T_flux 37.8 at 0.720 / um). The case gives `src_rel` (the
     amplitude spectrum of the source over its peak) and reports T_mode and T_flux as null where src_rel < 1e-3
     (`cases.SRC_REL_MIN`).
3. **2-D Si ring** (n 3.4 in air, inner radius 1.5 um, width 0.22 um; Ez), a short dipole in the ring, `Harminv` on
   Ez after the sources. Reference: the exact resonances of the dielectric ring (Bessel and Hankel functions,
   `references.ring_resonance`), azimuthal orders m = 15 to 21. E is parallel to the interfaces, so the averaged
   rule is the arithmetic mean.
   - Frequency error of the strongest mode of each m, averaged: -0.17 to -0.34 % at res 40 (2.6 s), -0.08 to
     -0.15 % at res 60 (4.1 s), -0.04 to -0.09 % at res 80 (8.8 s): second order (res 40 to 80: 1.957 to 2.015 for
     each m). What remains is mostly the dispersion of the Yee grid (the frequencies are low). Staircase: -0.43 to
     -0.69 %, -0.02 to -0.20 %, -0.09 to -0.14 % (no clear order: the staircase error and the dispersion change
     sign against each other).
   - Free spectral range of neighbor orders, averaged: 0.0275 to 0.0280 (res 40), 0.0277 to 0.0281 (res 60),
     0.0277 to 0.0282 (res 80); staircase 0.0266 to 0.0285, 0.0274 to 0.0287, 0.0275 to 0.0283; exact 0.0278 to
     0.0282.
   - Q (exact: above 1e9): averaged 1.2e5 to 1.4e7 (res 40), 1.1e7 to 8.6e7 (res 60), 1.4e7 to 1.2e8 (res 80);
     staircase 5.6e3 to 1.5e4, 5.0e4 to 6.0e5, 4.6e4 to 4.2e5 (the rough staircase circle scatters, and a mode can
     split into a pair; the check uses the strongest mode of each m). At a Q above about 1e6 the fit can give a
     small growth (Q < 0); the cases use abs(Q).
   - The test (res 60) requires errors below 0.5 % for the three strongest modes and the free spectral range within
     2 %; `test_averaging.py` requires the order above 1.7 (res 40 to 80) for m = 17 to 20, errors below 0.12 % at
     res 80 and Q above 5e4 at res 40.
4. **3-D point dipole** (Ez) in a homogeneous medium, flux through a closed box of half width 1 um, 5 frequencies
   from 0.4 to 0.6 / um. Reference: the analytic radiated power n omega^2 abs(I)^2 / (12 pi) of a dipole current I
   (Jackson), with the source spectrum of `GaussianSource.fourier_transform` (Purcell factor 1).
   - n = 1: 1.0022 to 1.0060 at res 10, 1.0006 to 1.0015 at res 20 (1.0 M cells, 1.7 s), 1.0002 to 1.0007 at
     res 30: second order in the cell size. n = 1.5, res 20: 1.0029 to 1.0066.
   - A box of half width 0.5 um, res 10: 0.99996 to 1.0036.
   - The test requires the error below 3e-3 at res 20 and an observed order above 1.7 (res 10 to 20) in vacuum,
     and below 1 % for n = 1.5 at res 20.
5. **1-D current sheet** (Ex) in a homogeneous medium, flux plane 3 um away, 7 frequencies from 0.4 to 1.6 / um.
   Reference: the exact flux abs(I)^2 / (4 n) per side, and E and H in phase.
   - Vacuum: flux / exact 1.0014 to 1.0220 at res 20, 1.0006 to 1.0045 at res 40 (second order at the high
     frequencies: the Yee grid makes abs(E / H) low by up to 2.2 % at res 20, 1.6 / um). n = 1.5, res 20: 1.0037 to
     1.071. arg(Hy / Ex) below 1.8e-4 rad in all rows.

6. **Subpixel averaging: periodic slab** (n 3.45 in 1.45, width 0.37 um, so the faces are between grid lines;
   periodic along x with the period 0.5 um, so the guided mode has beta = 2 pi / 0.5 um; a dipole at the center,
   `Harminv` after the sources). Reference: `references.slab_bloch_frequency`, the exact slab mode with the exact Yee
   dispersion along x and in time, so the remaining error is the discretization across the slab (the geometry).
   Relative frequency error, averaged / staircase:

   | res | Ez (E parallel to the faces) | Hz (E normal to the faces, `--e-normal`) |
   |---|---|---|
   | 20 | -2.2e-4 / +4.3e-3 | +3.7e-3 / +1.24e-2 |
   | 40 | -2.3e-4 / -2.4e-3 | -2.0e-4 / -4.0e-3 |
   | 80 | +1.7e-5 / -1.9e-3 | -8.0e-5 / -3.5e-3 |

   As an effective index (neff = beta / (2 pi f)) at res 40: averaged +7.2e-4 (Ez) and +5.9e-4 (Hz), staircase
   +7.6e-3 and +1.17e-2. The staircase rounds the width to whole cells and stops converging. The ratio of the
   errors, staircase / averaged, is 20, 10 and 111 (Ez) and 3.4, 20 and 44 (Hz) at res 20, 40 and 80. The averaged
   error is second order in its envelope only: its coefficient changes with the position of the face in the cell
   (20 to 40 for Ez: no gain; 40 to 80: 14 times). The test (res 40) requires the averaged error below 1e-3 and the
   staircase error at least 4 times larger, for both polarizations.
7. **Subpixel averaging: Bragg mirror with faces between grid lines** (check 1 with lambda0 = 1.03 um, `--lam0
   1.03`: layers 0.12875 and 0.206 um). Max abs(R - R_exact) at res 40 / 80 / 160: averaged 0.114 / 0.0265 /
   0.0062 (order 2.1 and 2.1; what remains is the dispersion of the Yee grid, as in check 1), staircase 0.257 /
   0.049 / 0.047 (no gain from 80 to 160). abs(R + T - 1) below 3.4e-5 in all rows. The test requires the order
   above 1.8 (res 80 to 160), the averaged error below 1e-2 at res 160 and the staircase error 4 times larger.
8. **Subpixel averaging: 2-D disk with in-plane E** (n 3.4 in air, radius 1.0 um, Hz; an Ex dipole 0.1 um inside
   the rim; `--case disk`). Reference: the exact resonances of the dielectric disk (`references.disk_resonance`,
   Hz and (1 / eps) dHz/dr continuous). Here the normal turns through all angles, and the diagonal rule drops the
   off-diagonal terms of the tensor of Kottke et al. Frequency error of m = 8, 9, 10:

   | res | averaged | staircase |
   |---|---|---|
   | 40 | +0.061, 0.000, -0.054 % | +0.088, -0.014, +0.001 % |
   | 60 | +0.096, +0.074, +0.053 % | +0.119, +0.069, +0.069 % |
   | 80 | +0.092, +0.081, +0.067 % | +0.115, +0.055, +0.035 % |

   Neither converges at second order. The averaged errors fit e = G / res - D / res^2 (a first-order geometry error
   and the second-order dispersion of the grid; G and D from res 40 and 60 predict res 80 to 1e-4) with G = 0.12 to
   0.14 um^-1: about +0.16 % at res 80 before the dispersion, of the same size as the staircase error. This is the
   known limit of the diagonal rule (see "Differences"). The averaged circle has no staircase roughness: Q of
   m = 8 to 10 is 7.5e4 to 2.0e5 (res 40), 1.4e5 to 9.0e5 (res 60), 2.7e5 to 1.1e6 (res 80); staircase 4.8e3 to
   4.2e4, 9.3e3 to 1.6e4, 1.2e4 to 4.1e4. The test (res 60) requires errors below 0.2 % and the averaged Q of
   m = 8, 9 at least 10 times the staircase Q.
9. **Near-to-far-field transform** (`tests/test_near2far.py`; the cases `transform_exact`, `near2far_dipole` and
   `near2far_line`). References: the fields of a point dipole in spherical components (Balanis, Antenna Theory, 3rd
   ed., Sec. 4.2; `references.dipole_fields_3d`) and of a line current in the Hankel form
   (`references.line_source_fields_2d`), written independently of the Cartesian form of the transform; the analytic
   power in the DFT convention of the monitors (twice the time average): n omega^2 abs(I)^2 / (6 pi) in 3-D
   (`references.dipole_flux_3d`; check 4 gives the time average n omega^2 abs(I)^2 / (12 pi)), and per unit length
   omega abs(I)^2 / 4 (2-D, current along z) or omega abs(I)^2 / 8 (2-D, current in the plane).
   - Transform of exact fields (no FDTD, `--case transform`): the exact fields of a source at (0.1, -0.05, 0.07) um
     (z = 0 in 2-D) in n = 1.5 on a closed box or square of half width 1 um (trapezoidal rule, spacing h), f = 0.4
     and 0.6 / um, against the exact fields at 1000 um (max abs(error) / max abs(exact) of E):

     | h (um) | 3-D, Ez dipole | 3-D, Ex dipole | 2-D, Ez line | 2-D, Ex line |
     |---|---|---|---|---|
     | 0.1 | 9.5e-3 | 8.6e-3 | 7.5e-3 | 4.7e-3 |
     | 0.05 | 2.4e-3 | 2.2e-3 | 1.9e-3 | 1.2e-3 |
     | 0.025 | 5.9e-4 | 5.4e-4 | 4.7e-4 | 2.9e-4 |

     Second order: each halving of h divides the error by 4.00 to 4.04. At 3 um the errors are 1.040 to 1.154 times
     these values. Inside the surface (0.4 um from the source) max abs(E) / min abs(E_exact) is below 9.6e-4,
     2.4e-4 and 5.9e-5 (extinction theorem). The JAX sum (GPU, float64) and the NumPy sum differ by at most 8.3e-15.
     The test (h = 0.1 and 0.05) requires the errors below 3e-3 (1000 um) and 4e-3 (3 um), the order above 1.9, and
     the inside ratio below 2e-3.
   - FDTD, 3-D point dipole (Ez, n = 1, 5 frequencies from 0.4 to 0.6 / um, `--case near2far`): two closed boxes of
     half width 1.0 and 0.5 um in one run (PML 1 um, 0.5 um from the larger box). Far fields at 1000 um on a sphere
     (Gauss-Legendre, 24 x 48 points) and at 3 um (231 points), against the reference at the Yee point of the
     source. P_far is the power of the far fields on the sphere; "flux" is `get_fluxes` of the same box. Field
     error: max abs(E - E_ref) / max abs(E_ref) after one global phase per frequency; pattern error: max
     abs(abs(E)^2 - abs(E_ref)^2) / max abs(E_ref)^2; "boxes": max abs(E_1.0 - E_0.5) / max abs(E_1.0). Worst values
     of both boxes and all frequencies:

     | res | P_far / flux | P_far / exact | field error | pattern error | at 3 um | boxes | FDTD loop |
     |---|---|---|---|---|---|---|---|
     | 10 | 0.9879 to 1.0057 | 0.9938 to 1.0094 | 6.7e-3 | 1.2e-2 | 7.7e-3 | 1.3e-2 | 1.1 s |
     | 20 | 0.9970 to 1.0014 | 0.9985 to 1.0023 | 1.7e-3 | 2.8e-3 | 1.9e-3 | 3.1e-3 | 1.5 s |
     | 30 | 0.9987 to 1.0006 | 0.9993 to 1.0010 | 7.6e-4 | 1.3e-3 | 8.5e-4 | 1.4e-3 | 5.1 s |
     | 40 | 0.9992 to 1.0004 | 0.9996 to 1.0006 | 4.3e-4 | 7.2e-4 | 4.9e-4 | 7.7e-4 | 16.1 s |

     All fall at second order: what remains is the discretization of the FDTD near fields (the dispersion of the Yee
     grid; the flux of check 4 has the same order), not the transform. At res 20 the global phase is -omega dt / 2
     within 2.14e-3 rad (n = 1; see "Differences"). Ex dipole (res 20, `--component Ex`): P_far / flux 0.9972 to 1.0000,
     P_far / exact 0.9943 to 0.9993, field error 3.0e-3, pattern error 5.9e-3, at 3 um 4.0e-3, boxes 5.3e-3. n = 1.5
     (res 20, `--index 1.5`): P_far / flux 0.99738 to 1.00345, P_far / exact 1.0002 to 1.0089 (flux / exact 1.0023
     to 1.0066, as check 4), field error 1.5e-2, pattern error 2.1e-2, at 3 um 1.2e-2, boxes 1.6e-2. The field error
     is 9.0 times that at n = 1 and the same f, but it depends mostly on k = n omega (at the same dx and box), not on
     the medium: a vacuum run at the same k (n = 1, f = 0.6 to 0.9 / um, `--fcen 0.75 --df 0.3`) gives the field errors
     1.8e-3, 4.5e-3, 1.0e-2, 1.5e-2 and 1.6e-2 (box 1.0 um), against 1.5e-3, 4.7e-3, 9.9e-3, 1.4e-2 and 1.5e-2 in
     n = 1.5 at f = 0.4 to 0.6 / um (2-D: 1.1e-3, 4.9e-3, 1.0e-2, 1.3e-2, 1.2e-2 against 9.3e-4, 5.0e-3, 9.8e-3,
     1.2e-2, 1.1e-2). In this band the error grows steeply with k dx: 8.5 to 13 times from n f = 0.6 to 0.825 / um
     (the cube of the ratio of k is 2.6). The Courant number S = c dt / (n dx), lower in the medium, changes mainly
     the global phase rest, which the field error removes: 1.08e-2 rad in n = 1.5, 6.7e-3 in vacuum at the same k
     (2-D: 1.46e-2 and 1.10e-2). It also changes the other errors a little: at the same k the field errors of the
     1.0 um box above differ by up to 15 % (2-D: 18 %), and the box differences by up to 9 % (3-D: 3.3e-3 to 1.63e-2
     in n = 1.5 against 3.1e-3 to 1.50e-2 in vacuum; 2-D: up to 40 %). It is the FDTD discretization, not the
     transform: at res 40 (`--resolution 40 --index
     1.5`) the field error is 3.8e-3, the boxes 4.1e-3, the phase rest 2.65e-3 rad, P_far / flux 0.99935 to 1.00086
     (second order: the ratios 4.0, 4.0 and 4.1). `n2f.flux` through the six faces of a cube of half width 100 um
     (40 x 40 points per face) / P_far: 0.99968 at all resolutions (the trapezoidal rule of the flux points).
   - FDTD, 2-D line source (n = 1, `--case near2far2d`): squares of half width 1.0 and 0.5 um, a circle of 72 points
     at 1000 um and at 3 um, the same metrics:

     | res | source | P_far / flux | P_far / exact | field error | pattern error | at 3 um | boxes |
     |---|---|---|---|---|---|---|---|
     | 20 | Ez | 0.9964 to 1.0042 | 0.9986 to 1.0067 | 4.5e-3 | 9.0e-3 | 4.0e-3 | 4.9e-3 |
     | 40 | Ez | 0.9991 to 1.0010 | 0.9996 to 1.0017 | 1.1e-3 | 2.3e-3 | 1.0e-3 | 1.2e-3 |
     | 80 | Ez | 0.9998 to 1.0003 | 0.9999 to 1.0004 | 2.9e-4 | 5.7e-4 | 2.6e-4 | 3.2e-4 |
     | 20 | Ex | 0.9967 to 1.0029 | 0.9968 to 1.0014 | 2.0e-3 | 4.1e-3 | 2.3e-3 | 2.5e-3 |
     | 40 | Ex | 0.9992 to 1.0007 | 0.9992 to 1.0004 | 5.1e-4 | 1.0e-3 | 5.7e-4 | 6.5e-4 |
     | 80 | Ex | 0.9998 to 1.0002 | 0.9998 to 1.0001 | 1.3e-4 | 2.5e-4 | 1.4e-4 | 1.7e-4 |

     Second order. At res 20 (n = 1) the global phase is -omega dt / 2 within 2.16e-3 (Ez) and 2.41e-3 rad (Ex). Ez
     in n = 1.5 (`--index 1.5`): P_far / flux 0.99707 to 1.00455 (res 20), 0.99926 to 1.00112 (res 40), 0.99981 to
     1.00028 (res 80); field error 1.2e-2, 3.0e-3, 7.4e-4; boxes 1.2e-2, 3.0e-3, 7.8e-4; pattern error 1.8e-2 at res
     20: second order, as in 3-D. `n2f.flux` through a square of half width 100 um (200 points per side) / P_far:
     0.9999946.
   - Distance from the surface (`--case distance`; the 3-D Ez dipole, n = 1, res 20, one box of half width 1 um):
     max abs(error) / max abs(exact) of E (and of H) on cube shells m cells outside the box of the DFT points (9 x 9
     points per face), after the global phase of the far fields, the worst frequency. m = 1: 0.13 (0.096); 2: 8.4e-3
     (9.0e-3); 3: 5.1e-3 (3.4e-3); 5: 4.6e-3 (2.5e-3); 10: 4.0e-3 (2.4e-3); at 1000 um 1.6e-3. The transform of the
     exact fields on the same points (the quadrature alone): 0.13 (0.096), 8.0e-3 (8.9e-3), 3.3e-3 (3.4e-3),
     1.8e-3 (2.1e-3), 1.3e-3 (1.1e-3). So the quadrature limits the points to at least 5 cells from the surface.
   - Subtraction of incident fields (`--case subtract`; 2-D, Ez, n = 1, the cell interior +-2 um, a square of half
     width 1 um): run A has a line source S2 at (0, -1.6) um, outside the square; run B has S2 and a line source S1
     at the origin, inside it. Res 20 / 40, the worst frequency: the far fields of B with the DFT fields of A
     subtracted (`load_minus_near2far_data`) against the field of S1: 1.3e-3 / 3.0e-4; of B without the subtraction:
     6.5e-3 / 1.6e-3 (the transform of the outer source is not exactly zero outside the square: 6.8e-3 / 1.6e-3 of
     its far field); the transform of A at 0.5 um from the center, inside the square, against minus the field of S2:
     3.3e-3 / 8.2e-4. At these inner points (res 20) max abs(transform) / max abs(field of S2) is at most 3.8e-3
     for B - A (only S1 is left, and an inner source gives zero inside) and 0.995 to 0.999 for B (the -S2 part).
   - The tests (3-D at res 10 and 20; 2-D, Ez and Ex, at res 20 and 40) require at res 20: abs(P_far / flux - 1)
     below 6e-3, abs(P_far / exact - 1) below 1e-2, the pattern error below 1.5e-2, the field error, the error at
     3 um and the box difference below 7e-3, the phase -omega dt / 2 within 4e-3 rad, abs(n2f.flux / P_far - 1)
     below 1e-3; an order above 1.8 for the worst P_far / flux, field error and box difference; and in 3-D that
     get_farfield equals get_farfields_at and the shape of get_farfields. More tests: load_minus_near2far_data with
     the data of a second, equal run gives zero far fields (below 1e-6 of the far fields), and a surface that
     crosses a second material gives the warning; the distance test requires the errors at 5 and 10 cells below
     1e-2 (FDTD) and 4e-3 (exact fields), and above 5e-2 at 1 cell; the subtraction test (res 20) requires the error
     of B - A below 3e-3, of B and of the outer source outside below 1.5e-2, and inside below 7e-3, and at the inner
     points the transform of B - A below 1e-2 and that of B above 0.9 of max abs(field of S2). A Near2FarRegion
     whose normal is an axis of size 0 (for example direction=Z in an x-y cell) raises ValueError (test).

Checks 10 to 15 (`tests/test_dispersion.py`; the cases `film`, `uniform`, `dipole` with `sigma_d`, `halfspace` and
`near2far_medium`) use synthetic media with round numbers in the textbook forms of the Lorentz and Drude models
(Jackson, Classical Electrodynamics, 3rd ed., Sec. 7.5), written as (kind, f_n, gamma, sigma) in the units of the
MIT package. Two references: the exact eps(f) (the transfer matrix, the roots of eps = 0, sigma_D, the Fresnel
reflectance, the exact dipole and line-source fields with a complex index) and the exact discrete scheme
(`references.chi_yee`, the exact 1-D Yee grid `references.yee_rt` and `references.yee_halfspace`, the root of
eps_inf + chi_yee = 0); the second leaves only the float32 arithmetic and the end of the DFT. The numbers of checks
10 to 12 are from the rows of `benchmarks/2026-11-08/fdtd-gpu-dispersion.jsonl` (the commands from `--case film
--kind drude` to `--case uniform`, under "Tests and bench"), except the conductive rows of check 12, which are from
`benchmarks/2026-11-08/fdtd-gpu-dispersion-2.jsonl` (with the source correction of check 13); the numbers of checks
13 to 15 are from that second file (the commands from `--case dipole --sigma-d 0.5` on).

10. **Dispersive film: Drude** (eps_inf 1, f_n 1 / um, gamma 0.05 / um, sigma 1: the plasma edge at 1 / um; 0.3 um
    thick, faces on cell faces, in vacuum, normal incidence; 41 frequencies from 0.6 to 1.4 / um; `--case film
    --kind drude`). R and T as in check 1. The source has fwidth 1.6 / um, so its amplitude at the band edges is
    0.29 of its peak. R falls from 0.738 at 0.6 / um to 0.102; the absorption is at most 0.113. Max abs error of R
    and T over the band, against: the transfer matrix (FDTD); the exact Yee grid with float64 and with float32 ADE
    coefficients (FDTD); and, without FDTD, the exact Yee grid against the transfer matrix ("grid only"):

    | res | transfer matrix | Yee grid, float64 | Yee grid, float32 | grid only | FDTD loops |
    |---|---|---|---|---|---|
    | 100 | 4.25e-4 | 6.3e-6 | 4.7e-6 | 4.25e-4 | 2.1 s |
    | 200 | 1.06e-4 | 1.4e-5 | 1.3e-5 | 1.06e-4 | 3.0 s |
    | 400 | 3.28e-5 | 1.06e-5 | 1.19e-5 | 2.65e-5 | 5.1 s |

    The error against the transfer matrix falls at second order (orders 2.00 and 1.70; the grid alone 2.00 and
    2.00), and the FDTD agrees with the exact Yee grid to 1.4e-5. With a narrower source (fwidth 0.8 / um: the
    amplitude at the band edges is 7e-3 of the peak) earlier runs, not in this file, had larger errors at the band
    edges; the wider source removes them. The mechanism of that error was not identified.
11. **Dispersive film: Lorentz slab** (eps_inf 2.25, f_n 1 / um, gamma 0.05 / um, sigma 0.5; `--case film --kind
    lorentz`; the same band and source): absorption up to 0.813 (at 1.12 / um), T down to 1.2e-6, R up to 0.461.
    Thickness 0.5 um (faces on cell faces), the same columns as check 10:

    | res | transfer matrix | Yee grid, float64 | Yee grid, float32 | grid only | FDTD loops |
    |---|---|---|---|---|---|
    | 100 | 1.71e-3 | 3.7e-5 | 1.9e-5 | 1.68e-3 | 2.3 s |
    | 200 | 5.25e-4 | 1.68e-4 | 2.6e-5 | 4.19e-4 | 3.5 s |
    | 400 | 3.61e-4 | 3.12e-4 | 3.4e-5 | 1.05e-4 | 6.0 s |

    The grid alone converges at second order (2.00 and 2.00). The FDTD error against the transfer matrix falls at
    the order 1.71 from res 100 to 200 and only by 1.46 times from 200 to 400: there the float32 ADE coefficients
    limit it. They move the resonance (see "Differences"): the FDTD stays within 3.4e-5 of the Yee grid with the
    float32 coefficients, but moves to 3.1e-4 from the grid with float64 coefficients at res 400. The staircase run
    (`--no-avg`, res 100) gives 1.71e-3, 3.5e-5 and 1.6e-5: with the faces on cell faces the arrays are the same.

    Thickness 0.504 um (`--thickness 0.504 --face-rule fill,average,staircase`): the first face is on a cell face,
    the second face fills 0.400, 0.800 and 0.600 of its cell at res 100, 200 and 400. Each entry: the FDTD against
    the transfer matrix / the grid only (the exact Yee grid of the rule against the transfer matrix, no FDTD); in
    every row the FDTD agrees with the Yee grid of its rule (float32 coefficients) to better than 4.5e-5:

    | res | "fill" (default) | "average" (MIT package) | "staircase" |
    |---|---|---|---|
    | 100 | 1.71e-3 / 1.68e-3 | 7.13e-3 / 7.14e-3 | 1.16e-2 / 1.16e-2 |
    | 200 | 5.27e-4 / 4.19e-4 | 1.99e-3 / 1.94e-3 | 3.07e-3 / 3.02e-3 |
    | 400 | 3.43e-4 / 1.05e-4 | 1.97e-3 / 1.89e-3 | 3.07e-3 / 2.99e-3 |

    With the rule "fill" the grid-only error is that of the 0.5 um slab on the grid at each resolution (second order,
    2.00 and 2.00): the face inside the cell adds no error at these digits. The FDTD error falls at the order 1.70
    from res 100 to 200; from 200 to 400 the float32 coefficients limit it, as for the slab on the grid. The rules
    "average" and "staircase" are worse at each resolution (4.2 and 6.8 times at res 100, 3.8 and 5.8 at res 200,
    5.7 and 8.9 at res 400). Their errors do not fall from res 200 to 400 (1.99e-3 to 1.97e-3, 3.07e-3 to 3.07e-3),
    because the fill fraction of the face cell changes with the resolution (0.8, then 0.6). So these rows give no
    order for them; a series with a constant fill fraction would be needed (not logged).
12. **Uniform field (k = 0)** in a periodic 1-D cell (`--case uniform`): a current over the whole cell, so
    curl H = 0. Drude (eps_inf 1, f_n 0.5, gamma 0.02, sigma 1) and Lorentz (eps_inf 2.25, f_n 0.5, gamma 0.02,
    sigma 0.5): the free oscillation (Harminv) against the root of eps(f) = 0 (exact: f = sqrt(f_n^2 (sigma / eps_inf
    + 1_Lorentz) - gamma^2 / 4) - i gamma / 2). Conductive (eps_inf 2.25, sigma_D 0.2): the decay rate of Ex (a fit
    of log abs(Ex) over 25 um/c) against sigma_D, and against the semi-implicit update ((2 / dt) atanh(sigma_D dt /
    2)).

    | medium | res | f (Harminv) | error against the exact root | against the discrete root | decay: exact, discrete |
    |---|---|---|---|---|---|
    | Drude | 20 | 0.500068 | +3.37e-4 | +1.2e-7 | -5.8e-6, -6.9e-6 |
    | Drude | 40 | 0.499942 | +8.33e-5 | -8.2e-7 | +3.7e-5, +3.6e-5 |
    | Lorentz | 20 | 0.552908 | +4.12e-4 | +6.5e-7 | +1.1e-5, +9.9e-6 |
    | Lorentz | 40 | 0.552738 | +1.03e-4 | +5.1e-7 | +1.7e-5, +1.6e-5 |

    The exact roots are 0.4999 - 0.01i and 0.552680 - 0.01i (/ um). The frequency error against the exact root falls
    at second order (orders 2.02 and 2.00); against the root of the discrete chi it is below 1e-6. "decay" is
    the relative error of Im f; at res 40 the float32 c2, which carries the damping, limits it (6e-8 / (g dt) =
    3.3e-5). Conductive medium, decay rate of Ex / sigma_D - 1 (rows of the second file: the source over the whole
    cell is now divided by 1 + b, check 13; the first file had +6.1e-6, +1.5e-5, +3.4e-6, +1.4e-5 and 1.9e-6):
    +6.0e-6 (res 20) and +1.4e-5 (res 40); against the semi-implicit update +3.3e-6 and +1.4e-5 (the float32 factors
    of the update); the fit residual of log abs(Ex) is below 1.5e-6, and Ex keeps its sign.
13. **Point dipole in a conductive medium** (`--case dipole --sigma-d 0.5`: check 4 with eps_inf 1 and sigma_D 0.5 /
    um, also in the PML). Reference: the flux of the exact dipole fields with the complex index n(f) = sqrt(1 + i
    sigma_D / omega) through the box of the monitor planes (the co-located field points, the source at its Yee point;
    `references.dipole_flux_surface`, Gauss-Legendre with 40 x 40 points per face). In a lossy medium this flux
    depends on the surface: here it is 0.564 to 0.591 of the power of the same dipole in vacuum. Flux / reference:
    1.001034 to 1.002586 (res 10), 1.000260 to 1.000675 (res 20), 1.000118 to 1.000298 (res 30): second order (orders
    1.94 to 2.02 from res 10 to 20, 1.84 to 2.02 from res 20 to 30). Without the correction (`--no-source-fix`, the
    current of fdtdx alone): 1.0299 to 1.0314 (res 10) and 1.0146 to 1.0150 (res 20), first order (1.03 to 1.07),
    and the ratio of the two runs is (1 + b)^2 within 4.4e-6 (b = 0.0143 at res 10, 0.0071 at res 20). The test
    (res 10 and 20) requires the error below 1.5e-3 at res 20, the order above 1.8, the ratio (1 + b)^2 within 1e-5,
    and the error without the correction above 1.2e-2.
14. **Half-space through the PML** (`--case halfspace`; 1-D, normal incidence, Ex): the face of the half-space is on
    a cell face 1 um after the source, the medium fills 2 um before the PML of 1 um and runs through the PML to the
    end of the cell. The pulse of check 10 comes from two current sheets of opposite sign 0.1 um apart (no
    zero-frequency part). Media: the Lorentz medium of check 11, the conductive medium of check 12 and the Drude
    medium of check 10 (R from 0.0022 to 0.463, 0.040 to 0.041 and 0.031 to 0.883 over 0.6 to 1.4 / um). References:
    the Fresnel reflectance of the face, and the exact reflectance of the 1-D Yee grid at the face of a half-space
    (`references.yee_halfspace`, with the discrete eps of the run and the float32 ADE coefficients): a wave that the
    PML sends back changes R from it. Flux planes 0.5 and 1.5 um deep give the flux in the medium (against
    (1 - R) exp(-4 pi f Im(n) d)), the ratio of their DFT fields (against the forward wave of the grid; where the
    second plane has at least 1e-2 of the field of the first) and r_pml, the backward wave over the forward wave at
    the start of the PML (where it is above the float32 noise and the 2 x 2 fit is not near singular). Max abs
    errors over the band:

    | medium | res | R: Fresnel | R: Yee half-space | grid only | flux in the medium | field ratio | r_pml |
    |---|---|---|---|---|---|---|---|
    | Lorentz | 50 | 6.74e-3 | 4.7e-6 | 6.74e-3 | 2.68e-3 | 2.1e-4 | 7.4e-3 |
    | Lorentz | 100 | 1.69e-3 | 1.6e-5 | 1.68e-3 | 6.47e-4 | 5.0e-4 | 1.8e-2 |
    | conductive | 50 | 9.46e-4 | 4.0e-7 | 9.46e-4 | 2.68e-3 | 5.6e-6 | 8.4e-6 |
    | conductive | 100 | 2.34e-4 | 5.2e-7 | 2.33e-4 | 6.64e-4 | 2.1e-6 | 3.2e-6 |
    | Drude | 50 | 2.20e-3 | 3.8e-4 | 2.22e-3 | 1.96e-3 | 6.1e-3 | 2.2e-2 |
    | Drude | 100 | 6.47e-4 | 4.4e-4 | 5.56e-4 | 9.06e-4 | 6.7e-3 | 2.7e-2 |

    In the vacuum run of the same cell r_pml is below 2.1e-6 (2.5e-6 with one sheet). Lorentz: R agrees with the Yee
    half-space to the float32 level of check 11 (below 1 / um: 2.1e-6 and 4.6e-6; the largest values are near the
    resonance, where the wave dies before the PML). The field ratio and r_pml are not resolved near the resonance (the
    largest r_pml is at 1.1 / um, where the strong absorption between the planes leaves mostly float32 noise); where the
    second plane has at least 0.1 of the field of the first, r_pml is below 5.6e-4 (res 50) and 2.3e-3 (res 100), not
    analyzed further. Conductive: the PML sends back at most 8.4e-6 of the wave, and R agrees with the Yee grid to
    5.2e-7. The flux in the medium falls at second order for both (orders 2.05 and 2.01). Drude: above its plasma
    edge (1.0 to 1.4 / um, Re n = 0.16 to 0.70) the PML sends back up to 1.25e-2 (res 50) and 1.38e-2 (res 100) at
    1.0 / um, falling to 2.4e-5 and 4.8e-5 at 1.38 / um; this is 0.42 to 0.81 (res 50) and 0.78 to 0.90 (res 100)
    of the round trip 1e-6^Re(n) exp(-4 pi f Im(n) 1 um) of the stretched coordinate (see "Media in the PML"), so
    the reflection does not fall with the resolution. It changes R by up to 4.4e-4 (at 1.04 to 1.16 / um), and the
    flux in the medium differs from the continuum by 1.96e-3 and 9.1e-4 (order 1.1, not 2). Below the plasma edge the
    wave is evanescent (the largest r_pml of the table, at 0.68 / um, comes from that weak wave), and R agrees with
    the Yee grid to 2.5e-5. With one source sheet (`--pair 0`, Lorentz) R changes by at most 7.3e-6 (res 50) and
    1.27e-5 (res 100).
    The test (res 50) requires R against the Yee half-space below 2e-5 (Lorentz), 5e-6 (conductive) and 1e-3
    (Drude), the flux in the medium below 5e-3, for the conductive medium the field ratio below 2e-5 and r_pml below
    5e-5, and for the Drude medium r_pml between 0.3 and 1.5 times the round trip above (1.0 to 1.4 / um) and the
    PML warning of run() (only for the Drude medium). `test_correct_sources_per_cell` checks the correction of
    check 13 in a face cell (b of the component), with eps_inf 2.25 (b = sigma_D dt / 2 in the bulk) and for a
    source in vacuum (the same to the last bit), from the build of the run.
15. **Near-to-far-field transform in a dispersive or conductive medium** (`--case near2far-medium`; 2-D, an Ez line
    source in a uniform medium that fills the cell and the PML, a square of half width 1 um, 36 points on each of
    the circles of radius 2 and 4 um about the source, 5 frequencies from 0.4 to 0.6 / um). Media: the Lorentz
    medium of check 11 (in this band n = 1.687 to 1.741, Im n = 4.2e-3 to 1.05e-2) and the conductive medium of
    check 12. Reference: the exact line-source fields with the complex index (`references.line_source_fields_2d`);
    one global phase per frequency for all points, so a wrong k gives a phase error between the two circles. Field
    error (as in check 9), Lorentz: 1.5e-2, 3.7e-3 and 9.1e-4 at res 20, 40 and 80 (orders 1.95 to 2.09), pattern
    error 2.8e-2, 6.9e-3, 1.7e-3; the global phase is -omega dt / 2 within 2.0e-2, 5.0e-3 and 1.3e-3 rad.
    Conductive (Re n = 1.50, Im n = 0.040 to 0.060): 1.3e-2 and 3.2e-3 at res 20 and 40 (orders 2.01 to 2.12), pattern
    error 2.2e-2, 5.5e-3, phase within 1.4e-2 and 3.5e-3 rad. The same transform with eps_inf (the medium of a
    lossless, non-dispersive run) has the field error 0.55 to 0.95 (Lorentz) and 0.28 to 0.31 (conductive).
    Transform of exact fields in a lossy medium (no FDTD, `--case transform
    --index-imag 0.05`: n = 1.5 + 0.05i, the surfaces of the transform of exact fields in check 9): the error of E at
    3 um is 9.0e-3, 2.2e-3 and 5.6e-4 for h = 0.1, 0.05 and 0.025 (3-D, Ez dipole; 2.9e-4 to 4.5e-4 at h = 0.025 for
    the 3-D Ex dipole and the 2-D line sources), and each halving of h divides it by 4.00 to 4.03; inside the
    surface below 6.3e-4; the JAX and NumPy sums agree to 1e-13. The test
    (res 20, and res 40 for the Lorentz medium; the exact fields at h = 0.1 and 0.05) requires the field error below
    3e-2, the pattern error below 5e-2, the phase within 4e-2 rad, the error with eps_inf above 0.2, the order above
    1.8, `n2f.eps` equal to eps(f), and for the exact fields the error at 3 um below 3e-3 (h = 0.05), the order above
    1.9 and inside below 2e-3.

Tests: 83 (46 without FDTD runs: geometry, Prism center, 2-D parities, Harminv, the averaged permittivity against
exact fill fractions and the formula of Kottke et al. (2-D and 3-D, both diagonals), periodic faces, NumPy against
JAX, the transfer to fdtdx, the near-to-far-field transform of exact fields (also in a lossy medium, and a negative
eps with a signed zero), the points of get_farfields; the units of the susceptibilities against the fdtdx poles,
the ADE coefficients of fdtdx against chi_yee, the exact Yee reference against the transfer matrix, the Yee
half-space reference, the dipole flux through a sphere and a box, the PML round trip and the rule of its warning,
the roots of eps = 0, the float32 coefficients, the three face rules in a face cell, the stability check with the
values of each cell, the errors of Medium; 37 GPU checks, some with more than one run, one with the build only).

## Tests and bench (on the GPU computer)

```bash
bash tools/gpu_run.sh fdtd_gpu env PYTHONPATH=src python -m pytest -q
B="env PYTHONPATH=src python bench/bench_fdtd.py"
bash tools/gpu_run.sh fdtd_gpu $B --case all          # bragg 200, waveguide 20, ring 40, dipole 20, sheet 20
bash tools/gpu_run.sh fdtd_gpu $B --case waveguide --three-d
bash tools/gpu_run.sh fdtd_gpu $B --case waveguide --three-d --src-freq 0.5701613
bash tools/gpu_run.sh fdtd_gpu $B --case ring --resolution 60
for a in "--resolution 10" "--resolution 30" "--resolution 20 --index 1.5" "--resolution 10 --half 0.5"; do
  bash tools/gpu_run.sh fdtd_gpu $B --case dipole $a; done
for a in "--resolution 40" "--resolution 20 --index 1.5"; do bash tools/gpu_run.sh fdtd_gpu $B --case sheet $a; done
bash tools/gpu_run.sh fdtd_gpu $B --case ring --resolution 40,60,80 --both          # check 3, averaged and staircase
bash tools/gpu_run.sh fdtd_gpu $B --case slab --resolution 20,40,80 --both          # check 6 (Ez)
bash tools/gpu_run.sh fdtd_gpu $B --case slab --resolution 20,40,80 --both --e-normal   # check 6 (Hz)
bash tools/gpu_run.sh fdtd_gpu $B --case bragg --resolution 40,80,160 --both --lam0 1.03   # check 7
bash tools/gpu_run.sh fdtd_gpu $B --case disk --resolution 40,60,80 --both          # check 8
bash tools/gpu_run.sh fdtd_gpu $B --case setup --resolution 20,50                   # time of the average
bash tools/gpu_run.sh fdtd_gpu $B --case transform                                  # check 9, exact fields
bash tools/gpu_run.sh fdtd_gpu $B --case near2far --resolution 10,20,30,40           # check 9, 3-D dipole
bash tools/gpu_run.sh fdtd_gpu $B --case near2far --resolution 20 --component Ex
bash tools/gpu_run.sh fdtd_gpu $B --case near2far --resolution 20 --index 1.5
bash tools/gpu_run.sh fdtd_gpu $B --case near2far2d --resolution 20,40,80            # check 9, 2-D line source
bash tools/gpu_run.sh fdtd_gpu $B --case near2far2d --resolution 20,40,80 --component Ex
bash tools/gpu_run.sh fdtd_gpu $B --case near2far2d --resolution 20 --index 1.5
bash tools/gpu_run.sh fdtd_gpu $B --case near2far --resolution 40 --index 1.5        # check 9, second file
bash tools/gpu_run.sh fdtd_gpu $B --case near2far2d --resolution 40,80 --index 1.5
bash tools/gpu_run.sh fdtd_gpu $B --case distance --resolution 20
bash tools/gpu_run.sh fdtd_gpu $B --case subtract --resolution 20,40
bash tools/gpu_run.sh fdtd_gpu $B --case near2far --resolution 20 --fcen 0.75 --df 0.3   # check 9, third file
bash tools/gpu_run.sh fdtd_gpu $B --case near2far2d --resolution 20 --fcen 0.75 --df 0.3
bash tools/gpu_run.sh fdtd_gpu $B --case subtract --resolution 20                    # with the inner points
bash tools/gpu_run.sh fdtd_gpu $B --case film --kind drude --resolution 100,200,400   # check 10 (dispersion file)
bash tools/gpu_run.sh fdtd_gpu $B --case film --kind lorentz --resolution 100,200,400  # check 11
bash tools/gpu_run.sh fdtd_gpu $B --case film --kind lorentz --resolution 100 --no-avg
bash tools/gpu_run.sh fdtd_gpu $B --case film --kind lorentz --thickness 0.504 --resolution 100,200,400 \
  --face-rule fill,average,staircase                                                 # check 11, a face in a cell
for k in drude lorentz conductive; do
  bash tools/gpu_run.sh fdtd_gpu $B --case uniform --kind $k --resolution 20,40; done
bash tools/gpu_run.sh fdtd_gpu $B --case dipole --sigma-d 0.5 --resolution 10,20,30   # check 13 (second file)
bash tools/gpu_run.sh fdtd_gpu $B --case dipole --sigma-d 0.5 --resolution 10,20 --no-source-fix
for k in lorentz conductive drude; do
  bash tools/gpu_run.sh fdtd_gpu $B --case halfspace --kind $k --resolution 50,100; done   # check 14
bash tools/gpu_run.sh fdtd_gpu $B --case halfspace --kind lorentz --resolution 50,100 --pair 0   # one source sheet
bash tools/gpu_run.sh fdtd_gpu $B --case halfspace --kind drude --resolution 50,100 --r-asymptotic 1e-15   # not logged
bash tools/gpu_run.sh fdtd_gpu $B --case near2far-medium --kind lorentz --resolution 20,40,80   # check 15
bash tools/gpu_run.sh fdtd_gpu $B --case near2far-medium --kind conductive --resolution 20,40
bash tools/gpu_run.sh fdtd_gpu $B --case transform --index-imag 0.05
bash tools/gpu_run.sh fdtd_gpu $B --case uniform --kind conductive --resolution 20,40   # check 12 with check 13
```

If the package is not installed in `.venv-rocm`, use `PYTHONPATH=src`. It needs fdtdx 0.6.3 (with tidy3d for its mode
solver), JAX, NumPy and SciPy.

## License

The code of this package is MIT-licensed (the `LICENSE` file at the root of the repository). fdtdx is MIT-licensed;
its mode solver uses tidy3d (LGPL-2.1 or later) as a library. The interface follows the published documentation of
the MIT FDTD package (GPL); no code of that package is used.
