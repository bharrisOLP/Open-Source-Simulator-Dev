# photonsuite

One device description for the GPU solvers of this repository (the idea of a Tidy3D-like suite, for
cross-sections): a `CrossSection` of rectangles with materials that carry the properties of every physics, and
solvers that sample it on their own windows and grids.

```python
from photonsuite import CrossSection, Material, Rect            # the import enables float64 in JAX
from photonsuite.physics import Window, mode_index, phase_shifter_ppi, modulator_vpil

si = Material("Si", n=3.476, dn_dT=1.86e-4, kappa=148.0, eps_r=11.7, semiconductor=True)
sio2 = Material("SiO2", n=1.444, dn_dT=0.95e-5, kappa=1.4, eps_r=3.9)
metal = Material("heater", n=1.444, kappa=100.0)
xs = CrossSection(sio2, (Rect(-5, 5, -5, -2, si), Rect(-0.25, 0.25, 0, 0.22, si), Rect(-1, 1, 2.2, 2.3, metal, tag="heater")))

neff = mode_index(xs, Window(-1.5, 1.5, -0.9, 1.1, 0.02), wavelength=1.55, num_modes=2)     # meow-gpu
p_pi, dneff, th = phase_shifter_ppi(xs, Window(-1.5, 1.5, -0.9, 1.1, 0.02), Window(-5, 5, -5, 2.3, 0.05),
                                    "heater", 1.55)                                          # thermal-gpu + meow-gpu
```

| Function | Solvers | Output |
|---|---|---|
| `mode_index` | meow-gpu (FDE) | complex mode indices, with an optional index change dn(x, y) |
| `temperature` | thermal-gpu | temperature rise (interpolator), heater power per length |
| `phase_shifter_ppi` | thermal-gpu + meow-gpu | P_pi = lambda / (2 dneff/dP') |
| `carriers` | charge-gpu | electrons and holes at a reverse bias (contacts: every semiconductor cell on the left and right window sides; each contact takes v_n or v_p from the sign of its mean doping) |
| `modulator_vpil` | charge-gpu + meow-gpu | V_pi L and the loss change per bias (negative: fewer carriers) |

Units: um for geometry, cm^-3 for doping, W/m for heater power per length.

- Every solve is checked: a mode, thermal or charge solve that does not converge raises
  `photonsuite.physics.ConvergenceError`. The thermal check uses the true residual |b - A T| / |b|.
- The solvers need float64. `import photonsuite` enables `jax_enable_x64`; each function raises RuntimeError if it
  is off.
- A `Rect` holds [x0, x1) x [y0, y1) (with a 1e-9 um tolerance): a grid sample on a left or bottom edge is inside, on
  a right or top edge outside, independent of the rounding of the grid. There is no sub-pixel averaging yet.

## Checks (`tests/test_workflows.py`)

The suite reproduces the standalone examples on the same grids, within 1e-3 relative: P_pi 18.997 mW
(`examples/thermo_optic_phase_shifter.py`, which prints 19.0) and V_pi L 1.386, 1.600, 1.955 V cm at 1, 2 and 4 V
(`examples/pn_modulator.py`). Also: the edge rule of `Rect`, the float64 check, and a thermal solve that does not
converge.

## Next

- Polygons and layer stacks (from a GDS cross-section); FDTD (fdtdx, MIT, JAX on ROCm) for 3-D devices; EME
  (meow + meow-gpu) for propagation along z; band structures (bands-gpu) for periodic waveguides.
