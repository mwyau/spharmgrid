# Comparison with related tools

spharmgrid overlaps most directly with [windspharm](https://ajdawson.github.io/windspharm/) and NCL's SPHEREPACK-based spherical harmonic routines. spharmgrid uses xarray objects and DUCC0 spherical harmonic transforms.

The table maps current spharmgrid operations to the closest windspharm and NCL counterparts. The numerical conventions are not identical: grids, normalization, truncation, Earth radius, vector conventions, and metadata handling can differ.

For NCL, `F` denotes fixed-grid routines and `G` denotes Gaussian-grid routines. Lower-case procedure forms and `_Wrap` variants also exist for many functions.

| Task | spharmgrid | NCL | windspharm |
| --- | --- | --- | --- |
| Spectral filtering / truncation | `field.sg.filter("T42")` | spherical harmonic analysis/synthesis with truncation; `exp_tapersh()` for tapering | `w.truncate(field, truncation=42)` |
| Scalar spectral regridding | `field.sg.regrid(target_grid)` | `g2gsh*`, `g2fsh*`, `f2gsh*`, `f2fsh*` | — |
| Vector spectral regridding | `ds.sg.regrid_vector(target_grid)` | `g2gshv*`, `g2fshv*`, `f2gshv*`, `f2fshv*` | — |
| Scalar gradient | `field.sg.gradient()` | `gradsF`, `gradsG` | `w.gradient(field)` |
| Inverse scalar gradient | `sg.inverse_gradient(gx, gy)` | `igradsF`, `igradsG` | — |
| Scalar Laplacian | `field.sg.laplacian()` | `lapsF`, `lapsG` | — |
| Inverse scalar Laplacian | `field.sg.inverse_laplacian()` | `ilapsF`, `ilapsG` | — |
| Relative vorticity | `ds.sg.vorticity()` | `uv2vrF`, `uv2vrG` | `w.vorticity()` |
| Divergence | `ds.sg.divergence()` | `uv2dvF`, `uv2dvG` | `w.divergence()` |
| Vorticity and divergence | `ds.sg.kinematics()` | `uv2vrdvF`, `uv2vrdvG` | `w.vrtdiv()` |
| Streamfunction | `sg.streamfunction(u, v)` | `uv2sfvpF`, `uv2sfvpG` returns both potentials | `w.streamfunction()` |
| Velocity potential | `sg.velocity_potential(u, v)` | `uv2sfvpF`, `uv2sfvpG` returns both potentials | `w.velocitypotential()` |
| Streamfunction and velocity potential | `ds.sg.potentials()` | `uv2sfvpF`, `uv2sfvpG` | `w.sfvp()` |
| Helmholtz decomposition | `ds.sg.helmholtz()` | `uv2vrdv*` followed by rotational/divergent synthesis | `w.helmholtz()` |
| Rotational wind | `sg.rotational_wind(...)` | `vr2uvF`, `vr2uvG`, or streamfunction synthesis | `w.nondivergentcomponent()` from the input wind |
| Divergent wind | `sg.divergent_wind(...)` | `dv2uvF`, `dv2uvG`, or velocity-potential synthesis | `w.irrotationalcomponent()` from the input wind |
| Wind from vorticity and divergence | `sg.wind(vo, d)` | `vrdv2uvF`, `vrdv2uvG` | — |
| Wind from streamfunction and velocity potential | `sg.wind(strf, vp)` | `sfvp2uvf`, `sfvp2uvg` | — |
| Vector Laplacian | `sg.vector_laplacian(u, v)` | `lapvf`, `lapvg` | — |
| Inverse vector Laplacian | `sg.inverse_vector_laplacian(u, v)` | `ilapvf`, `ilapvg` | — |

See the {doc}`api` reference for direct functions and xarray accessors.

## NCL grid names

NCL's fixed grid is an equally spaced global latitude–longitude grid. spharmgrid's pole-including Clenshaw–Curtis grid also has equally spaced latitudes, but the transform conventions should be compared explicitly when reproducing NCL results. Gaussian-grid comparisons likewise require matching latitude nodes and spectral conventions.

## Numerical conventions

Reproducing results across packages requires matching the grid definition, normalization, truncation, Earth radius, vector sign conventions, latitude ordering, and treatment of the degree-zero mode.

spharmgrid supports full rectangular Gauss–Legendre (GL) and pole-including Clenshaw–Curtis (CC) grids. See {doc}`grids` for their definitions and {doc}`filtering` for spectral-range conventions.
