# Comparison with related packages

Several packages provide spherical-harmonic transforms or atmospheric wind
diagnostics. spharmgrid overlaps most directly with pyspharm, windspharm, and
NCL's spherical-harmonic routines. PySHTOOLS has a broader coefficient-level
and geophysical scope.

## Atmospheric workflows

[pyspharm](https://github.com/jswhit/pyspharm) exposes SPHEREPACK through
Python. [windspharm](https://ajdawson.github.io/windspharm/) builds a
higher-level wind-analysis API on top of pyspharm. NCL also provides a large
set of SPHEREPACK-based spherical-harmonic procedures for global fixed and
Gaussian grids. spharmgrid operates directly on xarray objects and currently
uses DUCC0 for the numerical transforms.

The table maps each current spharmgrid scientific operation to the closest
counterpart. These are functional comparisons, not drop-in replacements:
grid definitions, normalization, truncation, metadata handling, Earth radius,
and vector conventions can differ.

For NCL, `F` denotes its fixed-grid routines and `G` the corresponding Gaussian
grid routines. Lower-case procedure forms and `_Wrap` metadata-preserving
variants also exist for many functions.

| Task | spharmgrid | windspharm | pyspharm | NCL |
| --- | --- | --- | --- | --- |
| Spectral filtering / truncation | `field.sg.filter("T42")` | `w.truncate(field, truncation=42)` | transform, truncate coefficients, synthesize; `specsmooth()` for spectral smoothing | spherical-harmonic analysis/synthesis plus truncation; `exp_tapersh()` for tapering |
| Scalar spectral regridding | `field.sg.regrid(target_grid)` | — | `regrid()` | `g2gsh*`, `g2fsh*`, `f2gsh*`, `f2fsh*` as appropriate |
| Vector spectral regridding | `ds.sg.regrid_vector(target_grid)` | — | compose vector analysis/synthesis | vector harmonic analysis/synthesis (`vha*`, `vhs*`) |
| Scalar gradient | `field.sg.gradient()` | `w.gradient(field)` | lower-level spectral operations | `gradsF`, `gradsG` |
| Inverse scalar gradient | `sg.inverse_gradient(gx, gy)` | — | lower-level spectral operations | `igradsF`, `igradsG` |
| Scalar Laplacian | `field.sg.laplacian()` | — | coefficient-space operation | `lapsF`, `lapsG` |
| Inverse scalar Laplacian | `field.sg.inverse_laplacian()` | — | coefficient-space operation | `ilapsF`, `ilapsG` |
| Relative vorticity | `ds.sg.vorticity()` | `w.vorticity()` | `getvrtdivspec()` plus synthesis | `uv2vrF`, `uv2vrG` |
| Divergence | `ds.sg.divergence()` | `w.divergence()` | `getvrtdivspec()` plus synthesis | `uv2dvF`, `uv2dvG` |
| Vorticity and divergence together | `ds.sg.kinematics()` | `w.vrtdiv()` | `getvrtdivspec()` | `uv2vrdvF`, `uv2vrdvG` |
| Streamfunction | `sg.streamfunction(u, v)` | `w.streamfunction()` | `getpsichi()` | `uv2sfvpF`, `uv2sfvpG` (returns both potentials) |
| Velocity potential | `sg.velocity_potential(u, v)` | `w.velocitypotential()` | `getpsichi()` | `uv2sfvpF`, `uv2sfvpG` (returns both potentials) |
| Streamfunction and velocity potential | `ds.sg.potentials()` | `w.sfvp()` | `getpsichi()` | `uv2sfvpF`, `uv2sfvpG` |
| Helmholtz decomposition | `ds.sg.helmholtz()` | `w.helmholtz()` | compose spectral transforms | `uv2sfvp*` followed by `vr2uv*` / `dv2uv*`, or equivalent vector transforms |
| Rotational wind from vorticity | `sg.rotational_wind(vo)` | `w.nondivergentcomponent()` computes it from the original wind | inverse vector transform | `vr2uvF`, `vr2uvG` |
| Divergent wind from divergence | `sg.divergent_wind(d)` | `w.irrotationalcomponent()` computes it from the original wind | inverse vector transform | `dv2uvF`, `dv2uvG` |
| Wind from vorticity and divergence | `sg.wind(vo, d)` | — | `getuv()` from spectral vorticity/divergence coefficients | `vrdv2uvF`, `vrdv2uvG` |
| Wind from streamfunction and velocity potential | compose spharmgrid wind transforms | — | compose spectral transforms | `sfvp2uvf`, `sfvp2uvg` |
| Vector Laplacian | `sg.vector_laplacian(u, v)` | — | coefficient-space composition | `lapvf`, `lapvg` |
| Inverse vector Laplacian | `sg.inverse_vector_laplacian(u, v)` | — | coefficient-space composition | `ilapvf`, `ilapvg` |

spharmgrid also accepts direct function calls for accessor operations; see the
{doc}`api` reference.

### NCL naming

NCL often exposes the same SPHEREPACK operation in several forms. For example,
`uv2vrf` is a procedure for a fixed grid, `uv2vrF` is the corresponding
function form, and `uv2vrF_Wrap` retains metadata. Gaussian-grid routines use
`g` or `G` instead. The table uses the function-style `F`/`G` names where they
make the mapping easiest to read.

NCL's "fixed grid" is its equally spaced global latitude--longitude grid. It
should not be assumed to be numerically identical to spharmgrid's
pole-including Clenshaw--Curtis grid merely because both include regularly
structured latitudes. Gaussian-grid comparisons likewise require matching the
actual nodes and transform conventions.

## PySHTOOLS

[PySHTOOLS](https://shtools.github.io/SHTOOLS/) is broader than the atmospheric
workflow above. It provides spherical-harmonic coefficient and grid classes,
transforms, spectral analysis, localization and Slepian methods, and extensive
gravity and magnetic-field functionality. It is better viewed as a general
spherical-harmonic and geophysical toolkit than as a windspharm-style
atmospheric API.

| Area | PySHTOOLS | spharmgrid |
| --- | --- | --- |
| Primary Python objects | spherical-harmonic coefficients and grids | xarray `DataArray` / `Dataset` |
| General coefficient manipulation | extensive | internal implementation detail |
| Atmospheric `u`, `v` diagnostics | possible from spherical-harmonic operations, but not the main high-level interface | first-class API |
| CF coordinate discovery and metadata preservation | not the primary abstraction | first-class API |
| Atmospheric truncation notation such as `T42` and `T6-42` | degree / bandwidth controls | first-class API |
| Spectral scalar and vector regridding | available through transform/grid operations | first-class API |
| Gravity and magnetic-field analysis | extensive | out of scope |
| Slepian / localization analysis | extensive | out of scope |

Use PySHTOOLS when coefficient-level spherical-harmonic analysis, gravity,
magnetics, localization, or its broader geophysical toolset is the goal. Use
spharmgrid when the input is an atmospheric xarray field and the desired
operation is filtering, regridding, differentiation, or wind decomposition
while retaining coordinates and metadata.

## Grid and convention differences

Function names alone are not enough to establish numerical equivalence. When
reproducing results from another package, compare the grid definition,
normalization, truncation, Earth radius, vector sign conventions, latitude
ordering, and treatment of the degree-zero mode.

spharmgrid currently supports full rectangular Gauss--Legendre (GL) and
pole-including Clenshaw--Curtis (CC) grids and validates those geometries before
transforming them. See {doc}`grids` for the definitions and {doc}`filtering`
for spectral-range conventions.
