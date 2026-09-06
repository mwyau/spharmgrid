# Comparison with related packages

Several Python packages provide spherical-harmonic transforms or atmospheric
wind diagnostics. spharmgrid overlaps with parts of these packages, but has a
different interface and scope: xarray-first operations on global atmospheric
fields, with grid detection, spectral filtering and regridding, differential
operators, and wind diagnostics in one API.

## Atmospheric workflows

The closest comparisons are
[pyspharm](https://github.com/jswhit/pyspharm) and
[windspharm](https://ajdawson.github.io/windspharm/).
pyspharm exposes a Python interface to SPHEREPACK. windspharm builds a
higher-level wind-analysis API on top of pyspharm. spharmgrid instead operates
directly on xarray objects and currently uses DUCC0 for the numerical
transforms.

The table below gives approximate equivalents for common atmospheric tasks.
The calls are not intended to be drop-in replacements; defaults, accepted
grids, coefficient conventions, metadata handling, and truncation semantics
can differ between packages.

| Task | pyspharm | windspharm | spharmgrid |
| --- | --- | --- | --- |
| Spectral filtering / truncation | transform coefficients and truncate; `specsmooth()` for spectral smoothing | `truncate()` | `field.sg.filter("T42")` |
| Scalar spectral regridding | `regrid()` | not a primary API | `field.sg.regrid(target_grid)` |
| Vector spectral regridding | compose vector transforms | not a primary API | `ds.sg.regrid_vector(target_grid)` |
| Relative vorticity | `getvrtdivspec()` plus synthesis | `vorticity()` | `ds.sg.vorticity()` |
| Divergence | `getvrtdivspec()` plus synthesis | `divergence()` | `ds.sg.divergence()` |
| Vorticity and divergence together | `getvrtdivspec()` | separate diagnostics | `ds.sg.kinematics()` |
| Streamfunction and velocity potential | `getpsichi()` | `sfvp()` | `ds.sg.potentials()` |
| Helmholtz decomposition | compose spectral transforms | `helmholtz()` | `ds.sg.helmholtz()` |
| Wind from vorticity and divergence | `getuv()` from spectral coefficients | lower-level reconstruction through the underlying transform stack | `sg.wind(vo, d)` |
| Rotational or divergent wind | compose inverse vector transforms | returned by `helmholtz()` | `sg.rotational_wind(...)`, `sg.divergent_wind(...)` |
| Scalar gradient | lower-level SPHEREPACK operations | not a primary API | `field.sg.gradient()` |
| Laplacian / inverse Laplacian | coefficient-space operations | not a primary API | `field.sg.laplacian()`, `field.sg.inverse_laplacian()` |

spharmgrid also accepts direct function calls for the accessor operations; see
the {doc}`api` reference.

## PySHTOOLS

[PySHTOOLS](https://shtools.github.io/SHTOOLS/) is broader than the atmospheric
workflow above. It provides spherical-harmonic coefficient and grid classes,
transforms, spectral analysis, localization and Slepian methods, and extensive
gravity and magnetic-field functionality. It is therefore better viewed as a
general spherical-harmonic and geophysical toolkit than as a direct
windspharm-style atmospheric API.

A rough comparison of emphasis is:

| Area | PySHTOOLS | spharmgrid |
| --- | --- | --- |
| Primary Python objects | spherical-harmonic coefficients and grids | xarray `DataArray` / `Dataset` |
| General coefficient manipulation | extensive | internal implementation detail |
| Atmospheric `u`, `v` diagnostics | possible from spherical-harmonic operations, but not the main high-level interface | first-class API |
| CF coordinate discovery and metadata preservation | not the primary abstraction | first-class API |
| Atmospheric truncation notation such as `T42` and `T6-42` | use coefficient bandwidth / degree controls | first-class API |
| Spectral scalar and vector regridding | available through transform/grid operations | first-class API |
| Gravity and magnetic-field analysis | extensive | out of scope |
| Slepian / localization analysis | extensive | out of scope |

Use PySHTOOLS when coefficient-level spherical-harmonic analysis, gravity,
magnetics, localization, or its broader geophysical toolset is the goal. Use
spharmgrid when the input is an atmospheric xarray field and the desired
operation is filtering, regridding, differentiation, or wind decomposition
while retaining coordinates and metadata.

## Grid scope

Grid names alone do not imply identical numerical conventions across
libraries. spharmgrid currently supports full rectangular Gauss--Legendre (GL)
and pole-including Clenshaw--Curtis (CC) grids and validates those geometries
before transforming them. See {doc}`grids` for the exact definitions and
{doc}`filtering` for spharmgrid's spectral-range conventions.

When reproducing results from another package, compare the grid definition,
normalization, truncation, Earth radius, vector sign conventions, and treatment
of the degree-zero mode rather than translating only the function name.
