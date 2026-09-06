# spharmgrid plan and current contract

## Status

The initial spharmgrid implementation is in place. This file records the current
scientific/API contract and the boundaries for follow-up work.

The package is an xarray-first layer around DUCC0 for spherical-harmonic
operations on full global rectangular Gauss--Legendre (GL) and
Clenshaw--Curtis (CC) grids. Atmospheric wind diagnostics and inverse
transforms are part of the package's core scope.

The package name is expanded as:

> **spharmgrid (SPherical HARMonics GRIDding)**

A concise description is:

> Spherical harmonics regridding tool for atmospheric science

DUCC0 supplies the numerical spherical-harmonic transforms. spharmgrid does not
implement a new SHT engine.

---

# Scientific and software roles

Keep these responsibilities distinct:

```text
published methods       scientific definitions
NCL/SPHEREPACK          established atmospheric semantics and parity reference
DUCC0                    numerical scalar and spin-weighted SHT engine
PyStormTracker           source implementation for the initial extraction
spharmgrid               xarray/CF atmospheric operations layer
```

PyStormTracker remains useful source context for shared numerical behavior, but
spharmgrid is its own package contract. Changes should be justified by
spharmgrid's tests, numerical definitions, and user-facing API.

---

# Current scope

## Supported grids

Support exactly two full rectangular global grid families:

- **GL**: Gauss--Legendre latitude nodes with equally spaced longitude on every
  latitude ring;
- **CC**: equally spaced latitude including both poles, with equally spaced
  longitude on every latitude ring.

Use lowercase Python values:

```python
"gl"
"cc"
```

A generic regular latitude--longitude grid is not automatically CC. A supported
longitude coordinate is one complete uniformly spaced, non-duplicated cycle.
Latitude may be ascending or descending. Equivalent cyclic longitude
representations, including `[0, 360)` and signed conventions, must represent
the same physical grid and produce the same numerical result after coordinate
restoration.

`T42` is a spectral truncation, not a physical-grid name.

## Out of scope unless a concrete new requirement is established

Do not add speculative architecture for:

- reduced Gaussian grids;
- HEALPix;
- regional DCT or generic regional transforms;
- polar stereographic output;
- arbitrary scattered synthesis points;
- MPI;
- a public backend abstraction;
- selectable SHT backends;
- public raw `alm` arrays;
- coefficient/grid object hierarchies;
- plugin systems;
- unit-conversion frameworks;
- custom calendar/time representations;
- cache or concurrency frameworks around DUCC.

These are not prohibited forever; they require a specific scientific or user
need and should not complicate the current GL/CC model in advance.

---

# Public model

The package should remain understandable as:

```text
global atmospheric field on GL/CC grid
    -> filter / regrid / differentiate
    -> wind diagnostics / potentials / inverse wind
```

The xarray accessor and direct functions use the same numerical implementation:

```python
field.sg.filter("T6-42")
sg.filter(field, "T6-42")
```

Importing spharmgrid registers `.sg` on `xarray.DataArray` and
`xarray.Dataset`.

Keep the top-level namespace small and close to the documented public API.
Internal DUCC coefficient helpers, gufunc kernels, E/B arrays, metadata
registries, layouts, and accessor classes are implementation details.

---

# Spectral filtering and regridding

## Spectral notation

Support triangular total-wavenumber notation:

```text
T42       -> l = 0..42
T6-42     -> l = 6..42
```

Parsing is case-insensitive and accepts an en dash. Explicit `lmin`/`lmax`
bounds are equivalent. Do not mix `truncation=` with explicit bounds in one call.

## Filtering

The default is a hard spectral selection with no taper. There is no hidden
smoothing.

The only supported taper is the Sardeshmukh--Hoskins exponential response. The
public argument remains:

```python
taper: float | None
```

with `0 < taper <= 1`. `taper=0.1` means the response at the upper retained
degree is 0.1. Modes outside an explicit retained range are zero.

When no explicit spectral range is supplied, the transform uses the complete
bandwidth representable by the grid. A supplied taper then uses that transform
`lmax` as its endpoint.

## Regridding

Support all pairwise combinations:

```text
GL -> GL
GL -> CC
CC -> GL
CC -> CC
```

Filtering and regridding share one transform cycle:

```text
source spatial field
    -> one spherical-harmonic analysis
    -> optional coefficient selection/taper
    -> one synthesis on the target grid
```

Do not implement a combined operation by calling public `filter()` followed by
public `regrid()`.

Without an explicit `Tn` range, retain the latitude and azimuthal content jointly
representable by source and target sampling under DUCC's actual limits. Explicit
`Tn` input remains triangular and must raise if the requested range cannot be
represented.

---

# Scalar differential operators

Use spectral operators, not finite differences.

The default spherical Earth radius is:

```python
EARTH_RADIUS_M = 6_371_220.0
```

`gradient()` returns physical eastward/northward derivatives. `laplacian()`
applies the spherical-harmonic eigenvalue `-l(l+1)/R^2`.
`inverse_laplacian()` uses the corresponding inverse for positive degree and
sets the singular degree-zero coefficient to zero, defining the zero-mean
solution.

---

# Atmospheric wind transforms

These operations are a core spharmgrid use case and should remain prominent in
README/docs/package metadata.

Canonical quantities are:

| Quantity | Short name | CF `standard_name` |
| --- | --- | --- |
| Eastward wind | `u` | `eastward_wind` |
| Northward wind | `v` | `northward_wind` |
| Relative vorticity | `vo` | `atmosphere_relative_vorticity` |
| Divergence | `d` | `divergence_of_wind` |
| Streamfunction | `strf` | `atmosphere_horizontal_streamfunction` |
| Velocity potential | `vp` | `atmosphere_horizontal_velocity_potential` |

The supported transform graph is:

```text
u, v -> vo
u, v -> d
u, v -> vo + d
u, v -> strf + vp

vo -> rotational wind
d -> divergent wind
strf -> rotational wind
vp -> divergent wind

vo + d -> full u, v
strf + vp -> full u, v
```

Use **rotational wind** and **divergent wind** as the scientific terms.

The adopted potential convention is:

```math
\zeta = \nabla^2\psi,
\qquad
\delta = \nabla^2\chi.
```

The DUCC spin-1 component/sign mapping must stay covered by analytic rotational
and divergent fields, forward/inverse round trips, and independent parity tests.
Do not change signs, E/B ordering, latitude orientation, or radius factors based
on intuition alone.

Related outputs should share one analysis where practical: `kinematics()`
shares the vector analysis for vorticity/divergence and `potentials()` shares it
for streamfunction/velocity potential.

---

# xarray, CF, and coordinates

Coordinate discovery order is:

```text
exact CF standard_name
    -> canonical coordinate name
    -> optional cf-xarray assistance when installed
    -> explicit error
```

Optional `cf-xarray` support means spharmgrid may import the installed package
lazily when its fallback is needed; users should not need to import cf-xarray
first solely to register its accessor.

Dataset physical-variable discovery is:

```text
explicit argument
    -> unique exact CF standard_name
    -> canonical short name
    -> explicit error
```

Do not maintain broad heuristic aliases without a demonstrated interoperability
need.

Same-quantity operations such as filtering and regridding preserve variable
name and semantic attributes. Derived quantities receive exact CF metadata when
an exact standard name exists. Rotational/divergent wind components should not
be assigned broader full-wind standard names when CF has no exact component
name.

Transforms operate only over discovered horizontal dimensions and preserve
arbitrary leading dimensions and coordinates. xarray owns time/calendar
representation; spharmgrid does not convert or normalize time.

---

# Dask and thread behavior

Dask remains optional. Dask-backed xarray inputs should remain lazy through the
horizontal transform graph. Do not introduce a spharmgrid backend selector.

`nthreads` is the only DUCC execution control. With Dask-backed input and no
explicit value, use one DUCC thread per task to avoid nested oversubscription.
Do not add thread/process pools around DUCC without measured need.

---

# CLI and I/O

The CLI is a thin file-oriented helper over the same package functions. Keep the
commands:

```text
spharmgrid info
spharmgrid filter
spharmgrid regrid
spharmgrid kinematics
spharmgrid potentials
spharmgrid wind
```

The core runtime is limited to `numpy`, `xarray`, and `ducc0`. The CLI is an
optional file-I/O capability with one user-facing `cli` extra containing
`h5netcdf[h5py]`, `zarr`, and `cfgrib`. A path ending in `.zarr` is opened with
`xarray.open_zarr()` and written with `Dataset.to_zarr()`. Other input paths use
`xarray.open_dataset()`, allowing installed engines such as h5netcdf or cfgrib
to handle them. Other outputs are written as NetCDF through
`engine="h5netcdf"`; GRIB is input-only through cfgrib. Actual decoding and
encoding remain delegated to xarray and its backend packages.

The CLI should not introduce a separate file abstraction or numerical path.
Python callers may still supply already-open xarray objects or use other xarray
backends themselves.

The `spharmgrid` console entry point remains available from a core-only install
for `--help` and `--version`. File-processing commands require the optional
`cli` backends and report an actionable `pip install "spharmgrid[cli]"`
message when those imports are unavailable. Optional backend imports stay lazy;
constructing the argument parser and importing `spharmgrid` do not require
`h5netcdf`, `zarr`, or `cfgrib`.

---

# Dependencies and repository engineering

Core runtime dependencies are:

```text
numpy
xarray
ducc0
```

Keep optional user-facing capabilities explicit:

```text
cf              optional cf-xarray coordinate discovery
dask            lazy Dask-backed xarray execution
cli             NetCDF and Zarr read/write plus GRIB input for the CLI
```

The ordinary test environment directly declares cftime, h5netcdf, and Zarr:
the calendar-preservation test imports cftime, and the CLI NetCDF and Zarr
read/write paths are exercised directly. GRIB remains optional and is checked
in a dedicated Linux backend lane because cfgrib/ecCodes have a heavier
platform footprint. cftime is test-only rather than a direct spharmgrid
runtime dependency. `pyspharm-syl` remains an independent parity-only
dependency.

uv's normal `dev` dependency group remains enabled so standard local commands
such as `uv run ruff`, `uv run ty check`, and `uv run pytest` work without
additional group flags. CI lanes that intentionally need a smaller environment
must use `--no-default-groups` explicitly.

Python 3.12--3.14 is the current supported matrix. The independent parity
dependency may use a narrower interpreter range without narrowing production
support.

The free-threaded compatibility probe checks each direct supported module in a
fresh Python process: the three core dependencies, `cf_xarray`, `dask`, the
three CLI backends, and `spharmgrid`. A package that cannot import or that
enables the GIL is reported as a warning so one incompatibility does not
prevent the remaining checks from running. Transitive implementation
dependencies are not probed separately.

---

# Validation requirements

Maintain three evidence layers:

1. deterministic analytic/constructed fields;
2. internal identities and inverse round trips;
3. parity against an independent implementation such as SPHEREPACK/pyspharm.

Current coverage should continue to include both GL and CC for the core analytic
and round-trip behavior, coordinate-order/cyclic-representation invariance,
Dask laziness, arbitrary leading dimensions, CF metadata, and accessor/direct
API equivalence.

CLI tests should cover NetCDF and Zarr read/write paths. GRIB support should be
checked through the cfgrib backend without making GRIB an unconditional runtime
or ordinary test dependency.

External parity should cover operations where the external grid and convention
can be aligned exactly. Do not weaken scientific tests merely to force equality
between implementations with different sampling or conventions.

Before accepting a numerical change, check at least:

- grid family and exact latitude nodes;
- longitude origin/order;
- latitude orientation;
- truncation and `mmax`;
- Earth radius;
- normalization;
- scalar/vector sign and component convention;
- units and precision.

---

# Documentation contract

Write for atmospheric scientists and xarray users. The landing material should
make the atmospheric capabilities visible immediately, including relative
vorticity, divergence, streamfunction, and velocity potential, while retaining
filtering/regridding and scalar operators as equal package capabilities.

Documentation must state:

- DUCC0 performs the numerical transforms;
- GL/CC definitions and coordinate restrictions;
- `T42`, `T6-42`, and taper semantics;
- regridding bandwidth behavior;
- Earth-radius and zero-mode conventions;
- wind sign/component conventions;
- CF discovery/metadata behavior;
- Dask/thread behavior;
- NetCDF and Zarr CLI read/write through the `cli` extra, plus optional GRIB
  input;
- NCL/SPHEREPACK's role as semantic/parity reference;
- PyStormTracker's role as the source of the initial extracted implementation.

Do not describe spharmgrid as a new SHT implementation or imply that it contains
SPHEREPACK.

---

# Completion and change gate

The initial scope is considered complete only while all of these remain true:

- ordinary tests pass across the supported Python matrix;
- minimum-direct-dependency tests pass;
- analytic and inverse/round-trip tests pass;
- independent parity tests pass in their supported environment;
- Ruff and ty checks pass;
- the strict Sphinx build passes;
- wheel and sdist build, and a fresh wheel import/CLI smoke test passes;
- the public namespace remains bounded to the documented API;
- no excluded grid/backend/concurrency abstraction enters without a concrete
  requirement.

A follow-up change may refine implementation details or strengthen engineering,
but scientific semantics and public behavior should change only with explicit
motivation, tests, and synchronized documentation.