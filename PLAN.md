# spharmgrid implementation plan

## Implementation target

Use **Terra Max** for the first implementation pass.

The work is bounded but numerically coupled: it combines DUCC0 spherical-harmonic conventions, GL/CC grid handling, xarray/Dask behavior, vector transforms, CF semantics, external parity tests, packaging, and a complete documentation build. Terra Max is preferred for the first pass because mistakes in vector signs, grid orientation, or API coupling can look plausible while being scientifically wrong.

Do not expand the package beyond this plan. In particular, do not add a backend framework, coefficient object model, plugin system, generic grid hierarchy, cache layer, or concurrency framework for possible future use. Prefer small functions and data types that serve the current GL/CC API.

Terra xhigh is acceptable if usage needs to be reduced. Luna Max is suitable for bounded follow-up work after the numerical/API structure is in place. If Luna Max is used for the full implementation, review vector-transform conventions, parity tests, and public API boundaries closely.

Do not release, tag, publish to PyPI, create Zenodo records, or add release automation as part of this plan.

## Read first

Read, in this order:

```text
AGENTS.md
skills/scientific-numerics/SKILL.md
skills/documentation/SKILL.md
skills/repository-engineering/SKILL.md
PLAN.md
```

The sibling repository is the primary source implementation:

```text
../PyStormTracker
```

Read its current `AGENTS.md` and current source before porting code. Do not assume snippets in this plan are newer than the local checkout.

Inspect at least:

```text
../PyStormTracker/src/pystormtracker/preprocessing/spectral.py
../PyStormTracker/src/pystormtracker/preprocessing/regrid.py
../PyStormTracker/src/pystormtracker/preprocessing/kinematics.py
../PyStormTracker/src/pystormtracker/models/geo.py
../PyStormTracker/src/pystormtracker/backends.py
```

Also inspect the corresponding tests and documentation when needed to understand behavior already validated in PyStormTracker.

Do not modify `../PyStormTracker`. Do not make PyStormTracker depend on spharmgrid.

---

# Goal

`spharmgrid` is a lightweight xarray-first wrapper/helper around `ducc0` for spherical-harmonic operations used in atmospheric and geophysical science.

`ducc0` supplies the numerical spherical-harmonic transform machinery. spharmgrid supplies:

- recognition and construction of common global atmospheric grids;
- xarray DataArray/Dataset integration through `.sg` accessors;
- equivalent direct Python functions under `spharmgrid`;
- spectral filtering and spectral regridding;
- scalar differential operators;
- vector-wind kinematics, potentials, and inverse wind transforms;
- CF-aware variable discovery and output metadata;
- lazy xarray/Dask execution where xarray supports it;
- a small file-oriented CLI that delegates I/O to xarray;
- analytic and SPHEREPACK/pyspharm parity tests;
- complete Sphinx/MyST documentation built by Read the Docs.

A central goal is to provide clear xarray-native equivalents of common **NCL/SPHEREPACK spherical-harmonic workflows**, implemented with `ducc0` rather than wrapping SPHEREPACK.

Do not describe spharmgrid as a new SHT implementation.

Suggested one-line description:

> Spherical-harmonic filtering, regridding, and atmospheric field operations for global xarray grids using ducc0.

---

# Scientific and software lineage

The first implementation should be extracted/generalized from the global spherical-harmonic layer already developed in PyStormTracker:

- scalar filtering from `preprocessing/spectral.py`;
- GL/CC spectral regridding from `preprocessing/regrid.py`;
- spin-1 vector transforms and vorticity/divergence from `preprocessing/kinematics.py`.

Use that code as the implementation starting point. Remove PyStormTracker-specific data loaders, tracker backends, regional paths, HEALPix paths, and tracking conventions.

Validate the extracted numerics independently. Relevant sources include:

- Sardeshmukh, P. D. and B. J. Hoskins, 1984: *Spatial Smoothing on the Sphere*, Monthly Weather Review, 112, 2524–2529, DOI `10.1175/1520-0493(1984)112<2524:SSOTS>2.0.CO;2`;
- SPHEREPACK and NCL documentation for established atmospheric transform semantics;
- Reinecke and Seljebotn (2013) and Ishioka (2018) for DUCC/libsharp numerical lineage where relevant;
- current CF Standard Name Table for semantic metadata.

Keep these roles distinct:

```text
published method       scientific definition/lineage
NCL/SPHEREPACK         established atmospheric API/behavior and parity reference
ducc0                   numerical transform engine
PyStormTracker          source implementation for extraction
spharmgrid              xarray/CF atmospheric operations layer
```

---

# Initial scope

## Supported grids

Support two rectangular global grid families.

### Gauss–Legendre (GL)

- Gaussian/Gauss–Legendre latitudes;
- constant number of equally spaced longitudes on every latitude;
- full rectangular `(lat, lon)` representation.

### Clenshaw–Curtis (CC)

- equally spaced latitude;
- both poles included;
- equally spaced longitude;
- full rectangular `(lat, lon)` representation.

Use **GL** and **CC** in prose, docs, comments, and scientific discussion. Use lowercase values in Python:

```python
"gl"
"cc"
```

Do not use `latlon` as a grid type. A generic regular latitude–longitude grid does not imply CC sampling.

## Out of scope

Do not add these in the first implementation:

- reduced Gaussian grids;
- HEALPix;
- PyStormTracker regional DCT filtering;
- polar stereographic output;
- arbitrary scattered synthesis points;
- MPI;
- PyStormTracker's public backend abstraction;
- selectable SHT backends;
- public raw `alm` arrays;
- `SHCoeffs`/`SHGrid`-style object hierarchies;
- unit-conversion frameworks;
- custom calendar/time representations;
- generic regional lat/lon transforms;
- release/tag/PyPI/Zenodo machinery.

Reduced Gaussian may be considered later. Do not complicate the initial rectangular xarray model for it.

---

# Terminology

Use **field** for a gridded physical scalar or vector quantity.

```text
field        gridded scalar/vector DataArray or Dataset
grid         horizontal sampling geometry
target_grid  output horizontal grid for regridding
data          generic xarray/file content when physical-field meaning is not needed
```

Avoid `frame` in the public API and user documentation. spharmgrid should work on a 2-D field or an xarray object with arbitrary leading dimensions such as:

```text
(time, level, member, lat, lon)
```

Each horizontal slice is transformed independently.

---

# Package shape

Use a normal `src/` layout. A compact target is:

```text
src/spharmgrid/
    __init__.py
    accessors.py
    grids.py
    spectral.py
    regrid.py
    operators.py
    kinematics.py
    metadata.py
    _ducc.py
    _xarray.py
    cli.py
```

Change filenames if a smaller organization is clearer. Keep numerical kernels separate from accessor wrappers.

The accessor classes should contain only:

- input/variable selection;
- argument normalization;
- calls into the same functions used by the direct API.

Do not put SHT mathematics in accessor classes.

---

# Public Python API

The two public styles are first-class and numerically equivalent:

```python
import spharmgrid as sg

field.sg.filter(...)
sg.filter(field, ...)
```

Register `.sg` on both `xarray.DataArray` and `xarray.Dataset` when spharmgrid is imported.

## Public namespace target

Keep the top-level namespace close to:

```python
sg.Grid
sg.SpectralRange
sg.EARTH_RADIUS_M

sg.gaussian_grid
sg.clenshaw_curtis_grid
sg.detect_grid
sg.parse_spectral

sg.filter
sg.regrid
sg.gradient
sg.laplacian
sg.inverse_laplacian

sg.vorticity
sg.divergence
sg.kinematics
sg.streamfunction
sg.velocity_potential
sg.potentials
sg.rotational_wind
sg.divergent_wind
sg.wind
```

Do not expose internal DUCC coefficient helpers, gufunc kernels, E/B arrays, metadata registries, or accessor classes without a demonstrated need.

---

# Grid API

Use one small immutable grid descriptor. Do not create separate GL/CC class hierarchies unless the implementation truly needs them.

Suggested public form:

```python
@dataclass(frozen=True)
class Grid:
    kind: Literal["gl", "cc"]
    latitude: np.ndarray
    longitude: np.ndarray
```

Constructors:

```python
sg.gaussian_grid(
    nlat: int,
    nlon: int,
    *,
    lon0: float = 0.0,
    latitude_order: Literal["ascending", "descending"] = "ascending",
) -> Grid

sg.clenshaw_curtis_grid(
    nlat: int,
    nlon: int,
    *,
    lon0: float = 0.0,
    latitude_order: Literal["ascending", "descending"] = "ascending",
) -> Grid
```

Detection:

```python
sg.detect_grid(field) -> Grid

field.sg.grid
field.sg.grid_type  # "gl" or "cc"
```

`regrid()` should accept either a `Grid` or an xarray object whose horizontal coordinates define a supported target grid:

```python
target = sg.gaussian_grid(64, 128)
field.sg.regrid(target)

field.sg.regrid(reference_field)
```

Do not make `T42` a grid constructor or target-grid shorthand. `T42` is a spectral truncation, not a unique physical grid.

## Coordinate discovery

Core coordinate discovery must work without `cf-xarray`:

1. exact CF latitude/longitude metadata when present;
2. canonical coordinate names `lat`/`latitude` and `lon`/`longitude`;
3. optional `cf-xarray` assistance when installed;
4. clear ambiguity/error otherwise.

Do not silently assume that the last two dimensions are latitude and longitude.

## CC detection

Require:

- equally spaced latitudes within a justified numerical tolerance;
- both poles represented;
- uniform globally cyclic longitude spacing;
- no duplicated cyclic longitude endpoint.

## GL detection

Require:

- latitude values matching the corresponding DUCC GL nodes within tolerance;
- either latitude order;
- uniform globally cyclic longitude spacing;
- no duplicated cyclic longitude endpoint.

Support both `[0, 360)` and `[-180, 180)`-style cyclic longitude representations by rolling/reordering internally when needed. Move coordinates and field values together.

For same-grid operations, preserve the user's coordinate order/convention. For a new `Grid` target, return the target coordinates represented by that object.

Confirm exact representable `lmax`/`mmax` constraints from the installed `ducc0` API and tests. Do not copy an unverified formula from older code.

---

# Spectral range

Expose:

```python
@dataclass(frozen=True)
class SpectralRange:
    lmin: int
    lmax: int
```

and:

```python
sg.parse_spectral("T42")
# SpectralRange(lmin=0, lmax=42)

sg.parse_spectral("T6-42")
# SpectralRange(lmin=6, lmax=42)
```

Parsing is case-insensitive:

```text
T42
t42
T6-42
t6-42
```

Accept an en dash as input normalization so `T6–42` also works.

Do not add rhomboidal or other truncation notation initially.

Validate:

```text
0 <= lmin <= lmax
```

A call may use either:

- `spectral=` / positional spectral notation; or
- explicit `lmin=` and `lmax=`.

Reject mixed use in the same call.

---

# Scalar filtering

Accessor examples:

```python
field.sg.filter("T42")
field.sg.filter("T6-42")
field.sg.filter(lmin=6, lmax=42)
field.sg.filter("T6-42", taper=0.1)
```

Direct equivalents:

```python
sg.filter(field, "T42")
sg.filter(field, "T6-42")
sg.filter(field, lmin=6, lmax=42)
sg.filter(field, "T6-42", taper=0.1)
```

Target signature:

```python
def filter(
    field: xr.DataArray,
    spectral: str | SpectralRange | None = None,
    *,
    lmin: int | None = None,
    lmax: int | None = None,
    taper: float | None = None,
    nthreads: int | None = None,
) -> xr.DataArray: ...
```

## Hard filter

Default:

```python
taper=None
```

For `T6-42`:

```text
l < 6       -> 0
6 <= l <=42 -> unchanged
l > 42      -> 0
```

There is no hidden smoothing.

## Taper

Support one taper only. The public parameter is simply:

```python
taper: float | None
```

Do not add a taper-name enum or `taper_response` parameter.

`taper=0.1` means the response at `lmax` for the Sardeshmukh–Hoskins exponential taper already implemented in PyStormTracker.

For total wavenumber `l`:

```math
w(l)=\exp\left[-K\{l(l+1)\}^2\right]
```

with:

```math
K=\frac{-\ln(\mathrm{taper})}{\{l_{\max}(l_{\max}+1)\}^2},
```

so:

```math
w(l_{\max})=\mathrm{taper}.
```

Apply the response only inside the retained band and hard-zero modes outside `[lmin, lmax]`.

Validate:

```text
0 < taper <= 1
```

`taper=1` gives unity weighting inside the retained band.

When `taper is None`, use the exact hard mask and skip the exponential weighting calculation.

---

# Spectral regridding

Accessor:

```python
field.sg.regrid(target_grid)
```

Direct:

```python
sg.regrid(field, target_grid)
```

Filtering and regridding must combine in one transform cycle:

```python
field.sg.regrid(
    target_grid,
    spectral="T6-42",
    taper=0.1,
)
```

Direct:

```python
sg.regrid(
    field,
    target_grid,
    spectral="T6-42",
    taper=0.1,
)
```

Also allow explicit bounds:

```python
field.sg.regrid(target_grid, lmin=6, lmax=42, taper=0.1)
```

Target signature:

```python
def regrid(
    field: xr.DataArray,
    target_grid: Grid | xr.DataArray | xr.Dataset,
    spectral: str | SpectralRange | None = None,
    *,
    lmin: int | None = None,
    lmax: int | None = None,
    taper: float | None = None,
    nthreads: int | None = None,
) -> xr.DataArray: ...
```

Implementation path:

```text
source field
  -> one scalar analysis
  -> optional spectral mask/taper
  -> one synthesis on target GL/CC grid
```

Do not implement combined regrid+filter by calling public `filter()` and then public `regrid()`.

Supported combinations:

```text
GL -> GL
GL -> CC
CC -> GL
CC -> CC
```

With no explicit spectral range, use the maximum transform content representable by the source/target geometry under DUCC's actual constraints. Do not impose an atmospheric `Tn` convention automatically.

If an explicit requested range cannot be represented, raise instead of silently clamping.

---

# Scalar differential operators

Use spectral operators, not finite differences.

Define:

```python
sg.EARTH_RADIUS_M = 6_371_220.0
```

matching the SPHEREPACK/NCL-style spherical Earth value currently used by PyStormTracker.

Allow a `radius` keyword on physical differential/kinematic operations when needed.

## Gradient

```python
grad = field.sg.gradient()
grad = sg.gradient(field)
```

Return a Dataset with default variables:

```text
gradient_eastward
gradient_northward
```

Allow:

```python
field.sg.gradient(
    eastward="dx",
    northward="dy",
)
```

The result is the physical horizontal gradient on a sphere, with units of input-units per meter when the input units are defined.

## Laplacian

```python
field.sg.laplacian()
sg.laplacian(field)
```

Apply:

```math
-\frac{l(l+1)}{R^2}.
```

## Inverse Laplacian

```python
field.sg.inverse_laplacian()
sg.inverse_laplacian(field)
```

For `l > 0`, apply:

```math
-\frac{R^2}{l(l+1)}.
```

The `l=0` mode is singular. Define the inverse as the zero-mean solution by setting the `l=0` output coefficient to zero and document that convention.

---

# Vector-wind API

Cover the useful transform graph associated with NCL/SPHEREPACK routines such as `uv2vr`, `uv2dv`, `uv2vrdv`, `uv2sfvp`, `vr2uv`, `dv2uv`, `vrdv2uv`, and `sfvp2uv`, but use descriptive Python names.

```text
u, v -> vo
u, v -> d
u, v -> vo + d
u, v -> strf + vp

vo -> rotational wind
d  -> divergent wind
vo + d -> full u, v

strf -> rotational wind
vp   -> divergent wind
strf + vp -> full u, v
```

Use **rotational wind** and **divergent wind** as the scientific terms.

---

# Dataset variable discovery

Canonical atmospheric quantities:

| Quantity | Short name | CF `standard_name` |
| --- | --- | --- |
| eastward wind | `u` | `eastward_wind` |
| northward wind | `v` | `northward_wind` |
| relative vorticity | `vo` | `atmosphere_relative_vorticity` |
| divergence | `d` | `divergence_of_wind` |
| streamfunction | `strf` | `atmosphere_horizontal_streamfunction` |
| velocity potential | `vp` | `atmosphere_horizontal_velocity_potential` |

Verify these exact standard names against the current official CF table during implementation.

Resolution order:

```text
explicit argument
  -> unique exact CF standard_name
  -> canonical short name
  -> error
```

Examples:

```python
ds.sg.vorticity()
```

uses `u` and `v` automatically when present.

This also works:

```python
ds["uwind"].attrs["standard_name"] = "eastward_wind"
ds["vwind"].attrs["standard_name"] = "northward_wind"

ds.sg.vorticity()
```

Explicit override:

```python
ds.sg.vorticity(u="ua", v="va")
```

If multiple variables match the same quantity, raise an ambiguity error listing candidates.

Do not maintain a broad alias table such as `UGRD`, `uwnd`, `vor`, `vort`, etc. Add aliases only if a real interoperability requirement appears later.

---

# Wind diagnostics

## Vorticity

```python
vo = ds.sg.vorticity()
vo = ds.sg.vorticity(output="vort")
vo = ds.sg.vorticity(u="ua", v="va", output="vort")

vo = sg.vorticity(u, v)
```

Default output name:

```text
vo
```

Return a DataArray.

## Divergence

```python
d = ds.sg.divergence()
d = sg.divergence(u, v)
```

Default output name:

```text
d
```

Return a DataArray.

## Combined kinematics

```python
kin = ds.sg.kinematics()
kin = sg.kinematics(u, v)
```

Return:

```text
Dataset
  vo
  d
```

Allow:

```python
kin = ds.sg.kinematics(
    u="ua",
    v="va",
    vorticity="vort",
    divergence="div",
)
```

Compute both outputs from one vector analysis whenever possible.

---

# Streamfunction and velocity potential

Individual:

```python
strf = ds.sg.streamfunction()
vp = ds.sg.velocity_potential()

strf = sg.streamfunction(u, v)
vp = sg.velocity_potential(u, v)
```

Defaults:

```text
strf
vp
```

Combined:

```python
pot = ds.sg.potentials()
pot = sg.potentials(u, v)
```

Return:

```text
Dataset
  strf
  vp
```

Allow:

```python
ds.sg.potentials(
    streamfunction="psi",
    velocity_potential="chi",
)
```

Compute both from one vector analysis whenever possible.

---

# Rotational and divergent wind

## Rotational wind

From relative vorticity:

```python
vo.sg.rotational_wind()
sg.rotational_wind(vo)
```

From streamfunction:

```python
strf.sg.rotational_wind()
sg.rotational_wind(strf)
```

Identify the scalar source from:

1. explicit `quantity=` when supplied;
2. CF `standard_name`;
3. canonical short name.

If ambiguous/unidentified, require:

```python
sg.rotational_wind(field, quantity="vorticity")
sg.rotational_wind(field, quantity="streamfunction")
```

Default output names:

```text
u_rotational
v_rotational
```

Return a Dataset.

## Divergent wind

From divergence:

```python
d.sg.divergent_wind()
sg.divergent_wind(d)
```

From velocity potential:

```python
vp.sg.divergent_wind()
sg.divergent_wind(vp)
```

Use the same source detection rules.

Default outputs:

```text
u_divergent
v_divergent
```

Return a Dataset.

---

# Full wind reconstruction

Dataset accessor:

```python
wind = ds.sg.wind()
```

Auto-detect when exactly one complete representation is present:

```text
vo + d     -> u + v
strf + vp  -> u + v
```

Return:

```text
Dataset
  u
  v
```

If both complete source representations are present, require:

```python
ds.sg.wind(source="vorticity_divergence")
ds.sg.wind(source="potentials")
```

Direct:

```python
sg.wind(vo, d)
sg.wind(strf, vp)
```

Infer semantics from metadata/name when unambiguous and provide an explicit `source=` escape hatch.

Allow output-name overrides without confusing them with source-variable selection. If necessary, use distinct keyword names internally/signature-wise rather than forcing one uniform accessor signature.

---

# Vector numerical implementation

Start from the current PyStormTracker spin-1 implementation and preserve its tested mapping until independent tests prove the convention.

PyStormTracker currently maps geographic wind into DUCC spin-1 components with the equivalent of:

```python
vec_map = np.stack((-v, u), axis=0)
```

and uses `ducc0.sht.analysis_2d(..., spin=1)`.

Its current vorticity/divergence scaling uses:

```math
\frac{\sqrt{l(l+1)}}{R}
```

with signs tied to DUCC's E/B convention.

Do not rewrite these signs from memory. Trace them from current source, then prove the result with analytic and independent parity tests.

Use the standard spherical relationships under the adopted sign convention:

```math
\zeta=\nabla^2\psi,
```

```math
\delta=\nabla^2\chi,
```

where `psi` is streamfunction and `chi` is velocity potential.

For `l > 0`:

```math
\psi_{lm}=-\frac{R^2}{l(l+1)}\zeta_{lm},
```

```math
\chi_{lm}=-\frac{R^2}{l(l+1)}\delta_{lm}.
```

Set `l=0` potential coefficients to zero because the additive constant is undefined.

For inverse wind from vorticity/divergence:

- invert the same E/B scaling used by the forward path;
- synthesize with `spin=1`;
- map DUCC components back to geographic `(u, v)` consistently.

For inverse wind from streamfunction/velocity potential:

- scalar-analyze the potential field(s);
- apply the spherical Laplacian to obtain vorticity/divergence coefficients;
- use the same vector synthesis path.

For streamfunction alone, set divergence coefficients to zero. For velocity potential alone, set vorticity coefficients to zero.

Share spectral coefficients between related outputs. Avoid repeated analyses/syntheses when one transform can produce the requested result.

---

# CF metadata

CF-compatible metadata is part of the API contract.

Do not require a CF library just to assign known metadata. Keep a small internal table for canonical outputs.

At minimum:

| Name | CF `standard_name` | `long_name` | units |
| --- | --- | --- | --- |
| `u` | `eastward_wind` | Eastward wind | `m s-1` |
| `v` | `northward_wind` | Northward wind | `m s-1` |
| `vo` | `atmosphere_relative_vorticity` | Relative vorticity | `s-1` |
| `d` | `divergence_of_wind` | Divergence of wind | `s-1` |
| `strf` | `atmosphere_horizontal_streamfunction` | Horizontal streamfunction | `m2 s-1` |
| `vp` | `atmosphere_horizontal_velocity_potential` | Horizontal velocity potential | `m2 s-1` |

Verify exact standard names against the current CF table before freezing tests.

An output-name override changes `DataArray.name`, not physical semantics:

```python
vo = ds.sg.vorticity(output="vort")
```

must still carry the vorticity `standard_name`.

## Rotational/divergent component metadata

Check the current CF table for exact names for rotational/divergent wind components. If no exact standard name exists, omit `standard_name` and use accurate `long_name` plus `units="m s-1"`.

Do not apply the broader `eastward_wind`/`northward_wind` standard names if that would misstate the component semantics.

## Same-quantity operations

Filtering and regridding do not change the physical quantity. Preserve:

- variable name;
- `standard_name`;
- `long_name`;
- `units`;
- other non-conflicting attrs.

## Generated spatial coordinates

Attach ordinary CF coordinate metadata:

```text
latitude:
    standard_name = latitude
    units = degrees_north
    axis = Y

longitude:
    standard_name = longitude
    units = degrees_east
    axis = X
```

Do not invent CF standard names for GL or CC sampling. Grid type is a numerical property.

---

# Optional cf-xarray

Make `cf-xarray` optional.

Core spharmgrid must directly understand the documented `standard_name` attributes and canonical names.

A likely user extra is:

```toml
[project.optional-dependencies]
cf = ["cf-xarray"]
```

When installed, `cf-xarray` may improve coordinate/axis discovery and interoperability. It must not change numerical results for an otherwise identical valid dataset.

Test the core path without `cf-xarray` and optional integration where useful.

---

# CF time and non-spatial dimensions

Do not implement a spharmgrid time representation.

xarray owns CF decoding/encoding of time and calendars. Preserve all non-spatial dimensions/coordinates while applying transforms only over the horizontal dimensions.

Support shapes including:

```text
(lat, lon)
(time, lat, lon)
(level, lat, lon)
(time, level, lat, lon)
(member, time, level, lat, lon)
```

Preserve, when present:

- NumPy datetime coordinates;
- `cftime` objects;
- proleptic Gregorian calendars;
- no-leap calendars;
- 360-day calendars.

Do not convert time to Unix time, integer milliseconds, or another package representation.

Do not add `cftime` as a direct dependency solely for spharmgrid. Let xarray's I/O stack provide it when required by a user's data.

---

# xarray and Dask behavior

The xarray API is primary. Numerical kernels should operate on NumPy arrays and be applied over xarray objects with `xr.apply_ufunc` or an equally direct mechanism.

Requirements:

- transform only detected horizontal dimensions;
- preserve arbitrary leading dimensions;
- eager NumPy-backed DataArrays work;
- Dask-backed DataArrays remain lazy when Dask is installed;
- Dask is not required as a core dependency solely for xarray support;
- horizontal core dimensions are rechunked only when needed;
- errors should state rechunking requirements clearly;
- preserve dtype where appropriate, but do not force float32 at the expense of transform correctness.

Do not copy PyStormTracker's `backend="serial"|"dask"|"mpi"` API.

Expose only:

```python
nthreads: int | None = None
```

for DUCC thread control where needed.

Default policy:

- eager data + `nthreads is None`: use normal DUCC behavior after confirming the installed DUCC convention;
- Dask-backed data + `nthreads is None`: use one DUCC thread per task to avoid oversubscription;
- explicit `nthreads`: honor it.

No MPI initially.

---

# Direct/accessor parity

These must be numerically equivalent:

```python
sg.filter(field, "T6-42", taper=0.1)
field.sg.filter("T6-42", taper=0.1)
```

and:

```python
sg.vorticity(ds.u, ds.v)
ds.sg.vorticity()
```

The Dataset accessor adds variable selection; it does not add a different numerical path.

Test accessor/direct equivalence explicitly.

---

# Minimal CLI

The CLI is a helper, not the main API.

Use standard-library `argparse` unless a concrete need appears for another dependency.

Initial commands:

```text
spharmgrid info
spharmgrid filter
spharmgrid regrid
spharmgrid kinematics
spharmgrid potentials
spharmgrid wind
```

Examples:

```bash
spharmgrid info input.nc

spharmgrid filter input.nc output.nc \
    --var msl \
    --spectral T6-42 \
    --taper 0.1

spharmgrid kinematics wind.nc output.nc
```

Do not require `--u u --v v` in the normal ERA5/CF case. Use the same variable discovery as the Dataset accessor.

Allow explicit input/output variable overrides where the Python API does.

## CLI I/O

Delegate file handling to xarray. Do not build a file abstraction.

At minimum, normal NetCDF should work through the installed xarray engine.

Optional engines may support:

- NetCDF variants;
- Zarr;
- GRIB/cfgrib reading.

GRIB writing is not required. Reading GRIB through xarray/cfgrib and writing a supported output format is sufficient.

The CLI must call the same public functions/accessors as Python users.

---

# Dependencies and tooling

Use `uv`, Hatchling, Ruff, pytest, and the repository's selected static type checker as described in `skills/repository-engineering/SKILL.md`.

Expected core runtime dependencies:

```text
numpy
xarray
ducc0
```

Optional concepts:

```text
cf-xarray       CF integration
Dask            lazy xarray execution
NetCDF/Zarr/GRIB engines
pyspharm-syl    SPHEREPACK parity environment only
```

Do not add Pint initially. Wind kinematic functions should document the expected SI wind-unit convention; unit conversion can be added later if it becomes a real requirement.

The parity dependency must not restrict production Python support. Run parity on a Python version where the comparison package installs reliably.

A package metadata version may use an unreleased placeholder required by Python packaging. Do not create a tag, release, changelog policy, or publishing workflow now.

---

# Testing strategy

Use three evidence layers.

## 1. Analytic tests

Use deterministic low-degree scalar harmonics and constructed vector fields.

Cover at least:

- GL scalar analysis/synthesis behavior;
- CC scalar analysis/synthesis behavior;
- GL -> CC and CC -> GL regridding;
- `Tn` hard truncation;
- `Tn-m` band pass;
- taper endpoint response, including `w(lmax) == taper`;
- gradient of known harmonics;
- Laplacian eigenvalue;
- inverse-Laplacian zero-mean convention;
- purely rotational wind with near-zero divergence;
- purely divergent wind with near-zero vorticity;
- forward/inverse full-wind reconstruction;
- streamfunction/vorticity relation;
- velocity-potential/divergence relation.

Do not rely only on random fields.

## 2. Internal identities and round trips

Test:

```text
u,v -> vo,d -> u,v
u,v -> strf,vp -> u,v
vo -> rotational_wind -> divergence ~= 0
d  -> divergent_wind  -> vorticity ~= 0
strf -> rotational_wind -> streamfunction ~= strf up to l=0 constant
vp   -> divergent_wind  -> velocity potential ~= vp up to l=0 constant
laplacian(inverse_laplacian(zero_mean_field)) ~= zero_mean_field
```

Run representative identities on GL and CC, ascending and descending latitude, and common longitude conventions.

Test arbitrary leading dimensions.

## 3. SPHEREPACK/pyspharm parity

Use a dedicated optional parity environment with a maintained SPHEREPACK wrapper such as `pyspharm-syl` if it installs reliably.

Generate deterministic test fields at runtime. Do not add large checked-in NCL/SPHEREPACK arrays.

Compare overlapping operations where grid definitions align:

- Gaussian scalar transforms;
- regular-grid scalar transforms where the SPHEREPACK grid matches the CC case being tested;
- regridding;
- gradient;
- vorticity/divergence;
- streamfunction/velocity potential;
- wind reconstruction.

Before comparing, align:

- exact grid definition;
- latitude nodes/order;
- longitude origin/order;
- spectral truncation and `mmax`;
- radius;
- normalization;
- sign/component convention;
- precision.

Use explicit tolerances and explain any materially loose tolerance.

A parity job may use a narrower Python version than the production matrix.

---

# Metadata and grid tests

Test:

- canonical variable discovery: `u`, `v`, `vo`, `d`, `strf`, `vp`;
- CF `standard_name` discovery with non-canonical variable names;
- explicit variable overrides;
- ambiguity errors;
- output-name overrides preserving semantic metadata;
- canonical output attrs;
- rotational/divergent component metadata without invented CF names;
- metadata preservation through filter/regrid;
- generated latitude/longitude attrs;
- non-spatial coordinate preservation;
- CF time/calendar preservation when available;
- accessor/direct API equality.

Grid detection tests:

```text
GL ascending latitude
GL descending latitude
CC ascending latitude
CC descending latitude
0..360 longitude
-180..180 longitude
```

Reject:

- duplicate cyclic endpoint;
- non-global longitude span;
- regular latitude grid omitting a pole while presented as CC;
- latitudes that do not match GL nodes;
- irregular longitude spacing;
- missing/ambiguous spatial coordinates.

Verify that internal rolls/reversals are restored correctly.

---

# Error behavior

Prefer explicit errors over silent fallback.

Raise clear errors for:

- unsupported/non-global grids;
- ambiguous latitude/longitude coordinates;
- ambiguous physical variables;
- incompatible `u`/`v` shapes or coordinates;
- invalid spectral notation/range;
- mixed `spectral` and explicit `lmin/lmax` arguments;
- requested `lmax` beyond source/target capability;
- invalid taper value;
- inverse scalar source that cannot be identified;
- Dataset `wind()` with both complete source representations and no `source=`.

Do not replace useful DUCC errors with generic catch-all messages. Add context and preserve the original exception as the cause.

---

# Documentation and Read the Docs

Documentation is part of the first implementation, not a later cleanup task.

Follow `skills/documentation/SKILL.md` and use the working PyStormTracker Read the Docs setup as a reference, but keep spharmgrid's configuration smaller.

## Required documentation files

Create at least:

```text
README.md
.readthedocs.yaml

docs/conf.py
docs/index.md
docs/quickstart.md
docs/grids.md
docs/filtering.md
docs/regridding.md
docs/operators.md
docs/kinematics.md
docs/cf.md
docs/cli.md
docs/api.md
docs/references.md
```

Pages may be combined when that improves readability. Do not split one concept across several tiny pages.

## Documentation stack

Use:

```text
Sphinx
MyST Markdown
sphinx_rtd_theme
sphinx.ext.autodoc
sphinx.ext.napoleon
sphinx.ext.viewcode
```

Add other Sphinx extensions only when the docs actually need them.

Do not copy PyStormTracker's Mermaid/Node/PDF configuration unless spharmgrid documentation uses those features.

Keep docs dependencies in a lightweight `docs` group. Read the Docs should install with uv and build `docs/conf.py`.

Use `.readthedocs.yaml` version 2. Follow the same general uv install pattern as PyStormTracker:

```yaml
python:
  install:
    - method: uv
      command: sync
      groups:
        - docs

sphinx:
  configuration: docs/conf.py
```

Choose the RTD Python version from spharmgrid's actual supported matrix rather than copying a version blindly.

A strict local build must pass, for example:

```bash
uv run --group docs sphinx-build -W --keep-going -b html docs docs/_build/html
```

Adjust the exact command to the final uv group configuration and document the real command.

## README

Keep README concise. Include:

- package purpose;
- `ducc0` backend statement;
- GL/CC support;
- GitHub installation while unreleased;
- `.sg` accessor quick start;
- direct API equivalent;
- `T42`, `T6-42`, `taper=0.1` example;
- wind/kinematics example;
- link to full docs;
- references/citation note appropriate to the current GitHub-only state.

Do not add release badges/version promises that do not exist.

## User guide content

The built docs must explain:

### Quickstart

- import/register `.sg`;
- inspect grid;
- filter;
- regrid;
- combined filter+regrid;
- compute vorticity/divergence;
- reconstruct wind.

### Grids

- GL/Gauss–Legendre definition;
- CC/Clenshaw–Curtis definition;
- why a generic lat/lon grid is not automatically CC;
- coordinate order and longitude conventions;
- transform bandwidth constraints.

### Filtering

- `T42` and `T6-42` parsing;
- hard truncation default;
- `taper=None` default;
- `taper=0.1` meaning;
- Sardeshmukh–Hoskins formula and citation.

### Regridding

- GL/CC pairwise combinations;
- target-grid construction;
- why `T42` is not a target-grid name;
- combined regrid+filter single-cycle behavior.

### Operators

- gradient;
- Laplacian;
- inverse Laplacian;
- Earth radius;
- units;
- `l=0` convention.

### Kinematics

- `u/v -> vo/d`;
- `u/v -> strf/vp`;
- rotational/divergent wind;
- full inverse wind;
- semantic relationship to NCL/SPHEREPACK functions;
- sign/component/radius conventions needed to interpret results.

### CF/xarray

- canonical short names;
- CF `standard_name` lookup order;
- output metadata;
- optional `cf-xarray`;
- preserved time/calendars/non-spatial coordinates;
- Dask behavior.

### CLI

- minimal commands;
- NetCDF example;
- optional format engines;
- variable overrides.

### API reference

Every public top-level function/type and accessor method must be documented. Use autodoc to reduce duplication but keep scientific semantics in maintained prose/docstrings.

### References

Include the scientific/numerical sources used by the implementation. Verify bibliographic details and make DOI/publisher links clickable.

## Documentation wording

Write for atmospheric scientists and xarray users. Use direct technical/scientific prose, not agent instructions.

State clearly:

- spharmgrid is a wrapper/helper around DUCC0;
- DUCC0 performs the numerical transforms;
- NCL/SPHEREPACK are semantic/parity references;
- PyStormTracker is the source implementation for the initial extraction.

Do not claim that spharmgrid contains SPHEREPACK or introduces a new transform algorithm.

---

# Package examples that must work

## Scalar workflow

```python
import xarray as xr
import spharmgrid as sg

field = xr.open_dataarray("msl.nc")

field.sg.grid_type
# "gl" or "cc"

filtered = field.sg.filter("T6-42")
filtered_tapered = field.sg.filter("T6-42", taper=0.1)

target = sg.gaussian_grid(64, 128)
regridded = field.sg.regrid(target)

filtered_regridded = field.sg.regrid(
    target,
    spectral="T6-42",
    taper=0.1,
)
```

Direct equivalents:

```python
sg.filter(field, "T6-42", taper=0.1)
sg.regrid(field, target, spectral="T6-42", taper=0.1)
```

## ERA5-style wind workflow

```python
ds = xr.open_dataset("wind.nc")

vo = ds.sg.vorticity()
d = ds.sg.divergence()
kin = ds.sg.kinematics()
pot = ds.sg.potentials()
```

Expected defaults:

```text
kin: vo, d
pot: strf, vp
```

No `u="u", v="v"` arguments are needed.

## CF-name discovery

```python
ds["uwind"].attrs["standard_name"] = "eastward_wind"
ds["vwind"].attrs["standard_name"] = "northward_wind"

kin = ds.sg.kinematics()
```

## Inverse wind

```python
rot = ds["vo"].sg.rotational_wind()
div = ds["d"].sg.divergent_wind()

wind_from_vort_div = xr.Dataset({"vo": vo, "d": d}).sg.wind()
wind_from_potentials = xr.Dataset({"strf": strf, "vp": vp}).sg.wind()
```

## Output names

```python
vo = ds.sg.vorticity(output="vort")

kin = ds.sg.kinematics(
    vorticity="vort",
    divergence="div",
)

pot = ds.sg.potentials(
    streamfunction="psi",
    velocity_potential="chi",
)
```

Output variable names may change; physical CF semantics must not.

---

# Implementation sequence

## Phase 1 — repository/package foundation

- create `pyproject.toml` and `src/spharmgrid`;
- establish uv/Hatchling/Ruff/pytest/type-check setup;
- create `uv.lock`;
- add only core dependencies;
- register `.sg` DataArray/Dataset accessors;
- add `EARTH_RADIUS_M`;
- establish strict local docs build skeleton early so API docs evolve with code.

## Phase 2 — grids and coordinate semantics

- implement CF/canonical coordinate discovery;
- implement `Grid`;
- implement GL and CC constructors;
- implement grid detection;
- handle latitude order and longitude rolling correctly;
- add grid tests.

## Phase 3 — scalar SHT/filter/regrid core

- extract/simplify global GL/CC scalar analysis/synthesis from PyStormTracker;
- implement `SpectralRange`/parser;
- implement hard filtering;
- implement optional `taper=float`;
- implement same-grid filtering;
- implement pairwise GL/CC regridding;
- combine filtering+regridding in one analysis/synthesis cycle;
- add analytic tests.

## Phase 4 — scalar differential operators

- implement gradient;
- implement Laplacian;
- implement inverse Laplacian with zero `l=0` mode;
- test on analytic harmonics;
- compare with SPHEREPACK/pyspharm where practical.

## Phase 5 — vector transforms

- extract/simplify PyStormTracker spin-1 convention;
- implement `vorticity`, `divergence`, `kinematics`;
- implement `streamfunction`, `velocity_potential`, `potentials`;
- implement `vo/d -> wind`;
- implement `strf/vp -> wind`;
- implement single-source `rotational_wind`/`divergent_wind`;
- test signs, radius factors, and inverse identities aggressively.

## Phase 6 — xarray, CF, and Dask

- implement Dataset variable auto-detection;
- implement canonical output metadata;
- implement output-name overrides;
- preserve leading dimensions and time/calendars;
- add Dask coverage;
- add optional `cf-xarray` integration/tests.

## Phase 7 — external parity

- add optional SPHEREPACK/pyspharm parity environment;
- compare deterministic synthetic fields on overlapping GL/regular cases;
- document convention differences;
- keep comparison-library restrictions out of production support.

## Phase 8 — CLI

- implement thin `argparse` CLI;
- use xarray for I/O;
- add `info`, `filter`, `regrid`, `kinematics`, `potentials`, `wind`;
- test that CLI uses package functions and normal variable discovery.

## Phase 9 — complete docs

- finish README and full `docs/` user guide;
- finish API reference/autodoc;
- add `.readthedocs.yaml`;
- verify GitHub Markdown and Sphinx/MyST rendering;
- verify citations/links;
- run strict Sphinx build with warnings as errors;
- ensure every public function and accessor method is documented.

## Phase 10 — final review

- run the full ordinary test suite;
- run Ruff and configured type checks;
- run the docs build;
- run optional parity tests in their supported environment;
- inspect public namespace;
- remove copied PyStormTracker abstractions that are not needed;
- check accessor/direct equivalence;
- check that no excluded feature or release machinery was added.

---

# Acceptance criteria

The first implementation is complete when:

1. `import spharmgrid as sg` works from an installable `src/` package.
2. Importing spharmgrid registers `.sg` on xarray DataArray and Dataset objects without a known accessor conflict.
3. GL and CC grids are generated and detected correctly.
4. Scalar filtering supports `T42`, `T6-42`, explicit bounds, hard cutoff, and optional `taper=0.1` semantics.
5. `regrid()` supports GL/CC in every pairwise direction and combines filtering+regridding in one transform cycle.
6. Gradient, Laplacian, and inverse Laplacian pass analytic tests.
7. `u/v -> vo/d` passes analytic and independent parity checks where comparable.
8. `u/v -> strf/vp` and inverse wind transforms pass round-trip/parity checks.
9. `vo -> rotational_wind` and `d -> divergent_wind` satisfy the expected zero-divergence/zero-vorticity properties within justified numerical tolerance.
10. ERA5-style `u`, `v`, `vo`, `d`, `strf`, and `vp` names are detected without explicit arguments.
11. Exact CF `standard_name` metadata is detected when variable names differ.
12. Output CF metadata is correct and output-name overrides do not change physical semantics.
13. Arbitrary non-spatial xarray dimensions are preserved.
14. CF time/calendar coordinates pass through unchanged.
15. NumPy-backed and Dask-backed xarray inputs are covered.
16. Accessor and direct API calls are numerically equivalent.
17. The CLI is a thin xarray helper with no separate numerical implementation.
18. Core runtime dependencies remain small; `cf-xarray`, I/O engines, Dask, and SPHEREPACK parity tooling are optional as appropriate.
19. Analytic tests, round-trip tests, and the dedicated external parity suite are present.
20. `README.md` and the complete Sphinx/MyST user documentation describe the implemented API and scientific conventions.
21. `.readthedocs.yaml` and `docs/conf.py` are present and a strict local Read-the-Docs-equivalent Sphinx build passes.
22. Every public function/type and accessor method is included in the built API documentation.
23. Documentation clearly states the DUCC0 backend, GL/CC definitions, NCL/SPHEREPACK relationship, taper method/reference, CF semantics, and inverse zero-mode conventions.
24. Reduced Gaussian, HEALPix, regional DCT, MPI, alternate SHT backends, and raw public coefficient APIs have not entered the first public scope.
25. No release, tag, PyPI publishing, Zenodo setup, PyStormTracker dependency, or PyStormTracker source modification is part of this implementation.

---

# Final implementation rule

Start from the actual current code in `../PyStormTracker`. Extract the validated global GL/CC numerical pieces, remove tracker-specific coupling, and expose them through the xarray/CF API defined here.

Use NCL/SPHEREPACK as the external semantic/parity reference and `ducc0` as the numerical transform engine.

The package should remain understandable from this model:

```text
atmospheric field on GL/CC grid
    -> filter / regrid / differentiate
    -> wind kinematics / potentials / inverse wind
```

Anything outside that model needs a concrete current requirement before it becomes public API.
