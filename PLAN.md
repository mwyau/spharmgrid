# spharmgrid implementation plan

## Implementation target

Use **Luna Max** for the first implementation pass. This is a small package, but the work is not a small mechanical extraction: it combines spherical-harmonic numerics, GL/CC grid detection, xarray/Dask handling, CF metadata, vector-wind conventions, inverse transforms, a thin CLI, and cross-implementation parity tests. Terra is suitable for later cleanup or isolated follow-up changes, not for the initial implementation.

Do not release, tag, publish to PyPI, or add release automation as part of this plan.

The implementation agent may and should use the sibling repository at:

```text
../PyStormTracker
```

as the primary implementation reference. Read the current source there before porting code. Do not assume snippets in this plan are newer than the local checkout.

Relevant PyStormTracker files include at least:

```text
../PyStormTracker/src/pystormtracker/preprocessing/spectral.py
../PyStormTracker/src/pystormtracker/preprocessing/regrid.py
../PyStormTracker/src/pystormtracker/preprocessing/kinematics.py
../PyStormTracker/src/pystormtracker/models/geo.py
../PyStormTracker/src/pystormtracker/backends.py
```

Also inspect the corresponding tests and current repository instructions in `../PyStormTracker/AGENTS.md` when useful.

Do **not** modify `../PyStormTracker` as part of this task. Do **not** make PyStormTracker depend on spharmgrid. The first spharmgrid implementation is a separate package derived from the global spherical-harmonic layer already developed and tested in PyStormTracker.

---

## Goal

`spharmgrid` is a lightweight xarray-first wrapper/helper around [`ducc0`](https://pypi.org/project/ducc0/) for spherical-harmonic operations used in atmospheric and geophysical science.

The numerical spherical-harmonic transforms come from `ducc0`. spharmgrid supplies:

- grid recognition and coordinate handling for common global atmospheric grids;
- xarray integration;
- filtering and spectral regridding;
- scalar differential operators;
- vector-wind kinematics and inverse transforms;
- CF-aware variable discovery and output metadata;
- a small direct Python API matching the xarray accessor behavior;
- a minimal file-oriented CLI using xarray for I/O.

A central goal is to provide clear xarray-native equivalents of commonly used **NCL/SPHEREPACK spherical-harmonic workflows**, implemented with `ducc0` rather than wrapping SPHEREPACK.

This package is **not** a new spherical-harmonic transform implementation and should not present itself as one. It is an atmospheric-science operations layer over `ducc0`.

Suggested one-line description:

> Spherical-harmonic filtering, regridding, and atmospheric field operations for global xarray grids using ducc0.

---

## Scientific and software lineage

The initial implementation should be extracted/generalized from the global spherical-harmonic code in PyStormTracker, especially:

- scalar filtering in `preprocessing/spectral.py`;
- GL/CC spectral regridding in `preprocessing/regrid.py`;
- spin-1 vector transforms and vorticity/divergence in `preprocessing/kinematics.py`.

PyStormTracker already uses `ducc0.sht` for these operations. Preserve tested sign conventions and transform ordering where they are correct, then validate independently against analytic fields and SPHEREPACK/pyspharm.

Relevant scientific/numerical references should be cited in source documentation where the corresponding method is implemented:

- Sardeshmukh, P. D. and B. J. Hoskins, 1984: *Spatial Smoothing on the Sphere*, Monthly Weather Review, 112, 2524–2529. DOI: `10.1175/1520-0493(1984)112<2524:SSOTS>2.0.CO;2`.
- SPHEREPACK documentation and NCL spherical-harmonic functions for operation semantics and parity targets.
- Reinecke and Seljebotn, 2013, and Ishioka, 2018, for the numerical SHT lineage used by `ducc0` where appropriate.

Do not describe spharmgrid as implementing SPHEREPACK internally. State that its public operations correspond to common NCL/SPHEREPACK workflows while using `ducc0` for the transforms.

---

## Scope for the first implementation

### Supported grids

Support only two structured global grid families initially:

1. **Gauss–Legendre (GL)** full Gaussian grid
   - Gaussian latitudes;
   - constant number of equally spaced longitudes on every latitude;
   - rectangular `(lat, lon)` representation.

2. **Clenshaw–Curtis (CC)** global regular latitude–longitude grid
   - equally spaced latitude;
   - both poles included;
   - equally spaced longitude;
   - rectangular `(lat, lon)` representation.

Use **GL** and **CC** in prose, documentation, comments, and mathematical discussion. Use lowercase string values in Python:

```python
"gl"
"cc"
```

The canonical public grid terminology is GL/CC. Do not use `latlon` as a grid type because a generic regular latitude–longitude grid does not imply Clenshaw–Curtis sampling.

### Explicitly out of scope initially

Do not include these in the first public implementation:

- reduced Gaussian grids;
- HEALPix;
- PyStormTracker's regional DCT filter;
- polar stereographic output;
- arbitrary scattered/general synthesis locations;
- MPI/backend abstractions copied from PyStormTracker;
- a public raw `alm`/coefficient object model;
- an `SHCoeffs`/`SHGrid` hierarchy;
- a custom calendar/time system;
- a unit-conversion framework;
- automatic support for arbitrary regional lat/lon grids;
- arbitrary user-selectable SHT backends;
- release/tag/PyPI automation.

Reduced Gaussian support may be considered later. `ducc0` can represent ring-specific longitude counts, but a reduced Gaussian field does not naturally fit the initial rectangular xarray `(lat, lon)` contract. Do not complicate the first data model for it.

---

## Terminology

Use **field** for a gridded scalar or vector quantity.

Preferred terms:

```text
field        physical scalar/vector field represented by a DataArray/Dataset
grid         horizontal sampling geometry
target_grid  output horizontal grid for regridding
data          generic xarray/file content when no physical-field meaning is needed
```

Avoid **frame** in the spharmgrid public API and documentation. `frame` is useful inside tracking/time-step code such as PyStormTracker, but spharmgrid operates equally on a 2-D field or on an xarray object with arbitrary leading dimensions such as `(time, level, member, lat, lon)`.

Each horizontal slice is transformed independently.

---

# Public Python API

The package should support both:

```python
import spharmgrid as sg
```

and xarray accessors registered as:

```python
field.sg
 ds.sg
```

Use `.sg` for both `xarray.DataArray` and `xarray.Dataset` accessors.

The accessor implementation must be thin. Accessor methods call the same public/core functions used by the direct API. Do not duplicate numerical logic in accessor classes.

Suggested module layout:

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

Exact file names may change if a smaller layout is clearer, but keep numerical kernels separate from xarray accessor wrappers.

---

## Grid API

Use a small immutable public grid descriptor. Do not create a class hierarchy for GL versus CC unless implementation experience demonstrates a real need.

Suggested shape:

```python
@dataclass(frozen=True)
class Grid:
    kind: Literal["gl", "cc"]
    latitude: np.ndarray
    longitude: np.ndarray
```

The actual implementation may store compact grid parameters internally instead of arrays, but the public object must identify the grid type and exact output coordinates.

Public constructors:

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

Public detection:

```python
sg.detect_grid(field) -> Grid

field.sg.grid      # Grid
field.sg.grid_type # "gl" or "cc"
```

`target_grid` accepted by regridding should be either:

- a `Grid` returned by spharmgrid; or
- an xarray `DataArray`/`Dataset` whose horizontal GL/CC coordinates can be detected.

This allows:

```python
target = sg.gaussian_grid(64, 128)
out = field.sg.regrid(target)
```

and:

```python
out = field.sg.regrid(reference_field)
```

without inventing atmospheric resolution names as grid objects.

Do not make `"T42"` mean a physical grid. T42 is a spectral truncation, not a unique horizontal grid.

### Grid detection

Coordinate detection should support standard atmospheric xarray data without requiring `cf-xarray`:

1. identify latitude/longitude from CF metadata when present;
2. fall back to canonical coordinate names `lat`/`latitude` and `lon`/`longitude`;
3. use optional `cf-xarray` integration for broader CF coordinate/axis resolution when installed;
4. raise a clear error if the horizontal coordinates remain ambiguous.

Do not silently use the last two dimensions as latitude/longitude in the public package.

For CC detection:

- latitude must be equally spaced within numerical tolerance;
- both `-90` and `+90` degrees must be represented;
- longitude must be uniformly spaced and globally cyclic without a duplicate endpoint.

For GL detection:

- latitude values must match the appropriate Gauss–Legendre latitude nodes from `ducc0` within numerical tolerance, allowing either latitude order;
- longitude must be uniformly spaced and globally cyclic without a duplicate endpoint.

Support common longitude conventions such as `[0, 360)` and `[-180, 180)` by normalizing/rolling internally as needed. Preserve user-facing coordinate order/convention for same-grid operations. For new target grids, use the coordinates represented by `Grid`.

Reject grids that only approximately look global but do not satisfy a supported DUCC geometry. Do not classify an arbitrary regular latitude grid that omits the poles as CC.

Confirm exact GL/CC representable `lmax`/`mmax` constraints against the installed `ducc0` API and tests rather than copying an unverified formula. PyStormTracker's current constraints are a starting point, not the final specification.

---

## Spectral-range API

Expose a small immutable value type:

```python
@dataclass(frozen=True)
class SpectralRange:
    lmin: int
    lmax: int
```

and parser:

```python
sg.parse_spectral("T42")
# SpectralRange(lmin=0, lmax=42)

sg.parse_spectral("T6-42")
# SpectralRange(lmin=6, lmax=42)
```

Parsing should be case-insensitive:

```text
T42
t42
T6-42
t6-42
```

It is reasonable to normalize an en dash as well so copied atmospheric notation such as `T6–42` works.

Do not add rhomboidal or other spectral truncation notation in the first implementation.

Validate:

```text
0 <= lmin <= lmax
```

A public operation may receive either:

- a spectral string/value object; or
- explicit `lmin` and `lmax`.

Do not allow both forms in the same call.

---

## Scalar filtering

Accessor:

```python
field.sg.filter("T42")
field.sg.filter("T6-42")
field.sg.filter(lmin=6, lmax=42)
field.sg.filter("T6-42", taper=0.1)
```

Direct API:

```python
sg.filter(field, "T42")
sg.filter(field, "T6-42")
sg.filter(field, lmin=6, lmax=42)
sg.filter(field, "T6-42", taper=0.1)
```

Suggested signature:

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

### Filter semantics

Without taper:

```python
field.sg.filter("T6-42")
```

means a hard total-wavenumber band pass:

- coefficients with `l < 6` are zero;
- coefficients with `6 <= l <= 42` are unchanged;
- coefficients with `l > 42` are zero.

The default is:

```python
taper=None
```

There is no hidden smoothing.

### Taper

Support one taper only in the first implementation. The public parameter is simply:

```python
taper: float | None
```

Do not expose a taper-name enum or a second `taper_response` parameter.

`taper=0.1` applies the Sardeshmukh–Hoskins exponential spectral taper used by the existing PyStormTracker implementation, with `taper` equal to the response at `lmax`.

Document/reference Sardeshmukh and Hoskins (1984) in the implementation and API documentation, but keep the public call concise:

```python
field.sg.filter("T6-42", taper=0.1)
```

For total wavenumber `l`, use the existing global spherical formulation from PyStormTracker:

```text
w(l) = exp(-K [l(l+1)]^2)
K = -ln(taper) / [lmax(lmax+1)]^2
```

inside the retained band, so:

```text
w(lmax) = taper
```

and apply hard zero outside `[lmin, lmax]`.

Validate:

```text
0 < taper <= 1
```

`taper=1` is equivalent to unity weighting inside the retained band.

When `taper is None`, do not run the exponential weighting calculation; use an exact hard band mask.

Do not carry PyStormTracker's current `0.1` default into spharmgrid. The general library default is no taper.

---

## Spectral regridding

Accessor:

```python
field.sg.regrid(target_grid)
```

Direct:

```python
sg.regrid(field, target_grid)
```

Regridding should optionally filter in the **same analysis/synthesis cycle**:

```python
field.sg.regrid(
    target_grid,
    spectral="T6-42",
)

field.sg.regrid(
    target_grid,
    spectral="T6-42",
    taper=0.1,
)
```

Equivalent direct calls:

```python
sg.regrid(field, target_grid, spectral="T6-42")
sg.regrid(field, target_grid, spectral="T6-42", taper=0.1)
```

Also support explicit bounds:

```python
field.sg.regrid(target_grid, lmin=6, lmax=42, taper=0.1)
```

Suggested signature:

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
  -> one scalar SHT analysis
  -> optional band mask/taper
  -> one scalar SHT synthesis on target GL/CC grid
```

Do not implement `filter(...).regrid(...)` internally for the combined case because that performs an unnecessary second transform cycle.

When no spectral range is supplied, preserve all modes that can be represented by both source and target grids. Determine the maximum usable transform degree from the actual supported DUCC geometry constraints. Do not silently truncate to an arbitrary atmospheric convention.

If the user explicitly requests a range that cannot be represented by the source or target grid, raise a clear error rather than silently clamping it.

Supported initial combinations:

```text
GL -> GL
GL -> CC
CC -> GL
CC -> CC
```

---

# Scalar differential operators

Include the following field operations because they are basic spectral operators and correspond to common SPHEREPACK/pyspharm workflows.

## Gradient

Accessor:

```python
grad = field.sg.gradient()
```

Direct:

```python
grad = sg.gradient(field)
```

Return an `xr.Dataset` with default variables:

```text
gradient_eastward
gradient_northward
```

Allow output-name overrides:

```python
field.sg.gradient(
    eastward="dx",
    northward="dy",
)
```

Use spherical-harmonic differentiation/spin synthesis, not finite differences.

The result is the physical horizontal gradient on a sphere of radius `radius` in units per meter. Default:

```python
sg.EARTH_RADIUS_M = 6_371_220.0
```

matching the standard SPHEREPACK/NCL-style spherical Earth value already used by PyStormTracker. All kinematic/differential functions should accept:

```python
radius: float = sg.EARTH_RADIUS_M
```

or an equivalent keyword default.

## Laplacian

Accessor/direct:

```python
field.sg.laplacian()
sg.laplacian(field)
```

Apply the spectral eigenvalue:

```text
-l(l+1) / radius^2
```

for each spherical-harmonic coefficient.

## Inverse Laplacian

Accessor/direct:

```python
field.sg.inverse_laplacian()
sg.inverse_laplacian(field)
```

For `l > 0`, apply:

```text
-radius^2 / [l(l+1)]
```

The `l=0` mode is singular. Define the inverse as the zero-mean solution by setting the `l=0` output coefficient to zero and document this explicitly.

Do not expose raw coefficient arrays as part of these APIs.

---

# Vector-wind API

The vector API should cover the useful transform graph exposed by NCL/SPHEREPACK, but use descriptive Python names rather than NCL's abbreviated routine names.

The core relationships are:

```text
u, v
  -> relative vorticity
  -> divergence
  -> vorticity + divergence
  -> streamfunction + velocity potential

relative vorticity
  -> rotational wind

divergence
  -> divergent wind

relative vorticity + divergence
  -> full u, v

streamfunction
  -> rotational wind

velocity potential
  -> divergent wind

streamfunction + velocity potential
  -> full u, v
```

These correspond conceptually to NCL/SPHEREPACK operations such as `uv2vr`, `uv2dv`, `uv2vrdv`, `uv2sfvp`, `vr2uv`, `dv2uv`, `vrdv2uv`, and `sfvp2uv`, without copying those function names into spharmgrid.

---

## Dataset input-variable discovery

For Dataset accessor methods, conventional ERA5/ECMWF names and CF `standard_name` metadata should remove boilerplate from the common case.

Canonical quantities:

| Quantity | Default short name | CF `standard_name` |
| --- | --- | --- |
| eastward wind | `u` | `eastward_wind` |
| northward wind | `v` | `northward_wind` |
| relative vorticity | `vo` | `atmosphere_relative_vorticity` |
| divergence | `d` | `divergence_of_wind` |
| streamfunction | `strf` | `atmosphere_horizontal_streamfunction` |
| velocity potential | `vp` | `atmosphere_horizontal_velocity_potential` |

Resolution order for a requested quantity:

```text
explicit variable argument
  -> unique matching CF standard_name
  -> canonical short name
  -> error
```

Examples:

```python
ds.sg.vorticity()
```

should find canonical `u` and `v` automatically.

This should also work:

```python
ds["uwind"].attrs["standard_name"] = "eastward_wind"
ds["vwind"].attrs["standard_name"] = "northward_wind"

ds.sg.vorticity()
```

Explicit overrides remain available:

```python
ds.sg.vorticity(u="ua", v="va")
```

If multiple variables match the same CF semantic quantity, raise an ambiguity error listing the candidates and require an explicit variable name.

Do not maintain a large heuristic alias table such as `UGRD`, `uwnd`, `vor`, `vort`, etc. CF metadata and the canonical short names are the initial contract.

The direct API receives DataArrays directly and therefore does not need Dataset variable-name discovery.

---

## Vorticity

Accessor:

```python
vo = ds.sg.vorticity()
vo = ds.sg.vorticity(output="vort")
vo = ds.sg.vorticity(u="ua", v="va", output="vort")
```

Direct:

```python
vo = sg.vorticity(u, v)
vo = sg.vorticity(u, v, output="vort")
```

Default output name:

```text
vo
```

Return an `xr.DataArray`.

## Divergence

Accessor:

```python
d = ds.sg.divergence()
```

Direct:

```python
d = sg.divergence(u, v)
```

Default output name:

```text
d
```

Return an `xr.DataArray`.

## Combined kinematics

Preferred call when both are required:

```python
kin = ds.sg.kinematics()
```

Direct:

```python
kin = sg.kinematics(u, v)
```

Return:

```text
Dataset
  vo
  d
```

Allow output names:

```python
kin = ds.sg.kinematics(
    vorticity="vort",
    divergence="div",
)
```

and explicit input names on Dataset accessors:

```python
kin = ds.sg.kinematics(
    u="ua",
    v="va",
    vorticity="vort",
    divergence="div",
)
```

Compute both from one vector analysis whenever possible.

---

## Streamfunction and velocity potential

Individual accessors:

```python
strf = ds.sg.streamfunction()
vp = ds.sg.velocity_potential()
```

Direct:

```python
strf = sg.streamfunction(u, v)
vp = sg.velocity_potential(u, v)
```

Default names:

```text
strf
vp
```

Combined preferred call:

```python
pot = ds.sg.potentials()
```

Direct:

```python
pot = sg.potentials(u, v)
```

Return:

```text
Dataset
  strf
  vp
```

Allow output-name overrides:

```python
ds.sg.potentials(
    streamfunction="psi",
    velocity_potential="chi",
)
```

Use one vector analysis for the combined result.

---

## Rotational wind

The scientific term is **rotational wind** / **rotational wind component**.

Support input from relative vorticity:

```python
vo.sg.rotational_wind()
sg.rotational_wind(vo)
```

and from streamfunction:

```python
strf.sg.rotational_wind()
sg.rotational_wind(strf)
```

Determine which scalar quantity was supplied from:

1. explicit `quantity=` override when given;
2. CF `standard_name`;
3. canonical short name.

If it cannot be identified, require:

```python
sg.rotational_wind(field, quantity="vorticity")
```

or:

```python
sg.rotational_wind(field, quantity="streamfunction")
```

Default outputs:

```text
u_rotational
v_rotational
```

Return an `xr.Dataset`.

---

## Divergent wind

Support input from divergence:

```python
d.sg.divergent_wind()
sg.divergent_wind(d)
```

and from velocity potential:

```python
vp.sg.divergent_wind()
sg.divergent_wind(vp)
```

Use the same semantic detection/override approach as `rotational_wind`.

Default outputs:

```text
u_divergent
v_divergent
```

Return an `xr.Dataset`.

---

## Full wind reconstruction

Dataset accessor:

```python
wind = ds.sg.wind()
```

When the Dataset contains only one complete supported source pair, detect it automatically:

```text
vo + d      -> u + v
strf + vp   -> u + v
```

Return:

```text
Dataset
  u
  v
```

If both complete source representations are present, do not guess. Require:

```python
ds.sg.wind(source="vorticity_divergence")
```

or:

```python
ds.sg.wind(source="potentials")
```

Direct API should accept the two source DataArrays and infer their semantic type from metadata/names, with an explicit `source=` escape hatch when needed:

```python
sg.wind(vo, d)
sg.wind(strf, vp)
```

Allow full-wind output-name overrides if needed:

```python
ds.sg.wind(u="ua", v="va")
```

For Dataset accessors where `u` and `v` would otherwise mean source-variable arguments, use distinct internal/signature names if necessary to avoid ambiguity. Prefer API clarity over forcing a single shared signature for all accessors.

---

# Vector numerical implementation

Use the existing PyStormTracker spin-1 implementation as the starting point, then validate signs and normalizations against analytic tests and SPHEREPACK parity.

PyStormTracker currently forms the DUCC spin-1 vector map as:

```python
vec_map = np.stack((-v, u), axis=0)
```

and analyzes it with `ducc0.sht.analysis_2d(..., spin=1)`.

Its current vorticity/divergence scaling is based on:

```text
sqrt(l(l+1)) / radius
```

with E/B coefficient sign conventions tied to DUCC's vector transform ordering.

Do not casually rewrite these signs from memory. Port the tested convention, then prove it with cross-library and analytic tests.

For scalar potentials, use the standard spherical identities:

```text
horizontal wind = rotational part + divergent part

vorticity  = Laplacian(streamfunction)
divergence = Laplacian(velocity potential)
```

under the package's adopted sign convention. In spectral space, for `l > 0`:

```text
psi_lm = -radius^2 / [l(l+1)] * vort_lm
chi_lm = -radius^2 / [l(l+1)] * div_lm
```

where `psi` is streamfunction and `chi` is velocity potential if the conventional Laplacian eigenvalue `-l(l+1)/radius^2` is used.

Set the `l=0` potential coefficients to zero because the additive constant is undefined.

For inverse vector reconstruction from vorticity/divergence, invert the same DUCC E/B scaling used by the forward path and synthesize with `spin=1`. Verify that the returned components are converted back from DUCC mathematical-vector ordering to geographic `(u, v)` consistently.

For reconstruction from streamfunction/velocity potential:

1. scalar-analyze the potential field(s);
2. apply the spherical Laplacian to obtain vorticity/divergence coefficients;
3. use the same vector synthesis path as the vorticity/divergence inverse.

For a single streamfunction, set divergence coefficients to zero. For a single velocity potential, set vorticity coefficients to zero.

Avoid unnecessary analyze/synthesize cycles. Build internal kernels that can produce multiple related outputs from the same spectral coefficients.

---

# CF metadata and variable semantics

CF-compatible output metadata is part of the package contract.

Do not add a required CF metadata library just to set six known quantities. Keep a small internal metadata table for canonical atmospheric outputs.

At minimum:

| Default name | `standard_name` | `long_name` | units |
| --- | --- | --- | --- |
| `u` | `eastward_wind` | Eastward wind | `m s-1` |
| `v` | `northward_wind` | Northward wind | `m s-1` |
| `vo` | `atmosphere_relative_vorticity` | Relative vorticity | `s-1` |
| `d` | `divergence_of_wind` | Divergence of wind | `s-1` |
| `strf` | `atmosphere_horizontal_streamfunction` | Horizontal streamfunction | `m2 s-1` |
| `vp` | `atmosphere_horizontal_velocity_potential` | Horizontal velocity potential | `m2 s-1` |

Verify the exact current CF standard names against the official CF Standard Name Table during implementation and test them as literals. Do not invent standard names.

An output-name override changes only `DataArray.name`, not the physical metadata:

```python
vo = ds.sg.vorticity(output="vort")
```

still has:

```text
standard_name = atmosphere_relative_vorticity
```

### Rotational/divergent component metadata

For `u_rotational`, `v_rotational`, `u_divergent`, and `v_divergent`, check the current CF table for exact semantic names. If CF does not provide exact standard names, omit `standard_name` and use accurate `long_name` plus `units="m s-1"` rather than assigning the broader `eastward_wind`/`northward_wind` names to a component with a more specific meaning.

### Same-quantity operations

For filtering and regridding, the physical quantity has not changed. Preserve variable metadata such as:

- `name`;
- `standard_name`;
- `long_name`;
- `units`;
- other non-conflicting user attributes.

Do not overwrite existing physical metadata with generic spharmgrid metadata.

### Coordinates

For generated target coordinates, attach normal CF coordinate attrs:

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

Use the actual coordinate names chosen for the returned object.

Do not invent a CF standard name for GL or CC sampling. Grid family is a numerical property, not a different physical latitude coordinate.

The grid can be rediscovered from the coordinate values. Avoid adding unnecessary custom global attrs unless a concrete need appears.

---

# Optional cf-xarray integration

`cf-xarray` should be an optional dependency, not a core requirement.

Suggested optional dependency group:

```toml
[project.optional-dependencies]
cf = ["cf-xarray"]
```

Core spharmgrid must understand its small set of required `standard_name` values directly from `.attrs`, so the common API works without `cf-xarray`.

When `cf-xarray` is installed, it may be used to improve coordinate/axis discovery and interoperability with more complex CF datasets. The numerical result for an otherwise identical valid dataset must not change merely because `cf-xarray` is installed.

Add tests both with the core path and, where useful, with the optional integration.

---

# CF time and non-spatial dimensions

Do not implement a spharmgrid time representation.

xarray owns CF decoding/encoding of time and calendars. spharmgrid should preserve non-spatial dimensions and coordinates unchanged while applying each transform over the horizontal dimensions.

This should work for fields shaped like:

```text
(lat, lon)
(time, lat, lon)
(level, lat, lon)
(time, level, lat, lon)
(member, time, level, lat, lon)
```

and other leading-dimension combinations.

Preserve xarray/cftime semantics including, when present:

- NumPy datetime values;
- `cftime` values;
- proleptic Gregorian calendars;
- no-leap calendars;
- 360-day calendars.

Do not convert timestamps to Unix time, milliseconds, or a package-specific representation.

Do not add `cftime` as a direct core dependency solely for spharmgrid. Let xarray's I/O stack resolve it when required by a user's dataset.

---

# xarray and Dask behavior

The xarray API is the primary user interface, but all numerical kernels should operate on NumPy arrays and be applied over xarray objects with `xr.apply_ufunc` or an equally clear mechanism.

Requirements:

- transform only the detected horizontal dimensions;
- preserve all leading dimensions;
- work eagerly with NumPy-backed DataArrays;
- work lazily with Dask-backed DataArrays when Dask is installed;
- do not require Dask as a core dependency solely to support xarray;
- rechunk horizontal core dimensions only when needed and make the requirement/error clear;
- preserve input dtype where scientifically/numerically appropriate, but do not sacrifice transform correctness to force float32.

Do not copy PyStormTracker's public `backend="serial"|"dask"|"mpi"` abstraction. spharmgrid does not need a backend selector.

Expose only a small DUCC thread control:

```python
nthreads: int | None = None
```

The implementation should avoid obvious Dask × DUCC oversubscription. A reasonable default policy is:

- eager field and `nthreads is None`: allow DUCC's normal thread behavior;
- Dask-backed field and `nthreads is None`: use one DUCC thread per Dask task;
- explicit `nthreads`: honor it.

Verify the exact `ducc0` convention for `nthreads=0`/default in the installed version instead of assuming it.

No MPI support in spharmgrid initially.

---

# Direct API and accessor parity

The direct and accessor APIs must call the same implementation and return equivalent xarray objects.

Examples:

```python
sg.filter(field, "T6-42", taper=0.1)
field.sg.filter("T6-42", taper=0.1)
```

must be equivalent.

Likewise:

```python
sg.vorticity(ds.u, ds.v)
ds.sg.vorticity()
```

must differ only in how inputs were selected, not in numerical implementation.

Test this explicitly.

The accessor classes should contain input selection, light argument normalization, and calls to public/core functions. They should not contain SHT mathematics.

---

# Minimal CLI

The CLI is a helper for file-oriented use, not the primary API and not a separate data model.

Use the Python standard library `argparse` unless there is a concrete reason to add a CLI dependency.

Initial commands:

```text
spharmgrid info
spharmgrid filter
spharmgrid regrid
spharmgrid kinematics
spharmgrid potentials
spharmgrid wind
```

`info` should report detected spatial coordinates, GL/CC grid type, shape, and supported transform degree information useful for diagnosis.

Examples:

```bash
spharmgrid info input.nc

spharmgrid filter input.nc output.nc \
    --var msl \
    --spectral T6-42 \
    --taper 0.1

spharmgrid kinematics wind.nc output.nc
```

Do not require `--u u --v v` for the normal ERA5/CF case. Use the same Dataset variable-discovery logic as the xarray accessor.

Allow explicit variable overrides when needed.

### CLI I/O

Delegate file handling to xarray. Do not build a custom file abstraction.

At minimum, support normal NetCDF through `xr.open_dataset`/`to_netcdf` when the user's installed xarray engine can handle it. Add thin suffix/engine handling for Zarr or GRIB only if it stays small and can be covered by optional dependencies.

Suggested optional groups may include:

```toml
io = [
    "netCDF4",
    "zarr",
    "cfgrib",
]
```

but do not make all I/O engines core dependencies.

For GRIB, reading through xarray/cfgrib and writing a different xarray-supported format is sufficient initially. Do not implement GRIB encoding.

The CLI should call the same public Python functions/accessors. No separate numerical paths.

---

# Core dependencies

Keep the required dependency set small:

```text
numpy
xarray
ducc0
```

Add only packaging/runtime dependencies that are actually required by the implementation.

Optional:

```text
cf-xarray       CF integration
Dask            lazy execution through xarray
NetCDF/Zarr/GRIB engines as I/O extras
pyspharm-syl    parity-test environment only
```

Do not add Pint/unit libraries initially. The package should expect wind kinematics in `m s-1` for canonical atmospheric outputs and document this clearly. If unit conversion becomes a real use case, it can be added later as optional integration.

---

# Testing strategy

Testing must establish both mathematical correctness and compatibility with established atmospheric spherical-harmonic workflows.

Do **not** add large checked-in NCL/SPHEREPACK numerical reference arrays.

Use three layers.

## 1. Analytic tests

Construct deterministic scalar spherical harmonics and simple vector fields with known behavior.

Cover at least:

- scalar GL analysis/synthesis round trips;
- scalar CC analysis/synthesis round trips;
- GL -> CC and CC -> GL regridding;
- hard `Tn` truncation;
- `Tn-m` band pass;
- taper endpoint response, including verifying `w(lmax) == taper` numerically;
- gradient of known harmonics;
- Laplacian eigenvalue `-l(l+1)/R^2`;
- inverse-Laplacian zero-mean convention;
- purely rotational wind with near-zero divergence;
- purely divergent wind with near-zero vorticity;
- forward/inverse wind reconstruction;
- streamfunction/vorticity relation;
- velocity-potential/divergence relation.

Use low-degree harmonics where expected values can be reasoned about directly. Do not rely only on random fields.

## 2. Internal identities and round trips

Test identities such as:

```text
u,v -> vo,d -> u,v
u,v -> strf,vp -> u,v
vo -> rotational_wind -> divergence ~= 0
d  -> divergent_wind  -> vorticity ~= 0
strf -> rotational_wind -> streamfunction ~= strf up to the l=0 constant
vp   -> divergent_wind  -> velocity potential ~= vp up to the l=0 constant
laplacian(inverse_laplacian(zero_mean_field)) ~= zero_mean_field
```

Test across GL and CC grids and both ascending/descending latitude coordinate order.

Also test non-spatial vectorization over time/level/member dimensions.

## 3. SPHEREPACK/pyspharm parity

Use a dedicated optional test environment with a maintained SPHEREPACK wrapper such as `pyspharm-syl` if it installs reliably on a supported CI Python version.

The parity dependency should not constrain spharmgrid's production Python support. If the oracle only builds on a subset such as one Python version, run the parity job there.

Compare deterministic synthetic fields for the overlapping operations:

- regular/CC-equivalent scalar transforms where the grid definitions align;
- Gaussian scalar transforms;
- regridding;
- gradient;
- vorticity/divergence;
- streamfunction/velocity potential;
- wind reconstruction.

SPHEREPACK/pyspharm uses its own regular-grid definitions and orientation conventions. Match the grid and coordinate conventions explicitly before comparing; do not treat a mismatch in grid definition as a numerical error.

Use tolerances justified by transform normalization, precision, and grid type. Record why any looser tolerance is required.

The current PyStormTracker NCL/SPHEREPACK parity work may be used to understand expected conventions, but spharmgrid tests should generate their own deterministic inputs instead of depending on PST's checked-in reference datasets.

---

# Metadata tests

Test:

- auto-detection from canonical names `u`, `v`, `vo`, `d`, `strf`, `vp`;
- auto-detection from CF `standard_name` with non-canonical variable names;
- explicit input-variable override;
- ambiguity errors when multiple variables match;
- output-name overrides preserving `standard_name`;
- canonical CF attrs for `u`, `v`, `vo`, `d`, `strf`, `vp`;
- no invented/incorrect `standard_name` for rotational/divergent components;
- same-quantity metadata preservation through filter/regrid;
- generated latitude/longitude CF metadata;
- time/level/member coordinate preservation;
- cftime/calendar preservation in an optional test environment when available.

---

# Grid tests

Test grid detection against exact generated grids and representative coordinate layouts:

```text
GL ascending latitude
GL descending latitude
CC ascending latitude
CC descending latitude
0..360 longitude
-180..180 longitude
```

Reject:

- duplicate cyclic longitude endpoint;
- non-global longitude span;
- regular latitude grid that omits one/both poles but is presented as CC;
- latitude values that do not match GL nodes;
- irregular longitude spacing;
- ambiguous/missing spatial coordinates.

Test that internal rolling/reordering is reversed correctly in outputs.

---

# Filter/regrid implementation notes from PyStormTracker

The existing PyStormTracker scalar SHT path already contains the essential implementation:

```text
map
 -> ducc0.sht.analysis_2d(... geometry="CC"/"GL")
 -> coefficient mask/taper
 -> ducc0.sht.synthesis_2d(... target geometry)
```

Port and simplify the **global GL/CC** logic only.

Do not port these PyStormTracker concerns into spharmgrid:

- reduced-grid `pseudo_analysis` in the first implementation;
- HEALPix synthesis;
- regional DCT filtering;
- polar stereographic synthesis;
- tracker-specific `Backend` selection;
- tracker-specific latitude/domain conventions;
- PST data-loader dependencies.

Replace `DataLoader` coupling with direct xarray/CF coordinate discovery appropriate to this package.

Filtering and regridding should share one scalar SHT kernel so `regrid(..., spectral=...)` does not duplicate work.

---

# Error behavior

Prefer explicit scientific/data-model errors over silent fallback.

Raise clear `ValueError`/`TypeError` messages for:

- unsupported grid;
- non-global grid;
- ambiguous latitude/longitude coordinates;
- ambiguous input variables;
- incompatible `u`/`v` dimensions or coordinates;
- invalid spectral range;
- user specifying both `spectral` and `lmin`/`lmax`;
- requested `lmax` beyond source/target capability;
- invalid taper response;
- inverse operation whose scalar quantity cannot be identified;
- Dataset `wind()` with multiple complete source representations and no `source=`.

Do not catch every DUCC exception and replace it with a generic message if doing so hides useful numerical diagnostics. Add context while preserving the original exception as the cause.

---

# Documentation to create during implementation

Keep documentation small for the first pass. At minimum create:

```text
README.md
```

with:

- what spharmgrid is;
- statement that `ducc0` is the numerical SHT backend;
- GL/CC definitions;
- quick xarray accessor examples;
- direct API equivalents;
- filter/regrid examples including `T6-42` and `taper=0.1`;
- vector examples;
- CF variable auto-detection;
- short note on NCL/SPHEREPACK operation correspondence;
- references.

If API documentation grows beyond a readable README, add a small `docs/` tree later. Do not build a large documentation framework just to complete the first implementation.

Comments/docstrings should explain transform conventions, grid assumptions, signs, coefficient scaling, and scientific lineage. Avoid comments that only restate the code.

---

# Package/API examples that should work

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

No `u="u", v="v"` arguments should be needed.

## CF-name discovery

```python
ds["uwind"].attrs["standard_name"] = "eastward_wind"
ds["vwind"].attrs["standard_name"] = "northward_wind"

kin = ds.sg.kinematics()
```

## Inverse wind operations

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

Physical `standard_name` metadata must remain correct regardless of the output variable name.

---

# Suggested implementation sequence

## Phase 1: package skeleton and grid model

- create `pyproject.toml` and `src/spharmgrid` layout;
- add core dependencies only;
- register `.sg` DataArray/Dataset accessors;
- implement CF/canonical coordinate discovery;
- implement GL/CC grid generation and detection;
- add grid tests;
- add `EARTH_RADIUS_M = 6_371_220.0`.

## Phase 2: scalar SHT core

- extract/simplify global GL/CC scalar analysis/synthesis from PyStormTracker;
- implement `SpectralRange` and `parse_spectral`;
- implement hard filter;
- implement optional `taper=float` using the existing Sardeshmukh–Hoskins form;
- implement same-grid filter;
- implement GL/CC regrid and combined filter+regrid;
- add analytic and PST-behavior tests.

## Phase 3: scalar differential operators

- implement gradient;
- implement Laplacian;
- implement inverse Laplacian with zero `l=0` mode;
- validate on analytic harmonics and against SPHEREPACK/pyspharm where practical.

## Phase 4: vector transforms

- extract and simplify the PyStormTracker spin-1 vector analysis convention;
- implement `vorticity`, `divergence`, `kinematics`;
- implement `streamfunction`, `velocity_potential`, `potentials`;
- implement vorticity/divergence -> wind;
- implement streamfunction/velocity-potential -> wind;
- implement single-source `rotational_wind` and `divergent_wind`;
- test forward/inverse identities and signs aggressively.

## Phase 5: xarray semantics and metadata

- add Dataset variable auto-detection;
- add canonical output metadata;
- add output-name overrides;
- ensure all non-spatial dimensions/coordinates survive;
- add Dask coverage;
- add optional `cf-xarray` integration and tests.

## Phase 6: SPHEREPACK parity

- add an optional parity-test dependency/environment;
- compare deterministic GL/Gaussian and regular-grid cases where definitions align;
- document any expected convention differences;
- keep oracle limitations out of the production dependency matrix.

## Phase 7: minimal CLI and README

- implement thin `argparse` CLI;
- use xarray directly for file I/O;
- add `info`, `filter`, `regrid`, `kinematics`, `potentials`, `wind`;
- write concise README examples and scientific references.

## Phase 8: cleanup

- run tests/lint/type checks;
- remove copied PST abstractions that are not needed;
- inspect the public namespace and keep it small;
- verify accessor/direct API parity;
- verify no release/tag/PyPI work was added.

---

# Acceptance criteria

The initial implementation is complete when all of the following are true:

1. `import spharmgrid as sg` works from a normal editable/installable package.
2. Importing spharmgrid registers `.sg` on xarray DataArray and Dataset objects without known namespace conflicts.
3. GL and CC grids are generated and detected correctly.
4. Scalar filtering supports `T42`, `T6-42`, explicit bounds, hard cutoff, and optional `taper=0.1` semantics.
5. `regrid()` supports GL/CC in every pairwise direction and combines regridding + filtering in one transform cycle.
6. Gradient, Laplacian, and inverse Laplacian pass analytic tests.
7. `u/v -> vo/d` passes analytic and SPHEREPACK/pyspharm parity checks.
8. `u/v -> strf/vp` and the inverse wind transforms pass round-trip/parity checks.
9. `vo -> rotational_wind` and `d -> divergent_wind` have the expected zero-divergence/zero-vorticity properties within numerical tolerance.
10. ERA5-style `u`, `v`, `vo`, `d`, `strf`, and `vp` names are detected without explicit arguments.
11. CF `standard_name` metadata is detected when variable names differ.
12. Output CF metadata is correct and output-name overrides do not alter physical semantics.
13. Arbitrary non-spatial xarray dimensions are preserved.
14. CF time/calendar coordinates pass through unchanged.
15. NumPy-backed and Dask-backed xarray inputs are both covered.
16. Accessor and direct API calls are numerically equivalent.
17. The CLI is a thin xarray helper and has no separate numerical implementation.
18. Core dependencies remain small; `cf-xarray` and SPHEREPACK parity tooling are optional.
19. Reduced Gaussian, HEALPix, regional DCT, MPI, and raw public coefficient APIs have not leaked into the initial scope.
20. No release, tag, PyPI publishing, or PyStormTracker dependency change is part of this implementation.

---

# Public namespace target

Keep the top-level namespace close to this set:

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

Do not expose internal DUCC coefficient helpers, xarray gufunc kernels, E/B coefficient arrays, metadata registries, or accessor implementation classes unless there is a concrete user need.

---

# Final implementation guidance

Start by reading the actual current code in `../PyStormTracker`, not by rewriting the mathematics from this plan. Extract the parts that are already validated, remove tracker-specific coupling, then broaden them into the xarray/CF API defined here.

Use NCL/SPHEREPACK as the external semantic/parity reference and `ducc0` as the numerical transform engine.

The package should remain small enough that a user can understand its purpose from the public API:

```text
atmospheric field on GL/CC grid
    -> filter / regrid / differentiate
    -> wind kinematics / potentials / inverse wind
```

Anything outside that model should need a clear reason before it becomes public API.
