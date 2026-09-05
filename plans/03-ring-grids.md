# Phase 3: reduced Gaussian and HEALPix SHT grids

## Goal

Extend spharmgrid beyond full rectangular GL/CC fields to global
isolatitude-ring grids that are important in atmospheric and spherical data:

```text
reduced Gaussian
HEALPix
```

Continue using DUCC0 as the only production SHT engine in this phase.

This remains an SHT feature. Do not add generic interpolation methods.

The key design change is that not every supported grid can be represented by a
rectangular `(lat, lon)` array with the same number of longitudes on every
latitude.

---

## 1. Research/verification gate before API changes

Before changing the public grid model, verify against the installed DUCC0
version:

- accepted general isolatitude-ring geometry arguments;
- scalar and spin transform support for non-rectangular ring grids;
- which operations are exact analysis, adjoint analysis, or iterative/pseudo
  analysis;
- how `lmax`, `mmax`, ring sampling and quadrature constrain accuracy;
- HEALPix ring geometry returned by `ducc0.healpix.Healpix_Base.sht_info()`;
- current `pseudo_analysis` / iterative least-squares behavior;
- whether a reduced Gaussian grid has a directly supported exact weighted
  analysis for the intended truncation or requires an iterative solve.

Do not describe an analysis as exact because the synthesis is exact.

Keep these concepts explicit:

```text
synthesis
    evaluate a band-limited harmonic series on target points/rings

adjoint analysis
    adjoint of synthesis; not automatically an exact inverse

quadrature/exact analysis
    coefficients recovered by a valid sampling/quadrature theorem

iterative/pseudo analysis
    approximate least-squares recovery to a stated tolerance
```

---

## 2. Replace the rectangular-only grid representation

The current public `Grid` is appropriate for GL/CC but cannot faithfully model
reduced Gaussian or HEALPix geometry.

Do not stretch it into a dataclass full of unrelated optional fields.

At this phase a small explicit grid type split becomes justified.

One acceptable direction is:

```python
Grid = RectangularGrid | RingGrid

@dataclass(frozen=True)
class RectangularGrid:
    kind: Literal["gl", "cc"]
    latitude: np.ndarray
    longitude: np.ndarray

@dataclass(frozen=True)
class RingGrid:
    kind: Literal["reduced_gaussian", "healpix"]
    theta: np.ndarray
    nphi: np.ndarray
    phi0: np.ndarray
    ringstart: np.ndarray
    npix: int
```

The exact public names may differ after reviewing current code.

Requirements:

- preserve the simple GL/CC API;
- do not expose DUCC's packed internal arrays unnecessarily;
- provide enough information to reconstruct xarray coordinates and map
  ordering;
- make grid equality/equivalence explicit;
- make backend transform geometry derivable without inspecting xarray data.

Because spharmgrid is still pre-1.0, prefer a coherent grid model over
preserving a weak abstraction solely for compatibility.

---

## 3. xarray representation

### Rectangular GL/CC

Keep:

```text
(..., lat, lon)
```

### Reduced Gaussian

Use a one-dimensional horizontal cell/point dimension for packed fields:

```text
(..., cell)
```

with auxiliary coordinates such as:

```text
latitude(cell)
longitude(cell)
```

and enough grid metadata to recover ring boundaries.

Do not force a ragged reduced grid into a rectangular array padded with NaNs.

### HEALPix

Use:

```text
(..., pixel)
```

with grid metadata containing at least:

```text
nside
ordering
```

Prefer RING ordering internally for DUCC SHT work.

If NESTED input is supported, reorder data and coordinates together and return
the user's requested/original ordering when appropriate.

Do not treat a HEALPix pixel index as latitude or longitude.

---

## 4. Reduced Gaussian support

### Scope

Support atmospheric reduced Gaussian grids as true SHT sampling geometries.

Initial target:

- accept explicit ring geometry / `pl` information from decoded datasets;
- detect a supported reduced Gaussian grid from coordinates plus ring
  structure when unambiguous;
- construct common ECMWF-style reduced Gaussian geometries only after the
  N/O naming and ring-count algorithms are verified from authoritative
  sources.

Do not guess `Nxxx` or `Oxxx` from array size alone.

### Operations

The final target is to make the Phase 2 physical operation set available on
reduced Gaussian fields where the analysis method is numerically justified:

```text
filter
regrid
regrid_vector

gradient
inverse_gradient
laplacian
inverse_laplacian
vector_laplacian
inverse_vector_laplacian

vorticity
divergence
kinematics
potentials
helmholtz
rotational_wind
divergent_wind
wind
```

If reliable analysis from a reduced grid requires an iterative solver, make
that solver/tolerance behavior explicit rather than silently pretending the
grid has GL quadrature.

### Regridding

Support, as justified by analysis capability:

```text
GL/CC -> reduced Gaussian
reduced Gaussian -> GL/CC
reduced Gaussian -> reduced Gaussian
reduced Gaussian <-> HEALPix
```

All are spectral regrids:

```text
analysis -> optional spectral operation -> synthesis
```

No bilinear fallback.

---

## 5. HEALPix support

Use DUCC0's HEALPix geometry helper rather than implementing HEALPix indexing
or ring geometry independently.

Public constructor target:

```python
sg.healpix_grid(
    nside: int,
    *,
    ordering: Literal["ring", "nested"] = "ring",
)
```

The exact name can be revised for consistency.

### Analysis semantics

HEALPix does not provide an exact spherical-harmonic sampling theorem in the
same sense as a quadrature-exact GL transform.

Therefore distinguish:

```text
HEALPix synthesis
    direct evaluation on HEALPix rings

HEALPix analysis
    adjoint or iterative/pseudo analysis with documented accuracy
```

Default to the analysis mode that gives a scientifically useful reconstruction
without surprising users about its cost.

If an iteration count/tolerance is public, use explicit numerical names such
as:

```text
max_iterations
rtol
```

Do not hide iteration behind an unrelated `accuracy=True` boolean.

### Vector/spin operations

Verify spin-1 analysis/synthesis on HEALPix independently before enabling wind
kinematics.

Do not infer vector component orientation from scalar HEALPix parity.

---

## 6. Transform-accuracy API

Do not force every grid through one false "exact" abstraction.

A transform/grid pair should know or report whether analysis is:

```text
quadrature
iterative
adjoint-only
```

This information can remain internal if no user choice is required, but
documentation must state it.

If users can control iterative analysis, keep the control on operations that
perform analysis rather than on grid construction.

Do not attach an iterative-analysis option to pure synthesis-only work.

---

## 7. Tests

### Geometry tests

For every new grid type:

- ring count;
- point/pixel count;
- longitude phase;
- ring offsets;
- latitude/colatitude ordering;
- cyclic equivalence;
- xarray coordinate reconstruction;
- round-trip ordering conversions.

### Analytic harmonic tests

Generate known scalar spherical harmonics, synthesize them onto each grid, and
verify recovered coefficients/fields at supported band limits.

Use multiple:

```text
l
m
```

including zonal and non-zonal modes.

### Vector/spin tests

Use analytic vector/spin-1 harmonics or independently derived
gradient/rotational fields.

Verify:

- sign;
- component orientation;
- pole behavior;
- E/B or divergent/rotational mapping;
- radius scaling.

### HEALPix parity

Use an independent HEALPix implementation such as healpy/libsharp where useful
for scalar map/alm parity, while keeping it an optional parity dependency.

Do not use DUCC calling DUCC through two code paths as the only parity evidence.

### Reduced Gaussian validation

Where no directly independent package exposes the identical transform:

- use analytic band-limited harmonics;
- compare synthesis values at the exact physical coordinates;
- measure analysis reconstruction error;
- document the transform/tolerance used.

---

## 8. Performance

Benchmark:

- full GL;
- reduced Gaussian of comparable spectral resolution;
- HEALPix of comparable pixel count/band limit.

Separate:

```text
analysis
synthesis
iterative-analysis iterations
xarray packing/unpacking
coordinate/order conversion
```

Do not present reduced storage alone as transform speedup.

---

## 9. Documentation

Add dedicated grid sections:

```text
docs/grids.md
docs/regridding.md
docs/kinematics.md
```

Explain:

- rectangular vs packed ring representation;
- reduced Gaussian ring geometry;
- HEALPix RING/NESTED ordering;
- exact/quadrature vs iterative analysis;
- which operations are available on each grid;
- any grid-specific accuracy/performance constraints.

Do not imply that every grid/back-end combination has identical mathematical
analysis semantics.

---

## 10. Acceptance criteria

Phase 3 is complete when:

- the grid model represents rectangular and packed ring grids without padding;
- DUCC ring geometry is the numerical source for new transforms;
- reduced Gaussian and HEALPix synthesis are correct on analytic harmonics;
- analysis semantics are explicitly classified and tested;
- scalar spectral regridding works across supported grid families;
- vector regridding and atmospheric vector operations are enabled only after
  spin-1 parity is demonstrated;
- HEALPix ordering conversions preserve data/coordinates;
- no non-SHT interpolation fallback exists;
- CPU DUCC remains the only production transform backend.

---

## Primary references for implementation research

- DUCC SHT overview:
  https://github.com/mreineck/ducc
- DUCC SHT documentation:
  https://gitlab.mpcdf.mpg.de/mtr/ducc/-/blob/ducc0/doc/sht.rst
- DUCC HEALPix support:
  https://github.com/mreineck/ducc
- Example of DUCC HEALPix ring geometry and iterative analysis:
  https://litebird-sim.readthedocs.io/en/latest/maps_and_harmonics.html
