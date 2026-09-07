# Phase 4: additional global SHT grids

## Goal

Extend spharmgrid beyond full rectangular Gauss--Legendre (GL) and
Clenshaw--Curtis (CC) fields after the Phase-3 backend boundary is established.

Add the two highest-value non-rectangular global grid families separately:

```text
4a. HEALPix
4b. reduced Gaussian
```

These grids should share spharmgrid's scientific API where the selected backend
provides a numerically valid analysis/synthesis path, but they do not need to
share identical transform algorithms or accuracy semantics.

The expected initial backend capability direction is:

```text
GL                DUCC0 + torch-harmonics + S2FFT where validated
CC                DUCC0 + torch-harmonics where validated
HEALPix           DUCC0 + S2FFT
reduced Gaussian  DUCC0 initially
```

Do not add generic interpolation methods as a fallback when a spherical-harmonic
analysis is unavailable.

---

## 1. Why grid expansion follows backend work

Do not change the backend architecture and public grid representation at the
same time.

Phase 3 first establishes transform/convention boundaries on the existing GL/CC
model. Phase 4 can then design new grid objects against more than one real SHT
engine where available.

This is especially important for HEALPix: both DUCC0 and S2FFT support HEALPix,
so its public spharmgrid representation should describe the scientific grid,
not merely mirror one engine's internal ring arguments.

Reduced Gaussian may remain DUCC-only until another engine provides a verified
equivalent transform.

---

## 2. Research and verification gate

Before changing the public grid model, verify the installed backend versions and
record their actual capabilities.

For DUCC0 verify:

- accepted general isolatitude-ring geometry;
- scalar and spin transform support for non-rectangular rings;
- exact analysis versus adjoint/pseudo/iterative analysis behavior;
- `lmax`/`mmax`, quadrature, and ring-sampling limits;
- HEALPix geometry from `ducc0.healpix.Healpix_Base.sht_info()`;
- current iterative/pseudo-analysis controls;
- reduced-Gaussian analysis semantics for the intended sampling.

For S2FFT verify:

- current HEALPix forward/inverse transform interfaces;
- JAX/CUDA HEALPix capability;
- supported spin values and coefficient conventions;
- ordering/indexing expectations;
- iterative-refinement behavior and accuracy.

Do not describe a transform as an exact inverse merely because synthesis is
exact.

Keep these concepts distinct:

```text
synthesis
    evaluate a band-limited harmonic representation on the grid

adjoint analysis
    adjoint of synthesis; not automatically an inverse

quadrature/exact analysis
    recover coefficients under a valid sampling/quadrature theorem

iterative/pseudo analysis
    approximate coefficient recovery to a stated numerical tolerance
```

---

## 3. Public grid model

The current `Grid` descriptor is intentionally simple for GL/CC. Do not stretch
it into a dataclass containing unrelated optional fields for every future grid.

At this phase a small explicit grid-type split is justified, but the public
objects should describe scientific sampling rather than backend call layouts.

A possible direction is:

```python
Grid = RectangularGrid | HealpixGrid | ReducedGaussianGrid
```

with the existing GL/CC behavior retained under the rectangular form.

The exact class names may differ after implementation research.

### Requirements

- preserve simple GL/CC construction and detection;
- make grid identity/equivalence explicit;
- keep public grid definitions independent of DUCC/S2FFT coefficient storage;
- derive backend transform geometry from the public grid object;
- preserve enough information to reconstruct xarray coordinates/order;
- do not expose DUCC `ringstart`, `theta`, `nphi`, or `phi0` arrays as the
  public essence of HEALPix merely because DUCC uses them internally;
- do not force HEALPix and reduced Gaussian into one generic `RingGrid` if their
  user-facing semantics are clearer as separate types.

Because spharmgrid is pre-1.0, prefer a coherent grid model over preserving an
abstraction that no longer represents the supported geometries cleanly.

---

# Phase 4a: HEALPix

## 4. HEALPix public representation

HEALPix has a natural scientific identity independent of any transform engine.

Target constructor shape:

```python
sg.healpix_grid(
    nside: int,
    *,
    ordering: Literal["ring", "nested"] = "ring",
)
```

The concrete return type may become `HealpixGrid`.

The public representation should minimally retain:

```text
nside
ordering
```

plus any additional information required to preserve user-visible coordinates
or metadata.

Do not make backend-specific ring arrays mandatory constructor arguments.

---

## 5. HEALPix xarray representation

Use one horizontal pixel dimension:

```text
(..., pixel)
```

Do not represent HEALPix as a rectangular latitude/longitude array and do not
treat pixel number itself as latitude or longitude.

Auxiliary latitude/longitude coordinates may be provided when useful, but the
HEALPix grid identity comes from its HEALPix metadata, not inferred equality of
those coordinates.

Prefer RING ordering internally when required by a backend SHT path.

If NESTED input is supported, reorder data and coordinates together and restore
or document the requested output ordering.

---

## 6. HEALPix backend behavior

Initial target engines:

```text
DUCC0
S2FFT
```

Implement backend-specific geometry/index translation inside the corresponding
adapters established in Phase 3.

Do not make torch-harmonics support a Phase-4 requirement unless the library
adds a verified HEALPix transform.

### Analysis semantics

HEALPix does not provide the same quadrature-exact analysis theorem as GL.
Document whether each backend uses:

```text
adjoint analysis
iterative/pseudo analysis
iterative refinement
```

and expose numerical controls only when users need them.

If public iterative controls are necessary, use explicit names such as:

```text
max_iterations
rtol
```

Do not hide iteration behind a generic `accuracy=True` switch.

### Vector/spin operations

Do not enable wind diagnostics merely because scalar HEALPix transforms work.
Verify spin-1/geographic-vector conventions independently for DUCC0 and S2FFT.

---

## 7. HEALPix scientific operations

Where backend analysis/synthesis is sufficiently accurate and tested, target
the established physical API:

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

Availability should be capability-based. Do not pretend all backends provide
identical numerical analysis semantics.

---

# Phase 4b: reduced Gaussian

## 8. Reduced-Gaussian public representation

Support atmospheric reduced Gaussian grids as true variable-ring SHT sampling
geometries.

Initial input target:

- explicit ring structure/`pl` from decoded datasets;
- unambiguous detection from coordinates plus ring metadata;
- common ECMWF-style named constructions only after N/O grid definitions and
  ring-count algorithms are verified from authoritative sources.

Do not infer an `Nxxx` or `Oxxx` grid from total point count alone.

A dedicated `ReducedGaussianGrid` should describe the sampling without requiring
users to pass DUCC internal ring arrays directly.

---

## 9. Reduced-Gaussian xarray representation

Use a packed one-dimensional horizontal dimension:

```text
(..., cell)
```

with auxiliary coordinates such as:

```text
latitude(cell)
longitude(cell)
```

and enough grid metadata to recover the ring structure.

Do not pad reduced Gaussian rows with NaNs to force a rectangular `(lat, lon)`
representation.

---

## 10. Reduced-Gaussian backend behavior

Use DUCC0 initially unless another backend has a verified equivalent transform.

Determine whether the intended reduced Gaussian sampling supports direct
quadrature analysis for the requested band limit or requires iterative/pseudo
analysis.

If reliable analysis requires an iterative solver, make the solver/tolerance
semantics explicit. Do not label the grid as equivalent to full GL merely
because the latitudes are Gaussian.

---

## 11. Reduced-Gaussian scientific operations

Where analysis is justified, target the same established physical API as other
grids:

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

Do not enable an operation until its required scalar or spin transform is
validated for the grid.

---

## 12. Cross-grid spectral regridding

All supported regridding remains spectral:

```text
source analysis
    -> optional spectral operation
    -> target synthesis
```

Potential Phase-4 combinations include:

```text
GL/CC <-> HEALPix
GL/CC <-> reduced Gaussian
HEALPix <-> reduced Gaussian
HEALPix -> HEALPix
reduced Gaussian -> reduced Gaussian
```

Only enable a direction when the source grid has a valid analysis path and the
target grid has a valid synthesis path for the selected backend/capability.

No bilinear, nearest-neighbor, spline, or conservative fallback belongs in this
API.

---

## 13. Transform-accuracy model

Analysis quality belongs to a backend/grid pair, not to the abstract grid name
alone.

The implementation should be able to classify or internally represent analysis
as appropriate, for example:

```text
quadrature
iterative
adjoint-only
```

This classification can remain internal when users have no choice to make, but
user documentation must state the actual semantics.

If users can control iterative analysis, keep those controls on operations that
perform analysis rather than embedding solver options in grid constructors.

---

## 14. Tests

### Grid/geometry tests

For every new grid type test:

- point/pixel count;
- ring structure where applicable;
- longitude phase;
- ordering conversion;
- xarray coordinate reconstruction;
- grid equality/equivalence;
- data and coordinate movement together.

### Analytic harmonic tests

Generate known low-degree scalar harmonics and verify synthesis/recovery at
supported band limits.

Use multiple `l` and `m`, including zonal and non-zonal modes.

### Vector/spin tests

Use independently derived gradient, divergent, and rotational fields to test:

- sign;
- component orientation;
- pole behavior;
- E/B or divergent/rotational mapping;
- radius scaling.

### HEALPix cross-backend validation

Where DUCC0 and S2FFT implement the same operation, compare them on the same
physical HEALPix field in addition to analytic tests.

An independent HEALPix implementation such as healpy/libsharp may be used as an
optional parity dependency where it adds genuinely independent evidence.

### Reduced-Gaussian validation

When no second package exposes the identical transform:

- use analytic band-limited harmonics;
- compare synthesis on exact physical coordinates;
- measure coefficient/field reconstruction error;
- document the analysis method and tolerance.

Do not treat DUCC through two spharmgrid code paths as independent parity.

---

## 15. Performance

Benchmark separately:

```text
analysis
synthesis
iterative-analysis iterations
backend conversion
xarray packing/unpacking
ordering conversion
```

For HEALPix, compare DUCC0 and S2FFT where both are appropriate.

For reduced Gaussian, compare storage and transform cost with full GL at a
meaningful common spectral resolution.

Do not describe reduced storage alone as transform acceleration.

---

## 16. Documentation

Update the grid, regridding, kinematics, API, and backend documentation to show
capabilities by grid/backend pair.

Explain:

- rectangular versus packed/pixel xarray representation;
- HEALPix `nside` and RING/NESTED ordering;
- reduced-Gaussian ring structure;
- exact/quadrature versus iterative/adjoint analysis;
- which operations are supported by which backend/grid combination;
- any bandwidth, accuracy, precision, or performance limits.

Do not imply that every grid/backend combination has identical mathematical
semantics.

---

## 17. Acceptance criteria

Phase 4 is complete when:

- the public grid model represents rectangular, HEALPix, and reduced-Gaussian
  sampling without padding or backend-specific constructor bundles;
- existing GL/CC public behavior remains coherent;
- HEALPix synthesis/analysis is validated with DUCC0 and S2FFT where supported;
- HEALPix ordering conversions preserve data and coordinates;
- reduced-Gaussian synthesis is correct on analytic harmonics;
- reduced-Gaussian analysis semantics and accuracy are explicitly classified;
- scalar regridding works across the grid combinations whose source analysis and
  target synthesis are valid;
- vector regridding and atmospheric wind operations are enabled only after
  spin-1 parity is demonstrated;
- backend/grid capability failures are explicit;
- no non-SHT interpolation fallback is introduced.

---

## Primary implementation references

- DUCC0: https://github.com/mreineck/ducc
- S2FFT: https://github.com/astro-informatics/s2fft
- HEALPix: https://healpix.sourceforge.io/

Verify installed/current APIs and authoritative grid definitions during
implementation rather than treating this roadmap as a frozen library reference.
