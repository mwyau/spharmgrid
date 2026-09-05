# Scientific numerics

Use this skill for spherical-harmonic transforms, GL/CC grids, filtering, regridding, differential operators, vector wind transforms, numerical conventions, scientific tests, and parity work.

## Start from the current contract

Read the current `PLAN.md`, relevant spharmgrid source/tests, and the current source implementation in `../PyStormTracker` before changing scientific behavior.

For the initial implementation, inspect at least:

```text
../PyStormTracker/src/pystormtracker/preprocessing/spectral.py
../PyStormTracker/src/pystormtracker/preprocessing/regrid.py
../PyStormTracker/src/pystormtracker/preprocessing/kinematics.py
../PyStormTracker/src/pystormtracker/models/geo.py
```

Use PyStormTracker as a source implementation, not as proof that a convention is mathematically correct. Verify the extracted behavior independently.

## Source hierarchy

Keep these roles distinct:

- published literature defines a named scientific method;
- NCL/SPHEREPACK define established atmospheric operation semantics and a parity target;
- `ducc0` defines the numerical transform API and conventions used by spharmgrid;
- PyStormTracker supplies the already-developed source wrapper and tested sign/coordinate choices;
- spharmgrid defines its own public xarray/CF API.

When these differ, document the difference rather than silently merging them.

## Grid and coordinate invariants

Initial support is limited to:

- full Gauss–Legendre (GL) grids;
- global pole-including Clenshaw–Curtis (CC) grids.

Do not classify a generic regular latitude-longitude grid as CC unless it satisfies the supported sampling contract. Verify GL latitude nodes against the `ducc0` geometry rather than by a loose regularity test.

Treat latitude order and cyclic longitude convention as representations of the same physical grid when they are mathematically equivalent. Reorder data and coordinates together and restore the requested/user-facing ordering after the transform.

Do not use positional last-dimension guessing when coordinates are ambiguous.

## Spectral filtering

The public default is hard spectral selection with no taper.

For a retained band `[lmin, lmax]`:

- set modes below `lmin` to zero;
- retain modes in the band;
- set modes above `lmax` to zero.

When `taper` is supplied, use the Sardeshmukh–Hoskins exponential response documented in `PLAN.md` and the corresponding PyStormTracker implementation. `taper` is the response at `lmax`; do not reinterpret it as an arbitrary strength parameter.

Filtering and regridding should share a transform path so a combined operation performs one analysis and one synthesis.

## Scalar operators

Implement gradient, Laplacian, and inverse Laplacian spectrally. Preserve the package Earth-radius convention and expose radius only where it changes the physical operator.

For inverse Laplacian, the `l=0` mode is singular. Use and document the zero-mean convention rather than inventing a constant.

## Vector transforms

Do not derive spin-1 signs or component ordering from memory.

Trace the current PyStormTracker mapping between geographic `(u, v)` and the `ducc0` spin-1 vector representation. Preserve that path initially, then verify it using:

- analytic purely rotational fields;
- analytic purely divergent fields;
- `u,v -> vo,d -> u,v` round trips;
- `u,v -> strf,vp -> u,v` round trips;
- independent SPHEREPACK/pyspharm comparisons where grid definitions align.

Check all of the following explicitly:

- geographic eastward/northward component order;
- sign of theta/latitude components;
- E/B coefficient order;
- `sqrt(l(l+1))/R` factors;
- Laplacian sign;
- `l=0` handling;
- latitude orientation before and after DUCC calls.

Do not hide a sign discrepancy behind a tolerance.

## Scientific metadata

CF metadata is semantic output, not decoration.

- Auto-detect documented quantities from explicit arguments, exact CF `standard_name`, then canonical short names.
- Do not maintain broad name heuristics without a real use case.
- Output-name overrides must not change the physical `standard_name`.
- Preserve metadata for same-quantity operations such as filter/regrid.
- For derived quantities, assign exact verified CF metadata when an exact standard name exists.
- If CF has no exact name for a rotational/divergent component, omit `standard_name` rather than assigning a broader but inaccurate one.

spharmgrid does not own CF time/calendar representation. Preserve non-spatial coordinates and objects through horizontal transforms.

## Tests and parity

Use three evidence layers:

1. analytic/constructed fields with known behavior;
2. internal identities and round trips;
3. parity against an independent implementation such as SPHEREPACK/pyspharm.

Do not rely only on random fields. Deterministic random fields may supplement, not replace, analytic cases.

Do not check in large numerical reference arrays when deterministic parity tests can generate the input at run time.

Before comparing two implementations, align:

- grid family and exact latitude nodes;
- longitude ordering and origin;
- latitude orientation;
- truncation and `mmax`;
- Earth radius;
- normalization;
- scalar/vector sign convention;
- units and precision.

State tolerances explicitly and justify any tolerance materially looser than floating-point/transform expectations.

## Performance

Correctness comes before optimization. Avoid repeated SHT analysis/synthesis when related outputs can share coefficients, but do not add caches, schedulers, backend abstractions, or concurrency layers without a measured need.

For Dask-backed xarray data, preserve lazy execution and avoid DUCC thread oversubscription. Do not copy PyStormTracker's MPI/backend architecture into this package.

## Completion check

Before considering a numerical change complete:

- the implementation matches the named method and documented grid contract;
- accessor and direct APIs use the same numerical path;
- analytic tests pass;
- inverse/round-trip identities pass where defined;
- parity tests agree where the external grid/convention is comparable;
- metadata and coordinate ordering remain correct;
- documentation states any numerical limitation or convention that a user must know.
