# Phase 3.5: Fejér-1 and Fejér-2 grids

## Status and goal

**Planned.** Add Fejér-1 (F1) and Fejér-2 (F2) support to spharmgrid's
existing global rectangular grid API. Implement both with DUCC0's existing
scalar and spin spherical harmonic transforms. This phase does not require
a new transform engine or a non-rectangular grid representation.

The Xarray/DUCC0 path is the initial target. Optional tensor backends
may support these grids only after their transforms are verified.

## 1. Grid geometry

For `N = nlat` latitude rings, north-to-south index `j = 0, ..., N-1`,
and colatitude `theta`:

```text
F1: theta[j] = pi * (j + 1/2) / N
F2: theta[j] = pi * (j + 1) / (N + 1)
```

Both exclude the geographic poles. They differ from Clenshaw–Curtis (CC),
which includes both poles.

As with existing GL/CC grids, require full global latitude coverage and
one complete, uniformly spaced, non-duplicated longitude cycle. Support
both latitude orders and equivalent cyclic longitude conventions without
changing the physical field.

Detect F1 and F2 from their actual latitude nodes, not merely uniform
latitude spacing or a generic regular-grid label. Reject regional,
incomplete, or incorrectly sampled grids.

## 2. DUCC0 transforms and bandwidth

DUCC0's `sht.analysis_2d` and `sht.synthesis_2d` accept
`geometry="F1"` and `geometry="F2"`. Reuse the existing spharmgrid
DUCC adapters for scalar and spin-1 operations.

DUCC0 documents the following maximum **inclusive** analysis degrees:

| Grid | Maximum degree |
| --- | --- |
| F1 | `nlat - 1` |
| F2 | `(nlat - 1) // 2` |

The longitude limit is `mmax <= min(lmax, (nlon - 1) // 2)`. F1 uses
additional meridional upsampling to attain its documented degree limit.
Do not assign the F1 limit to F2.

Verify these constraints against the installed DUCC0 release.
Explicit spectral truncation requests beyond supported limits must raise
a clear error, not silently clamp or change the grid. Preserve the
existing spectral normalization, wind conventions, zero-mode behavior,
and inverse-operation semantics.

## 3. API integration

- Extend the existing rectangular `Grid` kinds with `"f1"` and
  `"f2"` and provide consistent public grid constructors.
- Recognize both geometries through existing Xarray coordinate discovery.
  Preserve latitude/longitude coordinates, metadata, non-spatial
  dimensions, and Dask behavior where supported.
- Map grid kinds to DUCC's `"F1"`/`"F2"` geometry arguments without
  introducing another numerical implementation.
- Reuse the established scalar/vector filtering, differential,
  kinematics, analysis, and spectral regridding APIs where validated.
  Both source analysis and target synthesis must be supported for a
  regridding direction to be enabled.
- Do not claim F1/F2 support in `spharmgrid.torch` or `spharmgrid.jax`
  without validated matching numerical transforms. Unsupported
  combinations should fail explicitly.
- Do not introduce generic interpolation, automatic missing-data
  filling, or a new backend framework for these grids.

## 4. Tests and acceptance

Use deterministic analytic test fields rather than requiring external
datasets or network access.

1. Verify constructors, latitude nodes, automatic grid detection, and
   rejection of near-miss or partial global grids for F1 and F2.
2. Verify scalar analysis/synthesis, filtering, and GL/CC/F1/F2 spectral
   regridding at permitted degree/order limits.
3. Verify spin-1 transforms, scalar/vector operators, and geographic wind
   conventions using analytic fields and identities.
4. Test explicit limits, default bandwidth, and unsupported backend
   errors separately for F1 and F2.
5. Check both latitude orders, cyclic longitude representations,
   coordinate restoration, direct/accessor equivalence, leading
   dimensions, and Dask execution where applicable.
6. Keep existing GL/CC behavior and precision/error assertions intact.

The phase is complete when both F1 and F2 work through the supported
public scientific APIs, have validated numerical limits and tests, and
remain distinct from CC and from unsupported tensor-backend geometries.

## References

- [DUCC0 SHT Python API](https://github.com/mreineck/ducc/blob/ducc0/python/sht_pymod.cc)
- [Phase 3 backends](03-gpu-backends.md)
- [Phase 4 ring grids](04-ring-grids.md)
