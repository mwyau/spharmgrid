# Phase 3.5: Fejér-1 grids

## Status

**Implemented in Xarray/DUCC0.** Fejér-1 (F1) uses the existing scalar and spin-1
spherical harmonic transforms and rectangular `Grid` representation. JAX and PyTorch do
not support F1. Fejér-2 (F2) is not planned.

## 1. Grid geometry

For `N = nlat` latitude rings and north-to-south index `j = 0, ..., N-1`, the colatitude
nodes are:

```text
theta[j] = pi * (j + 1/2) / N
```

F1 excludes both geographic poles. Clenshaw–Curtis (CC) includes both poles.

As with existing GL/CC grids, require full global latitude coverage and one complete,
uniformly spaced, non-duplicated longitude cycle. Support both latitude orders and
equivalent cyclic longitude conventions without changing the physical field.

Detect F1 from its latitude nodes, not merely uniform latitude spacing or a generic
regular-grid label. Reject regional, incomplete, or incorrectly sampled grids.

## 2. DUCC0 transforms and bandwidth

DUCC0's `sht.analysis_2d` and `sht.synthesis_2d` accept `geometry="F1"`. Reuse the
existing spharmgrid DUCC adapters for scalar and spin-1 operations.

DUCC0's maximum inclusive F1 analysis degree is `nlat - 1`. The longitude limit is
`mmax <= min(lmax, (nlon - 1) // 2)`. F1 uses additional meridional upsampling to attain
this latitude degree limit.

Verify these constraints against the installed DUCC0 release. Explicit spectral
truncation requests beyond supported limits must raise a clear error, not silently clamp
or change the grid. Preserve the existing spectral normalization, wind conventions,
zero-mode behavior, and inverse-operation semantics.

## 3. API integration

- Extend the existing rectangular `Grid` kinds with `"f1"` and provide a public F1 grid
  constructor.
- Recognize F1 through existing Xarray coordinate discovery. Preserve latitude/longitude
  coordinates, metadata, non-spatial dimensions, and Dask behavior where supported.
- Map the F1 grid kind to DUCC's `"F1"` geometry argument without introducing another
  numerical implementation.
- Reuse the established scalar/vector filtering, differential, kinematics, analysis, and
  spectral regridding APIs where validated. Both source analysis and target synthesis
  must be supported for a regridding direction to be enabled.
- Do not claim F1 support in `spharmgrid.torch` or `spharmgrid.jax` without validated
  matching numerical transforms. Unsupported combinations should fail explicitly.
- Do not introduce generic interpolation, automatic missing-data filling, or a new
  backend framework for these grids.

## 4. Tests and acceptance

Use deterministic analytic test fields rather than requiring external datasets or
network access.

1. Verify the F1 constructor, latitude nodes, automatic grid detection, and rejection of
   near-miss or partial global grids.
2. Verify scalar and spin-1 analysis/synthesis at the documented F1 limit, including T71
   on 72×144 and T179 on 180×360, with independent SciPy harmonics. Verify filtering and
   GL/CC/F1 spectral regridding.
3. Verify scalar/vector operators and geographic wind conventions using analytic fields
   and identities. SHTns parity is optional when its F1 sampling and conventions match
   the tested geometry.
4. Test explicit limits, default bandwidth, and unsupported backend errors for F1.
5. Check both latitude orders, cyclic longitude representations, coordinate restoration,
   direct/accessor equivalence, leading dimensions, and Dask execution where applicable.
6. Keep existing GL/CC behavior and precision/error assertions intact.

The Xarray/DUCC implementation covers the listed F1 operations and grid limits. F1
remains distinct from CC and from the geometries supported by JAX and PyTorch.

## References

- [DUCC0 SHT Python API](https://github.com/mreineck/ducc/blob/ducc0/python/sht_pymod.cc)
- [Phase 3 backends](03-gpu-backends.md)
- [Phase 4 ring grids](04-ring-grids.md)
