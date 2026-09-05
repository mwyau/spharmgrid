# spharmgrid roadmap

`spharmgrid` is an xarray-first spherical-harmonic operations package for
atmospheric and geophysical fields. Its scope is intentionally limited to
operations whose numerical core is a spherical-harmonic transform or a direct
spectral-space operator.

It is not a generic interpolation or regridding package.

Long term, spharmgrid is intended to provide one scientific SHT API over three
numerical transform engines:

```text
DUCC0             CPU/reference implementation
torch-harmonics   PyTorch/GPU vector-SHT implementation
S2FFT             JAX/GPU arbitrary-spin and HEALPix implementation
```

The atmospheric and spectral mathematics belong to spharmgrid and should be
implemented once. Backend-specific code should be limited to transform
primitives, coefficient/convention translation, supported-grid checks, and
array/device integration.

## Plan structure

The implementation is organized into sequential plans:

1. `plans/01-core.md`
   - Initial DUCC0 implementation.
   - Full rectangular Gauss--Legendre (GL) and pole-including
     Clenshaw--Curtis (CC) grids.
   - Scalar spectral filtering and regridding.
   - Scalar gradient/Laplacian operators.
   - Wind vorticity/divergence, streamfunction/velocity potential, and inverse
     wind transforms.
   - xarray/CF integration.

2. `plans/02-sht-suite.md`
   - **Current next phase.**
   - Complete the high-value NCL/SPHEREPACK-style scalar/vector SHT operator
     suite without exposing a generic interpolation layer.
   - Use `pyspharm-syl`/SPHEREPACK as an independent parity implementation.

3. `plans/03-ring-grids.md`
   - Add reduced Gaussian and HEALPix grids through DUCC0's isolatitude-ring
     SHT interfaces.
   - Keep transform accuracy and analysis semantics explicit where exact
     quadrature is unavailable.

4. `plans/04-gpu-backends.md`
   - Add optional accelerator integrations only after CPU/grid semantics are
     stable.
   - Integrate both torch-harmonics and S2FFT as thin optional transform
     backends where their grid capabilities are valid.
   - Preserve one spharmgrid scientific API over DUCC0, torch-harmonics, and
     S2FFT, with backend-specific convention/parity tests.

## Scope rule

A feature belongs in spharmgrid when its core operation is one of:

- scalar or vector spherical-harmonic analysis/synthesis;
- spectral truncation/filtering;
- spectral regridding between supported global spherical grids;
- a scalar/vector differential or inverse differential operator implemented
  through spherical harmonics;
- a physical atmospheric transform derived from scalar/vector harmonic
  coefficients.

The following do not belong in spharmgrid merely for convenience:

- bilinear or bicubic interpolation;
- spline interpolation;
- nearest-neighbor interpolation;
- conservative remapping;
- regional DCT filtering;
- generic horizontal-grid conversion unrelated to SHTs.

Those are better handled by dedicated regridding packages.

## Backend direction

Phases 1--3 remain DUCC0-only in production. Do not build a speculative
backend framework before Phase 4. However, new scientific operations should
keep physical formulas, spectral multipliers, signs, radius conventions, and
metadata separate from direct DUCC calls so that Phase 4 can reuse the same
scientific implementation.

The intended Phase 4 design is not three copies of spharmgrid. It is one
scientific layer over small backend adapters exposing the scalar/vector or
spin analysis and synthesis primitives needed by the supported operation set.

## Current implementation contract

Until `plans/02-sht-suite.md` is implemented, the existing public behavior is
the Phase 1 contract. Moving the original long `PLAN.md` to
`plans/01-core.md` should preserve it unchanged as the record of that phase.

When plans are added to the repository, keep this root `PLAN.md` short. It is
the routing document that tells contributors which detailed plan governs the
requested work.
