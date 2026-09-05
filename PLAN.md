# spharmgrid roadmap

`spharmgrid` is an xarray-first spherical-harmonic operations package for
atmospheric and geophysical fields. Its scope is intentionally limited to
operations whose numerical core is a spherical-harmonic transform or a direct
spectral-space operator.

It is not a generic interpolation or regridding package.

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
   - Evaluate torch-harmonics for PyTorch/vector SHT workflows and S2FFT for
     JAX/spin/HEALPix workflows.
   - Preserve one spharmgrid scientific API and verify backend parity.

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

## Current implementation contract

Until `plans/02-sht-suite.md` is implemented, the existing public behavior is
the Phase 1 contract. Moving the original long `PLAN.md` to
`plans/01-core.md` should preserve it unchanged as the record of that phase.

When plans are added to the repository, keep this root `PLAN.md` short. It is
the routing document that tells contributors which detailed plan governs the
requested work.
