# spharmgrid roadmap

`spharmgrid` is an xarray-first spherical-harmonic operations package for
atmospheric and geophysical fields. Its scope is limited to operations whose
numerical core is a spherical-harmonic transform or a direct spectral-space
operator.

It is not a generic interpolation or regridding package.

Long term, spharmgrid is intended to provide one scientific SHT API over three
numerical transform engines:

```text
DUCC0             CPU/reference implementation
torch-harmonics   PyTorch/GPU scalar + vector SHT implementation
S2FFT             JAX/GPU arbitrary-spin implementation
```

The atmospheric and spectral mathematics belong to spharmgrid. Backend-specific
code should be limited to transform primitives, convention/coefficient
translation, supported-grid checks, and framework/device integration.

## Plan structure

The implementation is organized into sequential plans:

1. `plans/01-core.md` — **implemented baseline**
   - DUCC0 implementation on full rectangular Gauss--Legendre (GL) and
     pole-including Clenshaw--Curtis (CC) grids.
   - Scalar filtering/regridding and differential operators.
   - Atmospheric vorticity/divergence, streamfunction/velocity potential, and
     inverse wind transforms.
   - xarray/CF/Dask integration and the file-oriented CLI.

2. `plans/02-sht-suite.md` — **current next phase**
   - Complete the high-value scalar/vector SHT operator suite.
   - Add vector regridding, Helmholtz decomposition, inverse gradient, and
     vector Laplacian/inverse vector Laplacian.
   - Establish the complete scientific operation contract before alternate
     transform engines are introduced.

3. `plans/03-gpu-backends.md`
   - Introduce the smallest backend boundary required by the real DUCC0,
     torch-harmonics, and S2FFT implementations.
   - Keep GL/CC as the grid substrate while backend conventions are established.
   - Add torch-harmonics and S2FFT as optional accelerator/differentiable
     engines where their rectangular-grid capabilities are valid.
   - Use tensor-native `spharmgrid.torch` and `spharmgrid.jax` APIs for
     differentiable accelerator work; keep xarray as the scientific
     metadata/file-I/O layer.
   - Keep DUCC0 as the default/reference backend.

4. `plans/04-ring-grids.md`
   - Expand the public grid model only after the backend boundary is proven.
   - Add HEALPix first, using DUCC0 and S2FFT where appropriate.
   - Add reduced Gaussian as a separate subphase, initially through DUCC0 unless
     another engine provides a verified equivalent transform.
   - Keep analysis accuracy/iteration semantics explicit for non-quadrature
     grids.

## Why backends precede new grid families

Do not combine two large architectural changes unnecessarily.

The backend boundary should first be derived and validated while spharmgrid has
only the simple rectangular GL/CC model. This isolates coefficient layout,
normalization, spin/vector conventions, dtype/device behavior, and backend
bandwidth differences from the separate problem of representing packed ring or
pixel grids.

HEALPix also has more than one relevant engine: DUCC0 and S2FFT. Designing its
public grid object after the S2FFT adapter exists reduces the risk of exposing a
DUCC-specific ring-geometry representation as the package-level scientific API.

Reduced Gaussian can remain DUCC-only initially; it does not need to block the
accelerator architecture.

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

Phases 1 and 2 remain DUCC0-only in production. Do not build a speculative
backend framework during Phase 2.

Phase 3 is the point where an internal backend abstraction becomes justified,
because two real alternate engines are implemented against it. Start by
extracting only what the current DUCC path and alternate adapters actually need;
do not design a plugin framework in advance.

Keep one scientific definition for spectral masks, operator multipliers,
zero/null modes, radius factors, and atmospheric sign/component conventions.
Framework-native Torch/JAX array expressions are acceptable where required for
differentiability, provided they implement the same tested scientific formula.

The primary accelerator interfaces are tensor-native:

```text
spharmgrid.torch   torch.Tensor
spharmgrid.jax     jax.Array
```

Use spharmgrid grid objects as the scientific grid contract and keep
backend-specific sampling names inside adapters. For rectangular arrays, use
the trailing two dimensions as latitude/longitude and preserve arbitrary
leading/batch dimensions.

xarray remains the normal scientific I/O, CF metadata, coordinate, and
file-decoding layer. Device conversion should occur explicitly at the
application/data-loader boundary rather than inside every SHT call. Optional
JAX/xarray interoperability such as `xarray_jax` can be evaluated after the raw
JAX path is correct; optional xarray `backend=` convenience can be considered
later if benchmarks justify host/device transfers.

Never auto-select an accelerator merely because hardware is available.

## Grid direction

Phase 4 uses backend/grid capabilities rather than assuming a Cartesian product
of every engine and every grid.

Expected initial direction:

```text
GL                DUCC0 + torch-harmonics + S2FFT where validated
CC                DUCC0 + torch-harmonics where validated
HEALPix           DUCC0 + S2FFT
reduced Gaussian  DUCC0 initially
```

This is a capability target, not a claim that the same bandwidth, analysis
method, or precision applies to every backend/grid pair.

## Current implementation contract

Until `plans/02-sht-suite.md` is implemented, the existing public behavior is
the Phase-1 contract recorded in `plans/01-core.md`.

Keep this root `PLAN.md` short. It is the routing document; detailed scientific
and implementation decisions belong in the owning phase plan.
