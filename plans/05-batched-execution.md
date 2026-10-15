# Batched transform execution

## Goal

Preserve the scientific behavior and public signatures of the existing Xarray/DUCC,
PyTorch, and JAX APIs while making arbitrary leading dimensions a consistent input
contract. Xarray output must retain its original named dimensions, coordinates,
metadata, and `.sg` behavior.

## Current execution paths

- DUCC adapters in `src/spharmgrid/_ducc.py` accept one 2-D map (or the two components
  of one spin-1 map). Scalar/vector Xarray adapters in `_spectral_field.py`,
  `_vector.py`, `spectral.py`, `operators.py`, and `kinematics.py` use
  `xr.apply_ufunc(..., vectorize=True)` to dispatch one frame per callback.
- `regrid.py` routes scalar and vector transforms through those scalar/vector helpers.
- Torch accepts leading dimensions through torch-harmonics and spharmgrid's native Torch
  operations. JAX flattens leading dimensions and maps S2FFT over frames with
  `jax.vmap`.
- Accessor methods delegate to the direct Xarray APIs.

## Design

1. Extend the private DUCC adapters to accept arbitrary leading dimensions, flatten only
   those dimensions for execution, and restore them on coefficient and map outputs. Keep
   each DUCC transform's spin-component axis distinct from the batch axes.
2. Use a per-field loop in these adapters unless a DUCC operation is proven equivalent
   for batched quadrature analysis. DUCC0 0.41.0 documents `analysis_2d` and
   `synthesis_2d` inputs as one scalar map or two spin components. Its `pseudo_analysis`
   accepts `ntrans` but is iterative least-squares analysis, so it is not a substitute.
3. Change Xarray gufunc adapters to `vectorize=False`; numerical kernels will receive
   each complete available leading-dimension block. Keep horizontal dimensions as gufunc
   core dimensions and preserve current lazy Dask behavior and thread policy.
4. Verify all direct scalar/vector operations and reusable spectral-field analysis and
   synthesis use the block contract. Keep accessors as thin calls to these functions.
5. Audit Torch/JAX signatures and operations for arbitrary leading dimensions. Preserve
   their framework-native execution; add fixes only where shape tests expose a mismatch.
6. Add deterministic equivalence tests for individual fields and multi-axis batches,
   arbitrary Xarray dimension order, vector semantics, Dask laziness, accessor/direct
   equivalence, and backend differentiation/compiled paths already supported.
7. Document leading-dimension semantics and backend capability differences after the
   implementation is validated.

## Dask handling

Retain the existing complete-spatial-core requirement. First measure the current
`allow_rechunk=True` behavior with split spatial chunks and multiple leading chunk
layouts. Avoid changing rechunk policy unless tests show a safe way to reduce memory
costs without rejecting workflows that currently work. Do not add a DUCC worker pool or
nest transform threads inside Dask tasks.

## Acceptance

- Scalar and vector transforms, spectral fields, filtering, operators, and regridding
  preserve every leading dimension for Xarray/DUCC, Torch, and JAX.
- Xarray numerical callbacks receive a multi-field block when the available input block
  contains multiple fields, without changing coordinates, dimension order, metadata,
  Dask laziness, or `.sg` compatibility.
- Batched results agree with independently transformed fields at existing tolerances.
- No change is made to transform normalization, vector signs, coefficient conventions,
  truncation rules, supported grids, or backend precision requirements.
