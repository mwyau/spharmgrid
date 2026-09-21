# Changelog

## v0.3.0.dev0

### JAX

- Added the optional `spharmgrid.jax` array API backed by S2FFT, with the same
  19 scientific operations as `spharmgrid.torch` on exact GL and CC/MWSS
  grids.
- Added scalar and spin-1 spherical harmonic filtering, spectral regridding,
  differential operators, atmospheric kinematics, Helmholtz decomposition, and
  inverse wind transforms for JAX arrays.
- Added leading array dimensions, `jit`, `vmap`, and automatic
  differentiation. The JAX API requires JAX x64 mode and `float64` spatial
  inputs; spharmgrid does not change the process-wide JAX precision setting.
- JAX transforms cache S2FFT O(L²) Price–McEwen recursion precomputations.
  Scalar transforms use S2FFT's real-field path; spin-1 transforms use its
  complex path.

### Validation and interoperability

- Added numerical comparisons between the JAX and DUCC APIs across GL and
  CC/MWSS grids, scalar and vector operations, regridding, differential
  operators, and atmospheric wind diagnostics.
- Added JAX tests for `jit`, `vmap`, automatic differentiation, cold-cache
  precomputation, inverse-gradient projection, near-bandlimit regridding, and
  the supported dependency floor.
- Added development-only `gdm-xarray-jax` v0.1.1 tests for Xarray `DataArray`
  and `Dataset` PyTrees, including JIT, automatic differentiation, regridding,
  vector kinematics, leading dimensions, and static coordinates. Names,
  dimensions, and static coordinates survive the tested JAX transformations;
  ordinary Xarray attrs do not.
- Added dedicated JAX CI coverage and ran JAX, PyTorch, and pyspharm parity
  tests in a jointly compatible environment.

### Documentation

- Added separate PyTorch and JAX API reference pages for all public tensor and
  array operations and the four public `spharmgrid.torch.nn` modules.
- Added a separate Xarray accessor API reference for `DataArray.sg` and
  `Dataset.sg`.
- Documentation builds mock the optional PyTorch and JAX dependencies. PyTorch
  module constructor signatures are derived from their Python `__init__`
  methods, and all public module methods now have docstrings.

### Compatibility

- Raised the minimum Xarray version to 2025.1.1, including the Zarr v3 support
  used by the CLI.

## v0.2.0 - 2026-09-19

### Spectral truncation

- Added `Tnxm` trapezoidal and `Rn` symmetric rhomboidal truncation to the
  Xarray/NumPy filtering, regridding, and CLI APIs.
- Added public `TransformSpec` objects for parsed spectral degree/order limits
  and truncation type.
- The Torch API supports triangular `Tn` and total-degree band-pass `Ta-b`
  truncation. `Tnxm` and `Rn` are outside the coefficient domains supported
  by the current torch-harmonics transforms.

### PyTorch

- Added the optional `spharmgrid.torch` tensor API backed by
  `torch-harmonics`, covering filtering, scalar and vector regridding,
  differential operators, atmospheric wind diagnostics, Helmholtz
  decomposition, and inverse wind transforms.
- Added reusable `spharmgrid.torch.nn` modules: `SHTFilter`, `SHTRegrid`,
  `SHTVectorRegrid`, and `SHTOperators`.
- Added `float32` and `float64` support, CPU/CUDA execution, leading batch
  dimensions, dtype/device preservation, and autograd through the Torch
  operations.
- PyTorch and `torch-harmonics` are installed separately from spharmgrid;
  added installation documentation and dedicated CI coverage.

### API

- Renamed the `quantity=` argument of `rotational_wind()` and
  `divergent_wind()` to `source=`, aligning the Xarray and Torch APIs.

### Validation

- Added numerical comparisons between the Torch and DUCC APIs across the public
  Torch operations, supported GL/CC paths, and both supported floating dtypes.
- Added an NCL 6.6.2/SPHEREPACK parity suite for GL and CC scalar/vector
  operations, regridding, triangular, trapezoidal, and rhomboidal truncation,
  and hard and tapered filtering.

### Packaging

- Added SPDX BSD-3-Clause copyright and license headers to package and test
  files, with automated SPDX checks in development and CI tooling.

## v0.1.3 - 2026-09-13

### Execution

- Added `sht_threads` to spherical harmonic operations; Dask-backed operations
  use one DUCC thread per transform by default.
- Added CLI `--workers` and `--sht-threads` controls; the Python API no longer
  configures Dask workers.

## v0.1.2 - 2026-09-12

### Compatibility

- Restored Python 3.11 support; spharmgrid now supports Python 3.11–3.14.

### Execution

- Improved Dask worker scaling on systems with intermediate CPU counts.

### Testing and CI

- Added independent SPHEREPACK/pyspharm comparisons for `laplacian` and `inverse_laplacian` on GL and CC grids.
- Expanded Xarray `.sg` accessor tests and Codecov reporting for optional file backends.
- Added prek hooks, grouped Dependabot updates, and repository formatting/checks.

### Documentation

- Switched the documentation to the PyData Sphinx Theme and reorganized the guide and API reference.

---

## v0.1.1 - 2026-09-06

### Execution

- Eager transforms now let DUCC choose its default thread count when Dask is not installed. Dask-backed transforms retain four DUCC threads per transform with a bounded local worker count.

### Testing and CI

- Added analytic GL/CC tests for same-grid vector filtering, Helmholtz decomposition, `inverse_gradient`, and inverse-Laplacian behavior, and strengthened CLI tests against the corresponding Python API.
- Adjusted three CC tolerances for small macOS and conda-forge floating-point differences.
- Added Codecov branch coverage and test-result reporting to the existing unit and SPHEREPACK/pyspharm parity runs, plus lockfile freshness and stricter pytest and Ruff checks.

### Typing

- Added `py.typed` and tightened NumPy array and optional-metadata annotations, with stronger `ty` and Ruff checks.

### Documentation and distribution

- Added conda-forge installation alongside PyPI.
- Archived v0.1.1 on Zenodo with DOI [10.5281/zenodo.22559210](https://doi.org/10.5281/zenodo.22559210).
- Improved documentation navigation, theme controls, table rendering, and Edit on GitHub links, and expanded the API comparison with NCL and windspharm references.

---

## v0.1.0 - 2026-09-06

### Initial release

- Added spherical harmonic filtering and GL/CC spectral regridding with triangular truncation and optional Sardeshmukh–Hoskins tapering.
- Added scalar and vector differential operators, including gradient, Laplacian, Helmholtz decomposition, and vector regridding.
- Added relative vorticity, divergence, streamfunction, velocity potential, rotational/divergent wind, and inverse wind transforms.
- Added Xarray `.sg` accessors, coordinate preservation, optional Dask execution, and optional `cf-xarray` discovery.
- Added command-line workflows for filtering, regridding, atmospheric diagnostics, and NetCDF/Zarr/GRIB I/O.
- Released on PyPI with Read the Docs documentation and citation metadata.
