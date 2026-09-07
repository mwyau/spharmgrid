# Changelog

## v0.1.1 - 2026-09-06

v0.1.1 strengthens validation, execution, typing, and documentation without changing the scientific API.

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

______________________________________________________________________

## v0.1.0 - 2026-09-06

### Initial release

- Added spherical harmonic filtering and GL/CC spectral regridding with triangular truncation and optional Sardeshmukh–Hoskins tapering.
- Added scalar and vector differential operators, including gradient, Laplacian, Helmholtz decomposition, and vector regridding.
- Added relative vorticity, divergence, streamfunction, velocity potential, rotational/divergent wind, and inverse wind transforms.
- Added Xarray `.sg` accessors, coordinate preservation, optional Dask execution, and optional `cf-xarray` discovery.
- Added command-line workflows for filtering, regridding, atmospheric diagnostics, and NetCDF/Zarr/GRIB I/O.
- Released on PyPI with Read the Docs documentation and citation metadata.
