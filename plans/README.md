# Development plans

spharmgrid implements spherical harmonic operations on full global Gauss–Legendre (GL),
Clenshaw–Curtis (CC), and Fejér-1 (F1) grids.

## Implemented

- Xarray operations using DUCC0, including scalar and vector spectral filtering,
  regridding, differential operators, and atmospheric wind diagnostics.
- PyTorch operations using torch-harmonics on GL and CC grids.
- JAX operations using S2FFT on GL and CC grids.
- F1 transforms using Xarray/DUCC0. Fejér-2 (F2) is not planned.
- Execution across non-spatial dimensions, including Dask-backed Xarray arrays.

The [documentation](../docs/index.md) describes the public APIs, numerical conventions,
and backend limitations. Earlier implementation plans are available in the Git history.

## Planned

- [HEALPix and reduced Gaussian grids](ring-grids.md): nonrectangular grid
  representations, transform accuracy, and scientific validation.
