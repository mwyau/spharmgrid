# spharmgrid

[![CI](https://github.com/mwyau/spharmgrid/actions/workflows/ci.yml/badge.svg)](https://github.com/mwyau/spharmgrid/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/spharmgrid/badge/?version=latest)](https://spharmgrid.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/github/mwyau/spharmgrid/graph/badge.svg?token=5kDDQ8Ii0Z)](https://codecov.io/github/mwyau/spharmgrid)
[![PyPI version](https://img.shields.io/pypi/v/spharmgrid)](https://pypi.org/project/spharmgrid/)
[![PyPI Python Version](https://img.shields.io/pypi/pyversions/spharmgrid)](https://pypi.org/project/spharmgrid/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/spharmgrid)](https://anaconda.org/channels/conda-forge/packages/spharmgrid/overview)
[![GitHub License](https://img.shields.io/github/license/mwyau/spharmgrid)](https://github.com/mwyau/spharmgrid/blob/main/LICENSE)

Spherical harmonic tools for filtering, regridding, and kinematics in atmospheric science with Xarray.

**spharmgrid** (**sp**herical **harm**onic **grid**ding) implements spherical harmonic filtering, regridding, differential operators, and atmospheric kinematics for global Xarray fields. It computes relative vorticity, divergence, streamfunction, velocity potential, Helmholtz decomposition, and inverse wind transforms. [DUCC](https://gitlab.mpcdf.mpg.de/mtr/ducc) performs the numerical spherical harmonic transforms.

Supported grids are full rectangular Gauss–Legendre (GL) and Clenshaw–Curtis (CC) grids.

## Install

Install with either `pip`, `uv`, or `conda`:

```bash
pip install spharmgrid
```

```bash
uv add spharmgrid
```

```bash
conda install -c conda-forge spharmgrid
```

Optional groups are:

- `spharmgrid[dask]` — Dask-backed lazy execution;
- `spharmgrid[cf]` — optional `cf-xarray` coordinate discovery;
- `spharmgrid[cli]` — command-line NetCDF, Zarr, and GRIB I/O.

Without Dask installed, spharmgrid lets DUCC use its default thread count. With Dask support installed, DUCC uses four threads per transform; for the local Dask scheduler, spharmgrid sets `num_workers=max(1, os.cpu_count() // 4)` unless `num_workers` is configured.

For a standalone command-line installation:

```bash
uv tool install "spharmgrid[cli]"
```

For a project environment, install the CLI extra with either:

```bash
uv add "spharmgrid[cli]"
```

```bash
pip install "spharmgrid[cli]"
```

## Quick start

Importing `spharmgrid` registers the `.sg` accessor on Xarray objects. This example applies a T6–42 spectral filter to a `DataArray`:

```python
import xarray as xr
import spharmgrid

field = xr.open_dataarray("msl.nc")
filtered = field.sg.filter("T6-42")
```

See the [Quick start](https://spharmgrid.readthedocs.io/en/latest/quickstart.html) for regridding, atmospheric wind diagnostics, direct-function equivalents, and further examples.

## Documentation

The optional CLI reads NetCDF, Zarr, and GRIB and writes NetCDF and Zarr. See the [CLI documentation](https://spharmgrid.readthedocs.io/en/latest/cli.html) for installation and usage.

See the full [documentation](https://spharmgrid.readthedocs.io/) for grid requirements, coordinate handling, CF metadata, atmospheric kinematics, inverse transforms, zero-mode conventions, and command-line use.

## References

See the documentation [References](https://spharmgrid.readthedocs.io/en/latest/references.html) for the scientific literature and software cited by spharmgrid.

## License

spharmgrid is distributed under the [BSD 3-Clause License](https://github.com/mwyau/spharmgrid/blob/main/LICENSE).
