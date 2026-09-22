# spharmgrid

[![CI](https://github.com/mwyau/spharmgrid/actions/workflows/ci.yml/badge.svg)](https://github.com/mwyau/spharmgrid/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/spharmgrid/badge/?version=latest)](https://spharmgrid.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/github/mwyau/spharmgrid/graph/badge.svg?token=5kDDQ8Ii0Z)](https://codecov.io/github/mwyau/spharmgrid)
[![PyPI version](https://img.shields.io/pypi/v/spharmgrid)](https://pypi.org/project/spharmgrid/)
[![PyPI Python Version](https://img.shields.io/pypi/pyversions/spharmgrid)](https://pypi.org/project/spharmgrid/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/spharmgrid)](https://anaconda.org/channels/conda-forge/packages/spharmgrid/overview)
[![GitHub License](https://img.shields.io/github/license/mwyau/spharmgrid)](https://github.com/mwyau/spharmgrid/blob/main/LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.22559210-blue.svg)](https://doi.org/10.5281/zenodo.22559210)

Spherical harmonic tools for filtering, regridding, and kinematics in atmospheric science with Xarray, PyTorch, and JAX.

**spharmgrid** (**sp**herical **harm**onic **grid**ding) implements spherical harmonic filtering, regridding, differential operators, and atmospheric kinematics for global Xarray fields, PyTorch tensors, and JAX arrays. It computes relative vorticity, divergence, streamfunction, velocity potential, Helmholtz decomposition, and inverse wind transforms. The Xarray/NumPy API uses [DUCC](https://gitlab.mpcdf.mpg.de/mtr/ducc) (`ducc0`) for spherical harmonic transforms; the optional PyTorch API uses [torch-harmonics](https://github.com/NVIDIA/torch-harmonics), and the optional JAX API uses [S2FFT](https://github.com/astro-informatics/s2fft).

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

Optional extras are:

- `spharmgrid[dask]` — Dask-backed lazy execution;
- `spharmgrid[cf]` — optional `cf-xarray` coordinate discovery;
- `spharmgrid[cli]` — command-line NetCDF, Zarr, and GRIB I/O;
- `spharmgrid[jax]` — JAX arrays and S2FFT transforms;
- `spharmgrid[torch]` — PyTorch tensors and torch-harmonics transforms.

The `spharmgrid.torch` API can be installed with `spharmgrid[torch]`. See the
[PyTorch API documentation](https://spharmgrid.readthedocs.io/en/latest/torch.html)
for installation instructions.

The `spharmgrid.jax` API requires JAX and S2FFT. See the
[JAX API documentation](https://spharmgrid.readthedocs.io/en/latest/jax.html)
for supported GL and CC/MWSS dimensions, x64 and dtype requirements, and
examples. For CPU use, `spharmgrid[jax]` installs JAX and S2FFT. For GPU or
TPU use, install the appropriate JAX accelerator build first by following the
[official JAX installation instructions](https://docs.jax.dev/en/latest/installation.html),
then install `spharmgrid[jax]`; spharmgrid does not bundle or select CUDA or
TPU builds. The JAX API currently requires JAX x64 mode and `float64` spatial
inputs; it does not enable x64 globally.

Install `spharmgrid[cli,dask]` to use the transforming CLI commands.

For a standalone command-line installation:

```bash
uv tool install "spharmgrid[cli,dask]"
```

For a project environment, install the CLI and Dask extras with either:

```bash
uv add "spharmgrid[cli,dask]"
```

```bash
pip install "spharmgrid[cli,dask]"
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

For differentiable PyTorch workflows, install `spharmgrid[torch]` and use
`spharmgrid.torch`:

```python
import torch
import spharmgrid as sg
import spharmgrid.torch as sgt

grid = sg.gaussian_grid(64, 128)
field = torch.randn(grid.nlat, grid.nlon)
filtered = sgt.filter(field, grid=grid, truncation="T42")
```

See the [PyTorch API documentation](https://spharmgrid.readthedocs.io/en/latest/torch.html) for tensor dimensions, reusable `torch.nn` modules, device/autograd behavior, and PyTorch bandwidth limits.

For differentiable JAX workflows, configure JAX x64 mode before creating
arrays and use `spharmgrid.jax`:

```python
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import spharmgrid as sg
import spharmgrid.jax as sgj

grid = sg.gaussian_grid(64, 127)
field = jnp.ones((grid.nlat, grid.nlon))
filtered = sgj.filter(field, grid=grid, truncation="T42")
```

The last two array dimensions are latitude and longitude.

## Documentation

The optional CLI reads NetCDF, Zarr, and GRIB and writes NetCDF and Zarr. See the [CLI documentation](https://spharmgrid.readthedocs.io/en/latest/cli.html) for installation and usage.

See the full [documentation](https://spharmgrid.readthedocs.io/) for grid requirements, coordinate handling, CF metadata, atmospheric kinematics, inverse transforms, zero-mode conventions, and command-line use.

## Citation

If you use spharmgrid in research, please cite the software release DOI: [10.5281/zenodo.22559210](https://doi.org/10.5281/zenodo.22559210). Citation metadata are also provided in [`CITATION.cff`](https://github.com/mwyau/spharmgrid/blob/main/CITATION.cff).

## References

See the documentation [References](https://spharmgrid.readthedocs.io/en/latest/references.html) for the scientific literature and software cited by spharmgrid.

## License

spharmgrid is distributed under the [BSD 3-Clause License](https://github.com/mwyau/spharmgrid/blob/main/LICENSE).
