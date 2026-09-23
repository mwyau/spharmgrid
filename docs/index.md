# spharmgrid

Spherical harmonic tools for filtering, regridding, and kinematics in atmospheric
science with Xarray, JAX, and PyTorch.

**spharmgrid** (**sp**herical **harm**onic **grid**ding) implements spherical harmonic
filtering, regridding, differential operators, and atmospheric kinematics for global
Xarray fields, JAX arrays, and PyTorch tensors. It computes relative vorticity (`vo`),
divergence (`d`), streamfunction (`strf`), velocity potential (`vp`), Helmholtz
decomposition, and inverse wind transforms. The Xarray API uses
[DUCC](https://gitlab.mpcdf.mpg.de/mtr/ducc) (`ducc0`) for spherical harmonic
transforms; the optional JAX API uses
[S2FFT](https://github.com/astro-informatics/s2fft), and the optional PyTorch API uses
[torch-harmonics](https://github.com/NVIDIA/torch-harmonics).

Supported grids are full rectangular Gauss–Legendre (GL) and Clenshaw–Curtis (CC) grids.
Xarray operations preserve leading dimensions and coordinates; JAX and PyTorch
operations preserve leading array dimensions.

[GitHub](https://github.com/mwyau/spharmgrid) ·
[PyPI](https://pypi.org/project/spharmgrid/) ·
[conda-forge](https://anaconda.org/conda-forge/spharmgrid)

DOI: [10.5281/zenodo.22559210](https://doi.org/10.5281/zenodo.22559210)

## Install

Install with either pip or uv:

```bash
pip install spharmgrid
```

```bash
uv add spharmgrid
```

The JAX API requires JAX and S2FFT, installed with `spharmgrid[jax]`; see {doc}`jax`.
For CPU use, `spharmgrid[jax]` installs JAX and S2FFT. For GPU or TPU use, install the
appropriate JAX accelerator build first by following the
[official JAX installation instructions](https://docs.jax.dev/en/latest/installation.html),
then install `spharmgrid[jax]`; spharmgrid does not bundle or select CUDA or TPU builds.
The JAX API currently requires JAX x64 mode and `float64` spatial inputs; it does not
enable x64 globally.

The PyTorch API is installed with `spharmgrid[torch]`; see {doc}`torch`. For
accelerator-specific PyTorch builds, install the appropriate PyTorch build first, then
install `spharmgrid[torch]`.

## Quick start

Importing `spharmgrid` registers the `.sg` accessor on Xarray objects. This example
applies a T6–42 spectral filter to a `DataArray`:

```python
import xarray as xr
import spharmgrid

field = xr.open_dataarray("msl.nc")
filtered = field.sg.filter("T6-42")
```

For JAX arrays, enable x64 mode before creating arrays:

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

For PyTorch tensors, use `spharmgrid.torch`:

```python
import torch
import spharmgrid as sg
import spharmgrid.torch as sgt

grid = sg.gaussian_grid(64, 128)
field = torch.randn(grid.nlat, grid.nlon)
filtered = sgt.filter(field, grid=grid, truncation="T42")
```

```{toctree}
---
hidden: true
maxdepth: 2
---
guide
reference
changelog
```

## Citation

If you use spharmgrid in research, please cite the software release DOI:
[10.5281/zenodo.22559210](https://doi.org/10.5281/zenodo.22559210).

## Bibliography

See {doc}`references` for the scientific literature and software cited by spharmgrid.

## License

spharmgrid is distributed under the
[BSD 3-Clause License](https://github.com/mwyau/spharmgrid/blob/main/LICENSE).
