# spharmgrid

Spherical harmonic tools for filtering, regridding, and kinematics in atmospheric science with Xarray and PyTorch.

**spharmgrid** (**sp**herical **harm**onic **grid**ding) implements spherical harmonic filtering, regridding, differential operators, and atmospheric kinematics for global Xarray fields and PyTorch tensors. It computes relative vorticity (`vo`), divergence (`d`), streamfunction (`strf`), velocity potential (`vp`), Helmholtz decomposition, and inverse wind transforms. The Xarray/NumPy API uses [DUCC](https://gitlab.mpcdf.mpg.de/mtr/ducc) (`ducc0`) for spherical harmonic transforms; the optional PyTorch API uses [torch-harmonics](https://github.com/NVIDIA/torch-harmonics).

Supported grids are full rectangular Gauss–Legendre (GL) and Clenshaw–Curtis (CC) grids. Xarray operations preserve leading dimensions and coordinates; PyTorch operations preserve leading tensor dimensions.

[GitHub](https://github.com/mwyau/spharmgrid) · [PyPI](https://pypi.org/project/spharmgrid/) · [conda-forge](https://anaconda.org/conda-forge/spharmgrid)

DOI: [10.5281/zenodo.22559210](https://doi.org/10.5281/zenodo.22559210)

## Install

Install with either pip or uv:

```bash
pip install spharmgrid
```

```bash
uv add spharmgrid
```

The PyTorch API requires PyTorch and `torch-harmonics`, installed separately; see {doc}`torch`.

## Quick start

Importing `spharmgrid` registers the `.sg` accessor on Xarray objects. This example applies a T6–42 spectral filter to a `DataArray`:

```python
import xarray as xr
import spharmgrid

field = xr.open_dataarray("msl.nc")
filtered = field.sg.filter("T6-42")
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

If you use spharmgrid in research, please cite the software release DOI: [10.5281/zenodo.22559210](https://doi.org/10.5281/zenodo.22559210).

## Bibliography

See {doc}`references` for the scientific literature and software cited by spharmgrid.

## License

spharmgrid is distributed under the [BSD 3-Clause License](https://github.com/mwyau/spharmgrid/blob/main/LICENSE).
