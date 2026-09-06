# spharmgrid

Spherical harmonic tools for filtering, regridding, and kinematics in atmospheric science with Xarray.

**spharmgrid** (**sp**herical **harm**onic **grid**ding) implements spherical harmonic filtering, regridding, differential operators, and atmospheric kinematics for global Xarray fields. It computes relative vorticity (`vo`), divergence (`d`), streamfunction (`strf`), velocity potential (`vp`), Helmholtz decomposition, and inverse wind transforms. [DUCC](https://gitlab.mpcdf.mpg.de/mtr/ducc) performs the numerical spherical harmonic transforms.

Supported grids are full rectangular Gauss–Legendre (GL) and Clenshaw–Curtis (CC) grids. Leading dimensions and Xarray coordinates are preserved, so fields such as `(time, level, lat, lon)` can be transformed without reshaping them first.

**Project:** [GitHub](https://github.com/mwyau/spharmgrid) · [PyPI](https://pypi.org/project/spharmgrid/) · [conda-forge](https://anaconda.org/conda-forge/spharmgrid)

## Install

Install with either pip or uv:

```bash
pip install spharmgrid
```

```bash
uv add spharmgrid
```

## Quick start

Importing `spharmgrid` registers the `.sg` accessor on Xarray objects. This example applies a T6–42 spectral filter to a `DataArray`:

```python
import xarray as xr
import spharmgrid

field = xr.open_dataarray("msl.nc")
filtered = field.sg.filter("T6-42")
```

## Contents

```{toctree}
:maxdepth: 1

quickstart
filtering
regridding
kinematics
operators
grids
cf
cli
api
comparison
references
```

## References

See {doc}`references` for the scientific literature and software cited by spharmgrid.

## License

spharmgrid is distributed under the [BSD 3-Clause License](https://github.com/mwyau/spharmgrid/blob/main/LICENSE).
