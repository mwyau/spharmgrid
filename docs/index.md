# spharmgrid

Spherical harmonic tools for filtering, regridding, and kinematics in atmospheric science with xarray.

**spharmgrid** (**sp**herical **harm**onic **grid**ding) implements spherical harmonic filtering, regridding, differential operators, and atmospheric kinematics for global xarray fields. It computes relative vorticity (`vo`), divergence (`d`), streamfunction (`strf`), velocity potential (`vp`), Helmholtz decomposition, and inverse wind transforms. [DUCC](https://gitlab.mpcdf.mpg.de/mtr/ducc) performs the numerical spherical harmonic transforms.

Supported grids are full rectangular Gauss–Legendre (GL) and pole-including Clenshaw–Curtis (CC) grids. Arbitrary leading dimensions and xarray coordinates are preserved, so fields such as `(time, level, lat, lon)` can be transformed without reshaping them first.

## Install

Install with either pip or uv:

```bash
pip install spharmgrid
```

```bash
uv add spharmgrid
```

## Quick start

Importing spharmgrid registers the `.sg` accessor on xarray objects. This example applies a T6–42 spectral filter to a `DataArray`:

```python
import xarray as xr
import spharmgrid

field = xr.open_dataarray("msl.nc")
filtered = field.sg.filter("T6-42")
```

## Contents

```{toctree}
:maxdepth: 2

quickstart
filtering
regridding
kinematics
operators
grids
cf
comparison
cli
api
references
```

## References

See {doc}`references` for the scientific literature and software cited by spharmgrid.

## License

spharmgrid is distributed under the [BSD 3-Clause License](https://github.com/mwyau/spharmgrid/blob/main/LICENSE).
