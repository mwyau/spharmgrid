# spharmgrid

Spherical harmonics regridding tool for atmospheric science.

**spharmgrid** (**sp**herical **harm**onics **grid**ding) provides spherical-harmonic filtering, regridding, differential operators, and atmospheric wind diagnostics for global xarray fields. It computes relative vorticity, divergence, streamfunction, velocity potential, Helmholtz decomposition, and inverse wind transforms. [DUCC0](https://gitlab.mpcdf.mpg.de/mtr/ducc) supplies the numerical spherical-harmonic transforms.

Supported grids are full rectangular Gauss--Legendre (GL) and pole-including Clenshaw--Curtis (CC) grids.

## Install

```bash
uv add git+https://github.com/mwyau/spharmgrid.git
```

Optional integrations can be installed separately:

```bash
uv add "spharmgrid[cf] @ git+https://github.com/mwyau/spharmgrid.git"
uv add "spharmgrid[dask] @ git+https://github.com/mwyau/spharmgrid.git"
uv add "spharmgrid[netcdf] @ git+https://github.com/mwyau/spharmgrid.git"
uv add "spharmgrid[zarr] @ git+https://github.com/mwyau/spharmgrid.git"
uv add "spharmgrid[grib] @ git+https://github.com/mwyau/spharmgrid.git"
```

## Quick start

Importing spharmgrid registers `.sg` on xarray `DataArray` and `Dataset` objects.

```python
import xarray as xr
import spharmgrid as sg

field = xr.open_dataarray("msl.nc")

filtered = field.sg.filter("T6-42")
tapered = field.sg.filter("T6-42", taper=0.1)

target = sg.gaussian_grid(64, 128)
regridded = field.sg.regrid(target)
combined = field.sg.regrid(target, spectral="T6-42", taper=0.1)
```

The same operations are available as functions:

```python
filtered = sg.filter(field, "T6-42", taper=0.1)
regridded = sg.regrid(field, target)
```

For atmospheric wind fields, a Dataset with `u` and `v` variables or their exact CF standard names can compute wind diagnostics and inverse transforms:

```python
ds = xr.open_dataset("wind.nc")

kin = ds.sg.kinematics()   # vo: relative vorticity; d: divergence
pot = ds.sg.potentials()   # strf: streamfunction; vp: velocity potential

reconstructed = xr.Dataset({"vo": kin.vo, "d": kin.d}).sg.wind()
target_wind = ds.sg.regrid_vector(target, spectral="T42")
parts = ds.sg.helmholtz()
gradient = field.sg.gradient()
recovered_field = gradient.gradient_eastward.sg.inverse_gradient(
    gradient.gradient_northward,
)
wind_laplacian = ds.sg.vector_laplacian()
```

The command-line interface delegates file decoding to xarray. NetCDF and Zarr output are supported directly when their optional backends are installed; GRIB input is available through the optional `grib` extra.

`inverse_gradient()` sets the scalar degree-zero coefficient to zero and returns the irrotational projection when the supplied vector contains a rotational component. `inverse_vector_laplacian()` sets its degree-zero vector-harmonic slots to zero. See the operator documentation for these conventions.

`T42` retains total degrees 0 through 42. `T6-42` retains degrees 6 through 42. Filtering uses a hard spectral selection by default. `taper=0.1` applies the Sardeshmukh--Hoskins exponential response with value 0.1 at the upper retained degree.

See the full [documentation](https://spharmgrid.readthedocs.io/) for grid requirements, coordinate handling, CF metadata, wind diagnostics, inverse transforms, zero-mode conventions, and command-line use.

## References

See the documentation [references](https://spharmgrid.readthedocs.io/en/latest/references.html) for scientific and software citations. Related spherical-harmonic code is also used in [PyStormTracker](https://github.com/mwyau/PyStormTracker).
