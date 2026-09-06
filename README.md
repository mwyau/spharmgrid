# spharmgrid

Spherical harmonic tools for filtering, regridding, and kinematics in atmospheric science with xarray.

**spharmgrid** (**sp**herical **harm**onic **grid**ding) provides spherical harmonic filtering, regridding, differential operators, and atmospheric wind diagnostics for global xarray fields. It computes relative vorticity, divergence, streamfunction, velocity potential, Helmholtz decomposition, and inverse wind transforms. [DUCC0](https://gitlab.mpcdf.mpg.de/mtr/ducc) supplies the numerical spherical harmonic transforms.

Supported grids are full rectangular Gauss--Legendre (GL) and pole-including Clenshaw--Curtis (CC) grids.

## Install

```bash
pip install spharmgrid
uv add spharmgrid
```

The core `spharmgrid` install provides the numerical Python/xarray API.
Optional capabilities are:

- `spharmgrid[dask]` — Dask-backed lazy execution;
- `spharmgrid[cf]` — optional cf-xarray coordinate discovery;
- `spharmgrid[cli]` — command-line NetCDF, Zarr, and GRIB I/O.

For a standalone command-line installation, use the isolated tool environment:

```bash
uv tool install "spharmgrid[cli]"
```

For a normal project environment, use either:

```bash
uv add "spharmgrid[cli]"
pip install "spharmgrid[cli]"
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

The command-line interface is an optional file-I/O capability. Install
`spharmgrid[cli]` for NetCDF and Zarr read/write plus GRIB read support. File
decoding and encoding are delegated to xarray and its installed backend
packages; GRIB output is not supported. The core-installed `spharmgrid`
executable still supports `spharmgrid --help` and `spharmgrid --version`; file
processing reports how to install the CLI extra when its backends are absent.

`inverse_gradient()` sets the scalar degree-zero coefficient to zero and returns the irrotational projection when the supplied vector contains a rotational component. `inverse_vector_laplacian()` sets its degree-zero vector-harmonic slots to zero. See the operator documentation for these conventions.

`T42` retains total degrees 0 through 42. `T6-42` retains degrees 6 through 42. Filtering uses a hard spectral selection by default. `taper=0.1` applies the Sardeshmukh--Hoskins exponential response with value 0.1 at the upper retained degree.

See the full [documentation](https://spharmgrid.readthedocs.io/) for grid requirements, coordinate handling, CF metadata, wind diagnostics, inverse transforms, zero-mode conventions, and command-line use.

## References

See the documentation [references](https://spharmgrid.readthedocs.io/en/latest/references.html) for scientific and software citations. Related spherical harmonic code is also used in [PyStormTracker](https://github.com/mwyau/PyStormTracker).