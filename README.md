# spharmgrid

Spherical harmonic tools for filtering, regridding, and kinematics in atmospheric science with xarray.

**spharmgrid** (**sp**herical **harm**onic **grid**ding) implements spherical harmonic filtering, regridding, differential operators, and atmospheric kinematics for global xarray fields. It computes relative vorticity, divergence, streamfunction, velocity potential, Helmholtz decomposition, and inverse wind transforms. [DUCC](https://gitlab.mpcdf.mpg.de/mtr/ducc) performs the numerical spherical harmonic transforms.

Supported grids are full rectangular Gauss–Legendre (GL) and pole-including Clenshaw–Curtis (CC) grids.

## Install

Install with either pip or uv:

```bash
pip install spharmgrid
```

```bash
uv add spharmgrid
```

Optional extras are:

- `spharmgrid[dask]` — Dask-backed lazy execution;
- `spharmgrid[cf]` — optional cf-xarray coordinate discovery;
- `spharmgrid[cli]` — command-line NetCDF, Zarr, and GRIB I/O.

Dask-backed transforms execute lazily. DUCC uses four threads per transform. For the local Dask scheduler, spharmgrid sets `num_workers=max(1, os.cpu_count() // 4)` unless `num_workers` is configured.

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

Importing spharmgrid registers `.sg` on xarray `DataArray` and `Dataset` objects.

```python
import xarray as xr
import spharmgrid as sg

field = xr.open_dataarray("msl.nc")

filtered = field.sg.filter("T6-42")
tapered = field.sg.filter("T6-42", taper=0.1)

target = sg.gaussian_grid(64, 128)
regridded = field.sg.regrid(target)
combined = field.sg.regrid(target, truncation="T6-42", taper=0.1)
```

```python
filtered = sg.filter(field, "T6-42", taper=0.1)
regridded = sg.regrid(field, target)
```

For atmospheric wind fields, a Dataset with `u` and `v` variables or their exact CF standard names can compute kinematic quantities and inverse transforms:

```python
ds = xr.open_dataset("wind.nc")

kin = ds.sg.kinematics()   # vo: relative vorticity; d: divergence
pot = ds.sg.potentials()   # strf: streamfunction; vp: velocity potential

reconstructed = xr.Dataset({"vo": kin.vo, "d": kin.d}).sg.wind()
target_wind = ds.sg.regrid_vector(target, truncation="T42")
parts = ds.sg.helmholtz()
gradient = field.sg.gradient()
recovered_field = gradient.gradient_eastward.sg.inverse_gradient(
    gradient.gradient_northward,
)
wind_laplacian = ds.sg.vector_laplacian()
```

The optional CLI reads NetCDF, Zarr, and GRIB and writes NetCDF and Zarr. See the [CLI documentation](https://spharmgrid.readthedocs.io/en/latest/cli.html) for installation and usage.

`T42` retains total degrees 0 through 42. `T6-42` retains degrees 6 through 42. Filtering uses a hard spectral selection by default. `taper=0.1` applies the Sardeshmukh–Hoskins exponential response with value 0.1 at the upper retained degree.

See the full [documentation](https://spharmgrid.readthedocs.io/) for grid requirements, coordinate handling, CF metadata, atmospheric kinematics, inverse transforms, zero-mode conventions, and command-line use.

## References

See the documentation [references](https://spharmgrid.readthedocs.io/en/latest/references.html) for scientific and software citations. Related spherical harmonic code is also used in [PyStormTracker](https://github.com/mwyau/PyStormTracker).
