# Command-line interface

The `spharmgrid` command applies filtering, regridding, and atmospheric wind diagnostics to xarray-supported files.

NetCDF input/output is available in the base install. Install other file backends only when needed:

```bash
uv add "spharmgrid[zarr] @ git+https://github.com/mwyau/spharmgrid.git"
uv add "spharmgrid[grib] @ git+https://github.com/mwyau/spharmgrid.git"
```

Input decoding is delegated to xarray and its installed backends. NetCDF input/output uses h5netcdf. Output paths ending in `.zarr` are written as Zarr stores when the `zarr` extra is installed. GRIB input is available through the `grib` extra.

```bash
spharmgrid info input.nc

spharmgrid filter input.nc output.nc \
  --var msl \
  --spectral T6-42 \
  --taper 0.1

spharmgrid filter input.zarr output.zarr \
  --var msl \
  --spectral T42

spharmgrid regrid input.grib output.zarr \
  --var msl \
  --grid gl --nlat 64 --nlon 128

spharmgrid kinematics wind.nc kinematics.nc
spharmgrid potentials wind.zarr potentials.zarr
```

`kinematics` computes relative vorticity and divergence. `potentials` computes streamfunction and velocity potential. Pass `--u` and `--v` when CF metadata or canonical short names do not identify the wind variables uniquely.

Reconstruct wind from vorticity and divergence with:

```bash
spharmgrid wind diagnostics.nc wind.nc \
  --source vorticity_divergence \
  --vorticity vo --divergence d
```

Use `spharmgrid --help` or `spharmgrid <command> --help` for all options.
