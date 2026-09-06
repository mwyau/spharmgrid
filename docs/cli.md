# Command-line interface

The `spharmgrid` command applies filtering, regridding, and atmospheric wind diagnostics to xarray-supported files.

The `spharmgrid` executable is installed with the core package, so
`spharmgrid --help` and `spharmgrid --version` work without the CLI extra.
File-processing commands require the optional CLI backends. If they are not
installed, the command exits with an actionable installation message.

Install the complete command-line I/O environment with:

```bash
uv tool install "spharmgrid[cli]"
```

This installs the standalone `spharmgrid` executable in an isolated uv tool
environment. In a normal project environment, use either:

```bash
uv add "spharmgrid[cli]"
pip install "spharmgrid[cli]"
```

The CLI supports NetCDF read/write, Zarr read/write, and GRIB input. Actual
decoding and encoding are delegated to xarray and the backend packages in the
`cli` extra: `h5netcdf`, `zarr`, and `cfgrib`. A path ending in `.zarr` uses
the Zarr path; other inputs use xarray's normal dataset-opening dispatch, and
other outputs are written as NetCDF through `h5netcdf`. GRIB output is not
supported.

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
