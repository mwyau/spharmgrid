# Command-line interface

The `spharmgrid` command applies filtering, regridding, and atmospheric kinematics to files supported by Xarray.

The `spharmgrid` executable is installed with the core package, so
`spharmgrid --help` and `spharmgrid --version` work without the CLI extra.
File-processing commands require the optional CLI backends. If they are not
installed, the command exits with an installation message.

Install the command-line I/O dependencies with:

```bash
uv tool install "spharmgrid[cli]"
```

This installs the standalone `spharmgrid` executable in an isolated uv tool
environment. In a normal project environment, use either:

```bash
uv add "spharmgrid[cli]"
pip install "spharmgrid[cli]"
```

The CLI reads and writes Zarr through Xarray's Zarr methods. Other input paths use `xarray.open_dataset()`, so installed backends such as `h5netcdf` and `cfgrib` handle NetCDF and GRIB input. Non-Zarr outputs are written as NetCDF with `h5netcdf`. GRIB output is not supported.

```bash
spharmgrid info input.nc

spharmgrid filter input.nc output.nc \
  --var msl \
  --truncation T6-42 \
  --taper 0.1

spharmgrid filter input.zarr output.zarr \
  --var msl \
  --truncation T42

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
