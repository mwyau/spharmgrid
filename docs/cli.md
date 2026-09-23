# Command-line interface

The `spharmgrid` command applies filtering, regridding, and atmospheric kinematics to
files supported by Xarray.

The `spharmgrid` executable is installed with the core package, so `spharmgrid --help`
and `spharmgrid --version` work without the CLI extra. File-processing commands require
the optional CLI backends. If they are not installed, the command exits with an
installation message. Transforming commands also require Dask; install both extras for
the complete CLI.

Install the command-line dependencies with:

```bash
uv tool install "spharmgrid[cli,dask]"
```

This installs the standalone `spharmgrid` executable in an isolated uv tool environment.
In a normal project environment, use either:

```bash
uv add "spharmgrid[cli,dask]"
pip install "spharmgrid[cli,dask]"
```

The CLI reads and writes Zarr through Xarray's Zarr methods. Other input paths use
`xarray.open_dataset()`, so installed backends such as `h5netcdf` and `cfgrib` handle
NetCDF and GRIB input. Non-Zarr outputs are written as NetCDF with `h5netcdf`. GRIB
output is not supported.

Transform commands use Dask-backed Xarray inputs. `--workers` sets Dask task concurrency
and `--sht-threads` sets DUCC threads per spherical harmonic transform. When values are
omitted, the CLI resolves them from the available CPU count.

| `--workers` | `--sht-threads` | resolved behavior                      |
| ----------- | --------------- | -------------------------------------- |
| omitted     | omitted         | `T=min(4,C)`, `W=ceil(C/T)`            |
| `W`         | omitted         | `T=ceil(C/W)`                          |
| omitted     | `T`             | `W=ceil(C/T)`                          |
| `W`         | `T`             | both requested values are used exactly |

Here `C` is the available CPU count. Ceiling division can oversubscribe the available
CPU count. The informational `info` command has neither option.

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

`kinematics` computes relative vorticity and divergence. `potentials` computes
streamfunction and velocity potential. Pass `--u` and `--v` when CF metadata or
canonical short names do not identify the wind variables uniquely.

Reconstruct wind from vorticity and divergence with:

```bash
spharmgrid wind diagnostics.nc wind.nc \
    --source vorticity_divergence \
    --vorticity vo --divergence d
```

Use `spharmgrid --help` or `spharmgrid <command> --help` for all options.
