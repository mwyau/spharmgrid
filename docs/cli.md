# Command-line interface

The `spharmgrid` command is a small file-oriented helper for the same spectral
filtering, regridding, and atmospheric wind diagnostics exposed by the Python
API. It delegates input decoding to xarray and installed xarray backends, calls
the same package functions as Python users, and writes NetCDF output through an
installed NetCDF-capable xarray engine.

Normal NetCDF input and output work with the `io` extra:

```bash
uv add "spharmgrid[io] @ git+https://github.com/mwyau/spharmgrid.git"
```

Other input formats, such as Zarr or GRIB, can be read when xarray can infer an
installed backend for the supplied path. The CLI does not provide GRIB or Zarr
output; use the Python API and xarray directly when another output format is
required.

```bash
spharmgrid info input.nc

spharmgrid filter input.nc output.nc \
  --var msl \
  --spectral T6-42 \
  --taper 0.1

spharmgrid regrid input.nc output.nc \
  --var msl \
  --grid gl --nlat 64 --nlon 128

spharmgrid kinematics wind.nc kinematics.nc
spharmgrid potentials wind.nc potentials.nc
```

`kinematics` computes relative vorticity and divergence; `potentials` computes
streamfunction and velocity potential. Both commands use the same CF/canonical
variable discovery as `Dataset.sg`. Pass `--u` and `--v` only when their input
names need an override.

Reconstruct wind with an explicit source representation:

```bash
spharmgrid wind diagnostics.nc wind.nc \
  --source vorticity_divergence \
  --vorticity vo --divergence d
```

Use `spharmgrid --help` or `spharmgrid <command> --help` for all input and
output naming options. The CLI does not implement a separate numerical path or
a file abstraction beyond xarray.
