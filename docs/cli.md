# Command-line interface

The `spharmgrid` command is a small file-oriented helper. It delegates reading
and writing to xarray and calls the same package functions as Python users.
Normal NetCDF works when an xarray NetCDF engine is installed; optional engines
can also support Zarr or GRIB input.

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

`kinematics` and `potentials` use the same CF/canonical variable discovery as
`Dataset.sg`. Pass `--u` and `--v` only when their input names need an
override.

Reconstruct wind with an explicit source representation:

```bash
spharmgrid wind diagnostics.nc wind.nc \
  --source vorticity_divergence \
  --vorticity vo --divergence d
```

Use `spharmgrid --help` or `spharmgrid <command> --help` for all input and
output naming options. The CLI does not implement a separate numerical path or
a file abstraction beyond xarray.
