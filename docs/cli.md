# Command-line interface

The `spharmgrid` command provides file-based access to filtering, regridding,
and atmospheric wind diagnostics.

Install NetCDF input/output support with:

```bash
uv add "spharmgrid[io] @ git+https://github.com/mwyau/spharmgrid.git"
```

Input is read through xarray. Other formats such as Zarr or GRIB can be read
when the corresponding xarray backend is installed. CLI output is NetCDF.

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

`kinematics` computes relative vorticity and divergence. `potentials` computes
streamfunction and velocity potential. Pass `--u` and `--v` when the wind
variable names cannot be identified from CF metadata or canonical short names.

Reconstruct wind from vorticity and divergence with:

```bash
spharmgrid wind diagnostics.nc wind.nc \
  --source vorticity_divergence \
  --vorticity vo --divergence d
```

Use `spharmgrid --help` or `spharmgrid <command> --help` for all options.
