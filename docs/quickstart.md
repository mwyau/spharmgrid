# Quick start

Import spharmgrid to register `.sg` on xarray `DataArray` and `Dataset`
objects.

## Filtering and regridding

```python
import xarray as xr
import spharmgrid as sg

field = xr.open_dataarray("msl.nc")
print(field.sg.grid_type)  # "gl" or "cc"

filtered = field.sg.filter("T6-42")
tapered = field.sg.filter("T6-42", taper=0.1)

target = sg.gaussian_grid(64, 128)
regridded = field.sg.regrid(target)
combined = field.sg.regrid(target, spectral="T6-42", taper=0.1)
```

### Direct functions

```python
filtered = sg.filter(field, "T6-42", taper=0.1)
regridded = sg.regrid(field, target)
```

## Atmospheric wind diagnostics

With canonical `u` and `v` variable names or exact CF standard names, Dataset
methods identify the wind components automatically.

```python
wind = xr.open_dataset("wind.nc")

kin = wind.sg.kinematics()  # vo: relative vorticity; d: divergence
pot = wind.sg.potentials()  # strf: streamfunction; vp: velocity potential

reconstructed = xr.Dataset({"vo": kin.vo, "d": kin.d}).sg.wind()
```

The individual diagnostics are also available as `vorticity()`, `divergence()`,
`streamfunction()`, and `velocity_potential()`. See {doc}`kinematics` for wind
reconstruction and sign conventions.

All operations preserve non-spatial dimensions and xarray coordinates. See
{doc}`grids` for supported sampling geometries and coordinate requirements.
