# Quick start

Import spharmgrid once to register `.sg` on xarray `DataArray` and `Dataset`
objects. The accessor is the primary interface; each operation has an
equivalent direct function.

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

The direct equivalents call the same numerical implementation:

```python
filtered = sg.filter(field, "T6-42", taper=0.1)
regridded = sg.regrid(field, target)
```

## Atmospheric wind diagnostics

For a Dataset containing canonical `u` and `v` wind variables, or variables
with the corresponding exact CF standard names, no input-variable arguments
are needed.

```python
wind = xr.open_dataset("wind.nc")

kin = wind.sg.kinematics()  # vo: relative vorticity; d: divergence
pot = wind.sg.potentials()  # strf: streamfunction; vp: velocity potential

reconstructed = xr.Dataset({"vo": kin.vo, "d": kin.d}).sg.wind()
```

The individual diagnostics are also available directly as `vorticity()`,
`divergence()`, `streamfunction()`, and `velocity_potential()`. See
{doc}`kinematics` for rotational/divergent wind reconstruction and sign
conventions.

All operations preserve non-spatial dimensions and their xarray coordinates.
Read {doc}`grids` before applying an operation to a new data source: only
valid global GL and CC sampling is accepted.
