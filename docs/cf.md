# CF metadata, xarray dimensions, and Dask

## Dataset variable discovery

Dataset wind methods identify each physical quantity in this order:

1. an explicit argument, such as `u="ua"`;
2. one unique exact CF `standard_name` match;
3. the canonical short name;
4. an error if no unique match exists.

| Quantity | Short name | CF `standard_name` |
| --- | --- | --- |
| Eastward wind | `u` | `eastward_wind` |
| Northward wind | `v` | `northward_wind` |
| Relative vorticity | `vo` | `atmosphere_relative_vorticity` |
| Divergence | `d` | `divergence_of_wind` |
| Streamfunction | `strf` | `atmosphere_horizontal_streamfunction` |
| Velocity potential | `vp` | `atmosphere_horizontal_velocity_potential` |

```python
ds["uwind"].attrs["standard_name"] = "eastward_wind"
ds["vwind"].attrs["standard_name"] = "northward_wind"

kin = ds.sg.kinematics()
```

If more than one variable matches the same quantity, spharmgrid reports the candidates and requires an explicit variable name.

## Output metadata

Derived wind diagnostics use the CF metadata in the table. Renaming an output does not change its physical metadata:

```python
vo = ds.sg.vorticity(output="vort")
assert vo.attrs["standard_name"] == "atmosphere_relative_vorticity"
```

Filtering and regridding preserve the input variable name and attributes. Generated target coordinates include CF latitude and longitude metadata.

## Dimensions and time

Transforms preserve leading dimensions and coordinate alignment, including arrays such as `(time, level, lat, lon)`. NumPy datetimes and `cftime` calendar objects remain xarray coordinates through the operation.

## cf-xarray

Install the optional `cf-xarray` extra with:

```bash
uv add "spharmgrid[cf]"
```

When installed, `cf-xarray` provides an additional latitude/longitude discovery path after exact CF metadata and canonical coordinate names are checked.

## Dask

Install Dask support with:

```bash
uv add "spharmgrid[dask]"
```

Dask-backed xarray fields remain lazy. Horizontal core dimensions are rechunked when required by xarray generalized ufunc execution. The default is one DUCC thread per Dask task; pass `nthreads=` to use another value.
