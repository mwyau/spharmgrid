# CF metadata, xarray dimensions, and Dask

## Dataset variable discovery

Dataset wind methods resolve each physical quantity in this order:

1. an explicit argument, such as `u="ua"`;
2. one unique exact CF `standard_name` match;
3. the canonical short name;
4. an error if no unique match exists.

The canonical identities are:

| Quantity | Short name | CF `standard_name` |
| --- | --- | --- |
| Eastward wind | `u` | `eastward_wind` |
| Northward wind | `v` | `northward_wind` |
| Relative vorticity | `vo` | `atmosphere_relative_vorticity` |
| Divergence | `d` | `divergence_of_wind` |
| Streamfunction | `strf` | `atmosphere_horizontal_streamfunction` |
| Velocity potential | `vp` | `atmosphere_horizontal_velocity_potential` |

For example:

```python
ds["uwind"].attrs["standard_name"] = "eastward_wind"
ds["vwind"].attrs["standard_name"] = "northward_wind"

kin = ds.sg.kinematics()
```

If more than one variable matches the same quantity, spharmgrid lists the
candidates and requires an explicit argument. It does not maintain a broad
heuristic alias list such as `UGRD` or `uwnd`.

## Output semantics

Derived wind diagnostics carry the exact CF metadata in the table. Renaming an
output does not change its semantic attributes:

```python
vo = ds.sg.vorticity(output="vort")
assert vo.attrs["standard_name"] == "atmosphere_relative_vorticity"
```

Filtering and regridding preserve the input variable name and attributes
because they retain the same physical quantity. Generated target coordinates
carry ordinary CF latitude/longitude metadata; GL and CC are numerical
sampling properties, not CF standard names.

## Dimensions, time, and optional cf-xarray

Transforms operate only on detected horizontal dimensions. They preserve
arbitrary leading dimensions and coordinate alignment, including shapes such
as `(time, level, lat, lon)`. xarray remains responsible for CF time decoding
and encoding, so NumPy datetimes and `cftime` calendar objects pass through
unchanged.

`cf-xarray` is optional. spharmgrid does not need it for exact CF metadata or
canonical coordinate names, but consults it as a later coordinate-discovery
aid when installed:

```bash
uv add "spharmgrid[cf] @ git+https://github.com/mwyau/spharmgrid.git"
```

## Dask

Dask is optional as well:

```bash
uv add "spharmgrid[dask] @ git+https://github.com/mwyau/spharmgrid.git"
```

For Dask-backed xarray fields, transforms remain lazy. Horizontal core
dimensions are rechunked only when xarray needs that to invoke a gufunc. The
default uses one DUCC thread per Dask task to avoid nested oversubscription;
pass `nthreads=` to make an explicit choice. Eager calls also use DUCC's
documented one-thread Python default when `nthreads=None`.
