# Spectral regridding

`regrid()` supports all combinations of GL and CC grids:

```text
GL -> GL    GL -> CC
CC -> GL    CC -> CC
```

```python
target = sg.clenshaw_curtis_grid(65, 128)
result = field.sg.regrid(target)

# A reference xarray object can supply target horizontal coordinates.
result = sg.regrid(field, reference_field)
```

For a `Grid` target, spharmgrid keeps the source horizontal dimension names and
adds CF latitude/longitude metadata. For an xarray target, the target's
horizontal dimension and coordinate names are used.

## Filtering during regridding

A spectral range and taper can be applied in the same call:

```python
result = field.sg.regrid(
    target,
    spectral="T6-42",
    taper=0.1,
)
```

Without an explicit spectral range, regridding retains the content jointly
representable by the source and target grids. An explicit `Tn` range must be
representable by both grids.

## Vector regridding

`regrid_vector()` has the same supported GL/CC source-target combinations and
the same `spectral`, `lmin`, `lmax`, and `taper` options as scalar `regrid()`:

```python
wind_on_target = ds.sg.regrid_vector(target, spectral="T6-42", taper=0.1)

# The direct equivalent accepts the eastward and northward components.
wind_on_target = sg.regrid_vector(u, v, target, spectral="T6-42", taper=0.1)
```

Dataset accessors discover `u` and `v` by exact CF standard names or canonical
short names.

Vector regridding applies the selection or taper to both vector-harmonic
families. It does not scalar-regrid the eastward and northward components
independently. Output names default to `u` and `v`; use `eastward=` and
`northward=` to change them.
