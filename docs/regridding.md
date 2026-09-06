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
