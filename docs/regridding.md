# Spectral regridding

`regrid()` supports every pairwise combination of GL and CC grids:

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

For a `Grid` target, spharmgrid uses the source horizontal dimension names and
attaches generated CF coordinate metadata. For an xarray target, it uses the
target's horizontal dimension and coordinate names.

## One analysis and one synthesis

Filtering and regridding share one transform cycle:

```python
result = field.sg.regrid(
    target,
    spectral="T6-42",
    taper=0.1,
)
```

This performs one source analysis, applies the optional selection/taper to the
coefficients, and performs one target synthesis. It does not call public
`filter()` followed by public `regrid()`.

Without a requested range, the operation uses content jointly representable by
the source and target geometry. If an explicit `Tn` range cannot be represented
on either grid, it raises instead of silently clamping the range.
