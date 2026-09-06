# Spectral regridding

`regrid()` supports all source-target combinations of Gauss–Legendre (GL) and Clenshaw–Curtis (CC) grids:

```text
GL -> GL    GL -> CC
CC -> GL    CC -> CC
```

```python
target = sg.clenshaw_curtis_grid(65, 128)
result = field.sg.regrid(target)

result = sg.regrid(field, reference_field)
```

A `Grid` target keeps the source horizontal dimension names and adds CF latitude/longitude metadata. An xarray target supplies the target horizontal dimensions and coordinates.

## Filtering during regridding

A spectral range and taper can be applied during regridding:

```python
result = field.sg.regrid(
    target,
    truncation="T6-42",
    taper=0.1,
)
```

Without an explicit spectral range, regridding retains the spherical harmonic content represented by both source and target grids. An explicit `Tn` range must be representable on both grids.

## Vector regridding

`regrid_vector()` uses the same GL/CC combinations and the same `truncation`, `lmin`, `lmax`, and `taper` arguments as scalar `regrid()`:

```python
wind_on_target = ds.sg.regrid_vector(target, truncation="T6-42", taper=0.1)
wind_on_target = sg.regrid_vector(u, v, target, truncation="T6-42", taper=0.1)
```

Dataset accessors identify `u` and `v` from exact CF standard names or canonical short names.

Vector regridding uses vector spherical harmonics for the eastward and northward components and applies the spectral selection or taper to both harmonic families. Output names default to `u` and `v`; use `eastward=` and `northward=` to change them.
