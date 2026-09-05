# Supported grids and coordinates

spharmgrid supports two full rectangular global sampling geometries. `ducc0`
uses their geometry labels internally; spharmgrid exposes lowercase Python
labels, `"gl"` and `"cc"`.

## Gauss--Legendre (GL)

A GL grid has Gaussian latitude nodes and an equal number of equally spaced
longitudes on every latitude ring. The latitude nodes are generated from
`ducc0.misc.GL_thetas`.

```python
grid = sg.gaussian_grid(64, 128, lon0=0.0, latitude_order="ascending")
```

## Clenshaw--Curtis (CC)

A CC grid has equally spaced latitudes including both poles and equally spaced
longitudes.

```python
grid = sg.clenshaw_curtis_grid(65, 128, latitude_order="descending")
```

A generic regular latitude--longitude grid is not necessarily CC. Detection
requires an equally spaced latitude coordinate with `-90` and `90` present,
plus a globally cyclic non-duplicated longitude coordinate. A grid that omits
a pole is rejected rather than silently treated as CC.

## Coordinate discovery and cyclic representations

Latitude and longitude are discovered in this order:

1. unique exact CF `standard_name` values `latitude` and `longitude`;
2. the canonical coordinate names `lat`/`latitude` and `lon`/`longitude`;
3. optional `cf-xarray` assistance when installed.

Ambiguous coordinates raise an error. spharmgrid does not guess that the last
two dimensions are spatial.

Latitude may be ascending or descending. Longitudes may use `[0, 360)` or a
`[-180, 180)`-style cycle and may begin at another regular cyclic point. The
implementation moves longitude values and data columns together into DUCC's
cyclic-eastward order, then restores the source representation. A same-grid
operation therefore preserves the user’s coordinate order and convention.

## Transform limits

DUCC documents the latitude analysis limit as `nlat - 2` for CC and `nlat - 1`
for GL. The azimuthal limit is `(nlon - 1) // 2`. An explicit `Tn` request is
triangular, so both the source and target must represent all degrees and orders
through `n`; otherwise spharmgrid raises an error. Without an explicit range,
regridding retains the content jointly representable by its source and target
latitude and longitude sampling.

`T42` is a spectral truncation, not a physical-grid name. Construct a target
grid explicitly instead.
