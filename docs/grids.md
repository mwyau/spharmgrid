# Grids and coordinates

spharmgrid supports full rectangular Gauss–Legendre (`"gl"`) and Clenshaw–Curtis (`"cc"`) grids.

## Gauss–Legendre (GL)

A GL grid uses Gaussian latitude nodes and equally spaced longitudes on every latitude ring. Latitude nodes are generated from `ducc0.misc.GL_thetas`.

```python
grid = sg.gaussian_grid(64, 128, lon0=0.0, latitude_order="ascending")
```

## Clenshaw–Curtis (CC)

A CC grid uses equally spaced latitudes from −90° to 90° and equally spaced longitudes.

```python
grid = sg.clenshaw_curtis_grid(65, 128, latitude_order="descending")
```

CC detection requires equally spaced latitudes spanning −90° to 90°, together with a globally cyclic longitude coordinate without a duplicated endpoint.

## Coordinate discovery

Latitude and longitude are identified in this order:

1. unique exact CF `standard_name` values `latitude` and `longitude`;
2. coordinate names `lat`/`latitude` and `lon`/`longitude`;
3. optional `cf-xarray` discovery when installed.

Ambiguous coordinates raise an error.

Latitude may be ascending or descending. Longitude may use `[0, 360)`, `[-180, 180)`, or another regularly spaced cyclic origin. Same-grid operations preserve the input coordinate order and longitude convention.

## Transform limits

DUCC gives the latitude analysis limit as `nlat - 2` for CC and `nlat - 1` for GL. The azimuthal limit is `(nlon - 1) // 2`. A triangular `Tn` range requires both source and target grids to represent every degree and order through `n`.

Without an explicit spectral range, regridding retains the content represented by both source and target sampling.

`T42` denotes spectral truncation, not a physical grid. Construct the target grid explicitly.
