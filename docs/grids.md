# Grids and coordinates

spharmgrid's Xarray API supports full rectangular Gauss–Legendre (`"gl"`),
Clenshaw–Curtis (`"cc"`), and Fejér-1 (`"f1"`) grids. The JAX and PyTorch APIs support
GL and CC grids.

## Gauss–Legendre (GL)

A GL grid uses Gaussian latitude nodes and equally spaced longitudes on every latitude
ring. Latitude nodes are generated from `ducc0.misc.GL_thetas`.

```python
grid = sg.gaussian_grid(64, 128, lon0=0.0, latitude_order="ascending")
```

## Clenshaw–Curtis (CC)

A CC grid uses equally spaced latitudes from −90° to 90° and equally spaced longitudes.

```python
grid = sg.clenshaw_curtis_grid(65, 128, latitude_order="descending")
```

CC detection requires equally spaced latitudes spanning −90° to 90°, together with a
globally cyclic longitude coordinate without a duplicated endpoint.

## Fejér-1 (F1)

An F1 grid uses equally spaced longitudes and colatitudes
`theta[j] = pi * (j + 0.5) / nlat`, with `j` ordered from north to south. It excludes
both geographic poles.

```python
grid = sg.fejer1_grid(72, 144, lon0=0.0, latitude_order="descending")
```

F1 detection matches these latitude nodes within coordinate storage tolerance. The
longitude coordinate must contain one complete, uniformly spaced cycle without a
duplicated endpoint.

## Coordinate discovery

Latitude and longitude are identified in this order:

1. unique exact CF `standard_name` values `latitude` and `longitude`;
2. coordinate names `lat`/`latitude` and `lon`/`longitude`;
3. optional `cf-xarray` discovery when installed.

Ambiguous coordinates raise an error.

Latitude may be ascending or descending. Longitude may use `[0, 360)`, `[-180, 180)`, or
another regularly spaced cyclic origin. Same-grid operations preserve the input
coordinate order and longitude convention.

## Transform limits

DUCC gives the latitude analysis limit as `nlat - 2` for CC and `nlat - 1` for GL and
F1. The independent longitude-order limit is `(nlon - 1) // 2`, bounded by `lmax`. F1
accepts any valid longitude count; `nlon = 2 * nlat` is common but not required. An
explicit truncation must fit the latitude and longitude limits on both source and target
grids. Thus `Tn` requires both limits through `n`, `Tnxm` uses `lmax=n` and `mmax=m`,
and `Rn` uses `lmax=2n` and `mmax=n`.

Without an explicit truncation, regridding retains the content represented by both
source and target sampling.

`T42` denotes spectral truncation, not a physical grid. Construct the target grid
explicitly.
