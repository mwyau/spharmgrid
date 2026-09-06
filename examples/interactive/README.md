# Interactive ERA5 example

This example is a self-contained JupyterLab and Panel application for exploring
ERA5 850-hPa wind with spharmgrid. The left map shows the original field; the
right map shows the selected vector spherical harmonic processing on the same
timestamp. The application exposes wind speed and arrows, relative vorticity,
divergence, streamfunction, velocity potential, rotational wind, and divergent
wind.

## Run the example

From the repository checkout:

```bash
cd examples/interactive
uv sync --locked
uv run jupyter lab spharmgrid_era5.ipynb
```

The same notebook can be served as a standalone local Panel application:

```bash
uv run panel serve spharmgrid_era5.ipynb --show
```

The first notebook run downloads the small pinned wind asset into Pooch's user
cache. The optional ERA5 vorticity asset is downloaded only when the
consistency-check button is used. No CDS credentials or CDS API access are
required.

## Environment and data

The nested `uv` project installs spharmgrid from this checkout as an editable
path dependency. Visualization and notebook dependencies are isolated here;
they are not root-package dependencies or extras. The direct example
dependencies are:

- `cartopy`
- `geoviews`
- `h5netcdf[h5py]`
- `holoviews`
- `jupyterlab`
- `panel`
- `pooch`
- `spharmgrid` from `../..`

The notebook uses the immutable
[PyStormTracker-Data `v0.2.0-data` release](https://github.com/mwyau/PyStormTracker-Data/releases/tag/v0.2.0-data):

- `era5_uv850_2025-2026_djf_2.5x2.5.nc`, SHA-256
  `43cbc346a52c5230ac34eb22c7a640800fbffad40da4058686c8042a76bc5965`;
- `era5_vo850_2025-2026_djf_2.5x2.5.nc`, SHA-256
  `46ce78cd3b065d3777c2d628cdc2311d68a9fcb4d3a3b9948db7c7376ae7a6aa`.

The inspected wind file contains `u` and `v` with dimensions
`(valid_time, pressure_level, latitude, longitude) = (360, 1, 73, 144)`. The
singleton pressure level is 850 hPa. Timestamps are six-hourly from December
2025 through February 2026. The latitude coordinate is descending from 90° to
-90° and longitude is stored in `[0, 360)` at 2.5° spacing. `sg.detect_grid()`
identifies the field as a pole-including CC grid with a documented triangular
limit of T71.

The optional fixed display target is a public `sg.gaussian_grid(72, 144)` with
matching latitude order. The approximately 1.4-GB 0.25° ERA5 asset in the
release is intentionally not downloaded or exposed by this example.

## Scientific design

Wind processing always calls `sg.regrid_vector(u, v, ...)`. Selecting `ERA5 CC`
as the target performs vector spherical harmonic analysis, degree selection or
the Sardeshmukh–Hoskins taper, and inverse vector synthesis on the same physical
grid. It does not filter `u` and `v` as independent scalar fields. Selecting
`Gauss–Legendre` adds the same vector processing while synthesizing on the fixed
GL target.

`T0–42` retains total degrees 0 through 42; `T6–42` removes degrees 0 through
5. With tapering enabled, `0.1` is the retained endpoint response, not a
cutoff. Spectral controls use Panel's throttled slider values, and a small
in-memory LRU cache avoids repeating the same processed-field computation for
the two maps.

The consistency-check card exercises wind reconstruction, Helmholtz
decomposition, potential and Laplacian relationships, gradient and inverse
gradient, rotational and divergent wind reconstruction, and vector Laplacian
round trips. Its ERA5 `vo` comparison is an external reference comparison, not
ground truth or an exact parity test: the release's `u/v` and `vo` assets may
have passed through different processing paths.

## Attribution

The maps display ERA5 data from the Copernicus Climate Change Service / ECMWF,
distributed to this example through PyStormTracker-Data `v0.2.0-data`. This
example does not imply endorsement by ECMWF or Copernicus.
