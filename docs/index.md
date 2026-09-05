# spharmgrid

spharmgrid provides spherical-harmonic filtering, regridding, scalar
operators, and atmospheric wind transforms for global xarray fields. It is an
xarray/CF operations layer around [DUCC0](https://gitlab.mpcdf.mpg.de/mtr/ducc):
DUCC0 supplies the numerical spherical-harmonic transforms.

The initial scope is deliberately limited to full rectangular
Gauss--Legendre (GL) and pole-including Clenshaw--Curtis (CC) grids. It does
not reinterpret arbitrary regular latitude--longitude data as CC, and it does
not provide reduced Gaussian grids, HEALPix, regional transforms, MPI, or an
alternative transform backend.

```{toctree}
:maxdepth: 2

quickstart
grids
filtering
regridding
operators
kinematics
cf
cli
api
references
```

## Roles and lineage

- spharmgrid supplies xarray integration, grid validation, CF-aware variable
  discovery, and atmospheric operation semantics.
- DUCC0 supplies scalar and spin-weighted numerical transforms.
- NCL and SPHEREPACK are established atmospheric semantic and parity
  references. spharmgrid does not bundle or wrap SPHEREPACK.
- The initial DUCC0 wrapper is extracted and generalized from
  [PyStormTracker](https://github.com/mwyau/PyStormTracker).

The tested API is designed for fields with arbitrary leading dimensions, such
as `(time, level, lat, lon)`. Each horizontal slice is transformed
independently.

## Building these docs

From a repository checkout, install the lightweight documentation group and
run the strict local build:

```bash
uv sync --group docs
uv run --no-sync sphinx-build -W --keep-going -b html docs docs/_build/html
```
