# spharmgrid

**spharmgrid (SPherical HARMonics GRIDding)** is an xarray-first
spherical-harmonic tool for global atmospheric fields. It provides spectral
filtering, regridding, scalar operators, and atmospheric wind diagnostics and
inverse transforms, including relative vorticity, divergence, streamfunction,
and velocity potential. [DUCC0](https://gitlab.mpcdf.mpg.de/mtr/ducc) supplies
the numerical spherical-harmonic transforms.

For atmospheric winds, spharmgrid directly computes relative vorticity (`vo`),
divergence (`d`), streamfunction (`strf`), and velocity potential (`vp`), and
can reconstruct rotational, divergent, or full wind fields from those derived
quantities.

The initial scope is limited to full rectangular Gauss--Legendre (GL) and
pole-including Clenshaw--Curtis (CC) grids. It does not reinterpret arbitrary
regular latitude--longitude data as CC, and it does not provide reduced
Gaussian grids, HEALPix, regional transforms, MPI, or an alternative transform
backend.

```{toctree}
:maxdepth: 2

quickstart
filtering
regridding
kinematics
operators
grids
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
