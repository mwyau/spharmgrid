# spharmgrid

**spharmgrid** (**sp**herical **harm**onics **grid**ding) is an xarray-first
spherical-harmonic tool for global atmospheric fields. It provides spectral
filtering, regridding, scalar operators, and atmospheric wind diagnostics and
inverse transforms, including relative vorticity, divergence, streamfunction,
and velocity potential. [DUCC0](https://gitlab.mpcdf.mpg.de/mtr/ducc) supplies
the numerical spherical-harmonic transforms.

For atmospheric winds, spharmgrid computes relative vorticity (`vo`),
divergence (`d`), streamfunction (`strf`), and velocity potential (`vp`), and
can reconstruct rotational, divergent, or full wind fields from those derived
quantities.

Supported grids are full rectangular Gauss--Legendre (GL) and pole-including
Clenshaw--Curtis (CC) grids.

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

spharmgrid preserves arbitrary leading dimensions and xarray coordinates, so
fields such as `(time, level, lat, lon)` can be transformed directly.
