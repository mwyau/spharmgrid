# spharmgrid

**spharmgrid** (**sp**herical **harm**onics **grid**ding) provides spherical-harmonic filtering, regridding, differential operators, and atmospheric wind diagnostics for global xarray fields. It computes relative vorticity (`vo`), divergence (`d`), streamfunction (`strf`), velocity potential (`vp`), Helmholtz decomposition, and inverse wind transforms. [DUCC0](https://gitlab.mpcdf.mpg.de/mtr/ducc) supplies the numerical spherical-harmonic transforms.

Supported grids are full rectangular Gauss--Legendre (GL) and pole-including Clenshaw--Curtis (CC) grids. Arbitrary leading dimensions and xarray coordinates are preserved, so fields such as `(time, level, lat, lon)` can be transformed without reshaping them first.

```{toctree}
:maxdepth: 2

quickstart
filtering
regridding
kinematics
operators
grids
cf
comparison
cli
api
references
```
