# spharmgrid

**spharmgrid** (**sp**herical **harm**onics **grid**ding) is an xarray-first
spherical-harmonic tool for global atmospheric fields. It provides spectral
filtering, regridding, scalar operators, and atmospheric wind diagnostics and
inverse transforms, including relative vorticity, divergence, streamfunction,
velocity potential, Helmholtz decomposition, and vector differential
operators. [DUCC0](https://gitlab.mpcdf.mpg.de/mtr/ducc) supplies the numerical
spherical-harmonic transforms.

Supported grids are full rectangular Gauss--Legendre (GL) and pole-including
Clenshaw--Curtis (CC) grids.

## Install

Install from GitHub:

```bash
uv add git+https://github.com/mwyau/spharmgrid.git
```

## Quick start

Importing spharmgrid registers the xarray accessor:

```python
import xarray as xr
import spharmgrid as sg

field = xr.open_dataarray("msl.nc")

filtered = field.sg.filter("T6-42")
tapered = field.sg.filter("T6-42", taper=0.1)

target = sg.gaussian_grid(64, 128)
regridded = field.sg.regrid(target)
combined = field.sg.regrid(target, spectral="T6-42", taper=0.1)
```

### Direct functions

```python
filtered = sg.filter(field, "T6-42", taper=0.1)
regridded = sg.regrid(field, target)
```

For atmospheric wind fields, a Dataset with `u` and `v` variables or their
exact CF standard names can compute the common diagnostics:

```python
ds = xr.open_dataset("wind.nc")

kin = ds.sg.kinematics()   # vo: relative vorticity; d: divergence
pot = ds.sg.potentials()   # strf: streamfunction; vp: velocity potential

reconstructed = xr.Dataset({"vo": kin.vo, "d": kin.d}).sg.wind()

# These retain vector-harmonic rather than component-wise semantics.
target_wind = ds.sg.regrid_vector(target, spectral="T42")
parts = ds.sg.helmholtz()
gradient = field.sg.gradient()
recovered_field = gradient.gradient_eastward.sg.inverse_gradient(
    gradient.gradient_northward,
)
wind_laplacian = ds.sg.vector_laplacian()
```

`inverse_gradient()` sets the unrecoverable scalar degree-zero coefficient to
zero and returns the irrotational projection when the supplied vector also has
a rotational component. `inverse_vector_laplacian()` likewise zeroes its null
degree-zero vector-harmonic slots. See the operator documentation for the
precise conventions.

`T42` retains total degrees 0 through 42; `T6-42` retains degrees 6 through
42. Filtering is a hard selection by default. `taper=0.1` applies the
Sardeshmukh--Hoskins exponential response with value 0.1 at the upper
retained degree.

See the full [documentation](https://spharmgrid.readthedocs.io/) for grid
requirements, coordinate handling, CF metadata, wind diagnostics and inverse
transforms, inverse zero-mode conventions, and command-line use.

## References

See the documentation's
[references](https://spharmgrid.readthedocs.io/en/latest/references.html) for
scientific and software citations. spharmgrid grew out of spherical-harmonic
code used in [PyStormTracker](https://github.com/mwyau/PyStormTracker).
