# spharmgrid

**spharmgrid (SPherical HARMonics GRIDding)** is an xarray-first
spherical-harmonic tool for global atmospheric fields. It provides spectral
filtering, regridding, scalar operators, and atmospheric wind diagnostics and
inverse transforms, including relative vorticity, divergence, streamfunction,
and velocity potential. [DUCC0](https://gitlab.mpcdf.mpg.de/mtr/ducc) supplies
the numerical spherical-harmonic transforms.

The initial package supports full rectangular Gauss--Legendre (GL) grids and
pole-including Clenshaw--Curtis (CC) grids. It does not reinterpret an arbitrary
regular latitude--longitude field as CC.

## Install

The package is currently installed from its Git repository:

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

The direct functions use the same numerical path:

```python
filtered = sg.filter(field, "T6-42", taper=0.1)
regridded = sg.regrid(field, target)
```

For atmospheric wind fields, a Dataset with `u` and `v` variables (or their
exact CF standard names) can compute the common diagnostics directly:

```python
ds = xr.open_dataset("wind.nc")

kin = ds.sg.kinematics()   # vo: relative vorticity; d: divergence
pot = ds.sg.potentials()   # strf: streamfunction; vp: velocity potential

reconstructed = xr.Dataset({"vo": kin.vo, "d": kin.d}).sg.wind()
```

`T42` retains total degrees 0 through 42; `T6-42` retains degrees 6 through
42. Filtering is a hard selection by default. `taper=0.1` applies the
Sardeshmukh--Hoskins exponential response with value 0.1 at the upper
retained degree.

See the full [documentation](https://spharmgrid.readthedocs.io/) for grid
requirements, coordinate handling, CF metadata, wind diagnostics and inverse
transforms, inverse zero-mode conventions, and command-line use.

## Lineage and citation

DUCC0 performs the numerical transforms. NCL/SPHEREPACK are semantic and
parity references for the atmospheric operations; spharmgrid does not include
SPHEREPACK or implement a new transform engine. The initial wrapper is
extracted and generalized from PyStormTracker. See the documentation's
[references](https://spharmgrid.readthedocs.io/en/latest/references.html) for
scientific citations.
