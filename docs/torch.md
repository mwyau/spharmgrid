# PyTorch backend

The optional `spharmgrid.torch` namespace applies spharmgrid's spherical
harmonic operations to PyTorch tensors. Install it with the `torch` extra:

```bash
pip install "spharmgrid[torch]"
```

`spharmgrid[torch]` depends on PyTorch and `torch-harmonics`. Wheel and CUDA
availability varies by Python version and platform; see the upstream
[installation documentation](https://github.com/NVIDIA/torch-harmonics#installation)
for current installation options.

`torch-harmonics` computes the scalar and vector spherical harmonic transforms.
spharmgrid maps its `Grid` descriptors to those transforms and applies the
spectral selections, radius factors, and atmospheric vector conventions.
Import `spharmgrid.torch` to load the PyTorch API.

The Xarray/DUCC API handles file and metadata workflows:

```python
vo = sg.vorticity(u, v)
```

For individual tensor operations, use `spharmgrid.torch` directly. For repeated
model execution, use one of the reusable modules below.

## Functional API

The functional API accepts `torch.Tensor` objects. The last two dimensions are
latitude and longitude, in that order; all preceding dimensions are preserved
as independent leading or batch dimensions.

```python
import torch
import spharmgrid as sg
import spharmgrid.torch as sgt

grid = sg.gaussian_grid(128, 256)
field = torch.randn(4, grid.nlat, grid.nlon)
filtered = sgt.filter(field, grid=grid, truncation="T42")

u = torch.randn(grid.nlat, grid.nlon)
v = torch.randn(grid.nlat, grid.nlon)
vorticity, divergence = sgt.kinematics(u, v, grid=grid)
```

The namespace contains the following functions:

```text
filter, regrid, regrid_vector
gradient, inverse_gradient, laplacian, inverse_laplacian
vector_laplacian, inverse_vector_laplacian
vorticity, divergence, kinematics
streamfunction, velocity_potential, potentials, helmholtz
rotational_wind, divergent_wind, wind
```

`Grid` is required explicitly because tensors do not carry named coordinates.
Use `source_grid=` with `regrid()` and `regrid_vector()`; the target grid is a
positional argument. Vector arguments are geographic eastward `u` and
northward `v`. `wind()` and the single-source inverse wind functions require
an explicit `source=` because tensors do not carry CF metadata.

The default radius is `spharmgrid.EARTH_RADIUS_M`. Scalar inverse operators
set the degree-zero coefficient to zero, and vector inverse operations remove
the nonphysical degree-zero vector slot. The backend supports `float32` and
`float64` tensors. Tensor operations use PyTorch throughout, so gradients can
flow through the transforms and spectral multipliers.

## Reusable modules

`spharmgrid.torch.nn` contains four reusable module classes:

```python
import spharmgrid.torch.nn as sgnn

operators = sgnn.SHTOperators(grid)
vorticity, divergence = operators.kinematics(u, v)

low_pass = sgnn.SHTFilter(grid, "T42")
filtered = low_pass(field)

regrid_layer = sgnn.SHTRegrid(source_grid, target_grid, "T42")
target_field = regrid_layer(field)

vector_layer = sgnn.SHTVectorRegrid(source_grid, target_grid, "T42")
target_u, target_v = vector_layer(u, v)
```

`SHTOperators` exposes reusable transform and diagnostic methods and owns
reusable transform state. `SHTFilter`, `SHTRegrid`, and `SHTVectorRegrid` are
callable layers with fixed grids and spectral selections. Move a module and
its input tensors together with `.to(device)` or `.to(dtype=...)`.

`SHTOperators` builds a full same-grid transform state and therefore requires
full-bandwidth support for the grid. On CC grids, the fixed filtering and
regridding modules accept explicit supported `Tn` ranges.

## Grid and bandwidth capabilities

Supported grids are full rectangular GL grids and pole-including CC grids with
equally spaced latitudes from -90° to 90°.

The adapter uses triangular total-degree bands so that the torch-harmonics
coefficient domain matches spharmgrid's explicit `Tn` semantics. For an
explicit triangular request, the verified inclusive limits are

```text
GL
n <= min(nlat - 1, (nlon - 1) // 2)

CC
n <= min((nlat - 1) // 2, (nlon - 1) // 2)
```

For regridding, an explicit `Tn` must fit the limit of both source and target
grids. A full same-grid state is supported on GL when spharmgrid's requested
domain is representable by torch-harmonics. For CC, torch-harmonics supports
triangular bands through the CC limit above; spharmgrid's full same-grid domain
generally exceeds that limit.

CC filtering and regridding use explicit `Tn` bands within the documented
limit. Differential and wind functions require the full same-grid domain and
raise `ValueError` when it exceeds the limit. Requests outside the documented
bandwidth also raise `ValueError`.

CUDA execution requires compatible PyTorch and torch-harmonics builds. Use the
Xarray API for file I/O, CF metadata, and accessors. Learned spherical spectral
convolution is available from torch-harmonics and neural-operator packages.

On the tested Torch 2.11.0 and torch-harmonics 0.9.2 CUDA stack, default Inductor
compilation of `SHTFilter(...)` fails during Triton code generation with
`KeyError: 'complex64'`. A minimal compiled `torch_harmonics.RealSHT` reproduces
the same failure, while `InverseRealSHT` compiles. The CUDA compile test records
this torch-harmonics/Torch limitation as an expected failure.
