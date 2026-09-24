# PyTorch

The `spharmgrid.torch` namespace applies spherical harmonic operations to PyTorch
tensors using `torch-harmonics`.

The functional tensor API is documented in {doc}`torch_api`; reusable module classes are
documented in {doc}`torch_nn_api`.

Install the PyTorch API with:

```bash
pip install "spharmgrid[torch]"
```

This installs spharmgrid, PyTorch, and a compatible `torch-harmonics` wheel. For CUDA or
other accelerator-specific PyTorch builds, install PyTorch using the
[PyTorch installation instructions](https://pytorch.org/get-started/locally/) first,
then install `spharmgrid[torch]`.

## Development

For a spharmgrid development checkout, install the Torch development dependencies with:

```bash
uv sync --group torch-dev
```

`torch-harmonics` defines the Legendre/vector projections and normalization used by the
Torch backend, and provides the native transforms for small tables. spharmgrid maps its
`Grid` descriptors to the backend and applies spectral selections, radius factors, and
atmospheric vector conventions. Import `spharmgrid.torch` to load the PyTorch API.

The Xarray API uses DUCC and handles file and metadata workflows:

```python
vo = sg.vorticity(u, v)
```

For individual tensor operations, use `spharmgrid.torch` directly. For repeated model
execution, use one of the reusable modules below.

## Functional API

The functional API accepts `torch.Tensor` objects. The last two dimensions are latitude
and longitude, in that order; all preceding dimensions are preserved as independent
leading or batch dimensions.

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

For repeated spectral work, use the analyzed representations:

```python
spectral = sgt.analyze(field, grid=grid)
filtered = spectral.filter("T42").synthesize()
laplacian = spectral.laplacian().synthesize()
```

`SpectralVectorField` provides the corresponding reusable geographic vector
representation. Its coefficient-domain diagnostics such as `vorticity()` and
`divergence()` return `SpectralField` objects. These objects do not expose the
torch-harmonics coefficient layout.

`Grid` is required explicitly because tensors do not carry named coordinates. Use
`source_grid=` with `regrid()` and `regrid_vector()`; the target grid is a positional
argument. Vector arguments are geographic eastward `u` and northward `v`. `wind()` and
the single-source inverse wind functions require an explicit `source=` because tensors
do not carry CF metadata.

The default radius is `spharmgrid.EARTH_RADIUS_M`. Scalar inverse operators set the
degree-zero coefficient to zero, and vector inverse operations remove the nonphysical
degree-zero vector slot. The PyTorch API supports `float32` and `float64` tensors.
Tensor operations use PyTorch throughout, so gradients can flow through the transforms
and spectral multipliers.

## Spectral truncation support

The Torch API supports triangular `Tn` truncation and `Ta-b` total-degree bands.
spharmgrid also accepts `Tnxm` trapezoidal and `Rn` rhomboidal notation, but these
requests raise `NotImplementedError` because the current torch-harmonics transforms do
not support these coefficient domains.

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

`SHTOperators` exposes reusable transform and diagnostic methods and owns reusable
transform state. `SHTFilter`, `SHTRegrid`, and `SHTVectorRegrid` are callable layers
with fixed grids and spectral selections. Move a module and its input tensors together
with `.to(device=..., dtype=...)`, using the input dtype. Projection buffers are
constructed in float64 for accuracy, then retained in the requested transform dtype.

`SHTOperators` uses the full triangular domain supported by its grid. It raises
`ValueError` when the full grid domain is non-triangular.

## Grid and bandwidth capabilities

Supported grids are full rectangular GL grids and pole-including CC grids with equally
spaced latitudes from -90° to 90°.

`spharmgrid.torch` supports triangular coefficient domains. The inclusive limits for an
explicit triangular request are

```text
GL
n <= min(nlat - 1, (nlon - 1) // 2)

CC
n <= min(nlat - 2, (nlon - 1) // 2)
```

For regridding, an explicit `Tn` must fit the limit of both source and target grids. CC
analysis uses native torch-harmonics direct quadrature through degree `(nlat - 1) // 2`.
Above that limit, spharmgrid extends analysis to the full triangular grid bandwidth by
folded latitude resampling. It uses dense projection tables derived from native
torch-harmonics weights with the ordinary CC quadrature removed. Scalar and vector
synthesis use the native `InverseRealSHT` and `InverseRealVectorSHT` modules. Projection
tables are dense, so memory use grows with the requested bandwidth.

Trapezoidal (`Tnxm`) and rhomboidal (`Rn`) requests are unsupported. An explicit `Tn`
outside the grid limits raises `ValueError`. Without an explicit `Tn`, operations use
the full domain and raise `ValueError` when it is non-triangular.

CUDA execution requires compatible PyTorch and torch-harmonics builds. Use the Xarray
API for file I/O, CF metadata, and accessors. Learned spherical spectral convolution is
available from torch-harmonics and neural-operator packages.
