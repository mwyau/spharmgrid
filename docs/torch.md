# PyTorch backend

The optional `spharmgrid.torch` namespace provides tensor-native spherical
harmonic operations for PyTorch workflows. Install it with the `torch` extra:

```bash
pip install "spharmgrid[torch]"
```

The current `torch-harmonics` PyPI release provides prebuilt CPU wheels for
Linux x86-64 on only a limited set of CPython versions. CUDA-enabled upstream
packages and source builds have separate PyTorch, CUDA, and platform
requirements. See the upstream [installation documentation](https://github.com/NVIDIA/torch-harmonics#installation)
for the current wheel and build options; if no compatible upstream package can
be installed, installation fails rather than making the `torch` extra
available incompletely.

The backend calls the installed `torch-harmonics` scalar and vector transform
modules. spharmgrid translates its `Grid` descriptors and applies the
scientific spectral selections, radius factors, and atmospheric vector
conventions around those transforms. Importing `spharmgrid` alone does not
import PyTorch.

The existing Xarray/DUCC API remains the file and metadata workflow:

```python
vo = sg.vorticity(u, v)
```

For a tensor-native one-off, use `spharmgrid.torch` directly. For repeated
model execution, keep transform state in one of the reusable modules below.

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
an explicit `source=` or `quantity=` because tensors do not carry CF metadata.

The default radius is `spharmgrid.EARTH_RADIUS_M`. Scalar inverse operators
set the degree-zero coefficient to zero, and vector inverse operations remove
the nonphysical degree-zero vector slot. The backend supports `float32` and
`float64` tensors. Tensor operations remain in PyTorch, so gradients can flow
through the transforms and spectral multipliers.

## Reusable modules

`spharmgrid.torch.nn` contains four deliberately small module interfaces:

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

`SHTOperators` builds a full same-grid transform state. It therefore has the
same `truncation=None` capability restriction as full-bandwidth functional
calls; on a CC grid that cannot be represented, use a fixed layer with an
explicit supported `Tn` range for filtering or regridding.

## Grid and bandwidth capabilities

Only full rectangular GL and pole-including, equally spaced CC grids are
accepted. A CC grid is not a general regular latitude–longitude grid.

The adapter uses triangular total-degree bands so that the torch-harmonics
coefficient domain matches spharmgrid's explicit `Tn` semantics. For an
explicit request, the verified inclusive limits are

```text
GL: Tn where n <= min(nlat - 1, (nlon - 1) // 2)
CC: Tn where n <= min((nlat - 1) // 2, (nlon - 1) // 2)
```

For regridding, the limit is the minimum over both source and target grids.
`truncation=None` is accepted only when the full spharmgrid transform domain is
triangular and fits the verified torch-harmonics limit. In particular, a
full-bandwidth CC call usually requests more latitude modes than the current
equiangular torch-harmonics transform can represent. Such calls, and explicit
requests above the limits, raise `ValueError`; the backend never silently
clamps a requested band.

CUDA execution is available when the installed PyTorch and torch-harmonics
build support it. The optional test suite exercises CUDA conditionally and
keeps the normal CI job CPU-only. File I/O, CF metadata discovery, Xarray
accessors, S2FFT, HEALPix, reduced Gaussian grids, and SFNO layers are outside
this namespace. Use the existing Xarray API for scientific file and metadata
workflows, and use `torch-harmonics` or a neural-operator package for learned
spherical spectral convolution.

On the tested Torch 2.11.0 and torch-harmonics 0.9.2 CUDA stack, the real
Inductor path for `torch.compile(SHTFilter(...))` fails during Triton code
generation with `KeyError: 'complex64'`. A minimal compiled
`torch_harmonics.RealSHT` reproduces the same failure, while its inverse
transform compiles, so this is an upstream complex-coefficient compilation
limitation rather than a separate spharmgrid numerical path. The optional test
records this known limitation as an expected failure and still exercises the
real compiler when CUDA is available.
