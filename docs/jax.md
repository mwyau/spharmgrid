# JAX

The optional `spharmgrid.jax` namespace applies the spherical harmonic
operations to JAX arrays. S2FFT computes the scalar and spin-1 transforms;
the package applies the spectral selections, radius factors, and atmospheric
vector conventions.

The tensor-native API reference is in {doc}`jax_api`. The labeled Xarray API
reference, including `device_put()`, `device_get()`, and `.sgj` methods, is in
{doc}`jax_accessor_api`.

## Installation

Install the JAX extra with pip:

```bash
pip install "spharmgrid[jax]"
```

For CPU use, `spharmgrid[jax]` installs JAX and S2FFT. For GPU or TPU use,
install the appropriate JAX accelerator build first by following the [official
JAX installation instructions](https://docs.jax.dev/en/latest/installation.html),
then install `spharmgrid[jax]`. spharmgrid does not bundle or select CUDA or
TPU builds.

Import the array API separately from the Xarray API:

```python
import spharmgrid as sg
import spharmgrid.jax as sgj
```

## Arrays and grids

The functions accept `jax.Array` values. The last two dimensions are latitude
and longitude; all preceding dimensions are preserved.

`spharmgrid.jax` accepts these exact rectangular shapes and maps them to the
listed S2FFT samplings:

| spharmgrid grid      | dimensions    | S2FFT sampling |
| -------------------- | ------------- | -------------- |
| Gauss–Legendre (GL)  | `(L, 2L - 1)` | `gl`           |
| Clenshaw–Curtis (CC) | `(L + 1, 2L)` | `mwss`         |

CC latitudes are the pole-including equally spaced nodes from −90 to 90
degrees. Both latitude orders and cyclic longitude coordinate conventions are
accepted.

`spharmgrid.jax` requires JAX x64 mode. Public spatial inputs must be
`float64`; S2FFT coefficient and spin-transform arrays use `complex128`.
Configure JAX before creating arrays:

```python
import jax

jax.config.update("jax_enable_x64", True)
```

The package does not modify this process-wide setting. The tested ordinary
S2FFT MWSS scalar path produced about 9% interior relative error in `float32`
from L=8 through L=128, so `float32` and `complex64` inputs are rejected.
S2FFT also warns that disabling 64-bit precision can substantially affect
numerical accuracy at moderate L.

The JAX transforms cache S2FFT's O(L²) Price–McEwen recursion precomputations
for repeated static transform settings. Scalar transforms use S2FFT's
real-field path; spin-1 transforms use the complex path. The first call for a
new transform setting constructs the precomputations, and later calls reuse
them.

## Scalar operations

```python
import jax.numpy as jnp

grid = sg.gaussian_grid(16, 31)
field = jnp.ones((2, grid.nlat, grid.nlon), dtype=jnp.float64)
filtered = sgj.filter(field, grid=grid, truncation="T6")
laplacian = sgj.laplacian(filtered, grid=grid)
```

Grid and spectral configuration are static Python values. Close them over when
using JAX transformations rather than passing a `Grid` as a dynamic JIT
argument:

```python
compiled_filter = jax.jit(lambda values: sgj.filter(values, grid=grid, truncation="T6"))
filtered = compiled_filter(field)
```

The same array operations support `vmap` and automatic differentiation.

`regrid()` and `regrid_vector()` use S2FFT coefficient analysis and synthesis
when the source and target grids have different resolutions. The spectral
selection arguments accept the same triangular, trapezoidal, rhomboidal, and
taper options as the other functional APIs.

## Wind and kinematics

Vector arguments are geographic eastward `u` and northward `v` components:

```python
u = jnp.zeros((grid.nlat, grid.nlon), dtype=jnp.float64)
v = jnp.zeros_like(u)
vorticity, divergence = sgj.kinematics(u, v, grid=grid)
rotational_u, rotational_v = sgj.rotational_wind(
    vorticity,
    grid=grid,
    source="vorticity",
)
```

The inverse wind functions require `source=` because JAX arrays do not carry
CF metadata. The accepted sources are `"vorticity"`, `"streamfunction"`,
`"divergence"`, `"velocity_potential"`, `"vorticity_divergence"`, and
`"potentials"`, as appropriate for each function. The functions use the same
Earth-radius default and degree-zero conventions as the root API.

## Xarray convenience layer

The optional Xarray layer keeps labels, coordinates, and metadata around the
raw JAX arrays. Importing `spharmgrid.jax` registers the `.sgj` accessor on
Xarray `DataArray` and `Dataset` objects. Data transfer is explicit:

```python
import xarray as xr
import spharmgrid.jax as sgj

ds = xr.open_dataset("input.nc")
ds_jax = sgj.device_put(ds)

diagnostics = ds_jax.sgj.kinematics()
filtered = ds_jax["z"].sgj.filter("T42")
diagnostics_host = sgj.device_get(diagnostics)
```

`.sgj` delegates to the tensor-native `spharmgrid.jax` functions and returns
Xarray objects with the corresponding dimensions, coordinates, names, and
metadata. It requires the scientific input payloads to already contain
`jax.Array` values; an operation does not move data between host and device
implicitly. `device_put()` transfers data variables while leaving ordinary
coordinates on the host, and `device_get()` transfers numerical payloads back
to host arrays.

The `.sgj` layer does not require `xarray_jax`. Coordinates remain ordinary
Xarray/static metadata. The raw `spharmgrid.jax` array functions remain the
recommended interface for code compiled with `jax.jit`, differentiated with
`jax.grad`, or vectorized with `jax.vmap`.

The separately tested whole-container PyTree path uses optional `xarray_jax`.
It is not needed for `.sgj` operations and is not part of the published
`spharmgrid[jax]` extra.
