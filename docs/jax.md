# JAX

The optional `spharmgrid.jax` namespace applies the spherical harmonic operations to JAX
arrays. S2FFT computes the scalar and spin-1 transforms; the package applies the
spectral selections, radius factors, and atmospheric vector conventions.

The JAX array API reference is in {doc}`jax_api`. The labeled Xarray API reference,
including `device_put()`, `device_get()`, and `.sgj` methods, is in
{doc}`jax_accessor_api`.

## Installation

Install the JAX extra with pip:

```bash
pip install "spharmgrid[jax]"
```

For CPU use, `spharmgrid[jax]` installs JAX and S2FFT. For GPU or TPU use, install the
appropriate JAX accelerator build first by following the
[official JAX installation instructions](https://docs.jax.dev/en/latest/installation.html),
then install `spharmgrid[jax]`. spharmgrid does not bundle or select CUDA or TPU builds.

Import the grid API and JAX API:

```python
import spharmgrid as sg
import spharmgrid.jax as sgj
```

## Arrays and grids

The functions accept `jax.Array` values. The last two dimensions are latitude and
longitude; all preceding dimensions are preserved.

`spharmgrid.jax` accepts these regular rectangular shapes:

| spharmgrid grid      | dimensions              |
| -------------------- | ----------------------- |
| Gauss–Legendre (GL)  | `(L, nlon)`, `nlon ≥ 2` |
| Clenshaw–Curtis (CC) | `(L + 1, 2L)`           |

For GL grids, `L = nlat` sets the maximum spherical degree `L - 1`. Longitude sampling
sets `longitude_mmax = floor((nlon - 1) / 2)`, so the transform uses
`mmax = min(L - 1, longitude_mmax)`. The centered JAX coefficient array has shape
`(L, 2L - 1)` for every supported `nlon`.

Regular GL transforms currently require `s2fft==1.4.0` because spharmgrid uses that
release's internal latitude-transform functions. spharmgrid rejects other S2FFT versions
or an unexpected internal function signature.

CC latitudes are the pole-including equally spaced nodes from −90 to 90 degrees. Both
latitude orders and cyclic longitude coordinate conventions are accepted.

`spharmgrid.jax` requires JAX x64 mode. Public spatial inputs must be `float64`; S2FFT
coefficient and spin-transform arrays use `complex128`. Configure JAX before creating
arrays:

```python
import jax

jax.config.update("jax_enable_x64", True)
```

The package does not modify this process-wide setting. The tested ordinary S2FFT MWSS
scalar path produced about 9% interior relative error in `float32` from L=8 through
L=128, so `float32` and `complex64` inputs are rejected. S2FFT also warns that disabling
64-bit precision can substantially affect numerical accuracy at moderate L.

The JAX transforms cache S2FFT's O(L²) Price–McEwen recursion precomputations for
repeated static transform settings. Scalar transforms use S2FFT's real-field path;
spin-1 transforms use the complex path. The first call for a new transform setting
constructs the precomputations, and later calls reuse them.

## Scalar operations

```python
import jax.numpy as jnp

grid = sg.gaussian_grid(16, 31)
field = jnp.ones((2, grid.nlat, grid.nlon), dtype=jnp.float64)
filtered = sgj.filter(field, grid=grid, truncation="T6")
laplacian = sgj.laplacian(filtered, grid=grid)
```

Grid and spectral configuration are static Python values. Close them over when using JAX
transformations rather than passing a `Grid` as a dynamic JIT argument:

```python
compiled_filter = jax.jit(lambda values: sgj.filter(values, grid=grid, truncation="T6"))
filtered = compiled_filter(field)
```

The same array operations support `vmap` and automatic differentiation.

For repeated spectral work, use the PyTree-compatible analyzed representations:

```python
spectral = sgj.analyze(field, grid=grid)
filtered = spectral.filter("T6").synthesize()
laplacian = spectral.laplacian().synthesize()
```

`SpectralVectorField` provides the corresponding reusable geographic vector
representation. Its coefficient-domain diagnostics return `SpectralField` objects, while
coefficient storage is native to S2FFT and private.

PyTree reconstruction may express equivalent longitudes differently, such as values that
differ by 360 degrees. The result is `grids_equivalent()` to the input grid, although
the coordinate values may differ.

`regrid()` and `regrid_vector()` use S2FFT coefficient analysis and synthesis when the
source and target grids have different resolutions. The spectral selection arguments
accept the same triangular, trapezoidal, rhomboidal, and taper options as the Xarray
API.

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

The inverse wind functions require `source=` because JAX arrays do not carry CF
metadata. The accepted sources are `"vorticity"`, `"streamfunction"`, `"divergence"`,
`"velocity_potential"`, `"vorticity_divergence"`, and `"potentials"`, as appropriate for
each function. The functions use the same Earth-radius default and degree-zero
conventions as the root API.

## Xarray convenience layer

The optional Xarray layer keeps labels, coordinates, and metadata around the raw JAX
arrays. Importing `spharmgrid.jax` registers the `.sgj` accessor on Xarray `DataArray`
and `Dataset` objects. Data transfer is explicit:

```python
import xarray as xr
import spharmgrid.jax as sgj

ds = xr.open_dataset("input.nc")
ds_jax = sgj.device_put(ds)

diagnostics = ds_jax.sgj.kinematics()
filtered = ds_jax["z"].sgj.filter("T42")
diagnostics_host = sgj.device_get(diagnostics)
```

`.sgj` operates on Xarray objects whose numerical payloads are `jax.Array` values and
preserves their dimensions, coordinates, names, and metadata. Use `device_put()` before
`.sgj` methods and `device_get()` to copy results back to host arrays. Use the
`spharmgrid.jax` array functions with `jax.jit`, `jax.grad`, and `jax.vmap`.
