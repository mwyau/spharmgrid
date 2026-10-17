# Xarray API reference

The direct `spharmgrid` functions accept Xarray `DataArray` objects and use DUCC through
`ducc0` for spherical harmonic transforms. For labeled `.sg` methods, see
{doc}`accessor_api`. The optional JAX and PyTorch APIs are documented in {doc}`jax_api`
and {doc}`torch_api`.

## Typing

The public `spharmgrid` API is fully type annotated and the package ships a `py.typed`
marker for downstream static type checking.

## Leading dimensions

Xarray operations identify latitude and longitude from the field coordinates. Every
other dimension is an independent field dimension, and its name, coordinates, and
position are preserved. Latitude and longitude may appear anywhere in the input
dimension order. The `.sg` accessor uses the same operations and shape behavior as the
direct functions.

Xarray passes each available block of fields to the DUCC adapter, which applies one
spherical harmonic transform per field. Latitude and longitude can appear anywhere in
the dimension order.

For example, filtering a field with interleaved spatial and non-spatial dimensions
preserves its dimensions and coordinates:

```python
import numpy as np
import xarray as xr

import spharmgrid as sg

grid = sg.gaussian_grid(32, 64)
field = xr.DataArray(
    np.random.default_rng(0).standard_normal((3, grid.nlat, 2, grid.nlon)),
    dims=("time", "lat", "level", "lon"),
    coords={
        "time": [0, 1, 2],
        "lat": grid.latitude,
        "level": [850, 500],
        "lon": grid.longitude,
    },
)

filtered = field.sg.filter("T15")
assert filtered.dims == field.dims
```

Dask-backed inputs execute lazily. Each transform needs complete spatial dimensions
within its block, so Xarray may rechunk latitude and longitude.

The PyTorch and JAX APIs use the last two array axes for latitude and longitude; all
preceding axes are preserved as independent fields. Their grid and precision limits are
documented in {doc}`torch_api` and {doc}`jax_api`.

## Constants and descriptors

```{eval-rst}
.. autodata:: spharmgrid.EARTH_RADIUS_M

.. autoclass:: spharmgrid.Grid
   :members:

.. autoclass:: spharmgrid.TransformSpec
   :members:
```

## Grid and spectral selection

`parse_spectral()` returns a `TransformSpec` for the requested triangular, trapezoidal,
or rhomboidal coefficient domain.

```{eval-rst}
.. autofunction:: spharmgrid.gaussian_grid

.. autofunction:: spharmgrid.clenshaw_curtis_grid

.. autofunction:: spharmgrid.fejer1_grid

.. autofunction:: spharmgrid.detect_grid

.. autofunction:: spharmgrid.parse_spectral
```

## Reusable analyzed representations

The direct functions are convenient for one-shot operations. For repeated spectral work,
`analyze()` and `analyze_vector()` retain the analyzed representation so several
operations can reuse one forward transform. Coefficients remain private; use
`synthesize()` to return to the source grid and `regrid()` for another supported grid.

`spec` describes the currently available spectral domain. Filtering may narrow it;
discarded modes cannot be restored, and later selections or regridding cannot expand it.
Tapers modify the current coefficients and therefore compose when applied repeatedly.

```python
spectral = field.sg.analyze()
large_scale = spectral.filter("T5-42").synthesize()
laplacian = spectral.laplacian().synthesize()
```

```{eval-rst}
.. autoclass:: spharmgrid.SpectralField
   :members:

.. autoclass:: spharmgrid.SpectralVectorField
   :members:

.. autofunction:: spharmgrid.analyze

.. autofunction:: spharmgrid.analyze_vector
```

## Scalar operations

```{eval-rst}
.. autofunction:: spharmgrid.filter

.. autofunction:: spharmgrid.regrid

.. autofunction:: spharmgrid.gradient

.. autofunction:: spharmgrid.inverse_gradient

.. autofunction:: spharmgrid.laplacian

.. autofunction:: spharmgrid.inverse_laplacian
```

## Vector operations

```{eval-rst}
.. autofunction:: spharmgrid.regrid_vector

.. autofunction:: spharmgrid.helmholtz

.. autofunction:: spharmgrid.vector_laplacian

.. autofunction:: spharmgrid.inverse_vector_laplacian
```

## Atmospheric kinematics and wind transforms

```{eval-rst}
.. autofunction:: spharmgrid.vorticity

.. autofunction:: spharmgrid.divergence

.. autofunction:: spharmgrid.kinematics

.. autofunction:: spharmgrid.streamfunction

.. autofunction:: spharmgrid.velocity_potential

.. autofunction:: spharmgrid.potentials

.. autofunction:: spharmgrid.rotational_wind

.. autofunction:: spharmgrid.divergent_wind

.. autofunction:: spharmgrid.wind
```

## Execution

`sht_threads` sets the number of DUCC threads per spherical harmonic transform. With
`None`, eager operations use DUCC's default thread count and Dask-backed operations use
one thread per transform. The caller controls Dask worker and scheduler configuration.
