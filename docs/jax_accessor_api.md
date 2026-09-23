# JAX Xarray accessor API reference

Importing `spharmgrid.jax` registers `.sgj` on Xarray `DataArray` and `Dataset` objects.
Its methods apply `spharmgrid.jax` operations to JAX-backed data and return labeled
Xarray objects.

`device_put()` moves data-variable payloads to JAX arrays while coordinates remain
Xarray coordinates. `device_get()` copies JAX-backed payloads to host arrays. `.sgj`
methods operate on the current JAX-backed payloads.

For functions that accept and return `jax.Array` objects, see {doc}`jax_api`. The user
guide is in {doc}`jax`.

The entries below document `DataArray.sgj` and `Dataset.sgj`.

## Host/device transfer

```{eval-rst}
.. autofunction:: spharmgrid.jax.device_put

.. autofunction:: spharmgrid.jax.device_get
```

## DataArray.sgj

```{eval-rst}
.. container:: accessor-api

   .. autoclass:: spharmgrid.jax._accessors.DataArrayAccessor
      :members:
      :member-order: bysource
      :no-index-entry:
```

## Dataset.sgj

```{eval-rst}
.. container:: accessor-api

   .. autoclass:: spharmgrid.jax._accessors.DatasetAccessor
      :members:
      :member-order: bysource
      :no-index-entry:
```
