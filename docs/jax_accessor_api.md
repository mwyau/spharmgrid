# JAX Xarray accessor API reference

Importing `spharmgrid.jax` registers `.sgj` on Xarray `DataArray` and
`Dataset` objects. The accessor calls the `spharmgrid.jax` numerical functions
with JAX-backed data and returns labeled Xarray objects.

`device_put()` converts data-variable payloads to JAX arrays. Coordinates
remain Xarray coordinates. `device_get()` copies JAX-backed payloads to host
arrays. Scientific `.sgj` methods do not transfer data implicitly, and
`xarray_jax` is not required.

For tensor-native functions that accept and return `jax.Array` objects, see
{doc}`jax_api`. The user guide is in {doc}`jax`.

## Explicit data placement

```{eval-rst}
.. autofunction:: spharmgrid.jax.device_put

.. autofunction:: spharmgrid.jax.device_get
```

## DataArray.sgj

```{eval-rst}
.. autoclass:: spharmgrid.jax._accessors.DataArrayAccessor
   :members:
   :member-order: bysource
```

## Dataset.sgj

```{eval-rst}
.. autoclass:: spharmgrid.jax._accessors.DatasetAccessor
   :members:
   :member-order: bysource
```

The separately tested `xarray_jax` path supports whole-Xarray PyTree
transformations. That interoperability path is distinct from `.sgj`.
