# Xarray accessor API reference

Importing `spharmgrid` registers the `.sg` accessor on Xarray `DataArray` and `Dataset`
objects. Its methods preserve the coordinate and metadata behavior of the corresponding
`spharmgrid` functions.

Examples are given in {doc}`quickstart`, {doc}`filtering`, {doc}`regridding`,
{doc}`operators`, and {doc}`kinematics`.

The entries below document `DataArray.sg` and `Dataset.sg`.

## DataArray.sg

```{eval-rst}
.. container:: accessor-api

   .. autoclass:: spharmgrid._accessors.DataArrayAccessor
      :members:
      :member-order: bysource
      :no-index-entry:
```

## Dataset.sg

```{eval-rst}
.. container:: accessor-api

   .. autoclass:: spharmgrid._accessors.DatasetAccessor
      :members:
      :member-order: bysource
      :no-index-entry:
```

JAX-backed Xarray objects use the optional `.sgj` accessor; see {doc}`jax_accessor_api`.
