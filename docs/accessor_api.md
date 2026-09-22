# Xarray accessor API reference

Importing `spharmgrid` registers the `.sg` accessor on Xarray `DataArray` and
`Dataset` objects. The accessor calls the direct Xarray/NumPy functions and
preserves their coordinate and metadata behavior.

Examples are given in {doc}`quickstart`, {doc}`filtering`, {doc}`regridding`,
{doc}`operators`, and {doc}`kinematics`.

## DataArray.sg

```{eval-rst}
.. autoclass:: spharmgrid._accessors.DataArrayAccessor
   :members:
   :member-order: bysource
```

## Dataset.sg

```{eval-rst}
.. autoclass:: spharmgrid._accessors.DatasetAccessor
   :members:
   :member-order: bysource
```

JAX-backed Xarray objects use the optional `.sgj` accessor; see
{doc}`jax_accessor_api`.
