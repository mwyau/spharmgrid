# Xarray accessor API reference

Importing `spharmgrid` registers the `.sg` accessor on both Xarray `DataArray` and `Dataset` objects.

`DataArray.sg` provides field operations such as filtering, regridding, gradients, Laplacians, and vector transforms. `Dataset.sg` also supports multi-variable atmospheric diagnostics such as vorticity, divergence, streamfunction, velocity potential, and wind decomposition, including variable discovery from canonical names and CF metadata.

Accessor-based examples are given in {doc}`quickstart`, {doc}`filtering`, {doc}`regridding`, {doc}`operators`, and {doc}`kinematics`.

## DataArray.sg

```{eval-rst}
.. autoclass:: spharmgrid.accessors.DataArrayAccessor
   :members:
   :member-order: bysource
```

## Dataset.sg

```{eval-rst}
.. autoclass:: spharmgrid.accessors.DatasetAccessor
   :members:
   :member-order: bysource
```
