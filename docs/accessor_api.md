# Xarray accessor API reference

Importing `spharmgrid` registers the `.sg` accessor on Xarray `DataArray` and `Dataset`
objects. Its methods preserve the coordinate and metadata behavior of the corresponding
`spharmgrid` functions.

Examples are given in {doc}`quickstart`, {doc}`filtering`, {doc}`regridding`,
{doc}`operators`, and {doc}`kinematics`.

`DataArray.sg.analyze()` returns a reusable `SpectralField`, and
`DataArray.sg.analyze_vector(v)` returns a reusable `SpectralVectorField`.
`Dataset.sg.analyze_vector()` discovers `u` and `v` from exact CF metadata or canonical
names, with explicit `u=` and `v=` overrides.

```python
vector = eastward.sg.analyze_vector(northward, truncation="T6")
vector = dataset.sg.analyze_vector(u="eastward", v="northward")
```

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
