# API reference

## Xarray accessors

For Xarray workflows, the `.sg` accessor is usually the simplest way to use spharmgrid. Importing `spharmgrid` registers `.sg` on both Xarray `DataArray` and `Dataset` objects, so spherical harmonic operations can be called directly on the data being transformed while preserving Xarray dimensions, coordinates, and metadata.

`DataArray.sg` provides field operations such as filtering, regridding, gradients, Laplacians, and vector transforms. `Dataset.sg` also supports multi-variable atmospheric diagnostics such as vorticity, divergence, streamfunction, velocity potential, and wind decomposition, including variable discovery from canonical names and CF metadata.

The module-level functions documented below expose the same core operations for direct function calls. Accessor-based examples are given in {doc}`quickstart`, {doc}`filtering`, {doc}`regridding`, {doc}`operators`, and {doc}`kinematics`.

The Xarray/NumPy API uses DUCC through `ducc0` for spherical harmonic transforms.
The optional PyTorch tensor API uses `torch-harmonics`, is imported as
`spharmgrid.torch`, and is documented in {doc}`torch`.
The optional JAX array API uses S2FFT, is imported as `spharmgrid.jax`, and is
documented in {doc}`jax`.
It requires JAX x64 mode and `float64` input arrays; spharmgrid does not change
the process-wide JAX precision setting.

## Typing

The public `spharmgrid` API is fully type annotated and the package ships a `py.typed` marker for downstream static type checking.

## Constants and descriptors

```{eval-rst}
.. autodata:: spharmgrid.EARTH_RADIUS_M

.. autoclass:: spharmgrid.Grid
   :members:

.. autoclass:: spharmgrid.TransformSpec
   :members:
```

## Grid and spectral selection

`parse_spectral()` returns a `TransformSpec` for the requested triangular,
trapezoidal, or rhomboidal coefficient domain.

```{eval-rst}
.. autofunction:: spharmgrid.gaussian_grid

.. autofunction:: spharmgrid.clenshaw_curtis_grid

.. autofunction:: spharmgrid.detect_grid

.. autofunction:: spharmgrid.parse_spectral
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

`sht_threads` sets the number of DUCC threads per spherical harmonic transform.
With `None`, eager operations use DUCC's default thread count and Dask-backed
operations use one thread per transform. The caller controls Dask worker and
scheduler configuration.
