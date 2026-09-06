# API reference

```python
import spharmgrid as sg
```

## Typing

The public `spharmgrid` API is fully type annotated and the package ships a `py.typed` marker for downstream static type checking.

## Constants and descriptors

```{eval-rst}
.. autodata:: spharmgrid.EARTH_RADIUS_M

.. autoclass:: spharmgrid.Grid
   :members:

.. autoclass:: spharmgrid.SpectralRange
   :members:
```

## Grid and spectral selection

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

Without Dask installed, spharmgrid passes `nthreads=0` and lets DUCC use its default thread count. With Dask support installed, DUCC uses four threads per transform. For Dask-backed arrays, spharmgrid sets the local Dask worker count to `max(1, os.cpu_count() // 4)` only when `num_workers` is unset. A configured `num_workers` value takes precedence. Distributed clusters use their configured worker topology.

## Xarray accessors

Importing `spharmgrid` registers `.sg` on Xarray `DataArray` and `Dataset` objects. Examples are given in {doc}`quickstart`, {doc}`filtering`, {doc}`regridding`, {doc}`operators`, and {doc}`kinematics`.
