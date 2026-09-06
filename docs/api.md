# API reference

```python
import spharmgrid as sg
```

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

.. py:function:: spharmgrid.detect_grid(field)

   Detect a supported GL or CC grid from an xarray object.

.. autofunction:: spharmgrid.parse_spectral
```

## Scalar operations

```{eval-rst}
.. autofunction:: spharmgrid.filter

.. autofunction:: spharmgrid.regrid

.. autofunction:: spharmgrid.gradient

.. autofunction:: spharmgrid.laplacian

.. autofunction:: spharmgrid.inverse_laplacian
```

## Atmospheric wind diagnostics and transforms

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

## xarray accessors

Importing spharmgrid registers `.sg` on xarray `DataArray` and `Dataset`.
Accessor examples are shown throughout the {doc}`quickstart`, {doc}`filtering`,
{doc}`regridding`, {doc}`operators`, and {doc}`kinematics` pages.
