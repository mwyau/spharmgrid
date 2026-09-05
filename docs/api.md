# API reference

The top-level namespace is intentionally small. Import it as:

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

.. autofunction:: spharmgrid.detect_grid

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

## Wind operations

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

Importing spharmgrid registers `.sg` on both xarray types. The accessor methods
delegate to the direct operations above; they do not use separate numerical
kernels.

```{eval-rst}
.. autoclass:: spharmgrid.accessors.DataArrayAccessor
   :members:

.. autoclass:: spharmgrid.accessors.DatasetAccessor
   :members:
```
