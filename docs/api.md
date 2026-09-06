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

.. py:function:: spharmgrid.regrid(field, target_grid, spectral=None, *, lmin=None, lmax=None, taper=None, nthreads=None)

   Spectrally regrid a GL or CC field. A spectral range and taper can be
   applied during regridding.

.. py:function:: spharmgrid.gradient(field, *, eastward="gradient_eastward", northward="gradient_northward", radius=EARTH_RADIUS_M, nthreads=None)

   Compute the physical eastward and northward horizontal gradient.

.. autofunction:: spharmgrid.inverse_gradient

.. autofunction:: spharmgrid.laplacian

.. autofunction:: spharmgrid.inverse_laplacian
```

## Vector spectral operations

```{eval-rst}
.. autofunction:: spharmgrid.regrid_vector

.. autofunction:: spharmgrid.helmholtz

.. autofunction:: spharmgrid.vector_laplacian

.. autofunction:: spharmgrid.inverse_vector_laplacian
```

## Atmospheric wind diagnostics and transforms

```{eval-rst}
.. autofunction:: spharmgrid.vorticity

.. autofunction:: spharmgrid.divergence

.. py:function:: spharmgrid.kinematics(u, v, *, vorticity="vo", divergence="d", radius=EARTH_RADIUS_M, nthreads=None)

   Compute relative vorticity and horizontal divergence.

.. autofunction:: spharmgrid.streamfunction

.. autofunction:: spharmgrid.velocity_potential

.. py:function:: spharmgrid.potentials(u, v, *, streamfunction="strf", velocity_potential="vp", radius=EARTH_RADIUS_M, nthreads=None)

   Compute streamfunction and velocity potential.

.. autofunction:: spharmgrid.rotational_wind

.. autofunction:: spharmgrid.divergent_wind

.. autofunction:: spharmgrid.wind
```

## xarray accessors

Importing spharmgrid registers `.sg` on xarray `DataArray` and `Dataset`.
Accessor examples are shown throughout the {doc}`quickstart`, {doc}`filtering`,
{doc}`regridding`, {doc}`operators`, and {doc}`kinematics` pages.
