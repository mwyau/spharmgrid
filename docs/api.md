# Xarray/NumPy API reference

The Xarray/NumPy API uses DUCC through `ducc0` for spherical harmonic transforms.
For the `.sg` interface, see {doc}`accessor_api`. The optional PyTorch tensor
API uses `torch-harmonics`; see {doc}`torch` for usage and {doc}`torch_api`
for its API reference. The optional JAX array API uses S2FFT; see {doc}`jax`
for usage and {doc}`jax_api` for its API reference. The JAX API requires JAX
x64 mode; `float32` inputs are promoted to `float64` for S2FFT execution.
spharmgrid does not change the process-wide JAX precision setting.

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
