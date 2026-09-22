# JAX API reference

The `spharmgrid.jax` functions accept and return `jax.Array` objects. See {doc}`jax` for
installation, supported shapes, precision requirements, JAX transformations, and
examples. For JAX-backed Xarray objects, see {doc}`jax_accessor_api`.

## Reusable analyzed representations

`analyze()` and `analyze_vector()` retain native S2FFT coefficients for
workflows that need several spectral operations. The representations are JAX
PyTrees with coefficients as dynamic leaves and immutable transform metadata as
static data.

```python
spectral = sgj.analyze(field, grid=grid)
large_scale = spectral.filter("T5-42").synthesize()
laplacian = spectral.laplacian().synthesize()
```

```{eval-rst}
.. autoclass:: spharmgrid.jax.SpectralField
   :members:

.. autoclass:: spharmgrid.jax.SpectralVectorField
   :members:

.. autofunction:: spharmgrid.jax.analyze

.. autofunction:: spharmgrid.jax.analyze_vector
```

## Scalar operations

```{eval-rst}
.. autofunction:: spharmgrid.jax.filter

.. autofunction:: spharmgrid.jax.regrid

.. autofunction:: spharmgrid.jax.gradient

.. autofunction:: spharmgrid.jax.inverse_gradient

.. autofunction:: spharmgrid.jax.laplacian

.. autofunction:: spharmgrid.jax.inverse_laplacian
```

## Vector operations

```{eval-rst}
.. autofunction:: spharmgrid.jax.regrid_vector

.. autofunction:: spharmgrid.jax.helmholtz

.. autofunction:: spharmgrid.jax.vector_laplacian

.. autofunction:: spharmgrid.jax.inverse_vector_laplacian
```

## Atmospheric kinematics and wind transforms

```{eval-rst}
.. autofunction:: spharmgrid.jax.vorticity

.. autofunction:: spharmgrid.jax.divergence

.. autofunction:: spharmgrid.jax.kinematics

.. autofunction:: spharmgrid.jax.streamfunction

.. autofunction:: spharmgrid.jax.velocity_potential

.. autofunction:: spharmgrid.jax.potentials

.. autofunction:: spharmgrid.jax.rotational_wind

.. autofunction:: spharmgrid.jax.divergent_wind

.. autofunction:: spharmgrid.jax.wind
```
