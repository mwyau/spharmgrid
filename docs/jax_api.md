# JAX API reference

The `spharmgrid.jax` namespace applies spharmgrid operations to `jax.Array`
objects. See {doc}`jax` for installation, supported array shapes, precision
requirements, JAX transformations, and examples.

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
