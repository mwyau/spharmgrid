# JAX API reference

The `spharmgrid.jax` functions accept and return `jax.Array` objects. See {doc}`jax` for
installation, supported shapes, precision requirements, JAX transformations, and
examples. For JAX-backed Xarray objects, see {doc}`jax_accessor_api`.

## Reusable analyzed representations

`analyze()` and `analyze_vector()` retain native S2FFT coefficients for repeated
spectral work. The representations are JAX PyTrees with coefficients as dynamic leaves
and transform metadata as static data. PyTree reconstruction preserves grid geometry and
tensor-axis ordering.

`spec` describes the currently available spectral domain. Filtering may narrow it;
discarded modes cannot be restored, and later selections or regridding cannot expand it.
Tapers modify the current coefficients and compose when repeated.

`spharmgrid.jax.analyze()` is the backend-native reusable API; `.sgj` remains the
labeled Xarray interface for one-shot JAX operations.

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
