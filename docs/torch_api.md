# PyTorch API reference

The `spharmgrid.torch` namespace applies spharmgrid operations to `torch.Tensor`
objects. See {doc}`torch` for installation, grid and bandwidth constraints,
differentiability, and examples.

For reusable PyTorch modules, see {doc}`torch_nn_api`.

## Reusable analyzed representations

`analyze()` and `analyze_vector()` retain native torch-harmonics coefficients for
repeated spectral work. The coefficient layout remains private, and tensors stay on
their input device.

`spec` describes the currently available spectral domain. Filtering may narrow it;
discarded modes cannot be restored, and later selections or regridding cannot expand it.
Tapers modify the current coefficients and compose when repeated.

```python
spectral = sgt.analyze(field, grid=grid)
large_scale = spectral.filter("T5-42").synthesize()
laplacian = spectral.laplacian().synthesize()
```

```{eval-rst}
.. autoclass:: spharmgrid.torch.SpectralField
   :members:

.. autoclass:: spharmgrid.torch.SpectralVectorField
   :members:

.. autofunction:: spharmgrid.torch.analyze

.. autofunction:: spharmgrid.torch.analyze_vector
```

## Scalar operations

```{eval-rst}
.. autofunction:: spharmgrid.torch.filter

.. autofunction:: spharmgrid.torch.regrid

.. autofunction:: spharmgrid.torch.gradient

.. autofunction:: spharmgrid.torch.inverse_gradient

.. autofunction:: spharmgrid.torch.laplacian

.. autofunction:: spharmgrid.torch.inverse_laplacian
```

## Vector operations

```{eval-rst}
.. autofunction:: spharmgrid.torch.regrid_vector

.. autofunction:: spharmgrid.torch.helmholtz

.. autofunction:: spharmgrid.torch.vector_laplacian

.. autofunction:: spharmgrid.torch.inverse_vector_laplacian
```

## Atmospheric kinematics and wind transforms

```{eval-rst}
.. autofunction:: spharmgrid.torch.vorticity

.. autofunction:: spharmgrid.torch.divergence

.. autofunction:: spharmgrid.torch.kinematics

.. autofunction:: spharmgrid.torch.streamfunction

.. autofunction:: spharmgrid.torch.velocity_potential

.. autofunction:: spharmgrid.torch.potentials

.. autofunction:: spharmgrid.torch.rotational_wind

.. autofunction:: spharmgrid.torch.divergent_wind

.. autofunction:: spharmgrid.torch.wind
```
