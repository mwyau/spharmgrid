# PyTorch API reference

The `spharmgrid.torch` namespace applies spharmgrid operations to
`torch.Tensor` objects. See {doc}`torch` for installation, grid and bandwidth
constraints, differentiability, and examples.

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

## PyTorch modules

```{eval-rst}
.. autoclass:: spharmgrid.torch.nn.SHTOperators
   :members:
   :member-order: bysource

.. autoclass:: spharmgrid.torch.nn.SHTFilter
   :members:
   :member-order: bysource

.. autoclass:: spharmgrid.torch.nn.SHTRegrid
   :members:
   :member-order: bysource

.. autoclass:: spharmgrid.torch.nn.SHTVectorRegrid
   :members:
   :member-order: bysource
```
