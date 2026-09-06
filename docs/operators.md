# Scalar and vector operators

The differential operators act in spherical-harmonic space. The default spherical Earth radius is

```python
sg.EARTH_RADIUS_M  # 6_371_220.0 metres
```

Pass `radius=` in metres to use another spherical radius.

## Gradient

```python
gradient = field.sg.gradient()
# Dataset variables: gradient_eastward, gradient_northward

renamed = sg.gradient(field, eastward="dx", northward="dy")
```

The result is the physical horizontal gradient. If the input has a `units` attribute, each component is labeled with input units per metre.

## Inverse gradient

```python
potential = eastward_gradient.sg.inverse_gradient(northward_gradient)
potential = sg.inverse_gradient(eastward_gradient, northward_gradient)
```

`inverse_gradient()` returns the scalar potential associated with the irrotational part of the vector field. A rotational component does not contribute. The scalar degree-zero coefficient is set to zero, fixing the additive constant. If both inputs have the same gradient unit ending in `m-1`, the result uses the corresponding base unit.

## Laplacian

```python
lap = field.sg.laplacian()
```

For each scalar spherical harmonic,

```math
\nabla^2 Y_{\ell m}
= -\frac{\ell(\ell+1)}{R^2}Y_{\ell m}.
```

## Inverse Laplacian

```python
solution = field.sg.inverse_laplacian()
```

For $\ell>0$,

```math
(\nabla^2)^{-1}Y_{\ell m}
= -\frac{R^2}{\ell(\ell+1)}Y_{\ell m}.
```

The degree-zero mode is singular and is set to zero. Therefore `laplacian(inverse_laplacian(field))` recovers the field with its spatial mean removed.

## Vector Laplacian and inverse vector Laplacian

```python
lap = u.sg.vector_laplacian(v)
restored = lap.u.sg.inverse_vector_laplacian(lap.v)

lap = ds.sg.vector_laplacian()

lap = sg.vector_laplacian(u, v)
restored = sg.inverse_vector_laplacian(lap.u, lap.v)
```

Dataset accessors identify `u` and `v` from exact CF standard names or canonical short names.

The vector Laplacian acts on tangent vector harmonics. For both E and B families,

```math
\nabla_v^2: (E_{\ell m}, B_{\ell m}) \mapsto
-\frac{\ell(\ell+1)}{R^2}(E_{\ell m}, B_{\ell m}).
```

This differs from applying the scalar Laplacian independently to eastward and northward geographic components.

`inverse_vector_laplacian()` applies $-R^2/[\ell(\ell+1)]$ at positive degree. Degree-zero vector-harmonic slots do not represent tangent-vector modes and are set to zero. Applying the inverse after the vector Laplacian therefore recovers all representable positive-degree modes.
