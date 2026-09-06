# Scalar operators

The scalar operators act in spherical-harmonic space. The default spherical
Earth radius is:

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

The result is the physical horizontal gradient. When the input has a `units`
attribute, each component is labeled with input units per metre.

## Inverse gradient

```python
potential = eastward_gradient.sg.inverse_gradient(northward_gradient)

# Direct equivalent.
potential = sg.inverse_gradient(eastward_gradient, northward_gradient)
```

`inverse_gradient()` returns a scalar potential for the vector's irrotational
component. A rotational component is omitted, consistent with SPHEREPACK
inverse-gradient semantics. The scalar degree-zero coefficient is set to zero,
so the result is fixed up to an additive constant. When both inputs declare the
same gradient unit ending in `m-1`, the result uses the corresponding base
unit.

## Laplacian

```python
lap = field.sg.laplacian()
```

For every scalar spherical harmonic,

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

The degree-zero mode is singular, so its coefficient is set to zero. As a
result, `laplacian(inverse_laplacian(field))` recovers the field with its
spatial mean removed.

## Vector Laplacian and inverse vector Laplacian

```python
lap = u.sg.vector_laplacian(v)
restored = lap.u.sg.inverse_vector_laplacian(lap.v)

# Dataset accessors discover `u` and `v` from exact CF standard names or
# canonical short names.
lap = ds.sg.vector_laplacian()

# Direct equivalents.
lap = sg.vector_laplacian(u, v)
restored = sg.inverse_vector_laplacian(lap.u, lap.v)
```

The vector Laplacian is defined on tangent vector harmonics. For both E and B
families, it applies

```math
\nabla_v^2: (E_{\ell m}, B_{\ell m}) \mapsto
-\frac{\ell(\ell+1)}{R^2}(E_{\ell m}, B_{\ell m}),
```

This is not a scalar Laplacian applied independently to eastward and northward
geographic components.

`inverse_vector_laplacian()` applies
$-R^2/[\ell(\ell+1)]$ at positive degree. Vector-harmonic degree-zero slots
do not describe a tangent-vector mode and are set to zero. Thus applying the
inverse after the vector Laplacian recovers every representable positive-degree
mode, with those null slots fixed by convention.
