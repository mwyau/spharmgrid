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
