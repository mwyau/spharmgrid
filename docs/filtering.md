# Spectral filtering

`DataArray.sg.filter()` and `sg.filter()` analyze a supported global field,
apply a total-wavenumber coefficient selection, and synthesize on the same
grid. The data-variable name and attributes are preserved.

```python
low_pass = field.sg.filter("T42")
band_pass = field.sg.filter("T6-42")
explicit = field.sg.filter(lmin=6, lmax=42)
```

`T42` retains all modes with total degree $0 \leq l \leq 42$. `T6-42` retains
$6 \leq l \leq 42$. Parsing is case-insensitive and accepts an en dash, for
example `T6–42`. `spectral=` cannot be combined with explicit `lmin=` and
`lmax=` in the same call.

With no spectral range, spharmgrid uses the complete transform bandwidth
supported by the grid. This can include degrees above the largest represented
zonal order when the latitude sampling supports them.

## Hard selection

`taper=None` is the default. With an explicit retained range, the coefficient
selection is:

```text
l < lmin             zero
lmin <= l <= lmax    retained unchanged
l > lmax             zero
```

With no spectral range and no taper, `filter()` performs an analysis/synthesis
over the grid's available transform bandwidth.

## Sardeshmukh--Hoskins taper

Pass a response in `(0, 1]` to apply the exponential taper used by
Sardeshmukh and Hoskins (1984):

```python
tapered = field.sg.filter("T6-42", taper=0.1)
```

```math
w(l)=\exp[-K\{l(l+1)\}^2],
\qquad
K=\frac{-\ln(\mathrm{taper})}{\{l_{\max}(l_{\max}+1)\}^2}.
```

Thus `taper=0.1` means $w(l_{\max})=0.1$. With an explicit range, the response
is applied inside the retained range and modes outside it are zero. Without an
explicit range, the endpoint is the transform `lmax` available on the grid.
`taper=1` leaves retained modes unchanged.

See {doc}`references` for the original taper reference.
