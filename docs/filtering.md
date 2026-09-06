# Spectral filtering

`DataArray.sg.filter()` and `sg.filter()` analyze a supported global field, select spherical harmonic coefficients by total degree, and synthesize the filtered field on the same grid. The variable name and attributes are preserved.

```python
low_pass = field.sg.filter("T42")
band_pass = field.sg.filter("T6-42")
explicit = field.sg.filter(lmin=6, lmax=42)
```

`T42` retains modes with total degree $0 \leq l \leq 42$. `T6-42` retains $6 \leq l \leq 42$. Parsing is case-insensitive and accepts an en dash, for example `T6–42`. `spectral=` cannot be combined with explicit `lmin=` and `lmax=` in the same call.

With no spectral range, spharmgrid uses the transform bandwidth supported by the grid. The latitude sampling can support degrees above the largest represented zonal order.

## Hard selection

`taper=None` gives a hard spectral selection:

```text
l < lmin             zero
lmin <= l <= lmax    retained unchanged
l > lmax             zero
```

With no spectral range and no taper, `filter()` performs analysis and synthesis over the grid's available transform bandwidth.

## Sardeshmukh–Hoskins taper

Pass a response in `(0, 1]` to apply the exponential taper of Sardeshmukh and Hoskins (1984):

```python
tapered = field.sg.filter("T6-42", taper=0.1)
```

```math
w(l)=\exp[-K\{l(l+1)\}^2],
\qquad
K=\frac{-\ln(\mathrm{taper})}{\{l_{\max}(l_{\max}+1)\}^2}.
```

`taper=0.1` gives $w(l_{\max})=0.1$. With an explicit range, the taper is applied within the retained range and modes outside it are zero. Without an explicit range, the endpoint is the transform `lmax` supported by the grid. `taper=1` leaves retained modes unchanged.

See {doc}`references` for the taper reference.
