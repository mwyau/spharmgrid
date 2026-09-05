# Spectral filtering

`DataArray.sg.filter()` and `sg.filter()` analyze a supported global field,
apply a total-wavenumber coefficient selection, and synthesize on the same
grid. They preserve the data-variable name and attributes because filtering
does not change the physical quantity.

```python
low_pass = field.sg.filter("T42")
band_pass = field.sg.filter("T6-42")
explicit = field.sg.filter(lmin=6, lmax=42)
```

`T42` retains all modes with total degree $0 \leq l \leq 42$. `T6-42` retains
$6 \leq l \leq 42$. Parsing is case-insensitive and accepts an en dash, for
example `T6–42`. Do not mix `spectral=` notation with explicit `lmin=` and
`lmax=` in one call.

With no spectral range, spharmgrid uses the complete transform bandwidth
representable by the grid rather than inventing an implicit `Tn` truncation.
This can retain higher-degree zonally symmetric content when latitude sampling
supports it but longitude sampling limits the largest represented order.

## Hard selection is the default

`taper=None` is the default. With an explicit retained range it is an exact
hard selection:

```text
l < lmin             zero
lmin <= l <= lmax    retained unchanged
l > lmax             zero
```

There is no hidden smoothing. With no spectral range and no taper, `filter()`
performs an analysis/synthesis over the grid's available transform bandwidth.

## Optional Sardeshmukh--Hoskins taper

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
is applied only inside the retained range and modes outside it remain zero.
Without an explicit range, the endpoint is the transform `lmax` available on
the grid. `taper=1` leaves retained modes unchanged.

The taper is a coefficient response, not a generic smoothing-strength
parameter. Its published lineage is documented in {doc}`references`.
