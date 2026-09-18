# Spectral filtering

`DataArray.sg.filter()` and `sg.filter()` analyze a supported global field, select spherical harmonic coefficients by total degree, and synthesize the filtered field on the same grid. The variable name and attributes are preserved.

```python
low_pass = field.sg.filter("T42")
band_pass = field.sg.filter("T6-42")
explicit = field.sg.filter(lmin=6, lmax=42)
```

The supported truncation strings are:

| Form   | Retained coefficient domain                                                   |
| ------ | ----------------------------------------------------------------------------- |
| `Tn`   | triangular: $0 \leq m \leq l \leq n$                                          |
| `Ta-b` | triangular total-degree band-pass: $a \leq l \leq b$, with $0 \leq m \leq l$  |
| `Tnxm` | trapezoidal: $0 \leq m' \leq m$, with $m' \leq l \leq n$                      |
| `Rn`   | symmetric rhomboidal: $0 \leq m \leq n$, $m \leq l \leq 2n$, and $l-m \leq n$ |

For example, `T42` is triangular, `T5-42` is a total-degree band-pass,
`T42x10` is trapezoidal with maximum degree 42 and maximum order 10, and
`R42` is rhomboidal with maximum order 42 and maximum degree 84. These are
truncation and coefficient-selection strings, not grid names. Parsing is
case-insensitive and accepts an en dash, for example `T5–42`.

The notation intentionally does not encode compound cases such as a rhomboidal
band or a trapezoidal band. `truncation=` cannot be combined with explicit
`lmin=` and `lmax=` in the same call. The terminology follows the spectral
algorithm overview used in [CAM](https://acomstaff.acom.ucar.edu/tilmes/CAM_docs/doc/build/html/cam6_scientific_guide/chapter3.html#spectral-algorithm-overview).

With no explicit truncation, spharmgrid uses the transform bandwidth supported by
the grid. The latitude sampling can support degrees above the largest represented
zonal order.

## Hard selection

`taper=None` gives a hard spectral selection:

```text
l < lmin             zero
lmin <= l <= lmax    retained unchanged
l > lmax             zero
```

With no explicit truncation and no taper, `filter()` performs analysis and
synthesis over the grid's available transform bandwidth.

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
