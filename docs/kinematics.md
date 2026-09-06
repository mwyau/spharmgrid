# Atmospheric wind diagnostics and inverse transforms

spharmgrid computes relative vorticity, horizontal divergence, streamfunction,
and velocity potential from global wind fields. It can also reconstruct
rotational, divergent, or full wind fields from those quantities.

## Vorticity, divergence, streamfunction, and velocity potential

Given geographic eastward `u` and northward `v` wind:

```python
vo = sg.vorticity(u, v)
d = sg.divergence(u, v)
kin = sg.kinematics(u, v)  # Dataset: vo, d

strf = sg.streamfunction(u, v)
vp = sg.velocity_potential(u, v)
pot = sg.potentials(u, v)  # Dataset: strf, vp
```

For a Dataset, `u` and `v` can be identified from exact CF standard names or
canonical short names:

```python
kin = ds.sg.kinematics()
pot = ds.sg.potentials()
```

## Potentials and signs

The returned streamfunction `strf` ($\psi$) and velocity potential `vp`
($\chi$) use

```math
\zeta=\nabla^2\psi,
\qquad
\delta=\nabla^2\chi.
```

For positive degree,

```math
\psi_{\ell m}=-\frac{R^2}{\ell(\ell+1)}\zeta_{\ell m},
\qquad
\chi_{\ell m}=-\frac{R^2}{\ell(\ell+1)}\delta_{\ell m}.
```

The degree-zero coefficients are set to zero. Input wind is expected in SI
units of metres per second; relative vorticity and divergence are in `s-1`, and
the potentials are in `m2 s-1`.

## Rotational and divergent wind

```python
rot = vo.sg.rotational_wind()
div = vp.sg.divergent_wind()

rot = sg.rotational_wind(field, quantity="streamfunction")
div = sg.divergent_wind(field, quantity="divergence")
```

`rotational_wind()` accepts relative vorticity or streamfunction.
`divergent_wind()` accepts divergence or velocity potential. The returned
Datasets contain `u_rotational`/`v_rotational` or
`u_divergent`/`v_divergent`.

CF has no exact standard names for these component fields, so the outputs use
`long_name` metadata and `m s-1` units.

## Full inverse wind

```python
from_vort_div = sg.wind(vo, d)
from_potentials = sg.wind(strf, vp)

from_dataset = xr.Dataset({"vo": vo, "d": d}).sg.wind()
```

`wind()` identifies the source representation from exact CF metadata or
canonical short names when possible. Use `source="vorticity_divergence"` or
`source="potentials"` when the input fields cannot be identified. A Dataset
containing both complete representations also requires `source=`.
