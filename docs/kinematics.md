# Atmospheric kinematics and inverse transforms

spharmgrid computes relative vorticity, horizontal divergence, streamfunction, and velocity potential from global wind fields. It also reconstructs rotational, divergent, or full wind fields from these quantities.

## Vorticity, divergence, streamfunction, and velocity potential

For geographic eastward wind `u` and northward wind `v`:

```python
vo = sg.vorticity(u, v)
d = sg.divergence(u, v)
kin = sg.kinematics(u, v)  # Dataset: vo, d

strf = sg.streamfunction(u, v)
vp = sg.velocity_potential(u, v)
pot = sg.potentials(u, v)  # Dataset: strf, vp
```

Dataset accessors identify `u` and `v` from exact CF standard names or canonical short names:

```python
kin = ds.sg.kinematics()
pot = ds.sg.potentials()
```

## Potentials and signs

The returned streamfunction `strf` ($\psi$) and velocity potential `vp` ($\chi$) satisfy

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

Degree-zero coefficients are set to zero. Input wind is in metres per second. Relative vorticity and divergence are in `s-1`; streamfunction and velocity potential are in `m2 s-1`.

## Helmholtz decomposition

```python
parts = u.sg.helmholtz(v)
# Dataset variables: u_divergent, v_divergent, u_rotational, v_rotational

parts = ds.sg.helmholtz()
parts = sg.helmholtz(u, v)
```

`helmholtz()` separates a tangent wind field into divergent and rotational components. For a representable field, the two eastward components sum to the input eastward wind and the two northward components sum to the input northward wind.

The radius factors cancel in this decomposition. The four output fields use `long_name` metadata because CF has no exact standard names for divergent and rotational wind components.

## Rotational and divergent wind

```python
rot = vo.sg.rotational_wind()
div = vp.sg.divergent_wind()

rot = sg.rotational_wind(field, quantity="streamfunction")
div = sg.divergent_wind(field, quantity="divergence")
```

`rotational_wind()` accepts relative vorticity or streamfunction. `divergent_wind()` accepts divergence or velocity potential. The returned Datasets contain `u_rotational`/`v_rotational` or `u_divergent`/`v_divergent`.

## Full inverse wind

```python
from_vort_div = sg.wind(vo, d)
from_potentials = sg.wind(strf, vp)

from_dataset = xr.Dataset({"vo": vo, "d": d}).sg.wind()
```

`wind()` identifies vorticity/divergence or streamfunction/velocity-potential inputs from exact CF metadata or canonical short names. Use `source="vorticity_divergence"` or `source="potentials"` when the input fields cannot be identified uniquely. A Dataset containing both complete representations also requires `source=`.
