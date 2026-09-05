# Atmospheric wind diagnostics and inverse transforms

spharmgrid computes relative vorticity, horizontal divergence, streamfunction,
and velocity potential from global wind fields, and reconstructs rotational,
divergent, or full wind from those derived quantities. These operations are
descriptive xarray equivalents of common NCL/SPHEREPACK transform workflows.
NCL/SPHEREPACK define useful atmospheric semantics and parity comparisons;
DUCC0 performs spharmgrid's numerical spin-1 transforms.

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

For a Dataset, the accessor discovers `u` and `v` through exact CF standard
names or canonical short names:

```python
kin = ds.sg.kinematics()
pot = ds.sg.potentials()
```

`kinematics()` performs one vector analysis for relative vorticity and
divergence. `potentials()` also shares one vector analysis for streamfunction
and velocity potential.

## Potentials and signs

The returned streamfunction `strf` ($\psi$) and velocity potential `vp`
($\chi$) use

```math
\zeta=\nabla^2\psi,
\qquad
\delta=\nabla^2\chi.
```

For positive degree, their coefficients are therefore

```math
\psi_{\ell m}=-\frac{R^2}{\ell(\ell+1)}\zeta_{\ell m},
\qquad
\chi_{\ell m}=-\frac{R^2}{\ell(\ell+1)}\delta_{\ell m}.
```

The additive degree-zero coefficients are set to zero. Input wind is expected
in SI units of metres per second; the derived vorticity and divergence are in
`s-1`, and the potentials are in `m2 s-1`.

Internally, spharmgrid maps geographic components to DUCC's spin-1 map order
as $(v_\theta,v_\phi)=(-v,u)$. With DUCC E/B coefficients, it applies
$\sqrt{\ell(\ell+1)}/R$ with
$\delta_{\ell m}=-\sqrt{\ell(\ell+1)}E_{\ell m}/R$ and
$\zeta_{\ell m}=-\sqrt{\ell(\ell+1)}B_{\ell m}/R$. This convention is
tested with analytic rotational/divergent fields, round trips, and optional
pyspharm-syl/SPHEREPACK parity checks.

## Rotational and divergent wind

```python
rot = vo.sg.rotational_wind()
div = vp.sg.divergent_wind()

# Explicit semantics are available for an otherwise unidentified field.
rot = sg.rotational_wind(field, quantity="streamfunction")
div = sg.divergent_wind(field, quantity="divergence")
```

`rotational_wind()` accepts relative vorticity or streamfunction.
`divergent_wind()` accepts divergence or velocity potential. They return
Datasets with `u_rotational`/`v_rotational` or
`u_divergent`/`v_divergent`. CF has no exact standard names for those
components, so spharmgrid uses accurate `long_name` and `m s-1` units without
assigning the broader full-wind standard names.

## Full inverse wind

```python
from_vort_div = sg.wind(vo, d)
from_potentials = sg.wind(strf, vp)

from_dataset = xr.Dataset({"vo": vo, "d": d}).sg.wind()
```

`wind()` infers the source representation from exact CF metadata or canonical
short names when possible. Use `source="vorticity_divergence"` or
`source="potentials"` if the input fields cannot be identified. A Dataset
containing both complete representations must also specify `source=`.
