# References

## Scientific references

- Sardeshmukh, P. D., and B. J. Hoskins (1984): [Spatial Smoothing on the
  Sphere](https://doi.org/10.1175/1520-0493(1984)112%3C2524:SSOTS%3E2.0.CO;2),
  *Monthly Weather Review*, 112, 2524--2529. spharmgrid uses the exponential
  spectral taper described in this paper.
- Reinecke, M., and D. S. Seljebotn (2013): [Libsharp -- spherical harmonic
  transforms revisited](https://doi.org/10.1051/0004-6361/201321494),
  *Astronomy & Astrophysics*, 554, A112.
- Ishioka, K. (2018): [A New Recurrence Formula for Efficient Computation of
  Spherical Harmonic Transform](https://doi.org/10.2151/jmsj.2018-019),
  *Journal of the Meteorological Society of Japan*, 96, 241--249.

## Software and conventions

- [DUCC0](https://gitlab.mpcdf.mpg.de/mtr/ducc) supplies the scalar and
  spin-weighted spherical-harmonic transforms used by spharmgrid.
- [NCL spherical-harmonic function
  documentation](https://www.ncl.ucar.edu/Document/Functions/Spherepack/) and
  SPHEREPACK provide atmospheric operation semantics and parity references.
- [pyspharm's `Spharmt` interface](https://github.com/jswhit/pyspharm/blob/master/Lib/spharm.py)
  exposes SPHEREPACK workflows used by the optional parity tests.
- The [CF Standard Name
  Table](https://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html)
  defines the variable names used for CF discovery and output metadata.
- spharmgrid grew out of spherical-harmonic code used in
  [PyStormTracker](https://github.com/mwyau/PyStormTracker).

## NCL/SPHEREPACK correspondence

spharmgrid uses descriptive Python names rather than NCL's fixed-grid and
Gaussian-grid suffix families. This is a semantic map, not a claim that NCL is
the origin of the mathematics. An asterisk denotes the relevant NCL variants.

| spharmgrid | NCL/SPHEREPACK family |
| --- | --- |
| `gradient` | `gradsf`, `gradsg` |
| `inverse_gradient` | `igradsf`, `igradsg` |
| `laplacian` | `lapsf`, `lapsg` |
| `inverse_laplacian` | `ilapsf`, `ilapsg` |
| `vector_laplacian` | `lapvf`, `lapvg` |
| `inverse_vector_laplacian` | `ilapvf`, `ilapvg` |
| `vorticity` | `uv2vr*` |
| `divergence` | `uv2dv*` |
| `kinematics` | `uv2vrdv*` |
| `potentials` | `uv2sfvp*` |
| `rotational_wind` | `vr2uv*` or streamfunction synthesis |
| `divergent_wind` | `dv2uv*` or velocity-potential synthesis |
| `wind` | `vrdv2uv*`, `sfvp2uv*` |
| `helmholtz` | `uv2vrdv*` followed by rotational/divergent synthesis |
| `regrid` | `f2fsh`, `f2gsh`, `g2fsh`, `g2gsh` |
| `regrid_vector` | `f2fshv`, `f2gshv`, `g2fshv`, `g2gshv` |

`pyspharm-syl` is an optional parity dependency, not a runtime dependency. Its
tests compare spharmgrid GL grids with SPHEREPACK Gaussian sampling and CC
grids with its pole-including regular sampling.
