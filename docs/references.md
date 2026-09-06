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
  SPHEREPACK are used for comparison of atmospheric operations and
  conventions.
- The [CF Standard Name
  Table](https://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html)
  defines the variable names used for CF discovery and output metadata.
- spharmgrid grew out of spherical-harmonic code used in
  [PyStormTracker](https://github.com/mwyau/PyStormTracker).
