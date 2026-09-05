# References and implementation lineage

## Scientific method and numerical transform lineage

- Sardeshmukh, P. D., and B. J. Hoskins (1984): [Spatial Smoothing on the
  Sphere](https://doi.org/10.1175/1520-0493(1984)112%3C2524:SSOTS%3E2.0.CO;2),
  *Monthly Weather Review*, 112, 2524--2529. This is the lineage of the
  supported exponential spectral taper.
- Reinecke, M., and D. S. Seljebotn (2013): [Libsharp -- spherical harmonic
  transforms revisited](https://doi.org/10.1051/0004-6361/201321494),
  *Astronomy & Astrophysics*, 554, A112. This describes relevant numerical
  spherical-harmonic transform lineage.
- Ishioka, K. (2018): [A New Recurrence Formula for Efficient Computation of
  Spherical Harmonic Transform](https://doi.org/10.2151/jmsj.2018-019),
  *Journal of the Meteorological Society of Japan*, 96, 241--249.

## Software and semantic references

- [DUCC0](https://gitlab.mpcdf.mpg.de/mtr/ducc) supplies the numerical scalar
  and spin-weighted transform machinery used by spharmgrid.
- [NCL spherical-harmonic function
  documentation](https://www.ncl.ucar.edu/Document/Functions/Spherepack/) and
  SPHEREPACK provide established atmospheric operation semantics and a parity
  comparison point. spharmgrid does not contain SPHEREPACK.
- The [CF Standard Name
  Table](https://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html)
  defines the exact variable semantics used for discovery and generated
  metadata.
- [PyStormTracker](https://github.com/mwyau/PyStormTracker) is the source
  implementation from which the initial global GL/CC DUCC0 wrapper was
  extracted and generalized.

The test suite uses analytic harmonics and identities, plus dedicated optional
Gaussian-grid and compatible pole-including regular-grid (CC) comparisons
against pyspharm-syl/SPHEREPACK. These are implementation checks, not a claim
that any one implementation is scientific ground truth.
