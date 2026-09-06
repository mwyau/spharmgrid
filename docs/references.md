# References

- Sardeshmukh, P. D., and B. J. Hoskins (1984): [Spatial Smoothing on the Sphere](https://doi.org/10.1175/1520-0493(1984)112%3C2524:SSOTS%3E2.0.CO;2), *Monthly Weather Review*, 112, 2524–2529. spharmgrid uses the exponential spectral taper described in this paper.
- Reinecke, M. (2020): [DUCC: Distinctly Useful Code Collection](https://ascl.net/2008.023), *Astrophysics Source Code Library*, ascl:2008.023. DUCC performs the spherical harmonic transforms used by spharmgrid; the Python package is `ducc0`.
- [NCL spherical harmonic functions](https://www.ncl.ucar.edu/Document/Functions/Spherepack/) and SPHEREPACK define the atmospheric operations used for comparison in spharmgrid.
- [pyspharm](https://github.com/jswhit/pyspharm) is used by the optional SPHEREPACK parity tests.
- [PyStormTracker](https://github.com/mwyau/PyStormTracker) uses spherical harmonic processing for atmospheric fields and is a downstream user of spharmgrid.
- [CF Standard Name Table](https://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html) for standard-name based variable discovery and output metadata.

The {doc}`comparison` page lists the corresponding NCL/SPHEREPACK operations.
