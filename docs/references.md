# References

- Sardeshmukh, P. D., and B. J. Hoskins (1984): [Spatial Smoothing on the Sphere](https://doi.org/10.1175/1520-0493(1984)112%3C2524:SSOTS%3E2.0.CO;2), *Monthly Weather Review*, 112, 2524–2529. spharmgrid uses the exponential spectral taper described in this paper.
- Reinecke, M. (2020): [DUCC: Distinctly Useful Code Collection](https://ascl.net/2008.023), *Astrophysics Source Code Library*, ascl:2008.023. DUCC performs the spherical harmonic transforms used by spharmgrid; the Python package is `ducc0`.
- Ishioka, K. (2018): [A New Recurrence Formula for Efficient Computation of Spherical Harmonic Transform](https://doi.org/10.2151/jmsj.2018-019), *Journal of the Meteorological Society of Japan*, 96, 241–249. DUCC uses the accelerated recurrence described in this paper.
- [CF Standard Name Table](https://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html) for standard-name based variable discovery and output metadata.

The {doc}`comparison` page gives the corresponding NCL/SPHEREPACK operations and notes the optional SPHEREPACK parity tests.
