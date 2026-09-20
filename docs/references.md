# Bibliography

## Citing spharmgrid

If you use spharmgrid in research, please cite the software release DOI: [10.5281/zenodo.22559210](https://doi.org/10.5281/zenodo.22559210). Machine-readable citation metadata are provided in [`CITATION.cff`](https://github.com/mwyau/spharmgrid/blob/main/CITATION.cff).

## Scientific and software references

- Sardeshmukh, P. D., and B. J. Hoskins (1984): [Spatial Smoothing on the Sphere](<https://doi.org/10.1175/1520-0493(1984)112%3C2524:SSOTS%3E2.0.CO;2>), *Monthly Weather Review*, 112, 2524–2529. spharmgrid uses the exponential spectral taper described in this paper.
- Reinecke, M. (2020): [DUCC: Distinctly Useful Code Collection](https://ascl.net/2008.023), *Astrophysics Source Code Library*, ascl:2008.023. DUCC performs the spherical harmonic transforms for spharmgrid's Xarray/NumPy API through the `ducc0` package.
- [torch-harmonics](https://github.com/NVIDIA/torch-harmonics) performs the scalar and vector spherical harmonic transforms for the PyTorch API.
- [S2FFT](https://github.com/astro-informatics/s2fft) performs the scalar and spin-weighted spherical harmonic transforms for the JAX API.
- [NCL spherical harmonic functions](https://www.ncl.ucar.edu/Document/Functions/spherical.shtml) and SPHEREPACK are implementation references for the atmospheric operations compared with spharmgrid.
- [pyspharm](https://github.com/jswhit/pyspharm) is used by the optional SPHEREPACK parity tests.
- [CF Standard Name Table](https://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html) is used for standard-name based variable discovery and output metadata.

The {doc}`comparison` page lists the corresponding NCL/SPHEREPACK operations.
