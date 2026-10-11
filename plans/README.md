# spharmgrid development plans

The table summarizes implemented APIs and planned grid extensions.

| Phase                 | Status                   | Scope                                                                | Plan                                     |
| --------------------- | ------------------------ | -------------------------------------------------------------------- | ---------------------------------------- |
| 1 — Core              | Complete                 | DUCC0/Xarray, GL/CC, spectral and atmospheric operations             | [01-core.md](01-core.md)                 |
| 2 — Operator suite    | Complete                 | Vector regridding, Helmholtz, inverse gradient and vector Laplacians | [02-sht-suite.md](02-sht-suite.md)       |
| 3 — Accelerators      | Complete for GL/CC       | PyTorch/torch-harmonics and JAX/S2FFT                                | [03-gpu-backends.md](03-gpu-backends.md) |
| 3.5 — Fejér-1         | Complete for Xarray/DUCC | F1 rectangular global grids and spherical harmonic operations        | [03a-fejer-grids.md](03a-fejer-grids.md) |
| 4a — HEALPix          | Planned                  | Pixel/ring representation and DUCC0/S2FFT transforms                 | [04-ring-grids.md](04-ring-grids.md)     |
| 4b — Reduced Gaussian | Planned                  | Variable-length Gaussian rings, DUCC0 first                          | [04-ring-grids.md](04-ring-grids.md)     |

## Supported grids and operations

The Xarray API uses DUCC0 on full global Gauss–Legendre (GL), Clenshaw–Curtis (CC), and
Fejér-1 (F1) rectangular grids. The operator suite includes scalar and vector spectral
regridding, gradient and inverse gradient, scalar and vector Laplacians, vorticity,
divergence, streamfunction, velocity potential, Helmholtz decomposition, and wind
reconstruction.

The JAX and PyTorch APIs support GL and CC grids, with backend-specific bandwidth and
precision limits. Fejér-2 (F2) support is not planned.

## Other grid work

HEALPix and reduced Gaussian grids are described in [Phase 4](04-ring-grids.md). Neither
is implemented. Their grid representations and numerical transforms require separate
verification.
