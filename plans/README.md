# spharmgrid roadmap

This index distinguishes implemented capabilities from planned grid support.
The status reflects the code on `main` at v0.4.1 (October 2026).
A plan is not evidence that a feature is released.

| Phase | Status | Scope | Plan |
| --- | --- | --- | --- |
| 1 — Core | Complete | DUCC0/Xarray, GL/CC, spectral and atmospheric operations | [01-core.md](01-core.md) |
| 2 — Operator suite | Complete | Vector regridding, Helmholtz, inverse gradient and vector Laplacians | [02-sht-suite.md](02-sht-suite.md) |
| 3 — Accelerators | Complete for GL/CC | Optional PyTorch/torch-harmonics and JAX/S2FFT APIs; labeled JAX convenience | [03-gpu-backends.md](03-gpu-backends.md) |
| 3.5 — Fejér grids | Planned | Fejér-1 and Fejér-2 rectangular global grids, DUCC0 first | [03a-fejer-grids.md](03a-fejer-grids.md) |
| 4a — HEALPix | Planned | Pixel/ring representation and verified DUCC0/S2FFT transform paths | [04-ring-grids.md](04-ring-grids.md) |
| 4b — Reduced Gaussian | Planned | Variable-length Gaussian rings, DUCC0 first | [04-ring-grids.md](04-ring-grids.md) |

## Current capabilities

The published Xarray API uses DUCC0 on full global Gauss–Legendre (GL) and
pole-including Clenshaw–Curtis (CC) rectangular grids. The operator suite includes
scalar and vector spectral regridding, gradient and inverse gradient, scalar and
vector Laplacians, vorticity, divergence, streamfunction, velocity potential,
Helmholtz decomposition, and wind reconstruction. The optional Torch and JAX
namespaces implement the supported rectangular-grid scientific operation set with
their own backend-specific bandwidth and precision restrictions.

The historical Phase-1 and Phase-2 plans retain their original requirements
for reference. Their future-tense wording is not a current TODO list.

## Next grid work

Add Fejér-1 and Fejér-2 as full rectangular global grids using DUCC0's
existing F1/F2 transform support. Both share the established grid and
operation APIs, but have different sampling nodes and analysis bandwidths.
Keep backend-specific support explicit and test each geometry independently.

Phase 4 retains the separate HEALPix and reduced-Gaussian plans. Neither is
implemented by spharmgrid yet. New grid kinds must not be silently interpreted
as CC, and backend support must be enabled only after verified transforms.
