# Phase 4: optional GPU and differentiable SHT backends

## Goal

Add accelerator support only after the DUCC CPU API, SHT operator suite, and
new grid semantics are stable.

The purpose is not to replace DUCC0.

The purpose is to make the same spharmgrid scientific operations useful in
GPU-resident and differentiable workflows.

Initial candidates:

```text
torch-harmonics
S2FFT
```

Both provide the harmonic primitives needed to reproduce much of spharmgrid's
mathematics, but neither directly provides spharmgrid's atmospheric/xarray API.

---

## 1. What the candidate libraries provide

### torch-harmonics

Current torch-harmonics provides:

- differentiable real scalar SHT and inverse SHT;
- explicit real vector SHT and inverse vector SHT;
- PyTorch CPU/GPU execution;
- distributed SHT variants;
- multiple quadrature/sampling grids;
- resampling helpers;
- examples for spectral derivatives, Helmholtz-type PDEs and spherical
  physical simulation.

Relevant primitives include:

```text
RealSHT
InverseRealSHT
RealVectorSHT
InverseRealVectorSHT
```

This makes torch-harmonics the most direct candidate for a PyTorch-native
version of spharmgrid's existing wind operations.

### S2FFT

Current S2FFT provides:

- forward and inverse spin spherical-harmonic transforms;
- arbitrary spin, including spin 0 and spin 1 needed for scalar/vector work;
- JAX-native differentiable implementations;
- PyTorch wrappers around JAX transforms;
- Gauss--Legendre and several sampling theorems;
- HEALPix transforms;
- iterative refinement for forward/analysis transforms;
- a CUDA HEALPix primitive in current releases.

S2FFT is especially relevant to:

```text
JAX
differentiable spin transforms
HEALPix on accelerator hardware
```

### Important limitation

Neither package directly exposes the current spharmgrid API:

```text
filter
regrid / regrid_vector
gradient
laplacian / inverse_laplacian
vector Laplacian
vorticity / divergence
streamfunction / velocity potential
rotational / divergent wind
Helmholtz decomposition
```

They expose transforms and related numerical primitives.

spharmgrid must continue to own the physical definitions, metadata, operator
multipliers, variable semantics, signs and output conventions.

---

## 2. Do not assume identical grid support

Backend support should be capability-based.

DUCC should remain the reference/default backend and may support the broadest
set of spharmgrid grids.

Before adding another backend, verify exact sampling equivalence for:

```text
GL
CC
reduced Gaussian
HEALPix
```

Do not infer that a library's `"equiangular"` or `"lobatto"` label is exactly
the same grid spharmgrid calls CC without comparing coordinates, quadrature,
normalization and representable band limits.

Likewise:

- torch-harmonics core should not be assumed to provide HEALPix simply because
  PyTorch can run on a GPU;
- S2FFT should not be assumed to support reduced Gaussian simply because it
  supports HEALPix and GL;
- reduced Gaussian may remain DUCC-only unless a GPU backend gains a valid
  equivalent transform.

The public API should not promise every backend supports every grid.

---

## 3. First implementation decision: tensor-native API vs xarray transfer

The current package is xarray-first.

A naive GPU mode:

```text
xarray/NumPy CPU
    -> copy to GPU
    -> one transform
    -> copy back to CPU
```

may be slower than DUCC and is not differentiable across the xarray/NumPy
boundary.

Therefore do not immediately add:

```python
field.sg.filter(..., backend="torch")
```

as a public promise.

First prototype two execution models.

### Model A: xarray convenience acceleration

xarray DataArray enters on CPU, data are transferred to GPU, results return as
xarray/NumPy.

This is only worthwhile for sufficiently large/batched workloads.

### Model B: tensor-native differentiable API

Likely structure:

```python
import spharmgrid.torch as sgt
import spharmgrid.jax as sgj

filtered = sgt.filter(tensor, grid=grid, spectral="T42")
vo, div = sgt.kinematics(u, v, grid=grid)

filtered = sgj.filter(array, grid=grid, spectral="T42")
```

The exact namespace is not fixed by this plan, but a tensor-native interface is
likely cleaner for training/differentiation than forcing Torch/JAX through
xarray.

Do not expose either design before measuring real workloads.

---

## 4. Backend architecture becomes justified only here

The initial project correctly avoided a hypothetical backend framework.

At this phase, with two or more actual implementations, introduce the smallest
internal transform protocol that the supported operations need.

Conceptually:

```python
class SHTBackend(Protocol):
    def scalar_analysis(...): ...
    def scalar_synthesis(...): ...
    def vector_analysis(...): ...
    def vector_synthesis(...): ...
    def supports_grid(...): ...
```

Do not put atmospheric functions such as `vorticity()` into backend classes.

The layering should remain:

```text
spharmgrid scientific operation
        |
        +-- spectral mask/operator/sign/radius semantics
        |
        +-- backend primitive:
              DUCC
              torch-harmonics
              S2FFT
```

Keep one source of truth for:

- spectral range parsing;
- Sardeshmukh--Hoskins taper;
- scalar Laplacian multipliers;
- inverse zero modes;
- wind E/B or spin mapping;
- CF/xarray output metadata;
- operation names.

Where coefficient layouts differ, normalize only inside backend adapters.

Do not expose a supposedly universal raw `alm` object merely to simplify
backend code.

---

## 5. Candidate rollout

### 5.1 torch-harmonics first for PyTorch

Prototype torch-harmonics first for:

```text
GL
and only those CC/equiangular cases proven equivalent
```

Implement the full Phase 2 physical operation graph from:

```text
RealSHT / InverseRealSHT
RealVectorSHT / InverseRealVectorSHT
```

Target:

```text
filter
regrid
regrid_vector

gradient
inverse_gradient
laplacian
inverse_laplacian
vector_laplacian
inverse_vector_laplacian

vorticity
divergence
kinematics
potentials
helmholtz
rotational_wind
divergent_wind
wind
```

Do not add a GPU backend for an operation until cross-backend parity is
demonstrated.

### 5.2 S2FFT for JAX/spin/HEALPix

Use S2FFT where it provides a distinct capability rather than maintaining two
redundant PyTorch implementations.

Initial focus:

```text
JAX-native scalar/spin operations
GL
HEALPix
differentiability
```

Map geographic wind to/from S2FFT spin-1 coefficients explicitly and verify it
independently.

Do not reuse DUCC sign constants blindly.

If S2FFT's PyTorch wrapper provides no distinct benefit over torch-harmonics,
do not expose it as a second public PyTorch backend merely for completeness.

---

## 6. Cross-backend convention audit

For each backend document and test:

- spherical-harmonic normalization;
- Condon--Shortley phase convention;
- coefficient ordering;
- real-field storage;
- `lmax`/`mmax` conventions;
- spin basis definition;
- longitude phase/origin;
- colatitude/latitude orientation;
- physical vector component convention;
- gradient sign;
- rotational/divergent or E/B mapping;
- sphere radius application.

Use analytic harmonics to determine mappings.

Do not tune signs until a round trip works and then call it correct.
A paired forward/inverse sign error can cancel.

Compare against:

```text
analytic scalar harmonics
analytic vector/spin harmonics
DUCC spharmgrid
pyspharm/SPHEREPACK for GL/CC atmospheric operations
```

where applicable.

---

## 7. Differentiability

For tensor-native APIs:

- preserve autograd/JAX transformations through spharmgrid's spectral
  operations;
- do not convert tensors to NumPy inside a differentiable path;
- add gradient tests for representative scalar and vector operations;
- compare automatic gradients with finite differences on small problems where
  numerically appropriate.

Test at least:

```text
filter
laplacian
kinematics
wind reconstruction
```

Do not claim an operation is differentiable solely because the underlying SHT
library is differentiable.

---

## 8. Precision

Establish explicit cross-backend expectations for:

```text
float32
float64
complex precision used internally
```

Do not silently cast everything to float32 for GPU speed.

Do not require bitwise equality between independent implementations.

Use numerical tolerances justified by:

- dtype;
- transform band limit;
- grid;
- analysis method;
- iterative tolerance where applicable.

For inverse/iterative HEALPix analysis, report both solver tolerance and
physical-field error.

---

## 9. Performance acceptance

GPU support is valuable primarily when:

- many maps are batched;
- data are already GPU-resident;
- lmax is high enough to amortize launch/JIT overhead;
- the operation is repeated inside training/inference;
- differentiability is needed.

Benchmark at least:

```text
single small atmospheric field
batched T42/T63-type fields
0.25-degree-scale field
large batch / ML-style workload
```

Separate:

```text
host -> device transfer
JIT/compile or module setup
steady-state transform
device -> host transfer
full spharmgrid operation
```

Do not market a GPU speedup based on a steady-state kernel while excluding
transfer/JIT costs from a CPU-xarray use case.

For tensor-native workloads, benchmark with data already resident on device.

---

## 10. Optional dependencies

Keep accelerator stacks optional.

Conceptually:

```toml
[project.optional-dependencies]
torch = ["torch-harmonics>=..."]
jax = ["s2fft>=...", "jax>=..."]
```

Determine exact dependencies during implementation; do not pin speculative
versions in advance.

Do not make Torch, JAX, CUDA or S2FFT part of the base spharmgrid install.

CI should keep the normal CPU suite independent of accelerator availability.

Use small CPU tests for adapter import/API behavior and dedicated GPU CI only
when a reliable GPU runner is available.

---

## 11. Public backend selection

Do not expose a public `backend=` enum until at least one alternate backend is
implemented and tested.

When that point is reached, choose between:

```text
explicit backend= on xarray API
tensor-native spharmgrid.torch / spharmgrid.jax namespaces
both, with clearly different purposes
```

based on measured workflows.

Never auto-select a GPU backend merely because a GPU is present.

Explicit selection is required for reproducibility.

---

## 12. Acceptance criteria

Phase 4 is complete when:

- DUCC remains the default/reference CPU implementation;
- at least one GPU/tensor backend reproduces the supported spharmgrid
  scientific operation set within stated tolerances;
- geographic-vector/spin conventions are independently proven;
- differentiable paths contain no NumPy breaks;
- unsupported backend/grid combinations fail clearly;
- GPU performance is measured including realistic overhead;
- base installation remains free of Torch/JAX dependencies;
- no two redundant public accelerator backends are maintained without a
  distinct use case;
- documentation separates numerical backend capability from spharmgrid
  scientific semantics.

---

## Current research summary

### torch-harmonics

Repository:
https://github.com/NVIDIA/torch-harmonics

Key current capability:
scalar and vector spherical-harmonic transforms implemented in PyTorch,
including:

```text
RealSHT
InverseRealSHT
RealVectorSHT
InverseRealVectorSHT
```

### S2FFT

Repository:
https://github.com/astro-informatics/s2fft

Key current capability:
differentiable forward/inverse arbitrary-spin spherical-harmonic transforms,
JAX execution, PyTorch wrappers, Gauss--Legendre support, HEALPix support and
iterative refinement.

### DUCC

Repository:
https://github.com/mreineck/ducc

DUCC remains the CPU reference/default because it already provides efficient
scalar/spin transforms, general isolatitude-ring geometry, arbitrary-grid
support and iterative analysis.
