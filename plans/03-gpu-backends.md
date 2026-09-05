# Phase 3: multi-backend SHT execution

## Goal

Add the backend boundary and optional accelerator implementations after the
DUCC0 scientific operation suite is stable, while the public grid model is
still limited to rectangular Gauss--Legendre (GL) and Clenshaw--Curtis (CC)
grids.

The intended architecture is one spharmgrid scientific API over three transform
engines:

```text
DUCC0             CPU/reference implementation
torch-harmonics   PyTorch/GPU scalar + vector SHT implementation
S2FFT             JAX/GPU arbitrary-spin implementation
```

Phase 3 comes before reduced Gaussian and HEALPix support. Backend conventions
are easier to establish on the existing GL/CC model than while simultaneously
changing both transform execution and horizontal-grid representation.

DUCC0 remains the default/reference engine. spharmgrid should not replace or
reimplement the numerical SHT libraries. It owns the scientific operation
semantics, grid translation, atmospheric conventions, and user-facing APIs over
the engines.

---

## 1. Preconditions

Phase 2 should first establish the scientific operation set and semantics,
including where implemented:

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
streamfunction
velocity_potential
potentials
helmholtz
rotational_wind
divergent_wind
wind
```

Do not delay Phase 3 for reduced Gaussian or HEALPix. Those become Phase 4.

---

## 2. Refactor only as required by real backends

Extract the smallest internal transform boundary that the current DUCC
implementation and real alternate implementations require.

Conceptually:

```python
class SHTBackend(Protocol):
    def scalar_analysis(...): ...
    def scalar_synthesis(...): ...
    def vector_analysis(...): ...
    def vector_synthesis(...): ...
    def supports_grid(...): ...
```

The exact interface should be derived from the current code plus the installed
`torch-harmonics` and S2FFT APIs. Do not design a general plugin framework.

Before adding an alternate backend, route the current DUCC path through the
minimal boundary and prove that existing numerical results, metadata, Dask
behavior, and public APIs are unchanged.

Do not put atmospheric functions such as `vorticity()` or `helmholtz()` inside
backend classes.

The intended layering is:

```text
spharmgrid scientific operation
        |
        +-- spectral/physical definition
        |
        +-- backend transform primitive
              DUCC0
              torch-harmonics
              S2FFT
```

---

## 3. One scientific definition, framework-native execution

Keep one source of truth for the scientific definitions:

```text
spectral range parsing
Sardeshmukh--Hoskins taper
scalar Laplacian and inverse multipliers
zero/null-mode conventions
Earth-radius factors
gradient/inverse-gradient relationships
vorticity/divergence relationships
streamfunction/velocity-potential relationships
rotational/divergent/full-wind definitions
output quantity names and metadata
```

Do not fork those semantics by backend.

Do not force literal NumPy code reuse when it would break Torch/JAX
differentiability. Small framework-native implementations of coefficient masks,
degree multipliers, or array operations are acceptable when they implement the
same tested formula. The shared contract is the mathematics and semantics, not
one physical array-expression function for all frameworks.

Do not expose a universal raw `alm` representation merely to make the adapters
look identical. DUCC0, torch-harmonics, and S2FFT use different coefficient
layouts and conventions.

A backend-native coefficient result may be exposed inside a framework-specific
namespace only if a concrete tensor-native use case requires it. It must be
identified as backend-native rather than a package-wide coefficient format.

---

## 4. Backend capability is explicit

Backend support is capability-based. Do not promise every engine supports every
grid, bandwidth, dtype, or operation identically.

For Phase 3, focus on the existing rectangular grids:

```text
GL
CC where exact sampling equivalence and supported bandwidth are demonstrated
```

Current torch-harmonics uses triangular truncation and its
Clenshaw--Curtis/equiangular bandwidth behavior differs from DUCC's full
representable `lmax`/`mmax` behavior when spharmgrid is called without an
explicit `Tn` range.

Therefore:

- explicit `Tn` requests may map naturally when the backend can represent the
  same triangular band;
- `truncation=None` must not silently claim identical retained bandwidth across
  backends when the underlying engines differ;
- unsupported backend/grid/bandwidth combinations must fail clearly;
- do not silently clamp a user's explicit spectral request.

Do not infer equivalence from names such as `equiangular`, `lobatto`, or
`legendre-gauss`; compare actual nodes, quadrature, normalization, and
band-limit conventions.

---

## 5. torch-harmonics is the PyTorch numerical engine

Do not independently implement the torch-harmonics SHT inside spharmgrid.

Current torch-harmonics implements the scalar transforms in:

```text
torch_harmonics/sht.py
    RealSHT
    InverseRealSHT
```

and the vector transforms in the same numerical library:

```text
RealVectorSHT
InverseRealVectorSHT
```

The scalar transform itself combines a real FFT in longitude with the
latitude/associated-Legendre quadrature transform. torch-harmonics owns that
algorithm, its PyTorch autograd behavior, CUDA/cuFFT execution, coefficient
normalization/layout, precomputed quadrature data, and distributed variants.

spharmgrid should call those installed modules rather than copy their FFT,
Legendre, or SHT/ISHT implementation.

### 5.1 What the spharmgrid adapter owns

The adapter should translate:

```text
sg.Grid / spharmgrid sampling semantics
    -> torch-harmonics grid name and shape

spharmgrid spectral request
    -> supported torch-harmonics lmax/mmax

spharmgrid scientific operation
    -> RealSHT / InverseRealSHT or vector transforms
       + spharmgrid-owned coefficient operation
```

For example, a differentiable deterministic filter should conceptually remain:

```text
torch.Tensor field
    -> torch-harmonics RealSHT
    -> spharmgrid T-range mask / taper in Torch
    -> torch-harmonics InverseRealSHT
    -> torch.Tensor field
```

Likewise, Laplacians, inverse operators, and atmospheric wind relationships use
torch-harmonics for transforms and spharmgrid for the physical/spectral
multipliers and sign/radius conventions.

The adapter may cache or reuse transform modules where useful, but cache design
should follow measured workload needs rather than become a new public object
hierarchy.

### 5.2 Initial target

Start with:

```text
GL
torch-harmonics equiangular/CC only after exact coordinate and convention parity
```

For each operation, compare against DUCC spharmgrid on the same physical grid
and spectral content. Add analytic tests that can detect paired sign errors in
forward/inverse vector transforms.

Do not add an operation merely because a scalar analogue exists. Vector/spin
conventions must be independently verified.

---

## 6. SFNO and learned spherical spectral convolution

SFNO is a concrete near-term consumer for climate-variability research, but it
is distinct from spharmgrid's deterministic spectral filter.

Current torch-harmonics implements:

```text
torch_harmonics/spectral_convolution.py
    SpectralConvS2
```

`SpectralConvS2` performs:

```text
input channels on sphere
    -> RealSHT
    -> learned complex channel mixing in spectral space
    -> InverseRealSHT
    -> output channels on sphere
```

Its learned weights depend on spherical-harmonic degree and are part of a neural
operator. This is not the same operation as spharmgrid's fixed `T42`, `T6-42`,
or Sardeshmukh--Hoskins filter response.

### 6.1 `SpectralConvS2` is not required for the scientific operation API

Do not use `SpectralConvS2` to implement:

```text
filter
regrid
laplacian
vorticity/divergence
streamfunction/velocity potential
```

Those operations require only the forward/inverse scalar or vector transforms
plus spharmgrid's deterministic spectral mathematics.

### 6.2 Do not copy NVIDIA's spectral convolution

For SFNO model work, prefer using the maintained torch-harmonics
`SpectralConvS2` implementation directly or through a very thin spharmgrid
adapter if spharmgrid grid translation removes meaningful boilerplate.

Do not create an independent spharmgrid copy of the learned spectral
convolution, FFT, Legendre transform, or SFNO architecture merely to place it
under the spharmgrid namespace.

A future convenience API such as:

```python
import spharmgrid.torch as sgt

conv = sgt.nn.SpectralConvS2(
    in_channels=64,
    out_channels=64,
    grid=grid,
)
```

is acceptable only if it remains a thin adapter to torch-harmonics and provides
a real advantage such as `sg.Grid` translation, capability validation, or
consistent truncation semantics. It should not fork NVIDIA's implementation.

### 6.3 Existing higher-level SFNO packages

Before adding model-level code to spharmgrid, account for existing packages:

- `neuraloperator` already provides a ready-made `SFNO` model and
  `SphericalConv` layer and uses torch-harmonics for spherical transforms;
- NVIDIA Makani provides the weather/climate SFNO training/model stack and uses
  torch-harmonics;
- NVIDIA Earth2Studio wraps complete forecast models and workflows, including
  SFNO/Makani-based models.

These packages cover neural-operator architectures and forecast workflows.
spharmgrid should not duplicate them.

For climate-variability research, the preferred division is initially:

```text
xarray/CF preprocessing and atmospheric diagnostics    spharmgrid
PyTorch spherical scientific operators                 spharmgrid.torch
SHT/ISHT numerical kernels                             torch-harmonics
SFNO architecture                                      neuraloperator or Makani,
                                                       or research-local model code
```

If research establishes a missing reusable layer between torch-harmonics and
these model packages, add the smallest justified spharmgrid adapter then.

### 6.4 SFNO interoperability acceptance

Phase 3 should include at least one small interoperability example/test showing
that data and grids prepared for `spharmgrid.torch` can be used in an SFNO-style
PyTorch workflow without NumPy/device breaks.

This may use torch-harmonics `SpectralConvS2` directly or a current
`neuraloperator` SFNO/SphericalConv layer. It does not require spharmgrid to own
a full SFNO model.

---

## 7. S2FFT adapter

Implement S2FFT as the JAX/arbitrary-spin accelerator backend.

Initial Phase-3 target:

```text
JAX-native scalar and spin operations
GL
supported differentiable operation paths
```

Map geographic eastward/northward wind to and from S2FFT spin-1 coefficients
explicitly and verify the mapping independently.

S2FFT's HEALPix support is important, but HEALPix becomes a Phase-4 grid feature.
The Phase-3 S2FFT adapter should be designed so Phase 4 can add HEALPix without
replacing the backend interface.

Do not require S2FFT's PyTorch wrapper to become a second public PyTorch API if
torch-harmonics already covers that use case.

---

## 8. Cross-backend convention audit

For each engine document and test:

- spherical-harmonic normalization;
- Condon--Shortley phase convention;
- coefficient ordering and real-field storage;
- `lmax`/`mmax` inclusivity and truncation;
- spin basis definition;
- longitude phase/origin;
- latitude/colatitude orientation;
- physical vector component convention;
- gradient sign;
- E/B or divergent/rotational mapping;
- radius application;
- dtype/precision behavior.

Use analytic scalar harmonics and independently constructed vector fields.

A forward/inverse round trip is necessary but not sufficient: matching errors
can cancel. Where applicable compare against:

```text
analytic fields
DUCC spharmgrid
SPHEREPACK/pyspharm for GL/CC atmospheric operations
```

---

## 9. Public accelerator API and I/O boundary

Use separate interfaces for scientific file/metadata workflows and
accelerator-native differentiable computation.

### 9.1 Existing xarray API remains the scientific I/O interface

Keep the existing xarray API as the normal atmospheric-science interface:

```python
field.sg.filter("T42")
ds.sg.kinematics()
```

xarray remains responsible for:

```text
NetCDF / Zarr / GRIB decoding through installed engines
named dimensions and coordinates
CF metadata
variable discovery
time/calendar representation
scientific output assembly
```

The default xarray path remains DUCC0 unless a later measured convenience mode
justifies explicit accelerator execution.

Do not make torch-harmonics or S2FFT responsible for file I/O or CF semantics.

### 9.2 Tensor-native APIs are the primary accelerator interfaces

The primary Phase-3 accelerator APIs should be tensor/array native:

```python
import spharmgrid.torch as sgt
import spharmgrid.jax as sgj

filtered = sgt.filter(x, grid=grid, truncation="T42")
vo, div = sgt.kinematics(u, v, grid=grid)

filtered = sgj.filter(x, grid=grid, truncation="T42")
vo, div = sgj.kinematics(u, v, grid=grid)
```

Use native containers:

```text
spharmgrid.torch   torch.Tensor
spharmgrid.jax     jax.Array
```

Do not wrap Torch/JAX tensors in xarray merely to call the transform kernel.
The differentiable path must remain inside the native framework.

Use the spharmgrid grid description rather than exposing backend-specific grid
strings in the scientific API:

```python
grid = sg.gaussian_grid(128, 256)
sgt.filter(x, grid=grid, truncation="T42")
sgj.filter(x, grid=grid, truncation="T42")
```

Each adapter translates that grid into its backend's sampling terminology and
validates capability/bandwidth.

### 9.3 Tensor dimension convention

For rectangular tensor-native operations, use the last two dimensions as the
horizontal dimensions and treat all preceding dimensions as independent
leading/batch dimensions:

```text
(nlat, nlon)
(time, nlat, nlon)
(batch, channel, nlat, nlon)
(member, level, time, nlat, nlon)
```

Do not reproduce xarray's named-dimension machinery in the tensor API unless a
real use case requires it.

Phase 4 may use a one-dimensional trailing pixel/cell dimension for packed
HEALPix or reduced-Gaussian arrays where appropriate.

### 9.4 Explicit xarray <-> accelerator boundary

For accelerator workflows, conversion between scientific containers and device
arrays should be explicit at the data-loader or application boundary, not on
every SHT call.

Conceptually:

```text
NetCDF / Zarr / GRIB
        |
      xarray
        |
 host NumPy payload
        |
 torch.Tensor / jax.Array
        |
 repeated GPU/TPU model or SHT work
```

For PyTorch this may use `torch.as_tensor(...)`/device transfer. For JAX this
may use `jax.device_put(...)`. Convert back to xarray only when a labeled
scientific result or file output is required.

Do not add file-reading/writing methods to `spharmgrid.torch` or
`spharmgrid.jax`.

Small helpers such as `from_xarray()`/`to_xarray()` may be considered later if
real usage shows that they remove repeated boilerplate without hiding expensive
device transfers. They are not required for the initial accelerator API.

### 9.5 JAX/xarray interoperability is an optional higher-level path

Google DeepMind's `xarray_jax` demonstrates that xarray objects containing JAX
arrays can be registered as JAX PyTrees and used with `jit`, `grad`, `vmap`, and
sharding while retaining labels/coordinates.

This is useful precedent, but spharmgrid should not require `xarray_jax` merely
to expose S2FFT. The initial JAX kernel API should remain `jax.Array` native.

After the raw JAX API is correct and differentiable, evaluate optional
`xarray_jax` interoperability as a convenience layer. If adopted, keep it an
optional dependency and verify that spharmgrid operations remain JIT/grad-safe.
Do not assume ordinary xarray operations or the existing DUCC `apply_ufunc`
path are automatically safe for JAX transformations.

There is no need to force the PyTorch and JAX convenience layers to be
identical if their host framework ecosystems differ.

### 9.6 Optional xarray accelerator convenience may come later

A future explicit call such as:

```python
field.sg.filter("T42", backend="torch")
```

could be useful for a user who wants an xarray result and accepts host/device
transfer. It is not the primary accelerator API and should not be added until
benchmarks demonstrate a useful workload.

If added, document it as an xarray convenience path rather than a differentiable
model API. Never auto-select an accelerator because hardware is present.

---

## 10. Differentiability

For tensor-native paths:

- keep Torch autograd/JAX transformations intact;
- do not convert tensors to NumPy inside differentiable operations;
- add automatic-gradient tests for representative scalar and vector operations;
- compare with finite differences on small problems where numerically useful.

Test at least representative paths for:

```text
filter
laplacian
kinematics
wind reconstruction
```

Do not claim spharmgrid differentiability solely because the transform library
is differentiable.

---

## 11. Precision

Establish explicit expectations for:

```text
float32
float64
internal complex precision
```

Do not silently cast all GPU work to float32.

Do not require bitwise equality between independent implementations. Use
operation/grid/dtype-specific tolerances justified by analytic and cross-backend
error measurements.

---

## 12. Performance acceptance

Benchmark both xarray/file-boundary and tensor-native workflows.

Include at least:

```text
single small atmospheric field
batched T42/T63-type fields
0.25-degree-scale rectangular field
large batch / ML-style workload
```

Separate:

```text
host -> device transfer
JIT/module setup
steady-state transform
device -> host transfer
full spharmgrid operation
```

For tensor-native workloads, benchmark data already resident on device.

Do not report a GPU speedup for an xarray workflow while excluding transfer or
compile costs.

For the SFNO interoperability case, measure only enough to ensure the spharmgrid
adapter does not add material overhead beyond the underlying torch-harmonics
layer. Do not make model-training performance a spharmgrid benchmark suite.

---

## 13. Optional dependencies and CI

Keep accelerator and model stacks optional.

Conceptually:

```toml
[project.optional-dependencies]
torch = ["torch-harmonics>=..."]
jax = ["s2fft>=...", "jax>=..."]
jax-xarray = ["xarray-jax>=..."]  # only if this integration is adopted
```

`neuraloperator` and Makani should not become dependencies merely because they
are useful SFNO reference/consumer packages. If an optional interoperability
test requires neuraloperator, keep it in a dedicated test/research extra rather
than the runtime `torch` extra.

Determine exact supported versions during implementation. Do not add the
`jax-xarray` extra unless that integration is actually implemented and tested.

Do not make Torch, JAX, CUDA, torch-harmonics, S2FFT, xarray-jax,
neuraloperator, or Makani part of the base install. Normal CPU CI must remain
independent of accelerator availability.

Use small CPU adapter/import tests where possible. Add GPU CI only when a
reliable runner is available.

---

## 14. Relevant ecosystem precedent

Use current implementations as architectural evidence, not APIs to copy
blindly.

### 14.1 NVIDIA torch-harmonics

`torch-harmonics` was created to enable SFNO and is the low-level differentiable
PyTorch SHT library. Its SHTs are `torch.nn.Module` objects operating on batched
native tensors. This is the engine spharmgrid should wrap for PyTorch, not
reimplement.

### 14.2 neuraloperator

`neuraloperator` provides a higher-level generic neural-operator API including
`SFNO` and `SphericalConv`, with torch-harmonics supplying the spherical
transforms. It is a good first choice for research that needs a configurable
SFNO architecture rather than only raw SHT operations.

It is not an atmospheric xarray/CF diagnostics package, so it does not replace
spharmgrid's intended role.

### 14.3 NVIDIA Makani and Earth2Studio

Makani is a weather/climate model training and inference stack with SFNO models
and a direct torch-harmonics dependency. Earth2Studio sits higher still and
wraps complete forecast models, data sources, coordinates, and workflows.

They demonstrate that weather-model infrastructure and generic field-level SHT
operations are separate layers.

### 14.4 windspharm

`windspharm` is close to spharmgrid's atmospheric-science ergonomics: it
provides user-friendly vorticity/divergence/streamfunction and related wind
analysis with xarray/metadata interfaces. However, it is based on
pyspharm/SPHEREPACK rather than torch-harmonics and is not a differentiable GPU
PyTorch layer.

The combination spharmgrid is targeting remains distinct:

```text
atmospheric/xarray scientific API
        +
modern DUCC CPU implementation
        +
optional torch-harmonics PyTorch backend
        +
optional S2FFT JAX backend
```

### 14.5 Google DeepMind

NeuralGCM/Dinosaur uses a JAX-native spectral dynamical core with its own
spherical-harmonic implementation. The core SHT operates on `jax.Array`, with
explicit nodal/modal representations and accelerator-oriented JAX execution.
NeuralGCM provides xarray conversion utilities at the scientific API boundary.

DeepMind also provides `xarray_jax` for labeled JAX structures where direct
JAX-transformed xarray workflows are valuable.

GraphCast and GenCast do not solve their spherical model computation with an
SHT; they use spherical graph/icosahedral-mesh architectures. This is a
different way to avoid latitude/longitude convolution problems, not an
alternate SHT implementation.

The general design lesson for spharmgrid is:

```text
scientific metadata / files      xarray
accelerator numerical kernel     native Torch/JAX arrays
optional labeled JAX execution   xarray_jax-style integration if justified
model architecture               dedicated neural-operator/weather packages
```

---

## 15. Phase-4 handoff

Phase 3 should leave the package ready to add new grids without another backend
redesign.

Expected Phase-4 capability direction:

```text
GL                DUCC0 + torch-harmonics + S2FFT where validated
CC                DUCC0 + torch-harmonics where validated
HEALPix           DUCC0 + S2FFT
reduced Gaussian  DUCC0 initially
```

This table is a target capability map, not a promise that every operation has
identical analysis semantics or bandwidth on every engine.

---

## 16. Acceptance criteria

Phase 3 is complete when:

- current DUCC results and public CPU behavior remain unchanged through the new
  internal transform boundary;
- the backend abstraction is no larger than required by the three actual
  engines;
- torch-harmonics is used as the PyTorch numerical SHT implementation rather
  than copied or independently reimplemented;
- torch-harmonics is a tested optional backend on its accepted rectangular
  grids;
- S2FFT is a tested optional JAX/spin backend on its accepted Phase-3 grid(s);
- `spharmgrid.torch` and `spharmgrid.jax` provide tensor-native differentiable
  APIs for the supported scientific operation set;
- accelerator APIs use spharmgrid grid objects rather than backend sampling
  strings as their scientific grid contract;
- rectangular tensor APIs use trailing horizontal dimensions with arbitrary
  leading/batch dimensions;
- file/CF I/O remains an xarray responsibility rather than entering the tensor
  namespaces;
- the supported Phase-2 operation graph is shared semantically rather than
  duplicated as independent atmospheric implementations;
- geographic-vector/spin conventions are independently proven for each backend;
- differentiable tensor-native paths contain no NumPy breaks;
- unsupported grid/bandwidth/backend combinations fail clearly;
- explicit spectral requests are never silently clamped;
- accelerator precision and performance are measured with realistic overhead;
- an SFNO-style interoperability smoke test works without requiring spharmgrid
  to own a full SFNO model;
- `SpectralConvS2` is not used as a substitute for deterministic atmospheric
  filters/operators;
- model-level packages such as neuraloperator or Makani remain optional
  consumers/references, not core dependencies;
- the base installation remains free of Torch/JAX dependencies;
- no package-wide raw coefficient compatibility format is introduced;
- HEALPix and reduced Gaussian public grid expansion remain Phase-4 work.

---

## Current implementation references

- DUCC0: https://github.com/mreineck/ducc
- torch-harmonics: https://github.com/NVIDIA/torch-harmonics
- neuraloperator: https://github.com/neuraloperator/neuraloperator
- NVIDIA Makani: https://github.com/NVIDIA/makani
- NVIDIA Earth2Studio: https://github.com/NVIDIA/earth2studio
- S2FFT: https://github.com/astro-informatics/s2fft
- windspharm: https://github.com/ajdawson/windspharm
- Dinosaur/NeuralGCM dycore: https://github.com/neuralgcm/dinosaur
- DeepMind xarray/JAX integration: https://github.com/google-deepmind/xarray_jax
- WeatherNext: https://github.com/google-deepmind/weathernext

When implementing this phase, verify current library APIs and grid/bandwidth
behavior from the installed versions rather than relying on this planning
snapshot.