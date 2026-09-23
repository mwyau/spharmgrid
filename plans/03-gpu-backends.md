# Phase 3: multi-backend SHT execution

## Goal

Complete the rectangular-grid accelerator support by adding a JAX API backed by S2FFT
after the PyTorch API introduced in v0.2.0. The public grid model remains limited to
rectangular Gauss–Legendre (GL) and Clenshaw–Curtis (CC) grids for this phase.

The scientific operation set is shared across three transform engines:

```text
DUCC0             CPU/reference implementation
torch-harmonics   PyTorch/GPU scalar + vector SHT implementation
S2FFT             JAX/GPU arbitrary-spin implementation
```

Phase 3 comes before reduced Gaussian and HEALPix support. Backend conventions are
easier to establish on the existing GL/CC model than while simultaneously changing both
transform execution and horizontal-grid representation.

DUCC0 remains the default/reference engine. spharmgrid should not replace or reimplement
the numerical SHT libraries. It owns the scientific operation semantics, grid
translation, atmospheric conventions, and user-facing APIs over the engines.

---

## 1. Preconditions

Phase 2 should first establish the scientific operation set and semantics, including
where implemented:

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

## 2. Refactor only where the implementations actually share code

Add the S2FFT/JAX implementation without first routing DUCC0 and torch-harmonics through
a new common backend class. If the JAX work exposes an identical internal transform
interface that removes real duplication, extract only that shared piece.

Do not create a plugin framework or move atmospheric operations such as `vorticity()` or
`helmholtz()` into backend classes.

The implementation layers remain:

```text
spharmgrid scientific definition
        |
        +-- Xarray/NumPy implementation using DUCC0
        +-- PyTorch implementation using torch-harmonics
        +-- JAX implementation using S2FFT
```

---

## 3. One scientific definition, framework-native execution

Keep one scientific definition for:

```text
spectral range parsing
Sardeshmukh–Hoskins taper
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

Do not force literal NumPy code reuse when it would break Torch/JAX differentiability.
Small framework-native implementations of coefficient masks, degree multipliers, or
array operations are acceptable when they implement the same tested formula. The shared
contract is the mathematics and semantics, not one physical array-expression function
for all frameworks.

Do not expose a universal raw `alm` representation merely to make the adapters look
identical. DUCC0, torch-harmonics, and S2FFT use different coefficient layouts and
conventions.

A backend-native coefficient result may be exposed inside a framework-specific namespace
only if a concrete tensor-native use case requires it. It must be identified as
backend-native rather than a package-wide coefficient format.

---

## 4. Backend capability is explicit

Backend support is capability-based. Do not promise every engine supports every grid,
bandwidth, dtype, or operation identically.

For S2FFT in v0.3.0, both rectangular samplings are required:

```text
GL:
    L = nlat
    require nlon = 2L - 1
    map to S2FFT sampling="gl"

CC:
    L = nlat - 1
    require nlon = 2L
    map to S2FFT sampling="mwss"
```

The MWSS colatitudes are the same pole-including equally spaced latitude nodes as
spharmgrid CC for these dimensions. CC/MWSS support is a v0.3.0 requirement, not a later
extension. The JAX API also accepts atmospheric regular GL `(L, 2L)` with bandlimit `L`
and coefficient shape `(L, 2L - 1)`. This is the only additional GL longitude count;
other spharmgrid GL/CC shapes are unsupported unless exact S2FFT sampling equivalence is
established.

Current torch-harmonics uses triangular truncation and its Clenshaw–Curtis/equiangular
bandwidth behavior differs from DUCC's full representable `lmax`/`mmax` behavior when
spharmgrid is called without an explicit `Tn` range.

Therefore:

- explicit `Tn` requests may map naturally when the backend can represent the same
  triangular band;
- `truncation=None` must not silently claim identical retained bandwidth across backends
  when the underlying engines differ;
- unsupported backend/grid/bandwidth combinations must fail clearly;
- do not silently clamp a user's explicit spectral request.

Do not infer equivalence from names such as `equiangular`, `lobatto`, or
`legendre-gauss`; compare actual nodes, quadrature, normalization, and band-limit
conventions.

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
latitude/associated-Legendre quadrature transform. torch-harmonics owns that algorithm,
its PyTorch autograd behavior, CUDA/cuFFT execution, coefficient normalization/layout,
precomputed quadrature data, and distributed variants.

spharmgrid should call those installed modules rather than copy their FFT, Legendre, or
SHT/ISHT implementation.

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
torch-harmonics for transforms and spharmgrid for the physical/spectral multipliers and
sign/radius conventions.

The adapter may cache or reuse transform modules where useful, but cache design should
follow measured workload needs rather than become a new public object hierarchy.

### 5.2 Initial target

Start with:

```text
GL
torch-harmonics equiangular/CC only after exact coordinate and convention parity
```

For each operation, compare against DUCC spharmgrid on the same physical grid and
spectral content. Add analytic tests that can detect paired sign errors in
forward/inverse vector transforms.

Do not add an operation merely because a scalar analogue exists. Vector/spin conventions
must be independently verified.

---

## 6. SFNO and learned spherical spectral convolution

SFNO is a concrete near-term consumer for climate-variability research, but it is
distinct from spharmgrid's deterministic spectral filter.

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
operator. This is not the same operation as spharmgrid's fixed `T42`, `T6-42`, or
Sardeshmukh–Hoskins filter response.

### 6.1 `SpectralConvS2` is not required for the scientific operation API

Do not use `SpectralConvS2` to implement:

```text
filter
regrid
laplacian
vorticity/divergence
streamfunction/velocity potential
```

Those operations require only the forward/inverse scalar or vector transforms plus
spharmgrid's deterministic spectral mathematics.

### 6.2 Do not copy NVIDIA's spectral convolution

For SFNO model work, prefer using the maintained torch-harmonics `SpectralConvS2`
implementation directly or through a very thin spharmgrid adapter if spharmgrid grid
translation removes meaningful boilerplate.

Do not create an independent spharmgrid copy of the learned spectral convolution, FFT,
Legendre transform, or SFNO architecture merely to place it under the spharmgrid
namespace.

A future convenience API such as:

```python
import spharmgrid.torch as sgt

conv = sgt.nn.SpectralConvS2(
    in_channels=64,
    out_channels=64,
    grid=grid,
)
```

is acceptable only if it remains a thin adapter to torch-harmonics and provides a real
advantage such as `sg.Grid` translation, capability validation, or consistent truncation
semantics. It should not fork NVIDIA's implementation.

### 6.3 Existing higher-level SFNO packages

Before adding model-level code to spharmgrid, account for existing packages:

- `neuraloperator` already provides a ready-made `SFNO` model and `SphericalConv` layer
  and uses torch-harmonics for spherical transforms;
- NVIDIA Makani provides the weather/climate SFNO training/model stack and uses
  torch-harmonics;
- NVIDIA Earth2Studio wraps complete forecast models and workflows, including
  SFNO/Makani-based models.

These packages cover neural-operator architectures and forecast workflows. spharmgrid
should not duplicate them.

For climate-variability research, the preferred division is initially:

```text
xarray/CF preprocessing and atmospheric diagnostics    spharmgrid
PyTorch spherical scientific operators                 spharmgrid.torch
SHT/ISHT numerical kernels                             torch-harmonics
SFNO architecture                                      neuraloperator or Makani,
                                                       or research-local model code
```

If research establishes a missing reusable layer between torch-harmonics and these model
packages, add the smallest justified spharmgrid adapter then.

### 6.4 Optional SFNO interoperability check

An optional interoperability example/test may show that data and grids prepared for
`spharmgrid.torch` can be used in an SFNO-style PyTorch workflow without NumPy/device
breaks.

This may use torch-harmonics `SpectralConvS2` directly or a current `neuraloperator`
SFNO/SphericalConv layer. It does not require spharmgrid to own a full SFNO model.

---

## 7. S2FFT/JAX implementation

Add `spharmgrid.jax` as a JAX-array API backed by S2FFT. It should expose the same 19
scientific functions listed in Section 1 as `spharmgrid.torch`.

The v0.3.0 grid requirement was:

```text
GL       S2FFT "gl" sampling with shape (L, 2L - 1)
CC       S2FFT "mwss" sampling with shape (L + 1, 2L)
```

The current JAX grid capability also includes atmospheric regular GL `(L, 2L)`. Its
spectral bandlimit and coefficient shape are unchanged from GL `(L, 2L - 1)`. This path
uses the S2FFT 1.4.0 internal `ftm` latitude steps behind one version-checked
compatibility module. Keep the length-`2L` longitude FFT, Nyquist projection, and
longitude-origin handling in spharmgrid.

Support both scalar and spin-1 transforms. Map geographic eastward/northward wind to and
from S2FFT spin-1 coefficients explicitly and verify the mapping with analytic vector
fields and DUCC comparisons.

Regridding should resize the S2FFT coefficient domain in JAX before synthesis on the
target grid. Keep coefficient masking, degree multipliers, phase handling, and resizing
inside JAX so `jit`, `vmap`, and automatic differentiation are preserved.

Use S2FFT's ordinary public JAX transforms with externally generated Price–McEwen
recursion precomputations. Materialize the O(L²) arrays before passing them to the S2FFT
transform and cache them in a bounded Python LRU keyed by the static transform settings
`(bandlimit, sampling, spin, direction)`, with `maxsize=32`. Generate them through the
public `s2fft.generate_precomputes_jax` function. Scalar transforms use `reality=True`;
spin-1 transforms use `reality=False`. The scalar `reality=True` path still returns the
full centered coefficient array required by spharmgrid.

Measured comparisons showed that the external precomputations materially reduced warmed
CUDA transform time for the tested `L=16`, `64`, and `128` cases; this justifies
retaining the cache. Generated precompute arrays were uncommitted
(`Array.committed == False`) for all five arrays in each precompute tuple on Python 3.11
with JAX/jaxlib 0.5.0 on CPU and with JAX/jaxlib 0.10.2 on the RTX 5070 CUDA
environment. Direct S2FFT generation and the spharmgrid helper had the same placement
behavior, so the cache key does not include a device identifier for the tested execution
model. Only one CUDA device was available; multi-GPU behavior was not tested.

When the first cache miss occurs during an outer `jax.jit`,
`jax.ensure_compile_time_eval()` is required. Removing it leaves the S2FFT
precomputations as `DynamicJaxprTracer` values, and `block_until_ready()` then fails.
This reproduced on JAX/jaxlib 0.5.0 CPU and JAX/jaxlib 0.10.2 CUDA. With the context
restored, the outer-JIT check under `jax.checking_leaks()` passed with concrete cached
arrays.

HEALPix remains a Phase-4 grid feature.

---

## 8. Cross-backend convention audit

For each engine document and test:

- spherical-harmonic normalization;
- Condon–Shortley phase convention;
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

A forward/inverse round trip is necessary but not sufficient: matching errors can
cancel. Where applicable compare against:

```text
analytic fields
DUCC spharmgrid
SPHEREPACK/pyspharm for GL/CC atmospheric operations
```

---

## 9. Public accelerator API and I/O boundary

Use separate interfaces for scientific file/metadata workflows and accelerator-native
differentiable computation.

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

The default xarray path remains DUCC0 unless a later measured convenience mode justifies
explicit accelerator execution.

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

Do not wrap Torch/JAX tensors in xarray merely to call the transform kernel. The
differentiable path must remain inside the native framework.

Use the spharmgrid grid description rather than exposing backend-specific grid strings
in the scientific API:

```python
grid = sg.gaussian_grid(128, 255)
sgt.filter(x, grid=grid, truncation="T42")
sgj.filter(x, grid=grid, truncation="T42")
```

Each adapter translates that grid into its backend's sampling terminology and validates
capability/bandwidth.

### 9.3 Tensor dimension convention

For rectangular tensor-native operations, use the last two dimensions as the horizontal
dimensions and treat all preceding dimensions as independent leading/batch dimensions:

```text
(nlat, nlon)
(time, nlat, nlon)
(batch, channel, nlat, nlon)
(member, level, time, nlat, nlon)
```

Do not reproduce xarray's named-dimension machinery in the tensor API unless a real use
case requires it.

Phase 4 may use a one-dimensional trailing pixel/cell dimension for packed HEALPix or
reduced-Gaussian arrays where appropriate.

### 9.4 Explicit xarray \<-> accelerator boundary

For accelerator workflows, conversion between scientific containers and device arrays
should be explicit at the data-loader or application boundary, not on every SHT call.

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

For PyTorch this may use `torch.as_tensor(...)`/device transfer. For JAX this may use
`jax.device_put(...)`. Convert back to xarray only when a labeled scientific result or
file output is required.

Do not add file-reading/writing methods to `spharmgrid.torch` or `spharmgrid.jax`.

The post-v0.3.0 JAX convenience layer provides explicit `device_put()` and
`device_get()` helpers for Xarray `DataArray` and `Dataset` payloads. They transfer
numerical data only; ordinary coordinates and metadata remain on the host. They accept
an optional `device=` destination but do not expose JAX's lower-level source, donation,
or aliasing controls. SHT operations do not move data between host and device
implicitly.

### 9.5 JAX/xarray interoperability remains an optional higher-level path

Google DeepMind's `xarray_jax` demonstrates that xarray objects containing JAX arrays
can be registered as JAX PyTrees and used with `jit`, `grad`, `vmap`, and sharding while
retaining labels/coordinates.

This is useful precedent, but spharmgrid does not require `xarray_jax` merely to expose
S2FFT or use `.sgj`. The raw JAX kernel API remains `jax.Array` native and primary.
`.sgj` controls the ordinary Python/Xarray wrapper outside the numerical kernel and does
not promise whole-container PyTree transformations.

Keep `xarray_jax` as optional development/interoperability infrastructure for
complete-container `jit`, `grad`, `vmap`, and sharding characterization. Do not assume
ordinary Xarray operations or the existing DUCC `apply_ufunc` path are automatically
safe for those transformations.

There is no need to force the PyTorch and JAX convenience layers to be identical if
their host framework ecosystems differ.

### 9.6 Implemented Xarray accelerator convenience boundary

The implemented labeled JAX path is explicit:

```python
ds_jax = spharmgrid.jax.device_put(dataset)
result = ds_jax.sgj.kinematics()
host_result = spharmgrid.jax.device_get(result)
```

This is an Xarray convenience path rather than a differentiable model API. Never
auto-select an accelerator because hardware is present, and do not add a `backend=`
selector to the root `.sg` accessor. There is no `.sgt` accessor in this work.

---

## 10. Differentiability

For tensor-native paths:

- keep Torch autograd and JAX transformations intact;
- do not convert tensors to NumPy inside differentiable operations;
- test JAX execution under `jit` and `vmap`;
- add automatic-gradient tests for representative scalar and vector operations;
- compare with finite differences on small problems where numerically useful.

Test at least:

```text
filter
laplacian
kinematics
wind reconstruction
```

Do not infer spharmgrid differentiability solely from the transform library.

---

## 11. Precision

The supported `spharmgrid.jax` execution mode is float64/complex128 with JAX x64 enabled
by the application or test process. The adapter must not change the process-wide
`jax_enable_x64` setting and must reject x64-disabled or single-precision input before
entering the transform path.

Float32 characterization of the ordinary S2FFT GL and MWSS paths found that ordinary
MWSS scalar float32 has field-scale errors of order `10^-1`; GL and spin-1 cases are
more accurate, but float32 is unsupported so the JAX API has one conservative precision
contract. O(L²) precomputations do not repair the float32 Price–McEwen recurrence, and
O(L³) kernels are not a scalable remedy. Use analytic and cross-backend error
measurements to set float64 tolerances; do not require bitwise equality.

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

Do not report a GPU speedup for an xarray workflow while excluding transfer or compile
costs.

For the SFNO interoperability case, measure only enough to ensure the spharmgrid adapter
does not add material overhead beyond the underlying torch-harmonics layer. Do not make
model-training performance a spharmgrid benchmark suite.

---

## 13. Optional dependencies and CI

Keep accelerator and model stacks optional.

`spharmgrid.torch` uses separately installed PyTorch and `torch-harmonics`. The JAX API
should use both a user-facing extra and a development group:

```toml
[project.optional-dependencies]
jax = ["jax>=0.5.0", "s2fft>=1.4.0"]

[dependency-groups]
torch-dev = ["torch>=...", "torch-harmonics>=..."]
jax-dev = [
  { include-group = "test" },
  "jax>=0.5.0",
  "s2fft>=1.4.0",
]
cuda-dev = [
  { include-group = "jax-dev" },
  { include-group = "torch-dev" },
  "jax[cuda13]; sys_platform == 'linux'",
]
```

spharmgrid imports JAX directly, so JAX should be a direct optional dependency rather
than only a transitive S2FFT dependency. Determine the JAX lower bound from tests with
S2FFT 1.4.0 and the oldest supported Python version. The verified floor is Python
`>=3.11`, JAX/JAXLIB `>=0.5.0`, and S2FFT `>=1.4.0`.

For portable local JAX development and testing, use:

```bash
uv sync --group jax-dev
```

For local Linux NVIDIA development of both JAX and Torch, use the development only CUDA
group:

```bash
uv sync --group cuda-dev
```

The CUDA group is not part of the published `spharmgrid[jax]` extra.

Do not add a `jax-xarray` extra unless that integration is implemented and tested.
`neuraloperator` and Makani remain external consumer/reference packages rather than
spharmgrid dependencies.

Do not make Torch, JAX, CUDA, torch-harmonics, S2FFT, xarray-jax, neuraloperator, or
Makani part of the base install. Normal CPU CI must remain independent of accelerator
availability.

Use small CPU adapter/import tests where possible. Add GPU CI only when a reliable
runner is available.

---

## 14. Relevant ecosystem precedent

Use current implementations as architectural evidence, not APIs to copy blindly.

### 14.1 NVIDIA torch-harmonics

`torch-harmonics` was created to enable SFNO and is the low-level differentiable PyTorch
SHT library. Its SHTs are `torch.nn.Module` objects operating on batched native tensors.
This is the engine spharmgrid should wrap for PyTorch, not reimplement.

### 14.2 neuraloperator

`neuraloperator` provides a higher-level generic neural-operator API including `SFNO`
and `SphericalConv`, with torch-harmonics supplying the spherical transforms. It is a
good first choice for research that needs a configurable SFNO architecture rather than
only raw SHT operations.

It is not an atmospheric xarray/CF diagnostics package, so it does not replace
spharmgrid's intended role.

### 14.3 NVIDIA Makani and Earth2Studio

Makani is a weather/climate model training and inference stack with SFNO models and a
direct torch-harmonics dependency. Earth2Studio sits higher still and wraps complete
forecast models, data sources, coordinates, and workflows.

They demonstrate that weather-model infrastructure and generic field-level SHT
operations are separate layers.

### 14.4 windspharm

`windspharm` is close to spharmgrid's atmospheric-science ergonomics: it provides
user-friendly vorticity/divergence/streamfunction and related wind analysis with
xarray/metadata interfaces. However, it is based on pyspharm/SPHEREPACK rather than
torch-harmonics and is not a differentiable GPU PyTorch layer.

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
spherical-harmonic implementation. The core SHT operates on `jax.Array`, with explicit
nodal/modal representations and accelerator-oriented JAX execution. NeuralGCM provides
xarray conversion utilities at the scientific API boundary.

DeepMind also provides `xarray_jax` for labeled JAX structures where direct
JAX-transformed xarray workflows are valuable.

GraphCast and GenCast do not solve their spherical model computation with an SHT; they
use spherical graph/icosahedral-mesh architectures. This is a different way to avoid
latitude/longitude convolution problems, not an alternate SHT implementation.

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
GL                DUCC0 + torch-harmonics + S2FFT on validated shapes
CC                DUCC0 + torch-harmonics + S2FFT on MWSS-compatible shapes
HEALPix           DUCC0 + S2FFT
reduced Gaussian  DUCC0 initially
```

This table is a target capability map, not a promise that every operation has identical
analysis semantics or bandwidth on every engine.

---

## 16. Acceptance criteria

Phase 3 is complete when:

- existing DUCC and PyTorch results and public behavior are unchanged;
- shared backend code is extracted only where the implementations require it;
- torch-harmonics remains the PyTorch numerical SHT implementation;
- S2FFT 1.4.0 is the JAX numerical SHT implementation for the pinned GL `(L, 2L)`
  compatibility path;
- `spharmgrid.jax` supports GL `(L, 2L - 1)`, atmospheric regular GL `(L, 2L)`, and
  CC/MWSS `(L + 1, 2L)`;
- `spharmgrid.jax` exports the same 19 scientific functions as `spharmgrid.torch`;
- `spharmgrid.torch` and `spharmgrid.jax` provide tensor-native differentiable APIs for
  their supported grids;
- accelerator APIs use spharmgrid grid objects rather than backend sampling strings as
  their scientific grid contract;
- rectangular tensor APIs use trailing horizontal dimensions with arbitrary
  leading/batch dimensions;
- file/CF I/O remains an xarray responsibility rather than entering the tensor
  namespaces;
- the supported Phase-2 operation graph is shared semantically rather than duplicated as
  independent atmospheric implementations;
- geographic-vector/spin conventions are independently proven for each backend;
- JAX paths are tested under `jit`, `vmap`, and automatic differentiation without NumPy
  breaks;
- float32 accuracy is characterized and recorded but unsupported;
- float64/complex128 with JAX x64 enabled is the supported JAX execution mode;
- x64-disabled execution is tested for clean deterministic rejection;
- S2FFT O(L²) recursion precomputations are ordinary transform setup and are cached
  privately by static transform settings;
- O(L³) full transform kernels are not a production requirement;
- unsupported grid/bandwidth/backend combinations fail clearly;
- explicit spectral requests are never silently clamped;
- accelerator precision and performance are measured with recursion setup,
  trace/lowering/compile, transfer, and steady-state execution costs separated;
- `SpectralConvS2` is not used as a substitute for deterministic atmospheric
  filters/operators;
- model-level packages such as neuraloperator or Makani remain optional
  consumers/references, not core dependencies;
- the base installation remains free of Torch/JAX dependencies;
- no package-wide raw coefficient compatibility format is introduced;
- HEALPix and reduced Gaussian public grid expansion remain Phase-4 work.

---

## 17. v0.2 Torch slice

v0.2.0 added the optional PyTorch namespace using the installed `torch-harmonics`
`RealSHT`, `InverseRealSHT`, `RealVectorSHT`, and `InverseRealVectorSHT` modules. DUCC0
remains the reference engine for the Xarray API.

The implemented tensor API covers the scalar, vector, kinematic, potential, Helmholtz,
and inverse-wind operations listed in the preconditions, plus
`spharmgrid.torch.nn.SHTOperators`, `SHTFilter`, `SHTRegrid`, and `SHTVectorRegrid`.
Tensor inputs use the last two dimensions for latitude and longitude, preserve leading
dimensions, and keep coefficient masks and physical multipliers in Torch for autograd.

The adapter's independently checked vector mapping is
`(v_theta, v_phi) = (-v_northward, u_eastward)`. In the tested normalization, the Torch
spheroidal and toroidal coefficients map to spharmgrid's E and B coefficients as
`E = sqrt(l(l+1)) s` and `B = -sqrt(l(l+1)) t` for positive degree. Longitude-origin
phase handling and both latitude orders are tested against DUCC and analytic degree-one
fields.

The current torch-harmonics equiangular transform has a narrower CC latitude bandwidth
than DUCC. The adapter therefore accepts explicit triangular `Tn` ranges only through
`min((nlat - 1) // 2, (nlon - 1) // 2)` on CC (and the corresponding GL limit
`min(nlat - 1, (nlon - 1) // 2)`). It raises a clear error for a non-triangular or
over-wide request instead of clamping it.

---

## 18. v0.3 JAX/S2FFT slice

v0.3.0 adds `spharmgrid.jax` backed by S2FFT. The JAX API includes all 19 current
scientific functions on both exact S2FFT rectangular samplings:

```text
GL       (L, 2L - 1)
CC/MWSS  (L + 1, 2L)
```

CC/MWSS is supported alongside GL. Scalar and spin-1 conventions are tested
independently, and the complete JAX operation set is compared with DUCC on identical
fields. Float32 accuracy is characterized and recorded but unsupported; the supported
execution mode is float64/complex128 with JAX x64 enabled, and x64-disabled execution is
tested for clean rejection.

The JAX namespace uses trailing latitude/longitude dimensions with arbitrary leading
dimensions. Regridding resizes the spectral coefficient domain in JAX. The base package
does not import JAX or S2FFT.

### Atmospheric regular GL extension

The JAX API accepts atmospheric regular Gaussian grids with shape `(L, 2L)` alongside
the S2FFT-native GL shape `(L, 2L - 1)`. Both use bandlimit `L` and coefficient shape
`(L, 2L - 1)`. The adapter removes the length-`2L` longitude Nyquist mode before S2FFT's
latitudinal transform and inserts a zero mode before synthesis. S2FFT 1.4.0 is pinned
while this path uses its internal `ftm` functions; remove the compatibility module when
S2FFT exposes a public transform for `(L, 2L)`.

The v0.3.0 release itself does not add HEALPix, Flax/Equinox wrappers, or an xarray
`backend=` selector. Those additions require their own demonstrated use case or
grid-validation work.

### Post-v0.3.0 Xarray convenience layer

v0.3.0 established the raw `jax.Array`/S2FFT API. The optional `.sgj` accessors are a
thin Xarray wrapper over that existing implementation: they resolve grids and CF-aware
variables in Python, pass JAX payloads to the 19 raw JAX functions, and restore Xarray
dimensions, coordinates, names, and metadata. They do not add another SHT implementation
or numerical backend.

The public `device_put()` and `device_get()` helpers make host/device conversion
explicit. `.sgj` does not depend on `xarray_jax`; that package remains optional
development and interoperability infrastructure for treating complete Xarray containers
as JAX PyTrees across transformations. It is not part of the published `spharmgrid[jax]`
extra.

Raw `spharmgrid.jax` functions remain the primary transformation-safe API for `jit`,
`grad`, and `vmap`. The labeled convenience layer does not promise whole-container JAX
PyTree transformations and does not move numerical payloads between host and device
implicitly.

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

When implementing this phase, verify current library APIs and grid/bandwidth behavior
from the installed versions rather than relying on this planning snapshot.
