# Phase 2: complete the SHT operator and vector-transform suite

## Goal

Extend spharmgrid from its initial atmospheric transform set into a coherent,
high-value spherical-harmonic operator suite comparable in capability to the
useful NCL/SPHEREPACK and pyspharm workflows.

This phase remains **DUCC0-only** for production numerics.

Use `pyspharm-syl` as an independent SPHEREPACK-based parity implementation.
NCL/SPHEREPACK function names are semantic/reference mappings, not names to
copy into the Python API.

Do not add reduced Gaussian, HEALPix, GPU backends, raw public `alm` objects,
or non-spectral interpolation in this phase.

## Existing baseline

The current public API already includes:

```python
sg.filter
sg.regrid

sg.gradient
sg.laplacian
sg.inverse_laplacian

sg.vorticity
sg.divergence
sg.kinematics

sg.streamfunction
sg.velocity_potential
sg.potentials

sg.rotational_wind
sg.divergent_wind
sg.wind
```

These cover most of the commonly used NCL/SPHEREPACK wind conversion graph:

```text
u, v -> vorticity
u, v -> divergence
u, v -> vorticity + divergence
u, v -> streamfunction + velocity potential

vorticity -> rotational wind
divergence -> divergent wind
vorticity + divergence -> full wind

streamfunction -> rotational wind
velocity potential -> divergent wind
streamfunction + velocity potential -> full wind
```

The next phase should fill the remaining high-value gaps rather than adding
many aliases.

---

## 1. Add vector spectral regridding

This is the highest-value missing operation.

NCL/SPHEREPACK provides the vector regridding families:

```text
f2fshv
f2gshv
g2fshv
g2gshv
```

spharmgrid should expose one grid-independent API:

```python
sg.regrid_vector(
    u,
    v,
    target_grid,
    spectral=None,
    *,
    lmin=None,
    lmax=None,
    taper=None,
    eastward="u",
    northward="v",
    nthreads=None,
) -> xr.Dataset
```

Accessor form:

```python
ds.sg.regrid_vector(
    target_grid,
    u="u",
    v="v",
    spectral="T6-42",
    taper=0.1,
)
```

The numerical path must be:

```text
source u, v
    -> one spin-1/vector analysis
    -> optional degree mask/taper
    -> one vector synthesis on the target grid
    -> target u, v
```

Do not implement vector regridding by independently applying scalar
`regrid()` to `u` and `v`.

Do not implement it through an intermediate spatial vorticity/divergence pair
unless testing demonstrates that this is mathematically and numerically
equivalent and no extra analysis/synthesis cycles are introduced.

Required initial combinations remain:

```text
GL -> GL
GL -> CC
CC -> GL
CC -> CC
```

Use the same spectral-range and taper semantics as scalar `regrid()`.

---

## 2. Add a combined Helmholtz decomposition

The current API can recover rotational and divergent winds separately. Add a
single analysis-oriented operation that computes both from one input wind
field:

```python
sg.helmholtz(
    u,
    v,
    *,
    divergent_eastward="u_divergent",
    divergent_northward="v_divergent",
    rotational_eastward="u_rotational",
    rotational_northward="v_rotational",
    radius=EARTH_RADIUS_M,
    nthreads=None,
) -> xr.Dataset
```

Accessor:

```python
ds.sg.helmholtz()
```

Return four components:

```text
u_divergent
v_divergent
u_rotational
v_rotational
```

Use one vector analysis and the minimum required syntheses.

This maps conceptually to windspharm's `helmholtz`,
`irrotationalcomponent`, and `nondivergentcomponent`, while retaining
spharmgrid's existing terminology:

```text
divergent wind
rotational wind
```

Do not add non-SHT conveniences such as wind magnitude merely to reproduce the
complete windspharm method list.

---

## 3. Add inverse gradient

NCL/SPHEREPACK provides:

```text
igradsf
igradsg
```

Add:

```python
sg.inverse_gradient(
    eastward,
    northward,
    *,
    output=None,
    radius=EARTH_RADIUS_M,
    nthreads=None,
) -> xr.DataArray
```

This reconstructs the scalar potential whose horizontal spherical gradient is
the supplied tangent vector field.

The scalar additive constant is not recoverable. Define and document the same
kind of canonical zero-mode convention already used by
`inverse_laplacian()`:

```text
degree-zero coefficient = 0
```

Before implementation, verify the exact SPHEREPACK/NCL sign, component and
normalization conventions. Do not derive them from memory.

Reject or clearly define behavior for a supplied vector field with a
non-gradient/rotational component. The intended first behavior should follow
the corresponding SPHEREPACK operation rather than introducing an
undocumented projection rule.

---

## 4. Add vector Laplacian and inverse vector Laplacian

NCL/SPHEREPACK provides:

```text
lapvf
lapvg
ilapvf
ilapvg
```

Add:

```python
sg.vector_laplacian(
    u,
    v,
    *,
    eastward="u",
    northward="v",
    radius=EARTH_RADIUS_M,
    nthreads=None,
) -> xr.Dataset

sg.inverse_vector_laplacian(
    u,
    v,
    *,
    eastward="u",
    northward="v",
    radius=EARTH_RADIUS_M,
    nthreads=None,
) -> xr.Dataset
```

These operations must follow the established SPHEREPACK vector-Laplacian
definition.

Do **not** assume the scalar multiplier

```text
-l(l+1)/R^2
```

can simply be applied independently to two physical vector components. Trace
the vector-harmonic definition and verify it against SPHEREPACK/pyspharm or an
independent analytic derivation.

Document singular/null modes explicitly for the inverse.

---

## 5. Keep raw coefficient analysis/synthesis private for now

NCL exposes scalar and vector harmonic analysis/synthesis directly:

```text
sha*
shs*
vha*
vhs*
```

pyspharm similarly exposes:

```text
grdtospec
spectogrd
getvrtdivspec
getuv
```

These are useful reference interfaces, but spharmgrid should **not** expose a
raw public coefficient API in this phase.

Reasons:

- DUCC, SPHEREPACK, torch-harmonics and S2FFT use different coefficient
  layouts/conventions.
- A public coefficient representation becomes a compatibility contract.
- The current physical-field API already supports the high-value atmospheric
  workflows without requiring users to understand packed `alm` storage.
- A future accelerator layer should not be constrained by a prematurely
  public DUCC-specific coefficient layout.

Continue using internal analysis/synthesis helpers as needed.

Reconsider a public spectral-coefficient object only when a concrete use case
requires users to inspect or modify coefficients directly.

---

## 6. Do not add non-SHT operations

Do not add these in this phase:

```text
bilinear interpolation
bicubic interpolation
splines
nearest-neighbor interpolation
conservative remapping
wind speed
planetary vorticity
absolute vorticity
```

Some are useful atmospheric diagnostics, but they do not strengthen the
package's spherical-harmonic scope.

The package name and scientific boundary should remain literal:
`spharmgrid` performs spherical-harmonic operations on global spherical grids.

---

## 7. Independent pyspharm/SPHEREPACK verification

### Dependency

Keep `pyspharm-syl` in the existing parity dependency group only.

It must not become a runtime dependency.

The current compatibility restriction to Python <3.14 is acceptable for parity
jobs.

### Reference mappings

Use `pyspharm.Spharmt` as the independent reference where possible:

```text
spharmgrid                     pyspharm / SPHEREPACK
--------------------------------------------------------------
scalar analysis internals      Spharmt.grdtospec
scalar synthesis internals     Spharmt.spectogrd

gradient                       Spharmt.getgrad

u,v -> vo,d                    Spharmt.getvrtdivspec + spectogrd
vo,d -> u,v                    Spharmt.getuv

u,v -> strf,vp                 Spharmt.getpsichi

scalar regrid                  pyspharm.regrid
spectral filtering             grdtospec -> coefficient mask -> spectogrd
```

For vector regridding, construct the independent reference using two
independent `Spharmt` grid objects and SPHEREPACK vector analysis/synthesis:

```text
source u,v
    -> pyspharm/SPHEREPACK vector analysis
    -> preserve/truncate the reference spectral representation
    -> target SPHEREPACK vector synthesis
```

Do not call spharmgrid helpers to construct the reference result.

For inverse gradient and vector Laplacian operations not exposed directly by
the high-level pyspharm class, use one of:

1. the underlying independent SPHEREPACK operation available through
   pyspharm;
2. a small static NCL/SPHEREPACK reference generated outside the production
   spharmgrid code;
3. an analytic harmonic field with an independently derived exact answer.

Prefer 1 or 3 over storing large reference arrays.

### Grids

Run parity on both:

```text
pyspharm "regular"  <-> spharmgrid CC
pyspharm "gaussian" <-> spharmgrid GL
```

First verify that the sampling coordinates and latitude ordering correspond
exactly before interpreting numerical differences.

### Test fields

Use low-degree analytic scalar and vector harmonics where possible.

Also use deterministic smooth constructed atmospheric fields with multiple
degrees/orders to catch:

- normalization errors;
- E/B or rotational/divergent swaps;
- sign errors;
- latitude orientation errors;
- longitude phase errors;
- radius scaling errors;
- zero-mode errors;
- truncation errors.

### Round-trip identities

At minimum test:

```text
gradient(inverse_gradient(gradient(f))) ~= gradient(f)

inverse_laplacian(laplacian(f)) ~= f - mean_mode(f)

kinematics(wind(vo, d)) ~= (vo, d)       for representable nonzero modes

potentials(wind(strf, vp)) ~= (strf, vp) modulo degree zero

rotational + divergent wind ~= original representable wind

vector inverse Laplacian(vector Laplacian(wind))
    ~= wind modulo documented null modes

regrid_vector(source -> target -> source)
    ~= band-limited source within transform accuracy
```

Do not use a round trip as the only evidence; round trips can preserve a
matching sign/convention error on both sides.

---

## 8. NCL/SPHEREPACK semantic map

Document the correspondence without reproducing NCL's fixed/gaussian suffix
explosion.

Target documentation table:

| spharmgrid | NCL/SPHEREPACK family |
| --- | --- |
| `gradient` | `gradsf`, `gradsg` |
| `inverse_gradient` | `igradsf`, `igradsg` |
| `laplacian` | `lapsf`, `lapsg` |
| `inverse_laplacian` | `ilapsf`, `ilapsg` |
| `vector_laplacian` | `lapvf`, `lapvg` |
| `inverse_vector_laplacian` | `ilapvf`, `ilapvg` |
| `vorticity` | `uv2vr*` |
| `divergence` | `uv2dv*` |
| `kinematics` | `uv2vrdv*` |
| `potentials` | `uv2sfvp*` |
| `rotational_wind` | `vr2uv*` or streamfunction synthesis |
| `divergent_wind` | `dv2uv*` or velocity-potential synthesis |
| `wind` | `vrdv2uv*`, `sfvp2uv*` |
| `regrid` | `f2fsh`, `f2gsh`, `g2fsh`, `g2gsh` |
| `regrid_vector` | `f2fshv`, `f2gshv`, `g2fshv`, `g2gshv` |

Do not present NCL as the scientific origin of the mathematics. It is an
established atmospheric API/behavior and SPHEREPACK integration reference.

---

## 9. Implementation requirements

Keep accessor methods thin.

New direct functions and `.sg` methods must use identical numerical paths.

Share vector analysis whenever multiple requested outputs derive from the same
input wind.

Do not introduce a general backend framework yet.

Do not expose DUCC-specific geometry or coefficient objects publicly.

Keep Dask behavior consistent with existing gufunc/apply_ufunc execution.

Preserve arbitrary leading dimensions:

```text
(time, level, member, ..., lat, lon)
```

and coordinate/longitude conventions.

All new operations must support both ascending and descending latitude input
where the current package supports them.

---

## 10. Documentation

Update:

```text
README.md
docs/api.md
docs/kinematics.md
docs/operators.md
docs/regridding.md
docs/references.md
```

Add a compact NCL/SPHEREPACK correspondence table.

Emphasize physical/descriptive Python names rather than NCL function names.

Document the zero/null-mode convention for every inverse operator.

---

## 11. Acceptance criteria

Phase 2 is complete when:

- vector regridding is implemented through vector SHT analysis/synthesis;
- combined Helmholtz decomposition is implemented efficiently;
- inverse gradient is implemented and its zero mode documented;
- vector Laplacian and inverse vector Laplacian match the chosen
  SPHEREPACK definition;
- all current operations still pass existing tests;
- new operations have analytic tests;
- GL and CC parity is demonstrated against pyspharm/SPHEREPACK where
  independently available;
- direct/accessor results are equivalent;
- Dask tests cover the new operations where supported;
- no pyspharm runtime dependency is introduced;
- no non-SHT interpolation is added;
- no public raw coefficient format is introduced.

---

## Primary references for implementation research

- NCL built-in spherical-harmonic functions:
  https://www.ncl.ucar.edu/Document/Functions/Built-in/
- NCL vector analysis (`vhaeC`):
  https://www.ncl.ucar.edu/Document/Functions/Built-in/vhaeC-1.shtml
- pyspharm `Spharmt` source/API:
  https://github.com/jswhit/pyspharm/blob/master/Lib/spharm.py
- windspharm API:
  https://ajdawson.github.io/windspharm/api/windspharm.standard.html
