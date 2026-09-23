# JAX Gauss–Legendre (L, 2L) evaluation

This report compares two JAX transform adapters for atmospheric Gauss–Legendre grids.
The production path uses Candidate B for GL `(L, 2L)`; Candidate A is retained in the
benchmark directory for comparison. Normal spharmgrid operations do not expose a method
selector. S2FFT itself was not modified.

## Environment

| Item                                   | Value                                                      |
| -------------------------------------- | ---------------------------------------------------------- |
| Python                                 | 3.14.7                                                     |
| spharmgrid base                        | c5f488257a8e0992c6e300f9d7dded0d36228130 (main)            |
| validation code                        | 168a0f54c10901d366533daa73347349e31c10ae                   |
| performance harness and transform code | d9846ff3cd91c12487aa8455b760fb4db4e041b1                   |
| S2FFT                                  | 1.4.0, tag commit e140536880fc53081a98156fb0245d9d197d1e5d |
| JAX / jaxlib                           | 0.11.2 / 0.11.2, x64 enabled                               |
| DUCC0                                  | 0.41.0                                                     |
| Platform                               | Linux 7.0.14, x86_64, glibc 2.43                           |
| Device                                 | CPU backend, AMD Ryzen 9 5950X, device 0                   |

The machine reports NVIDIA hardware, but its installed jaxlib has no CUDA support, so no
accelerator timings were taken. The validation and performance JSON files record S2FFT
source metadata and device details.

## Implementations

**A — public Fourier resampling** is implemented in
`benchmarks/jax_gl_2l/_candidate_a.py`. It uses a length-2L longitude FFT, removes the
centered Nyquist slot, evaluates the remaining Fourier series on S2FFT's 2L-1 GL
longitude nodes, then calls public `s2fft.forward_jax`. Synthesis calls public
`s2fft.inverse_jax` and evaluates its Fourier series on the 2L grid after inserting a
zero Nyquist slot. All resampling stays in JAX.

**B — internal latitude step** is isolated in `src/spharmgrid/jax/_s2fft_gl_compat.py`.
S2FFT 1.4.0's public GL transform uses `(L, 2L - 1)` sampling. Candidate B performs one
length-2L longitude FFT/IFFT and enters S2FFT at its internal forward and inverse
latitudinal steps.

For even-length fftshift ordering, index zero is m=-L, the Nyquist mode equivalent to
m=+L on these samples. Candidate A removes that first centered bin. Candidate B removes
the same bin for complex input; for real input it removes the positive rfft Nyquist
endpoint. Both insert zero at the first centered slot for synthesis.

## Correctness

The raw validation file contains **3,973 metric rows**. It covers L=4, 8, 16, 32, 64,
128 with float64/complex128. DUCC spharmgrid transforms provide the primary independent
reference on the same physical (L, 2L) grids. Fixtures include analytic scalar and
vector modes, m=L-1, signed isolated spin modes, and seeded low-degree, high-degree,
random-phase, random-amplitude, and large-dynamic-range spectra. The spin fixtures are
also checked against public S2FFT synthesis on native (L, 2L-1) GL followed by
independent Fourier evaluation on 2L longitudes.

The matrix checks both latitude orders, zero/fractionally shifted/wrapped longitude
conventions, and leading shapes (L,2L), (1,L,2L), (batch,L,2L), and (time,level,L,2L).
It compares scalar, spin, vector, filter, regrid, gradient, Laplacian, kinematics,
potential, and wind operations. All 19 requested operation families pass for both
candidates. Experiment accessor checks filter a labeled DataArray with time and level
dimensions and CFTime no-leap coordinates through both adapters, preserving its
coordinates, dimension order, name, and attributes. Permanent tests exercise the
selected production path.

Metrics record maximum absolute error and RMS absolute error over every element,
relative RMS against the reference RMS, and maximum relative error where the reference
magnitude exceeds 1e-8 times the largest reference magnitude. The threshold is included
in every row; zero-reference cases have no relative maximum.

| Comparison                                                                                                    |  Candidate A |  Candidate B |
| ------------------------------------------------------------------------------------------------------------- | -----------: | -----------: |
| Maximum relative RMS over the matrix                                                                          |    1.281e-12 |    1.281e-12 |
| Maximum meaningful relative error                                                                             |     8.326e-6 |     8.325e-6 |
| Maximum relative RMS on ordinary analytic/random spectral fixtures, excluding the large-dynamic-range fixture | \<=1.281e-12 | \<=1.281e-12 |
| Maximum scalar/vector absolute error on ordinary analytic/random spectra                                      |     3.96e-10 |     3.96e-10 |
| Maximum large-dynamic-range field absolute error                                                              |     3.762e-3 |     3.762e-3 |
| Large-dynamic-range inverse field relative RMS at L=128                                                       |    6.540e-13 |    6.540e-13 |

The largest overall absolute comparison is inverse Laplacian at L=8: 0.252 for A and
0.250 for B, against a reference RMS of 5.03e12 (relative RMS about 2.02e-14). The
largest meaningful relative error is a small coefficient in the deliberately
large-dynamic-range spin fixture; its absolute maximum is about 2.0e-5 while the whole
coefficient array has RMS 1.13e7. Both absolute and relative measures are retained in
the raw results.

### Nyquist and spin/vector parity

For scalar fields and complex spin +1/-1 fields, a pure longitude Nyquist input produces
exactly zero coefficients for both candidates. Across all valid scalar and spin
synthesis cases, the synthesized length-2L Nyquist slot is at most 1.79e-15. No physical
longitude column is dropped.

Both candidates match DUCC scalar and vector analysis/synthesis and preserve spin signs,
geographic vector components, longitude phase, and existing spectral operations. For
isolated signed spin modes, forward coefficient error against native public S2FFT has
maximum absolute error 1.67e-13 and maximum relative RMS 9.09e-13. Their inverse
physical fields differ by at most 1.90e-15.

## JAX transformations

Both candidates pass eager execution, jit, vmap(jit(...)), jit(vmap(...)), and
reverse-mode grad through scalar analysis, synthesis, filter composition, spin, and
vector operations. The two batching forms give identical scalar transform results in the
small matrix.

Candidate B also passes forward-mode JVP for forward and inverse transforms. Candidate
A's forward and inverse JVPs are rejected because S2FFT 1.4.0 wraps its public latitude
step in custom_vjp; reverse-mode gradients pass. Candidate A therefore supports the
tested reverse-mode grad path, but not jvp/jacfwd through these public calls.

Gradient differences A versus B are at roundoff: maximum absolute differences are
1.25e-16 for scalar forward, 7.11e-15 for scalar inverse, 1.11e-15 for filter
composition, 1.76e-16 for spin, and 6.06e-28 for vector kinematics.

## Performance

The performance artifact has **148 successful timing rows** and no failed cases. It
benchmarks scalar forward/inverse, spin +1 forward/inverse, vector analysis/synthesis,
filter, and kinematics on identical CPU inputs for both candidates. Every operation is
measured at batch 1. Scalar forward/inverse also use batch sizes 1, 4, and 16 through
L=64; sizes 1 and 4 at L=128; and size 1 at L=256 and 512. Every timed call is
synchronized with .block_until_ready(). The harness records trace, lowering,
compilation, first execution, steady-state median, minimum, p25, p75, maximum, and
repetition count separately.

Ratios below are median time A/B; values above one favor B.

| Scope                                    |   A/B | Interpretation                                   |
| ---------------------------------------- | ----: | ------------------------------------------------ |
| All 56 batch-1 operation/size cases      | 0.998 | Effectively tied; A is 0.2% faster geometrically |
| Scalar forward/inverse, batch 4          | 1.138 | B is 12.1% faster geometrically                  |
| Scalar forward/inverse, batch 16         | 1.226 | B is 18.4% faster geometrically                  |
| All 74 measured cases, including batches | 1.039 | B is 3.7% faster geometrically                   |

Representative steady-state medians:

| L / operation / batch   | A (ms) | B (ms) | Faster           |
| ----------------------- | -----: | -----: | ---------------- |
| 256 scalar forward / 1  |  135.2 |  182.5 | A, 25.9%         |
| 512 scalar forward / 1  |  1,354 |  1,593 | A, 15.0%         |
| 512 spin +1 forward / 1 |  2,641 |  3,419 | A, 22.8%         |
| 512 vector analysis / 1 |  6,115 |  6,087 | Essentially tied |
| 512 kinematics / 1      |  8,267 |  6,908 | B, 16.5%         |
| 64 scalar forward / 16  |  8.214 |  7.194 | B, 12.4%         |
| 64 scalar inverse / 16  |  7.839 |  5.948 | B, 24.1%         |

At L=64 and batch 16, p25–p75 is 7.96–8.38 ms (A) versus 6.07–7.45 ms (B) for scalar
forward, and 7.59–8.19 ms (A) versus 5.07–6.39 ms (B) for scalar inverse. Timing spread
is retained for every case in the JSON file.

Compilation is similar. For batch-1 scalar forward at L=512, A/B trace times are 470/356
ms, lowering 90/86 ms, compile 2,492/2,482 ms, first execution 1,444/1,626 ms, and
steady-state median 1,354/1,593 ms. Across the eight batch-1 operations at L=512,
average compile time is 1.94/1.92 s. Shared S2FFT precompute generation takes 0.94–3.10
s over the tested sizes; L=512 Python setup is 36.6 ms and input device transfer is 2.5
ms.

## Memory and compiled representation

Candidate A adds a native-grid physical intermediate of shape (L, 2L-1) per transformed
scalar or spin field. At L=512 that payload is 4 MiB for real scalar and 8 MiB for
complex spin; vector operations can use two such spin intermediates. Candidate B enters
S2FFT at the (L, 2L-1) representable m domain and has no extra physical resampling
array. Both paths still hold an (L, 2L-1) complex ftm.

XLA does not retain the theoretical physical intermediate in every compiled case. At
L=512, memory-analysis temporary bytes are identical for scalar forward (44.05 MB) and
filter. B reduces spin forward from 54.54 to 50.36 MB and kinematics from 104.88 to
100.64 MB, while vector analysis is effectively identical. Averaged over the eight
batch-1 operations, reported temporary memory is 63.0 MiB (A) and 62.0 MiB (B).

At L=512, average StableHLO operation count is 4,513 (A) versus 4,427 (B); StableHLO
text sizes are almost identical, about 48.1 million characters. The extra FFTs and
physical intermediate matter for some batched/high-spin cases, but do not yield a
uniform runtime or memory advantage.

## Maintenance and dependency risk

| Item                        | A — public resampling                    | B — internal latitude step                                                                                       |
| --------------------------- | ---------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| Adapter implementation      | 84 lines                                 | 199-line compatibility module                                                                                    |
| S2FFT transform symbols     | Public forward_jax / inverse_jax         | Private-module latitude forward/inverse                                                                          |
| Convention logic owned here | Fourier resampling and Nyquist placement | FFT normalization, quadrature, GL theta, harmonic normalization, spin sign/zeroing, real-field Hermitian mapping |
| Wigner-d recursion copied   | None                                     | None                                                                                                             |
| Version guard               | Not needed                               | Exact 1.4.0 and function-signature check                                                                         |

Candidate B imports S2FFT internals only in the compatibility module. Its other
low-level helpers are s2fft.sampling.s2_samples.thetas and
s2fft.utils.quadrature_jax.quad_weights_transform. An explicit version test and
signature test make unreviewed upgrades fail with a targeted error.

This boundary is fragile across S2FFT upgrades: after the 1.4.0 baseline, S2FFT commit
95c8d355 replaced custom-VJP latitude steps with registered JAX primitives, and commit
4ef1ee5 moved transpose rules to the new primitive interface. The 1.4.0 otf_recursions
functions are therefore not a stable cross-version contract. A product dependency on
`s2fft==1.4.0` is required while the compatibility module calls these internals. The
module checks the installed version, required symbols, and function signatures. The
dependency can be relaxed after a S2FFT release exposes and validates a public path for
GL `(L, 2L)`.

At report time, a read-only S2FFT checkout has an unreleased `feat/oversampled-gl`
branch at `1b66d9d`; its `main` is at `86a4dff`, and the installed experiment baseline
is S2FFT 1.4.0. Commit `da51a7b` on the feature branch adds public `(L, 2L)` GL support
for [issue #406](https://github.com/astro-informatics/s2fft/issues/406): `forward_jax`
accepts the even-width input shape and `inverse_jax` accepts `nphi=2L`. Once a release
includes that API, spharmgrid can replace the compatibility module with those public
functions and relax the temporary pin. The high-level spharmgrid operations do not need
separate rewrites.

## Recommendation

**Recommend B, the pinned internal ftm path, for the supported implementation of this
shape.** It is the only candidate that passes the tested forward-mode JVP gate: A's
public S2FFT calls reject JVP for both forward and inverse, although reverse-mode grad
passes. B passes both modes, removes the physical resampling intermediates, and improves
the measured batch-4/16 scalar transforms by 12–18% geometrically (with a 24% inverse
improvement for L=64, batch 16). Kinematics at L=512 is 16.5% faster. Those benefits
justify one isolated private boundary and the temporary exact pin for workloads with
atmospheric leading dimensions.

The tradeoff is specific: batch-1 performance is tied overall, and A is 15–26% faster
for scalar forward at L=256–512 and 22.8% faster for L=512 spin forward. Memory savings
are real for selected transforms but small across the full compiled operation set. A is
faster for workloads that are almost entirely unbatched scalar analysis at the largest
measured sizes. It remains benchmark-only because the tested public S2FFT path does not
support forward-mode JVP with version 1.4.0.

## Reproduction and follow-up

From the worktree root, the exact commands used were:

```bash
JAX_ENABLE_X64=1 PYTHONPATH=src /home/albert/spharmgrid/.venv/bin/python \
    benchmarks/jax_gl_2l/validate.py \
    --output benchmarks/jax_gl_2l/results/validation.json
JAX_ENABLE_X64=1 PYTHONPATH=src /home/albert/spharmgrid/.venv/bin/python \
    benchmarks/jax_gl_2l/benchmark.py \
    --output benchmarks/jax_gl_2l/results/performance.json
```

The validator defaults to L=4, 8, 16, 32, 64, 128; the performance harness defaults to
L=8, 16, 32, 64, 128, 256, 512. Raw metrics and environment metadata are in
[validation.json](results/validation.json) and
[performance.json](results/performance.json).

Permanent tests cover scalar and spin ±1 round trips, vector parity with DUCC, scalar
modes m=0, 1, L-2, and L-1, scalar Nyquist projection, shifted longitude, both latitude
orders, leading dimensions, `jit`, `vmap`, `jvp`, and `grad`. The JAX/DUCC parity matrix
covers the scientific operation set on both GL shapes. The full A/B benchmark matrix
remains outside routine unit CI.

`uv.lock` resolves S2FFT 1.4.0 from PyPI on Python 3.11/3.12 and from the v1.4.0 source
tag on Python 3.13 and later. The source-build workaround for the 1.4.0 wheel filename
is still required on those newer Python versions.
