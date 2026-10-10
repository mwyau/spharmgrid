# NCL/SPHEREPACK parity

The tests compare spharmgrid's DUCC0-backed operations with NCL 6.6.2 using its built-in
SPHEREPACK routines. The [`validation.yml`](../../.github/workflows/validation.yml)
workflow generates the input fields and NCL outputs on pushes to `main` and by manual
dispatch. `pytest tests/parity` does not require NCL.

## Test fields

The tests use Gauss–Legendre (GL) grids of `64 × 128` points and Clenshaw–Curtis (CC)
grids of `73 × 144` points, with ascending latitude and non-cyclic longitude. Inputs
comprise analytic spherical harmonic mixtures near truncation boundaries and seeded
harmonic mixtures (`np.random.default_rng(42)`) through degree 50. Wind fields include a
`cos(latitude)` factor to ensure regularity at the CC poles.

Python generates identical input fields for both implementations. Normalization checks
the coordinates and converts NCL output to ascending latitude.

The spectral selections are:

- `T42`: `0 ≤ m ≤ l ≤ 42`;
- `T5-42`: `5 ≤ l ≤ 42`, `0 ≤ m ≤ l`;
- `T42x10`: `0 ≤ l ≤ 42`, `0 ≤ m ≤ min(l, 10)`;
- `R21`: `0 ≤ m ≤ 21`, `0 ≤ l - m ≤ 21`.

Each selection is tested with a hard cutoff and `taper=0.1`. Full-band analysis and
synthesis are also tested. The reference calculation applies coefficient masks for
`T42x10` and `R21`.

## NCL operations

| spharmgrid operation                             | NCL/SPHEREPACK routine                                                   |
| ------------------------------------------------ | ------------------------------------------------------------------------ |
| scalar analysis/synthesis, filtering, regridding | `shaec`/`shsec` (CC), `shagc`/`shsgc` (GL)                               |
| gradient and inverse gradient                    | `gradsf`/`gradsg`, `igradsf`/`igradsg`                                   |
| scalar Laplacian and inverse Laplacian           | `lapsf`/`lapsg`, `ilapsf`/`ilapsg`                                       |
| vorticity and divergence                         | `uv2vrf`/`uv2vrg`, `uv2dvf`/`uv2dvg`                                     |
| combined kinematics                              | `uv2vrdvf`/`uv2vrdvg`                                                    |
| streamfunction and velocity potential            | `uv2sfvpf`/`uv2sfvpg`                                                    |
| rotational and divergent wind                    | `vr2uvf`/`vr2uvg`, `dv2uvf`/`dv2uvg`                                     |
| wind from vorticity and divergence               | `vrdv2uvf`/`vrdv2uvg`                                                    |
| wind from streamfunction and velocity potential  | `sfvp2uvf`/`sfvp2uvg`                                                    |
| vector Laplacian and inverse vector Laplacian    | `lapvf`/`lapvg`, `ilapvf`/`ilapvg`                                       |
| Helmholtz wind components                        | combined vorticity/divergence analysis followed by `vr2uv*` and `dv2uv*` |
| vector regridding                                | `vhaec`/`vhsec` (CC), `vhagc`/`vhsgc` (GL)                               |

The NCL comparisons test numerical values through spharmgrid's Python API. CLI file
handling, metadata, and coordinate alignment are tested separately.

## Spectral taper

At `L=42` with `taper=0.1`, spharmgrid applies

```text
w_l = exp(log(0.1) * [l(l+1)/(42*43)]**2).
```

NCL defines

```text
S(l) = exp(-[l(l+1)/(N(N+1))]**2).
```

The equivalent NCL parameter is `N ≈ 34.002499526`. The tests obtain weights from
`exp_tapersh_wgts` and apply them at degree `l`. NCL 6.6.2 `exp_tapersh` shifts the
weight sequence by one row, so the reference calculation does not use it.

## Numerical agreement

The 276 NCL tests make 400 elementwise comparisons of analytic and seeded random fields.
All assertions use operation-specific absolute tolerances (`rtol=0`) defined in
`test_ncl.py`. Failures report maximum absolute, root-mean-square, and maximum relative
errors away from zero.

The largest maximum error normalized by the NCL field's maximum magnitude was
`3.71e-11`, for CC random scalar filtering with a hard `T5-42` cutoff (`2.16e-11`
absolute error). The largest absolute error, `1.67e-2`, occurred for the CC random
inverse vector Laplacian at a pole. The NCL reference magnitude there was `4.61e12`,
giving a normalized error of `3.62e-15`. Inverse Laplacian operations amplify
coefficient differences by the squared-radius factor.

These measurements describe agreement with NCL/SPHEREPACK on the tested grids and
fields, not an independent measure of physical accuracy.

## Running locally

Install NCL 6.6.2 and the Python test dependencies:

```bash
conda create --yes --name spharmgrid-ncl --channel conda-forge ncl=6.6.2
uv sync --no-default-groups --group test --extra cli --frozen
```

Generate the input fields and NCL outputs, then run the tests:

```bash
work=$(mktemp -d /tmp/spharmgrid-ncl.XXXXXX)
mkdir -p "$work/inputs" "$work/outputs" "$work/normalized"

uv run --no-sync python tests/ncl/generate_inputs.py --output-dir "$work/inputs"

for grid in gl cc; do
    for family in analytic random; do
        SPHARMGRID_NCL_INPUT="$work/inputs/input-${grid}-${family}.nc" \
            SPHARMGRID_NCL_OUTPUT="$work/outputs/ncl-${grid}-${family}.nc" \
            SPHARMGRID_NCL_GRID="$grid" \
            SPHARMGRID_NCL_FAMILY="$family" \
            conda run --no-capture-output --name spharmgrid-ncl \
            ncl -Q tests/ncl/generate_references.ncl
    done
done

uv run --no-sync python tests/ncl/normalize_references.py \
    --input-dir "$work/inputs" \
    --ncl-output-dir "$work/outputs" \
    --output-dir "$work/normalized"

SPHARMGRID_NCL_OUTPUT_DIR="$work/normalized" \
    uv run --no-sync pytest tests/ncl/test_ncl.py
```

The workflow uploads the inputs, NCL outputs, normalized arrays, and logs as GitHub
Actions artifacts for 14 days.
