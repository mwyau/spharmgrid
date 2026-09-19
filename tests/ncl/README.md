# NCL/SPHEREPACK parity

These tests compare spharmgrid with NCL 6.6.2/SPHEREPACK. They live under
`tests/ncl/`, so `pytest tests/parity` does not collect them. The
`.github/workflows/ncl-parity.yml` workflow runs on pull requests and can also
be started with `workflow_dispatch`. It generates all numerical data for each
run. No NCL-generated numerical data is stored in Git.

## Inputs

The comparison uses:

- Gauss–Legendre (GL) `64 × 128`;
- Clenshaw–Curtis (CC) `73 × 144`;
- ascending latitude and non-cyclic longitude;
- an analytic spherical-harmonic mixture spanning the truncation boundaries;
- broadband random harmonic mixtures from `np.random.default_rng(42)`.

Python evaluates the same seeded random harmonic coefficients on both grids,
then writes each input array once for NCL and spharmgrid. The random fields are
band-limited through degree 50 so the comparison does not depend on how the two
transform libraries project unresolved grid-point noise. The random seed is
stored with the input and copied to the NCL output. The normalizer checks the
coordinates and converts NCL output to ascending latitude before pytest reads
it.

## Spectral domains

The NCL coefficient masks implement the spharmgrid domains independently:

- `T42`: `0 <= m <= l <= 42`;
- `T5-42`: `5 <= l <= 42` and `0 <= m <= l`;
- `T42x10`: `0 <= l <= 42` and `0 <= m <= min(l, 10)`;
- `R21`: `0 <= m <= 21` and `0 <= l - m <= 21`.

Each domain is tested with a hard cutoff and with `taper=0.1`. The tests also
include an untruncated analysis/synthesis case. The `T42x10` and `R21`
masks are applied directly to NCL coefficients.

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

Tests for accessors, metadata, coordinate alignment, and command-line behavior
remain in the normal test suite because they have no NCL/SPHEREPACK numerical
equivalent.

## Taper

For `taper=0.1` at upper retained degree `L=42`, spharmgrid uses

```text
w_l = exp(log(0.1) * [l(l+1)/(42*43)]**2).
```

NCL uses
`S(l) = exp(-[l(l+1)/(N(N+1))]^2)`. Solving for the NCL mode gives
`N ≈ 34.002499526`. The suite gets these weights from
`exp_tapersh_wgts` and applies them explicitly by total degree. The tests
compare the NCL weights with the spharmgrid expression before comparing spatial
fields.

## Numerical comparison

The tests use element-wise comparisons with operation-specific tolerances.
Failures report maximum absolute error, root-mean-square error, and maximum
relative error away from numerical zero.

## Running locally

Create an NCL 6.6.2 environment and install the Python test dependencies:

```bash
conda create -n spharmgrid-ncl -c conda-forge ncl=6.6.2
uv sync --no-default-groups --group test --frozen
```

Generate the inputs and NCL output in a temporary directory:

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
            conda run --no-capture-output -n spharmgrid-ncl ncl -Q \
            tests/ncl/generate_references.ncl
    done
done

uv run --no-sync python tests/ncl/normalize_references.py \
    --input-dir "$work/inputs" \
    --ncl-output-dir "$work/outputs" \
    --output-dir "$work/normalized"

SPHARMGRID_NCL_OUTPUT_DIR="$work/normalized" \
    uv run --no-sync pytest tests/ncl/test_ncl.py
```

The dispatch workflow runs the same sequence and uploads the generated inputs,
NCL output, normalized arrays, logs, and pytest output as workflow artifacts.
