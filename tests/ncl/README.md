# NCL/SPHEREPACK parity

This suite compares spharmgrid with NCL 6.6.2/SPHEREPACK on the same scalar and wind
fields. It runs from `.github/workflows/validation.yml` on pushes to `main` or by manual
dispatch. Each run generates its inputs, NCL output, and normalized comparison arrays.
No NCL-generated numerical data is committed.

The suite lives under `tests/ncl/`, so `pytest tests/parity` does not require NCL.

## Inputs

The comparison uses:

- Gauss–Legendre (GL) `64 × 128`;
- Clenshaw–Curtis (CC) `73 × 144`;
- ascending latitude and non-cyclic longitude;
- an analytic spherical-harmonic mixture with modes around the truncation boundaries;
- seeded broadband harmonic mixtures from `np.random.default_rng(42)`.

Python generates the arrays used by both implementations. The random case draws
coefficients for modes through degree 50. The wind fields are multiplied by
`cos(latitude)` so they are regular at the CC poles. The resulting fields are resolved
by both test grids, which avoids comparing how NCL and DUCC project unresolved
grid-point white noise.

The random seed is stored in the input metadata and copied to the NCL output.
Normalization checks the coordinates and converts NCL output to ascending latitude
before pytest reads it.

## Spectral domains

NCL applies the spectral domains from their degree and order definitions:

- `T42`: `0 ≤ m ≤ l ≤ 42`;
- `T5-42`: `5 ≤ l ≤ 42` and `0 ≤ m ≤ l`;
- `T42x10`: `0 ≤ l ≤ 42` and `0 ≤ m ≤ min(l, 10)`;
- `R21`: `0 ≤ m ≤ 21` and `0 ≤ l - m ≤ 21`.

Each domain is compared with a hard cutoff and with `taper=0.1`. An untruncated
analysis/synthesis case tests the full transform bandwidth. `T42x10` and `R21` are
applied as explicit coefficient masks.

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

Accessor behavior, metadata handling, coordinate alignment, and command-line behavior
are tested in the normal test suite because they have no NCL/SPHEREPACK numerical
analogue.

## Taper

For `taper=0.1` at upper retained degree `L=42`, spharmgrid uses

```text
w_l = exp(log(0.1) * [l(l+1)/(42*43)]**2).
```

NCL defines

```text
S(l) = exp(-[l(l+1)/(N(N+1))]**2).
```

Solving for the NCL mode gives `N ≈ 34.002499526`. The suite obtains these weights from
`exp_tapersh_wgts` and applies each weight to coefficient row `l`. NCL 6.6.2
`exp_tapersh` applies the sequence with a one-row offset, so the parity calculation does
not call it directly.

The taper-weight test compares the NCL weights with the spharmgrid expression before
comparing the spatial fields.

## Numerical comparison

Parity is tested element-wise with operation-specific tolerances. Failures report the
maximum absolute error, root-mean-square error, and maximum relative error away from
numerical zero.

## Running locally

Conda supplies NCL 6.6.2. uv supplies spharmgrid and the Python test dependencies:

```bash
conda create --yes --name spharmgrid-ncl --channel conda-forge ncl=6.6.2
uv sync --no-default-groups --group test --extra cli --frozen
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

The workflow uploads the generated inputs, raw NCL output, normalized arrays, logs, and
pytest output as GitHub Actions artifacts for 14 days.
