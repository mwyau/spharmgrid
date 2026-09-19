NCL/SPHEREPACK parity

This directory contains the opt-in parity run against the independent NCL
6.6.2/SPHEREPACK implementation. It is deliberately outside `tests/parity/`:

```text
pytest tests/parity
```

does not collect or run NCL parity. The NCL run requires a separate Conda
environment and is therefore canonical only through the manual
`.github/workflows/ncl-parity.yml` dispatch. No NCL-generated reference data is
stored in Git; every output is generated during that run and uploaded as an
ephemeral workflow artifact.

## Scope

The inputs use the exact supported grids GL `64 x 128` and CC `73 x 144`, with
ascending latitude and non-cyclic longitude from 0 degrees through the final
point below 360 degrees. The Python input writer emits double-precision arrays
in spharmgrid's canonical south-to-north row order. The NCL generator uses that
explicit row order. The normalizer verifies monotonic coordinates and reverses
coordinates and every associated field together if an NCL output is returned
in the opposite order. No individual pytest compares silently reverse an
array.

Two deterministic input families are generated in Python:

- `analytic` is a smooth, explicitly defined real spherical-harmonic mixture
  containing terms below degree 5, at degrees 5, 21, and 42, above degree 42,
  low and high zonal orders, both sides of `m=10`, and terms distinguishing
  triangular and rhomboidal domains.
- `random` uses `np.random.default_rng(42)` for three float64 arrays.
  NCL does not generate random values; it consumes the exact serialized arrays
  written by Python.

The NCL-side coefficient masks are independent degree/order loops:

- `T42`: `0 <= m <= l <= 42`;
- `T5-42`: `5 <= l <= 42` and `0 <= m <= l`;
- `T42x10`: `0 <= l <= 42` and `0 <= m <= min(l, 10)`;
- `R21`: `0 <= m <= 21` and `0 <= l - m <= 21`.

Every explicit domain is run with a hard mask and with `taper=0.1`. A full
bandwidth analysis/synthesis baseline is included where the source and target
grid geometry makes it meaningful. The trapezoidal `T42x10` case is masked
directly in the NCL coefficient arrays; it does not call a spharmgrid mask or
depend on an NCL convenience function with a different domain.

## NCL mapping

The generator records the routine used for each output in its normalized
metadata. The implemented mappings are:

| spharmgrid operation                      | NCL/SPHEREPACK reference                                                                            |
| ----------------------------------------- | --------------------------------------------------------------------------------------------------- |
| scalar analysis/synthesis, filter, regrid | `shaec`/`shsec` (CC), `shagc`/`shsgc` (GL)                                                          |
| gradient and inverse gradient             | `gradsf`/`gradsg`, `igradsf`/`igradsg`                                                              |
| scalar Laplacian and inverse              | `lapsf`/`lapsg`, `ilapsf`/`ilapsg`                                                                  |
| vorticity, divergence, kinematics         | `uv2vrf`/`uv2vrg`, `uv2dvf`/`uv2dvg`, `uv2vrdvf`/`uv2vrdvg`                                         |
| streamfunction and velocity potential     | `uv2sfvpf`/`uv2sfvpg`                                                                               |
| rotational and divergent wind             | `vr2uvf`/`vr2uvg`, `dv2uvf`/`dv2uvg`                                                                |
| wind from vorticity/divergence            | `vrdv2uvf`/`vrdv2uvg`                                                                               |
| wind from potentials                      | `sfvp2uvf`/`sfvp2uvg`                                                                               |
| vector Laplacian and inverse              | `lapvf`/`lapvg`, `ilapvf`/`ilapvg`                                                                  |
| Helmholtz components                      | NCL vorticity/divergence analysis followed by the independent `vr2uv*` and `dv2uv*` reconstructions |
| vector regrid                             | `vhaec`/`vhsec` (CC), `vhagc`/`vhsgc` (GL)                                                          |

Accessors, CLI behavior, metadata discovery, coordinate alignment, and other
Xarray API contracts have no NCL/SPHEREPACK mathematical analogue and remain
covered by the package's ordinary tests.

## Taper and numerical comparison

Spharmgrid's fourth-order degree weight is

```text
exp(log(0.1) * [l(l+1)/(42*43)]**2).
```

The NCL parameter is solved independently so that NCL's
`exp_tapersh` curve has response exactly 0.1 at `l=42`; the resulting mode is
approximately `34.002499526`. The generator corrects NCL's packed coefficient
row offset so the total-degree weight applies to `l`, while the degree/order
mask remains separate. The pytest suite compares the calculated weights
directly in addition to comparing synthesized fields.

Comparisons are element-wise and operation-specific. Failures report maximum
absolute error, RMS error, and a relative error over values away from numerical
zero. The larger bounds for broadband random inverse/vector operations are
grid-specific measured bounds for independent high-degree GL/CC algorithms;
analytic and low-condition cases retain much tighter bounds. No correlation
criterion is used.

## Running locally

Normal tests do not require NCL:

```bash
uv run pytest tests/parity
```

To run the full NCL path locally, install NCL 6.6.2 from conda-forge and the
small Python-only NetCDF writer dependencies in the active uv environment:

```bash
conda create -n spharmgrid-ncl -c conda-forge ncl=6.6.2
uv sync --no-default-groups --group test --frozen
uv pip install --python .venv/bin/python h5netcdf h5py
```

Then create temporary directories, generate the Python inputs, run one NCL
process per grid/input family, normalize its output, and run pytest against
that same output directory:

```bash
work=$(mktemp -d /tmp/spharmgrid-ncl.XXXXXX)
mkdir -p "$work/inputs" "$work/outputs" "$work/references"
uv run --no-sync python tests/ncl/generate_inputs.py --output-dir "$work/inputs"
for grid in gl cc; do
    for family in analytic random; do
        SPHARMGRID_NCL_INPUT="$work/inputs/input-${grid}-${family}.nc" \
            SPHARMGRID_NCL_OUTPUT="$work/outputs/ref-${grid}-${family}.nc" \
            SPHARMGRID_NCL_GRID="$grid" \
            SPHARMGRID_NCL_FAMILY="$family" \
            conda run --no-capture-output -n spharmgrid-ncl ncl -Q \
            tests/ncl/generate_references.ncl
    done
done
uv run --no-sync python tests/ncl/normalize_references.py \
    --input-dir "$work/inputs" \
    --ncl-output-dir "$work/outputs" \
    --output-dir "$work/references"
SPHARMGRID_NCL_REFERENCE_DIR="$work/references" \
    uv run --no-sync pytest tests/ncl/test_ncl.py
```

The dispatch workflow performs this same fresh generation, runs the live
comparison, and uploads the Python inputs, raw NCL outputs/logs, normalized
reference outputs, and pytest error summary. It never commits or pushes any
generated file.
