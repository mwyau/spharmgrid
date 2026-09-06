# Repository engineering

Use this skill for package layout, `pyproject.toml`, dependency groups, uv, lint/type/test tooling, CLI packaging, documentation build configuration, CI, releases, and publishing.

## Package and environment

Use a normal `src/` layout and keep `pyproject.toml` authoritative for packaging and tool configuration.

Use:

- `uv` for environment/dependency management and reproducible development commands;
- Hatchling for the build backend unless a concrete package requirement makes another backend necessary;
- Ruff for lint/format checks;
- `ty` for static type checking if the supported dependency/toolchain works cleanly for the implemented xarray/NumPy code;
- pytest for tests.

Do not add a second dependency manager or parallel packaging configuration.

Choose the supported Python range from actual compatibility of `ducc0`, xarray, NumPy, optional Dask/cf-xarray, and the parity environment. Do not narrow the production Python range merely because the optional SPHEREPACK parity package supports fewer versions.

## Dependencies

Keep core runtime dependencies small. The expected core is approximately:

```text
numpy
xarray
ducc0
```

Add another core dependency only when the runtime implementation requires it.

Keep optional features separate. Expected groups/extras include concepts such as:

```text
cf       cf-xarray
io       optional NetCDF/Zarr/GRIB engines
docs     Sphinx/MyST/Read the Docs dependencies
test     test-only dependencies if not kept in a dev group
parity   pyspharm/SPHEREPACK comparison tooling
```

The exact pyproject organization may use dependency groups versus package extras according to whether users need to request the dependency at install time. Do not expose a package extra solely for internal developer tooling.

Do not add Pint, MPI, HEALPix, regional-grid libraries, or alternate SHT backends until the relevant implementation phase requires them.

## uv workflow

Track `uv.lock` once `pyproject.toml` exists.

Use `uv sync`/`uv run` for local commands. Keep documented commands consistent with the actual group configuration. Prefer frozen/locked CI and documentation builds after the initial lock is established.

## Code quality

Configure Ruff and typing once rather than scattering command flags across workflows.

Use explicit type annotations for public functions and meaningful internal boundaries. Do not introduce `Any`, broad casts, compatibility aliases, or wrappers only to silence the type checker. Small targeted casts at third-party typing boundaries are acceptable when the runtime contract is verified.

Keep the public top-level namespace limited to the implemented API defined by the current phase plans under `plans/` and documented in `docs/api.md`.

## Tests

Organize tests by scientific/API purpose rather than by coverage target. Use markers such as `parity` or `slow` only when they describe a real distinction.

The production test suite must not depend on network access or optional SPHEREPACK tooling. A dedicated parity job/environment may install the comparison package separately.

Do not store large generated comparison arrays in the repository when deterministic tests can regenerate the input.

## Dask and threads

Do not copy PyStormTracker's backend/MPI abstraction.

Use xarray/Dask's normal lazy model where supported. Keep DUCC thread control small and explicit. Avoid nested oversubscription by using one DUCC thread per Dask task when the user has not requested a different thread count.

Do not add thread pools, process pools, or task schedulers around DUCC without profiling and a concrete need.

## CLI

Use the standard library `argparse` unless the implemented CLI demonstrates a real need for another dependency.

The CLI must call the same package API as Python users. Do not implement separate numerical paths or a new I/O abstraction.

Delegate file reading/writing to xarray and installed engines. Keep optional format engines optional.

## Documentation build

Follow the repository-wide writing instructions in `AGENTS.md`, including `mwyau/write-like-a-scientist` for README, documentation, comments, docstrings, references, and technical/scientific prose.

Use Sphinx + MyST + `sphinx_rtd_theme` with `.readthedocs.yaml`. Keep documentation dependencies in a lightweight `docs` group and use a strict local Sphinx build.

Do not copy PyStormTracker's Node/Mermaid/PDF setup unless spharmgrid documentation actually needs those features.

## CI

Keep CI limited to checks that validate the supported package:

- ordinary Python tests across a reasonable supported matrix;
- Ruff and configured type checks;
- minimum-direct-dependency testing;
- strict documentation build;
- optional dedicated SPHEREPACK parity on a Python version where the comparison dependency installs reliably;
- non-blocking compatibility probes only when they provide useful forward-looking information without weakening release gates.

Do not make ordinary branch CI build release distributions on every push. Release packaging belongs in the release/publish workflow after the release-tag CI has succeeded.

## Release and publishing

Preserve the tested-artifact publishing model.

For a release tag:

1. the normal CI workflow must complete successfully for the tagged commit;
2. the publish workflow must build the wheel and source distribution from that exact tested commit;
3. install/import/CLI smoke tests and strict distribution metadata checks must run on those artifacts before publishing;
4. upload the validated distributions as a workflow artifact and publish those exact files rather than rebuilding them;
5. require exact tag/package-version agreement for tagged releases;
6. preserve Trusted Publishing/OIDC and build provenance attestations;
7. route development prereleases such as `.devN` to TestPyPI and stable releases to PyPI unless the owner explicitly changes that policy.

Normal pushes to `main` and pull requests should not create skipped publish workflow runs. Filter release workflow triggers before job creation where GitHub Actions supports it.

Manual publishing should identify a successful CI run and build from that run's exact source commit. Manual runs remain TestPyPI-only unless the owner explicitly requests a different release policy.

## Completion check

Before repository-engineering work is complete:

- `pyproject.toml` and `uv.lock` agree;
- editable/normal installation works;
- imports work without optional extras;
- tests, Ruff, and configured typing pass;
- the strict docs build passes;
- optional dependency absence is tested or handled clearly;
- package metadata and supported Python classifiers match actual behavior;
- release workflows, when changed, preserve tested-artifact publishing and tag/version agreement;
- no unintended PyStormTracker dependency or sibling-repository modification is introduced.
