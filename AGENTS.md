# Repository instructions

Keep this file to repository-wide invariants, task routing, and change control.
Implementation plans belong under `plans/`; user-facing scientific and technical
documentation belongs under `docs/`.

## Before changing code

- Inspect the current branch, working tree, and relevant files before editing.
  Current repository content overrides old chats, prompts, reports, and handoffs.
- Read the relevant file under `plans/` before changing the public API,
  supported grids, numerical behavior, backends, dependencies, tests, CLI, or
  documentation scope. Code and tests determine current behavior; user
  documentation must match them. Plans may also describe later work.
- Do not modify another repository unless the owner explicitly requests it.

## Writing

For README text, documentation, references, API prose, comments, docstrings, and
scientific explanations, follow
[mwyau/write-like-a-scientist](https://github.com/mwyau/write-like-a-scientist),
including its research-software profile and atmospheric-science guidance.
Repository-specific rules here take precedence.

Use `spherical harmonic` as a modifier (for example, `spherical harmonic
transform`) and `spherical harmonics` only as a plural noun. Never hyphenate
either form.

Preserve equations, signs, normalization, grid definitions, radius factors,
degree-zero conventions, units, coordinates, and API names when editing
scientific text.

## Scientific and API invariants

- spharmgrid exposes atmospheric and geophysical spherical harmonic operations
  on Xarray objects. `ducc0` performs the numerical spherical harmonic
  transforms; do not describe spharmgrid as a new transform implementation.
- Supported horizontal grids are full Gauss–Legendre (GL) and Clenshaw–Curtis
  (CC) grids. CC uses equally spaced latitudes from -90 to 90 degrees. Do not
  reinterpret another latitude-longitude grid as CC.
- Preserve cyclic-longitude equivalence by moving coordinates and data together.
  Coordinate order and longitude convention must not change the physical field.
- Direct functions and the `.sg` accessor must use the same numerical
  implementation.
- Spectral tapering is off unless `taper` is supplied.
- Preserve non-spatial dimensions, CF time/calendar objects, and coordinate
  alignment. spharmgrid does not define its own time representation.
- Automatic variable discovery uses exact CF `standard_name` metadata and
  documented canonical short names. Do not add heuristic aliases without a
  demonstrated interoperability need, and do not invent CF standard names.
- Do not document planned backends, grids, or API behavior as implemented until
  the code and tests support them.

## Scientific changes and tests

- Do not rewrite vector-transform signs, component ordering, normalization,
  radius factors, or latitude orientation from memory. Trace the implementation
  and check analytic behavior and an independent implementation where relevant.
- NCL/SPHEREPACK and pyspharm are implementation and parity references, not
  scientific ground truth.
- Do not weaken scientific assertions or tolerances to make a failing test pass;
  diagnose the cause first.
- Unit tests should protect numerical primitives or small API contracts with
  deterministic analytic or constructed fields. Parity tests compare against an
  identified independent implementation.
- Prefer analytic harmonics, identities, round trips, invariance checks, and
  explicit tolerances over large stored reference arrays.
- Hold grid, coordinates, normalization, radius, truncation, and comparison
  population fixed before attributing a difference to the implementation.
- Test accessor/direct equivalence, GL/CC behavior, both latitude orders, common
  cyclic longitude conventions, leading Xarray dimensions, and Dask behavior
  where supported.
- Production tests must not require network access or optional parity tooling.

## Package and tooling

- `pyproject.toml` defines package metadata, dependencies, build configuration,
  and tool configuration. Keep `uv.lock` synchronized, but do not generate,
  regenerate, or hand-edit it; if it is stale, ask the owner to run `uv lock`
  locally and commit the result.
- Do not create temporary GitHub Actions workflows or other automation to refresh
  the lockfile.
- Keep uv's normal `dev` group enabled so `uv run ruff`, `uv run ty check`, and
  `uv run pytest` work directly. Reduced CI environments should opt out with
  `--no-default-groups`.
- Use the existing Hatchling, uv, Ruff, ty, pytest, Sphinx/MyST, and Read the
  Docs setup unless a concrete requirement justifies changing a tool.
- Keep optional capabilities out of core runtime dependencies.
- Keep the wheel limited to `src/spharmgrid`. Keep repository-only material such
  as `.github/`, `AGENTS.md`, `plans/`, and agent instructions out of source
  distributions.
- Keep DUCC thread control explicit and avoid nested oversubscription.
- The CLI must call the same package API as Python users rather than a separate
  numerical path.

## CI and publishing

- CI covers the supported Python/OS matrix, minimum direct dependencies, Ruff,
  typing, strict documentation, and independent parity.
- `Python Publish` also tests package distributions. Publishing is restricted to
  release tags; development tags such as `.devN` publish to TestPyPI and stable
  tags publish to PyPI.
- Release tags must match the package version exactly. Build wheel and sdist,
  install and import the wheel, smoke-test the installed CLI, check distribution
  metadata, and verify repository-only files do not leak into the sdist.
- Use PyPI Trusted Publishing/OIDC. Grant `id-token: write` only to the publish
  job and publish the distributions produced by the successful package test.
- Keep the trusted-publishing workflow filename stable unless the corresponding
  PyPI/TestPyPI publisher configuration is updated.

## Change control

- Keep changes within the requested scope and preserve unrelated work.
- Develop new features on a dedicated branch rather than directly on `main`.
- Do not create pull requests or Git tags unless the owner explicitly requests
  that specific pull request or tag.
- Use Conventional Commits-style subjects such as `feat: ...`, `fix: ...`,
  `docs: ...`, `test: ...`, `ci: ...`, `refactor: ...`, or `chore: ...`.
- Do not add abstractions or extension points for hypothetical features.
- Do not force-push, reset shared history backwards, or overwrite newer
  unrelated work unless the owner explicitly requests a pre-release history
  rewrite.
- Do not commit or push unless the current task explicitly requests it. When
  commits are requested, keep them small and linear.
