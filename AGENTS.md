# Repository instructions

Keep this file limited to repository-wide invariants, task routing, and change
control. Implementation plans belong under `plans/`; user-facing technical and
scientific documentation belongs under `docs/`.

## Read first

- Inspect the current checkout, branch, working tree, and relevant files before
  editing. Current repository content overrides old chats, prompts, reports, and
  handoffs.
- Read the relevant phase plan before changing public APIs, supported grids,
  numerical behavior, backend behavior, dependencies, tests, CLI, or
  documentation scope:
  - `plans/01-core.md` — completed core GL/CC implementation.
  - `plans/02-sht-suite.md` — completed scalar/vector SHT operation suite and
    current public scientific contract.
  - `plans/03-gpu-backends.md` — current next phase: backend boundary,
    torch-harmonics, and S2FFT.
  - `plans/04-ring-grids.md` — later HEALPix and reduced-Gaussian work.
- PyStormTracker, NCL/SPHEREPACK, and pyspharm may be useful implementation or
  parity references when a task explicitly depends on them. Do not treat a
  sibling checkout as part of spharmgrid's required development environment.
- Do not modify another repository as part of spharmgrid work unless the owner
  explicitly requests it.

## Writing

For README, docs, references, API prose, comments, docstrings, and scientific or
technical explanations, follow
[mwyau/write-like-a-scientist](https://github.com/mwyau/write-like-a-scientist),
including its research-software profile and atmospheric-science guidance.
Repository-local scientific and API rules in this file take precedence when
more specific.

Use `spherical harmonic` as a modifier (for example, `spherical harmonic
transform`). Use `spherical harmonics` only as a plural noun. Never hyphenate
either form.

Keep atmospheric, mathematical, xarray, CF, and software terminology precise.
Preserve equations, signs, normalization, grid definitions, radius factors,
degree-zero conventions, units, and API names when editing prose. Distinguish
published methods, external implementation behavior, parity results,
repository-tested behavior, measured validation, and planned work. User
documentation describes implemented behavior; plans remain under `plans/`.

## Scientific and API invariants

- spharmgrid is an xarray-first wrapper/helper around `ducc0` for atmospheric
  and geophysical spherical harmonic operations. Do not describe it as a new
  spherical harmonic transform implementation.
- `ducc0` supplies the numerical SHT machinery. NCL/SPHEREPACK provide important
  operation semantics and parity references. Published methods remain distinct
  from external implementation behavior and spharmgrid behavior.
- The current supported horizontal grids are full Gauss--Legendre (GL) and
  pole-including Clenshaw--Curtis (CC) grids. Do not silently reinterpret another
  latitude-longitude grid as CC.
- Preserve cyclic-longitude equivalence by moving coordinates and data together.
  Coordinate ordering or longitude convention must not change the physical
  field.
- Keep the xarray `.sg` accessor thin. Accessor and direct `spharmgrid` functions
  must use the same numerical implementation.
- Keep scientific defaults explicit. Spectral tapering is off unless `taper` is
  supplied.
- Preserve xarray non-spatial dimensions, CF time/calendar objects, and
  coordinate alignment. spharmgrid does not define its own time representation.
- Use CF `standard_name` metadata and documented canonical short names for
  automatic variable discovery. Do not grow a heuristic alias table without a
  demonstrated interoperability need.
- Do not invent CF standard names. When no exact standard name exists, use
  accurate ordinary metadata instead.
- The current public behavior is the Phase-2 contract in
  `plans/02-sht-suite.md`. Planned backend and grid features are not current
  behavior until implemented and tested.

## Scientific correctness and evidence

- Verify literature-derived formulas and terminology against primary scientific
  sources when practical. Verify bibliographic details against the publisher or
  another authoritative source.
- Do not replace a named scientific method with an approximation while retaining
  its name.
- Do not rewrite vector-transform signs, component ordering, normalization,
  radius factors, or latitude orientation from memory. Trace the implementation
  and verify it with analytic tests and an independent implementation.
- Treat PyStormTracker, NCL/SPHEREPACK, and pyspharm results as identified
  implementation/parity references, not scientific ground truth.
- Do not claim parity, accuracy, performance, or external validation without
  stating the comparison and evidence.
- Scientific validation is cumulative: analytic behavior, internal identities
  and round trips, and independent-backend parity are separate evidence; parity
  alone does not establish mathematical correctness.
- Do not weaken scientific assertions or tolerances to make a failing test pass.
  Diagnose the cause first.

## Tests

- Unit tests should protect a numerical primitive or small API contract using
  deterministic analytic or constructed fields.
- Parity tests compare against an identified independent implementation.
- Prefer analytic harmonics, identities, round trips, invariance checks, and
  explicit tolerances over large stored reference arrays.
- Hold grid, coordinates, normalization, radius, truncation, and comparison
  population fixed before attributing a difference to the implementation.
- A failed parity test is information. Do not silently skip, loosen, or
  substitute another method.
- Test accessor/direct equivalence, GL/CC behavior, both latitude orders, common
  cyclic longitude conventions, leading xarray dimensions, and Dask behavior
  where supported.

## Package and tooling

- `pyproject.toml` is authoritative for package metadata, dependencies, build
  configuration, and tool configuration. Keep `uv.lock` synchronized.
- Do not generate, regenerate, or hand-edit `uv.lock`. When a change makes the
  lockfile stale, notify the owner that they must run `uv lock` locally and
  commit the resulting lockfile update.
- Never create temporary GitHub Actions workflows, throwaway CI jobs, or other
  repository automation solely to generate or refresh a lockfile.
- Keep uv's normal `dev` group enabled so standard local commands such as
  `uv run ruff`, `uv run ty check`, and `uv run pytest` work directly. CI jobs
  that require a reduced environment should opt out with
  `--no-default-groups` instead of disabling development dependencies globally.
- Use the existing Hatchling, uv, Ruff, ty, pytest, Sphinx/MyST, and Read the
  Docs setup unless a concrete requirement justifies changing a tool.
- Keep core runtime dependencies small and optional capabilities optional.
- Keep the wheel limited to `src/spharmgrid`.
- Keep repository-only material such as `.github/`, `AGENTS.md`, `plans/`, and
  agent instruction files out of source distributions. Tests and user-facing
  documentation may remain in the sdist.
- The production test suite must not require network access or optional parity
  tooling.
- Keep DUCC thread control small and explicit; avoid nested oversubscription and
  unmeasured concurrency abstractions.
- The CLI must use the same package API as Python callers rather than a separate
  numerical path.

## CI and publishing

- CI covers the supported Python/OS matrix, minimum direct dependencies, Ruff,
  typing, strict documentation, and independent parity. Forward-looking probes
  may be non-blocking when their failure is understood and reported clearly.
- `Python Publish` also serves as the package-distribution test workflow. It may
  run for pull requests and ordinary `main` pushes, but publishing itself is
  restricted to release tags.
- Package tests must build both wheel and sdist, install and import the wheel,
  smoke-test the installed CLI, check distribution metadata strictly, and
  verify repository-only files do not leak into the sdist.
- Release tags must match the package version exactly.
- Development tags such as `.devN` publish to TestPyPI; stable tags publish to
  PyPI.
- Use PyPI Trusted Publishing/OIDC. Grant `id-token: write` only to the publish
  job. Publish the exact distributions produced by the successful package-test
  job rather than rebuilding them.
- Keep the trusted publishing workflow filename stable unless the corresponding
  PyPI/TestPyPI Trusted Publisher configuration is updated.

## Change control

- Keep changes within the requested scope and preserve unrelated work.
- Develop new features on a dedicated feature branch rather than directly on
  `main`.
- Do not create pull requests or Git tags unless the owner explicitly instructs
  you to create that specific pull request or tag.
- Use Conventional Commits-style subjects such as `feat: ...`, `fix: ...`,
  `docs: ...`, `test: ...`, `ci: ...`, `refactor: ...`, or `chore: ...`.
- Do not build abstractions or extension points for hypothetical future
  features. Add structure when current supported behavior needs it.
- Do not force-push, reset shared history backwards, or overwrite newer unrelated
  work unless the owner explicitly requests a pre-release history rewrite.
- Do not commit or push unless the current task explicitly requests it. If
  commits are requested, keep them small and linear.
