# Repository instructions

Keep this file limited to repository-wide invariants, task routing, and change control. Detailed procedures belong under `skills/`; the current implementation contract belongs in `PLAN.md`; user-facing technical and scientific documentation belongs under `docs/`.

## Read first

- Inspect the current checkout, branch, working tree, and relevant files before editing. Current repository content overrides old chats, prompts, reports, and handoffs.
- Read `PLAN.md` for routing/current phase, and read the linked detailed phase plan before implementing or changing the initial public API, supported grids, numerical behavior, dependencies, tests, CLI, or documentation scope.
- The sibling repository `../PyStormTracker` is the primary source implementation for the initial extraction. Read its current `AGENTS.md` and the relevant preprocessing code before porting numerical behavior. Do not assume an old snippet still matches that checkout.
- Do not modify `../PyStormTracker` as part of spharmgrid work unless the owner explicitly requests it.

## Route work through the owning skill

- **Spherical-harmonic methods, GL/CC grids, filtering, regridding, differential operators, vector transforms, NCL/SPHEREPACK parity, scientific tests, or numerical behavior:** read `skills/scientific-numerics/SKILL.md`.
- **README, docs, references, API prose, comments, docstrings, Sphinx/MyST, or Read the Docs:** read `skills/documentation/SKILL.md`.
- **Packaging, dependencies, uv, pyproject configuration, lint/type/test tooling, CLI packaging, documentation build configuration, or CI:** read `skills/repository-engineering/SKILL.md`.

Any change to `README.md`, `docs/`, public docstrings, or explanatory repository prose must also follow `skills/documentation/SKILL.md`.

## Scientific and API invariants

- spharmgrid is an xarray-first wrapper/helper around `ducc0` for atmospheric and geophysical spherical-harmonic operations. Do not describe it as a new spherical-harmonic transform implementation.
- `ducc0` supplies the numerical SHT machinery. NCL/SPHEREPACK provide important operation semantics and parity references. Published methods remain distinct from NCL behavior, PyStormTracker behavior, and spharmgrid behavior.
- The initial supported horizontal grids are full Gauss–Legendre (GL) and pole-including Clenshaw–Curtis (CC) grids. Do not silently reinterpret another latitude-longitude grid as CC.
- Preserve cyclic-longitude equivalence by moving coordinates and data together. Coordinate ordering or longitude convention must not change the physical field.
- Keep the xarray `.sg` accessor thin. Accessor and direct `spharmgrid` functions must use the same numerical implementation.
- Keep scientific defaults explicit. In particular, spectral tapering is off unless `taper` is supplied.
- Preserve xarray non-spatial dimensions, CF time/calendar objects, and coordinate alignment. spharmgrid does not define its own time representation.
- Use CF `standard_name` metadata and the documented canonical short names for automatic variable discovery. Do not grow a heuristic alias table without a demonstrated interoperability need.
- Do not invent CF standard names. When no exact standard name exists, use accurate ordinary metadata instead.
- Keep the first public surface small. Reduced Gaussian grids, HEALPix, regional DCT filtering, MPI, arbitrary SHT backends, raw public coefficient objects, and release/publishing work are out of scope unless `PLAN.md` is explicitly changed.

## Scientific correctness and evidence

- Verify literature-derived formulas and terminology against primary scientific sources when practical. Verify bibliographic details against the publisher or another authoritative source.
- Do not replace a named scientific method with an approximation while retaining its name.
- Do not rewrite vector-transform signs, component ordering, normalization, radius factors, or latitude orientation from memory. Trace the current PyStormTracker implementation, then verify with analytic tests and an independent implementation.
- Treat PyStormTracker, NCL/SPHEREPACK, and pyspharm results as identified implementation/parity references, not scientific ground truth.
- Do not claim parity, accuracy, performance, or external validation without stating the comparison and evidence.
- Do not weaken scientific assertions or tolerances to make a failing test pass. Diagnose the cause first.

## Tests

- Unit tests should protect a numerical primitive or small API contract using deterministic analytic or constructed fields.
- Parity tests compare against an identified implementation such as SPHEREPACK/pyspharm or the source PyStormTracker behavior.
- Prefer analytic harmonics, identities, round trips, invariance checks, and explicit tolerances over large stored reference arrays.
- Hold grid, coordinates, normalization, radius, truncation, and comparison population fixed before attributing a difference to the implementation.
- A failed parity test is information. Do not silently skip, loosen, or substitute another method.
- Test accessor/direct equivalence, GL/CC behavior, both latitude orders, common cyclic longitude conventions, leading xarray dimensions, and Dask behavior where supported.

## Writing

- Use plain, direct English. Keep established atmospheric, mathematical, xarray, CF, and software terms when they are the precise terms.
- Write documentation, comments, and docstrings as technical or scientific descriptions of current behavior, methods, evidence, and limits.
- Comments should explain non-obvious numerical reasoning, sign or coordinate conventions, scientific lineage, constraints, or implementation choices rather than restating code.
- Avoid filler qualifiers and promotional wording. Do not call something robust, comprehensive, sophisticated, modern, clean, or similar without a specific measured meaning.
- Distinguish published methods, external implementation behavior, repository-tested behavior, measured validation, and planned work.

## Change control

- Keep changes within the requested scope and preserve unrelated work.
- Do not build abstractions or extension points for hypothetical future features. Add structure when current supported behavior needs it.
- Do not add release tags, PyPI publishing, release workflows, Zenodo integration, or a PyStormTracker dependency as part of the initial implementation.
- Do not force-push, reset shared history backwards, or overwrite newer unrelated work.
- Do not commit or push unless the current task explicitly requests it. If commits are requested, keep them small and linear.
