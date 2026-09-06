# Documentation

Use this skill for `README.md`, `docs/`, references, API prose, examples, comments, docstrings, Sphinx/MyST configuration, and Read the Docs.

Documentation is part of the public package contract. Keep it synchronized with the implemented API and tested behavior.

## Structure

Keep `README.md` short enough to scan. It should provide:

- what spharmgrid does;
- that `ducc0` supplies the numerical spherical-harmonic transforms;
- supported GL/CC grids;
- installation from the repository while the package is GitHub-only;
- a compact accessor-first quick start;
- direct API examples where useful;
- a link to the full documentation;
- citation/reference guidance when available.

Put detailed material under `docs/`. Keep the documentation compact and combine pages when that makes the result clearer.

## Read the Docs

Use Sphinx with MyST Markdown and the Read the Docs theme, following the working pattern in `../PyStormTracker` unless the local package requires a smaller configuration.

Required repository pieces:

```text
.readthedocs.yaml
docs/conf.py
```

Keep documentation dependencies in a separate `docs` dependency group. The hosted docs build should install only what is needed to import spharmgrid and build the docs.

Use a strict local build command such as:

```bash
uv run --group docs sphinx-build -W --keep-going -b html docs docs/_build/html
```

or the equivalent command supported by the repository configuration.

Do not add Mermaid, Node, PDF builds, or other documentation tooling unless the documentation uses it.

## Human-facing prose

Write for atmospheric scientists and Python/xarray users.

Published documentation should read as documentation for a researcher or user, not as instructions to an agent, a design review, a project-management brief, or a record of implementation decisions.

- Start with the scientific or technical subject.
- Put the current behavior, definition, or result before supporting detail.
- Keep constraints only when users need them to interpret results or avoid an invalid call.
- Put implementation plans, rejected alternatives, future architecture, repository-management rules, and contributor instructions in `PLAN.md`, `AGENTS.md`, or skills.
- Do not preserve chat history, migration narratives, implementation history, or intermediate design decisions in user-facing prose.
- Do not list absent features merely to define scope. Document supported behavior instead.
- Avoid routing and repository meta-language such as `source of truth`, `authoritative`, `owning page`, `this page owns`, `maintained in`, `future agents`, `future contributors`, `the API is designed to`, and explanations of why documentation is organized a certain way.
- Avoid contract narration when examples are clearer. Do not explain that two interfaces are equivalent, share the same numerical path, delegate to the same kernel, or are intentionally thin unless that fact is necessary for users.
- Avoid defensive constructions such as `rather than silently`, `does not guess`, `does not reinterpret`, `does not implement a separate path`, `not a claim that`, or `by design` unless the contrast is required to explain an actual error condition or scientific interpretation.
- Avoid project-management wording such as `initial scope`, `initial package`, `current phase`, `future work`, `planned`, `out of scope`, `roadmap`, `accepted evolution`, and `change gate` in user documentation.
- Avoid historical/meta terms such as `lineage`, `provenance`, `heritage`, `source implementation`, and `semantic reference` in normal user-facing prose. Cite the relevant software or paper directly instead. A short factual statement such as `spharmgrid grew out of spherical-harmonic code used in PyStormTracker` is enough when history is relevant.
- Avoid machine-facing wording such as `pipeline`, `framework`, `architecture`, `backend abstraction`, `contract`, `surface`, and `execution layer` when ordinary scientific/software wording is more precise. Keep these terms only when they name a real software object or established technical concept.
- Prefer ordinary words such as `analysis`, `check`, `comparison`, `method`, `sequence`, and `implementation` over `audit`, `framework`, `protocol`, `hierarchy`, or similar abstractions when no precision is lost.
- Avoid `-style` constructions when an exact method, package, or operation can be named.
- Remove filler adverbs and minimizers that add no meaning, including `simply`, `merely`, `deliberately`, `intentionally`, `directly`, and `essentially` when they are not technically needed.
- Avoid promotional wording such as `seamless`, `comprehensive`, `sophisticated`, `modern`, `clean`, `robust`, or `high-performance` unless supported by a specific property or measurement.
- Do not describe a method or feature as rejected, removed, excluded, or unsupported unless that fact is needed to explain current behavior.
- State the comparison and evidence when claiming parity, accuracy, performance, robustness, or validation.

Prefer concise statements such as:

- `spharmgrid supports full rectangular GL and CC grids.`
- `The l=0 coefficient is set to zero for the inverse Laplacian.`
- `ducc0 performs the spin-1 synthesis.`
- `CC detection requires both poles and a globally cyclic longitude coordinate.`

## API documentation

Show the xarray accessor naturally in examples:

```python
field.sg.filter(...)
```

Show the direct form where useful:

```python
sg.filter(field, ...)
```

Do not add prose solely to explain that both forms exist when the examples already make that clear.

Every public function/class in the top-level namespace must appear in `docs/api.md` or another clearly linked API reference page. Use Sphinx autodoc where it reduces duplication, but keep important scientific semantics in maintained prose/docstrings rather than relying on generated signatures alone.

Document canonical variable names and CF semantics for:

```text
u, v, vo, d, strf, vp
```

Explain automatic discovery order and ambiguity errors.

## Scientific content

Explain when relevant:

- what the operation computes;
- supported grid assumptions;
- units and radius conventions;
- coordinate and metadata behavior;
- zero-mode, truncation, or sign conventions needed to interpret output;
- a short runnable example.

For software and scientific references, state only what users need:

- `ducc0` supplies the numerical SHT routines;
- NCL/SPHEREPACK are useful atmospheric comparison references;
- Sardeshmukh and Hoskins (1984) defines the supported spectral taper;
- spharmgrid grew out of spherical-harmonic code used in PyStormTracker.

Do not imply that spharmgrid contains SPHEREPACK or implements a new transform engine.

Verify literature-derived claims against primary sources and bibliographic metadata against a publisher or other authoritative source when practical. Make DOI and paper links clickable.

## Equations

Repository Markdown should also render usefully on GitHub.

- Use inline `$...$` for short notation.
- Use GitHub-compatible fenced `math` blocks for standalone equations when practical.
- Keep equations consistent with the implementation. A rendering cleanup must not change a sign, normalization, exponent, radius factor, or index range.

## Comments and docstrings

Comments/docstrings should explain non-obvious transform conventions, coordinate handling, constraints, or scientific references. Do not narrate obvious code, preserve migration history, or justify implementation structure to the reader.

## Synchronization

When public behavior changes, update the implementation/docstring, API reference, and relevant user guide page in the same bounded change. Update README examples only when needed.

Before adding a new page, check whether an existing page already covers the subject. Cross-link rather than duplicate long parameter tables or method explanations.

## Documentation completion check

Before documentation work is complete:

- the Read the Docs configuration is coherent;
- the strict local Sphinx build passes;
- every public API is documented;
- quick-start examples use current signatures;
- GL/CC terminology is consistent;
- `T42`, `T6-42`, and `taper` semantics are correct;
- vector sign/radius conventions and inverse zero modes are described where relevant;
- CF metadata and time-preservation behavior are documented;
- links and references resolve;
- public pages contain no agent, roadmap, review, migration, repository-management, or unnecessary meta-historical prose.
