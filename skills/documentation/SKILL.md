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

Published documentation should read as documentation for a researcher or user, not as instructions to an agent, a design review, or a record of implementation decisions.

- Start with the scientific or technical subject.
- State current behavior directly.
- Keep constraints only when users need them to interpret results or avoid an invalid call.
- Put implementation plans, rejected alternatives, future architecture, and repository-management rules in `PLAN.md`, `AGENTS.md`, or skills.
- Do not preserve migration narratives or implementation history in user-facing prose unless provenance itself is relevant, such as the short PyStormTracker lineage note in the references.
- Do not list absent features merely to define scope. Document supported behavior instead.
- Avoid routing or meta-language such as `source of truth`, `owning page`, `this page documents`, `the API is designed to`, `the implementation intentionally`, or explanations of why the documentation is organized a certain way.
- Avoid contract narration when an example is clearer. Do not say that two APIs are equivalent, share the same path, or delegate to the same kernel unless that fact matters to a user-facing guarantee.
- Avoid defensive constructions such as `rather than silently`, `does not guess`, `does not reinterpret`, `does not implement a separate path`, or `not a claim that ...` unless the contrast is required to explain an actual error condition or scientific interpretation.
- Avoid project-management wording such as `initial scope`, `current phase`, `future work`, `planned`, `out of scope`, and `roadmap` in normal user documentation.
- Avoid machine-facing wording such as `pipeline`, `framework`, `architecture`, `backend abstraction`, or `contract` when ordinary scientific/software wording is more precise. Keep these terms when they name actual software objects or established technical concepts.
- Remove filler adverbs and minimizers that add no meaning, including `simply`, `merely`, `deliberately`, and `intentionally`.
- Avoid promotional wording such as `seamless`, `comprehensive`, `sophisticated`, `modern`, `clean`, `robust`, or `high-performance` unless supported by a specific property or measurement.

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

State software/scientific lineage plainly:

- `ducc0` supplies the numerical SHT engine;
- NCL/SPHEREPACK provide established atmospheric operation semantics and parity references;
- Sardeshmukh and Hoskins (1984) defines the supported spectral taper form;
- PyStormTracker is the source of the earlier DUCC0 wrapper from which spharmgrid grew.

Do not imply that spharmgrid contains SPHEREPACK or implements a new transform engine.

Verify literature-derived claims against primary sources and bibliographic metadata against a publisher or other authoritative source when practical. Make DOI and paper links clickable.

## Equations

Repository Markdown should also render usefully on GitHub.

- Use inline `$...$` for short notation.
- Use GitHub-compatible fenced `math` blocks for standalone equations when practical.
- Keep equations consistent with the implementation. A rendering cleanup must not change a sign, normalization, exponent, radius factor, or index range.

## Comments and docstrings

Comments/docstrings should explain non-obvious transform conventions, scientific lineage, coordinate handling, or constraints. Do not narrate obvious code, preserve migration history, or justify implementation structure to the reader.

## Synchronization

When a public behavior changes, update the owning implementation/docstring, API reference, and relevant user guide page in the same bounded change. Update README examples only when needed.

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
- public pages contain no agent, roadmap, review, migration, or repository-management prose.
