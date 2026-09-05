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
- direct API equivalents;
- a link to the full documentation;
- citation/reference guidance when available.

Put detailed material under `docs/`. For the initial implementation, create a complete but compact documentation set covering at least:

```text
docs/index.md
docs/quickstart.md
docs/grids.md
docs/filtering.md
docs/regridding.md
docs/operators.md
docs/kinematics.md
docs/cf.md
docs/cli.md
docs/api.md
docs/references.md
docs/conf.py
```

Combine pages when that makes the result clearer; do not create one page per function just to increase page count.

## Read the Docs

Use Sphinx with MyST Markdown and the Read the Docs theme, following the working pattern in `../PyStormTracker` unless the local package requires a smaller configuration.

Required repository pieces:

```text
.readthedocs.yaml
docs/conf.py
```

Keep documentation dependencies in a separate `docs` dependency group. The hosted docs build should install only what is needed to import spharmgrid and build the docs; do not pull parity-test or heavy optional I/O dependencies into the docs environment without need.

Use a strict local build command such as:

```bash
uv run --group docs sphinx-build -W --keep-going -b html docs docs/_build/html
```

or the equivalent command supported by the final uv configuration. The exact documented command must match the repository configuration.

Do not add Mermaid, Node, PDF builds, or other documentation tooling unless the documentation actually uses it.

## Human-facing content

Write for atmospheric scientists and Python/xarray users.

Start pages with the scientific or technical subject, not with repository-management language. Explain:

- what the operation computes;
- supported grid assumptions;
- units and radius conventions when relevant;
- coordinate and metadata behavior;
- any zero-mode, truncation, or sign convention users need to interpret output;
- a short runnable example.

Do not write user documentation as instructions to an agent. `PLAN.md` and skills are the place for implementation instructions.

## API documentation

Document both interfaces:

```python
field.sg.filter(...)
sg.filter(field, ...)
```

Use the accessor form first because it is the main interface, then show the direct equivalent where useful.

Every public function/class in the top-level namespace must appear in `docs/api.md` or another clearly linked API reference page. Use Sphinx autodoc where it reduces duplication, but keep important scientific semantics in maintained prose/docstrings rather than relying on generated signatures alone.

Document canonical variable names and CF semantics for:

```text
u, v, vo, d, strf, vp
```

Explain automatic discovery order and ambiguity errors.

## Scientific lineage and references

State the roles clearly:

- spharmgrid provides the xarray/atmospheric operations layer;
- `ducc0` provides the numerical SHT engine;
- NCL/SPHEREPACK provide established atmospheric operation semantics and parity references;
- Sardeshmukh and Hoskins (1984) defines the supported spectral taper form;
- PyStormTracker is the source implementation from which the initial wrapper is extracted/generalized.

Do not imply that spharmgrid contains SPHEREPACK or implements a new transform engine.

Verify literature-derived claims against primary sources and bibliographic metadata against a publisher or other authoritative source when practical.

Make DOI and paper links clickable. Prefer descriptive Markdown links over bare URLs.

## Equations

Repository Markdown should also render usefully on GitHub.

- Use inline `$...$` for short notation.
- Use GitHub-compatible fenced `math` blocks for standalone equations when practical.
- Keep equations consistent with the implementation. A rendering cleanup must not change a sign, normalization, exponent, radius factor, or index range.

For example:

````markdown
```math
\nabla^2 Y_{\ell m}
=
-\frac{\ell(\ell+1)}{R^2}Y_{\ell m}.
```
````

## Prose and code comments

Use plain, direct English and established scientific terms.

Avoid filler and promotional language. Prefer concrete statements such as:

- `The input must be a global GL or CC grid.`
- `The l=0 coefficient is set to zero for the inverse Laplacian.`
- `ducc0 performs the spin-1 synthesis.`

Comments/docstrings should explain non-obvious transform conventions, scientific lineage, coordinate handling, or constraints. Do not narrate obvious code or preserve migration history.

## Synchronization

When a public behavior changes, update in the same bounded change:

- the owning implementation/docstring;
- API reference;
- relevant user guide page;
- README example only if it would otherwise become wrong.

Before adding a new page, check whether an existing page already owns the subject. Cross-link rather than duplicate long parameter tables or method explanations.

## Documentation completion check

Before documentation work is complete:

- the Read the Docs configuration is present and coherent;
- the strict local Sphinx build passes;
- every public API is documented;
- quick-start examples use the actual current signatures;
- GL/CC terminology is consistent;
- `T42`, `T6-42`, and `taper` semantics are stated correctly;
- vector sign/radius conventions and inverse zero modes are described where relevant;
- CF metadata and time-preservation behavior are documented;
- links and references resolve;
- no release/version instructions were invented for the GitHub-only stage.
