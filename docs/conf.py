"""Sphinx configuration for spharmgrid."""

from __future__ import annotations

import inspect
from datetime import UTC, datetime

from docutils import nodes
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.util.inspect import stringify_signature

project = "spharmgrid"
author = "Albert M. W. Yau"
copyright = f"{datetime.now(UTC).year}, Albert M. W. Yau"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

autodoc_mock_imports = ["jax", "s2fft", "torch", "torch_harmonics"]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_title = project
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/mwyau/spharmgrid",
            "icon": "fa-brands fa-github",
        }
    ],
    "use_edit_page_button": True,
}
html_static_path = ["_static"]
html_css_files = ["layout.css"]
html_sidebars = {
    "changelog": [],
}
html_context = {
    "github_user": "mwyau",
    "github_repo": "spharmgrid",
    "github_version": "main",
    "doc_path": "docs/",
}

myst_heading_anchors = 3
myst_enable_extensions = ["dollarmath"]
myst_fence_as_directive = ["math"]


_ACCESSOR_NAMES = {
    "spharmgrid._accessors.DataArrayAccessor": "DataArray.sg",
    "spharmgrid._accessors.DatasetAccessor": "Dataset.sg",
    "spharmgrid.jax._accessors.DataArrayAccessor": "DataArray.sgj",
    "spharmgrid.jax._accessors.DatasetAccessor": "Dataset.sgj",
}


def _accessor_names(app: Sphinx, doctree: nodes.document) -> None:
    """Show accessor methods under their public Xarray names."""
    del app
    for signature in doctree.findall(addnodes.desc_signature):
        fullname = signature.get("fullname", "")
        module = signature.get("module")
        if f"{module}.{fullname}" in _ACCESSOR_NAMES:
            signature["_toc_name"] = ""
            continue

        owner, _, member = fullname.rpartition(".")
        public = _ACCESSOR_NAMES.get(f"{module}.{owner}")
        if public:
            prefix = next(iter(signature.findall(addnodes.desc_addname)), None)
            if prefix is not None:
                prefix.children[:] = [nodes.Text(f"{public}.")]
            else:
                name = next(iter(signature.findall(addnodes.desc_name)), None)
                if name is not None:
                    signature.insert(
                        signature.index(name),
                        addnodes.desc_addname("", f"{public}."),
                    )

            toc_name = signature.get("_toc_name", "")
            suffix = "()" if toc_name.endswith("()") else ""
            signature["_toc_name"] = f"{public}.{member}{suffix}"


def _torch_module_signature(
    app: Sphinx,
    obj_type: str,
    name: str,
    obj: object,
    options: object,
    signature: str | None,
    return_annotation: str | None,
) -> tuple[str, None] | None:
    """Use each mocked PyTorch module's real constructor signature."""
    del app, options, signature, return_annotation
    if obj_type != "class" or not name.startswith("spharmgrid.torch.nn."):
        return None

    init = obj.__dict__.get("__init__")
    if init is None:
        return None

    try:
        init_signature = inspect.signature(init, eval_str=True)
    except (NameError, TypeError, ValueError):
        return None

    parameters = list(init_signature.parameters.values())[1:]
    class_signature = init_signature.replace(
        parameters=parameters,
        return_annotation=inspect.Signature.empty,
    )
    return stringify_signature(class_signature, unqualified_typehints=True), None


def setup(app: Sphinx) -> None:
    """Register local Sphinx hooks."""
    app.connect("autodoc-process-signature", _torch_module_signature)
    app.connect("doctree-read", _accessor_names, priority=400)
