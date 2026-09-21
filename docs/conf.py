"""Sphinx configuration for spharmgrid."""

from __future__ import annotations

import inspect
from datetime import UTC, datetime

from sphinx.application import Sphinx

project = "spharmgrid"
author = "Albert Yau"
copyright = f"{datetime.now(UTC).year}, Albert Yau"

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
    return str(class_signature), None


def setup(app: Sphinx) -> None:
    """Register local Sphinx hooks."""
    app.connect("autodoc-process-signature", _torch_module_signature)
