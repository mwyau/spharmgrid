"""Sphinx configuration for spharmgrid."""

from __future__ import annotations

from datetime import UTC, datetime

project = "spharmgrid"
author = "Albert Yau"
copyright = f"{datetime.now(UTC).year}, Albert Yau"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["dark.css"]
html_js_files = ["theme.js"]
html_context = {
    "display_github": True,
    "github_user": "mwyau",
    "github_repo": "spharmgrid",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

myst_heading_anchors = 3
myst_enable_extensions = ["dollarmath"]
myst_fence_as_directive = ["math"]
