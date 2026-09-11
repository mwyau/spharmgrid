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
