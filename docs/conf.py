# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html
from __future__ import annotations

import importlib.metadata
import sys
from pathlib import Path

# Make ase_biaspot importable from the src layout when running sphinx-build
# directly (without `pip install -e .`).  In CI the package is installed, so
# this line is a no-op there.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ---------------------------------------------------------------------------
# Project information
# ---------------------------------------------------------------------------
project   = "ASE_Biaspot"
author    = "ss0832"
copyright = "2026, ss0832"  # noqa: A001

try:
    release = importlib.metadata.version("ase-biaspot")
except importlib.metadata.PackageNotFoundError:
    release = "unknown"

version = ".".join(release.split(".")[:2])

# ---------------------------------------------------------------------------
# General configuration
# ---------------------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",       # Core: pulls docstrings into RST pages
    "sphinx.ext.autosummary",   # Summary tables (no :toctree: = no stub generation)
    "sphinx.ext.napoleon",      # NumPy / Google docstring styles
    "sphinx.ext.viewcode",      # [source] links in API pages
    "sphinx.ext.intersphinx",   # Cross-links to NumPy, Python, ASE docs
    "myst_parser",              # Markdown (.md) support for narrative pages
]

# Accept both .rst and .md source files.
# RST is used for API reference pages (native autodoc directives).
# Markdown (MyST) is used for narrative/guide pages (easier to write).
source_suffix = {
    ".rst": "restructuredtext",
    ".md":  "markdown",
}

templates_path   = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# ---------------------------------------------------------------------------
# Autodoc
# ---------------------------------------------------------------------------
autodoc_default_options = {
    "members":          True,
    "undoc-members":    False,
    "show-inheritance": True,
    "special-members":  "__iter__, __len__, __repr__",
}

# autosummary_generate=False: stub .rst files are NOT auto-generated.
# Detailed API pages are maintained manually in docs/api/*.rst using
# .. autoclass:: / .. autofunction:: directives. The autosummary tables in
# docs/api/index.rst serve only as a navigation overview (no :toctree:),
# so no duplicate descriptions are created.
autosummary_generate = False

napoleon_numpy_docstring  = True
napoleon_google_docstring = False

suppress_warnings = [
    "ref.duplicate",
    "myst.header",
    # intersphinx can fail in offline or proxy-restricted environments.
    # The cross-links are a nice-to-have; a network failure must not break CI.
    "intersphinx.fetch_inventory",
]

# ---------------------------------------------------------------------------
# Intersphinx
# ---------------------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3",           None),
    "numpy":  ("https://numpy.org/doc/stable",        None),
    "ase":    ("https://wiki.fysik.dtu.dk/ase/",      None),
}

# ---------------------------------------------------------------------------
# HTML output — sphinx-book-theme
# ---------------------------------------------------------------------------
html_theme = "sphinx_book_theme"

html_theme_options = {
    "repository_url":        "https://github.com/ss0832/ASE_Biaspot",
    "use_repository_button": True,
    "use_issues_button":     True,
    "use_download_button":   True,
    "show_navbar_depth":     2,
    "navigation_with_keys":  True,
    "logo": {
        "text": "ASE_Biaspot",
        "alt":  "ASE_Biaspot",
    },
}

html_title = f"ASE_Biaspot {version}"
# html_static_path is intentionally omitted — no custom static assets exist yet.
# Add it back as ["_static"] if CSS overrides or custom JS are needed later.

# ---------------------------------------------------------------------------
# MyST (Markdown) extensions
# ---------------------------------------------------------------------------
myst_enable_extensions = [
    "colon_fence",   # ::: directive syntax
    "deflist",       # definition lists
    "dollarmath",    # $...$ inline math
]
