import os
import sys
from importlib.metadata import version as _get_version

sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -----------------------------------------------------
project = "saealib"
copyright = "2026, shlka"
author = "shlka"
release = _get_version("saealib")
myst_substitutions = {"version": release}

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxcontrib.mermaid",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- MyST-Parser -------------------------------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "substitution",
    "tasklist",
]

# -- Internationalization -----------------------------------------------------
language = "en"
locale_dirs = ["locale/"]
gettext_compact = False

# -- HTML output -------------------------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#2e7d32",
        "color-brand-content": "#2e7d32",
    },
    "dark_css_variables": {
        "color-brand-primary": "#66bb6a",
        "color-brand-content": "#66bb6a",
    },
    # "light_logo": "logo.svg",
}

# -- autodoc -----------------------------------------------------------------
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
napoleon_numpy_docstring = True
napoleon_google_docstring = False

# -- autosummary -------------------------------------------------------------
autosummary_generate = True
autosummary_imported_members = False

# -- warnings ----------------------------------------------------------------
suppress_warnings = ["toc.no_title"]

# -- sphinx-multiversion -----------------------------------------------------
smv_tag_whitelist = r"^v\d+\.\d+.*$"
smv_branch_whitelist = r"^main$"
smv_remote_whitelist = r"^origin$"
smv_outputdir_format = "{ref.name}"
smv_prefer_remote_refs = False
