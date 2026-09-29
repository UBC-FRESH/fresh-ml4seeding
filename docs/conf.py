"""Sphinx configuration for ml4seeding documentation."""

project = "ml4seeding"
copyright = "2026, UBC FRESH Lab"
author = "UBC FRESH Lab"
version = "0.1.0a1"
release = "0.1.0a1"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = []
