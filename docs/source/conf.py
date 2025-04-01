# -*- coding: utf-8 -*-
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

sys.path.insert(0, os.path.abspath("../../"))  # Path to your project root


project = "CellViT++ Inference"
copyright = "2025, Fabian Hörst, University Hospital Essen (AöR), Essen, Germany"
author = "Fabian Hörst"
release = "1.0.0b"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx_material",
    "sphinx_design",
    "sphinx_collapse",
    "notfound.extension",
    "sphinx_copybutton",
]
autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "inherited-members": True,
    "show-inheritance": True,
}
autodoc_mock_imports = ["cupy"]

# Set theme
html_theme = "sphinx_material"

# Material theme options
html_theme_options = {
    # Purple colors
    "color_primary": "purple",
    "color_accent": "deep-purple",
    # Other theme options
    "base_url": "https://tio-ikim.github.io/CellViT-Inference",
    "repo_url": "https://github.com/TIO-IKIM/CellViT",
    "repo_name": "CellViT/CellViT++ Inference",
    "nav_title": "CellViT/CellViT++ Inference Documentation",
    "logo_icon": "&#xe88a",
    "globaltoc_depth": 2,
    "globaltoc_collapse": True,
    "globaltoc_includehidden": True,
}

html_static_path = ["_static"]

html_sidebars = {
    "**": ["logo-text.html", "globaltoc.html", "localtoc.html", "searchbox.html"]
}

html_css_files = [
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css",
    "custom.css",
]

html_favicon = "favicon.ico"

suppress_warnings = ["autodoc.import_object", "config.cache", "ref.ref"]
intersphinx_mapping = {
    "torch": ("https://pytorch.org/docs/stable/", None),
}
