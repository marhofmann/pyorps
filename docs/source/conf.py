import os
import sys

sys.path.insert(0, os.path.abspath('../pyorps'))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'PYORPS'
copyright = '2025, Martin Hofmann'
author = 'Martin Hofmann'
release = '09.05.2025'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx_autodoc_typehints',
    'myst_parser',
    'sphinx.ext.autosummary',
]

autosummary_generate = True

myst_enable_extensions = [
    "colon_fence",  # For ::: fenced code blocks
    "linkify",      # Automatically convert URLs into links
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'private-members': True,
    'imported-members': True,
}

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"

# Optional: Customize the sidebar depth
html_theme_options = {
    "navigation_depth": 2,  # Controls how many levels of headings are shown
    "collapse_navigation": False,  # Keeps the sidebar expanded
    "sticky_navigation": True,  # Keeps the sidebar visible while scrolling
}

html_static_path = ['_static']
