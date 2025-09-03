import os
import sys

# Add both the project root and the parent directory to sys.path
sys.path.insert(0, os.path.abspath('../..'))  # Project root
sys.path.insert(0, os.path.abspath('..'))  # Parent of docs

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
    'sphinx.ext.intersphinx',
]

# Autosummary configuration - generate stub files for all modules
autosummary_generate = True
autosummary_generate_overwrite = True
autosummary_imported_members = True

# Add modules to be documented explicitly
autosummary_modules = [
    'pyorps',
    'pyorps.core',
    'pyorps.io',
    'pyorps.raster',
    'pyorps.graph',
    'pyorps.graph.api',
    'pyorps.utils',
]

myst_enable_extensions = [
    "colon_fence",  # For ::: fenced code blocks
    "linkify",  # Automatically convert URLs into links
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# Enhanced autodoc options to catch all modules and Cython code
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'private-members': False,
    # Changed to False to avoid cluttering with private methods
    'special-members': '__init__',
    'imported-members': True,
    'show-inheritance': True,
    'inherited-members': True,
}

# Type hints configuration
autodoc_typehints = 'description'
autodoc_type_aliases = {}

# Handle Cython modules and C extensions
# Add any problematic imports that fail on ReadTheDocs
autodoc_mock_imports = []

# Try to import Cython modules, if they fail, mock them
try:
    import pyorps.utils
except ImportError:
    autodoc_mock_imports.append('pyorps.utils')

try:
    import pyorps.graph.api
except ImportError:
    # If the module can't be imported, we'll still try to document it
    pass

# Detect if we're on ReadTheDocs
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'

if on_rtd:
    # On ReadTheDocs, we might need to build Cython extensions differently
    # or mock them if building fails
    import subprocess

    # Try to build Cython extensions
    try:
        # Change to project root
        project_root = os.path.abspath('../..')
        result = subprocess.run(
            ['python', 'setup.py', 'build_ext', '--inplace'],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=300
        )
        if result.returncode != 0:
            print(f"Warning: Failed to build Cython extensions: {result.stderr}")
            # Add Cython modules to mock imports if build fails
            autodoc_mock_imports.extend(['pyorps.utils', 'pyorps.graph._cgraph'])
    except Exception as e:
        print(f"Warning: Could not build Cython extensions: {e}")
        autodoc_mock_imports.extend(['pyorps.utils', 'pyorps.graph._cgraph'])

# Napoleon settings for Google and NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True

# Intersphinx mapping for cross-references
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
}

templates_path = ['_templates']
exclude_patterns = []


# Explicitly add modules that are not imported in __init__.py
# This helps autodoc find them
def setup(app):
    """Custom setup to ensure all modules are documented."""
    import inspect
    import pkgutil

    try:
        import pyorps

        # Walk through all submodules
        for importer, modname, ispkg in pkgutil.walk_packages(
                path=pyorps.__path__,
                prefix=pyorps.__name__ + '.',
                onerror=lambda x: None
        ):
            try:
                # Try to import each module to make it available to autodoc
                __import__(modname)
            except ImportError as e:
                print(f"Could not import {modname}: {e}")
                # Add to mock imports if it fails
                if modname not in autodoc_mock_imports:
                    autodoc_mock_imports.append(modname)
    except ImportError:
        print("Warning: Could not import pyorps package")


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"

# Optional: Customize the sidebar depth
html_theme_options = {
    "navigation_depth": 4,  # Increased to show more nested modules
    "collapse_navigation": False,  # Keeps the sidebar expanded
    "sticky_navigation": True,  # Keeps the sidebar visible while scrolling
    "includehidden": True,
    "titles_only": False,
}

html_static_path = ['_static']


# Add custom CSS if needed for better display
def add_custom_css(app):
    app.add_css_file('custom.css')


# Suppress specific warnings if needed
suppress_warnings = ['autodoc.import_error']
