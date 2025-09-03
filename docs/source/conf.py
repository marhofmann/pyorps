import os
import sys
from pathlib import Path

# Add paths to ensure modules can be found
project_root = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(project_root))

print(f"Python path includes: {sys.path[:3]}")
print(f"Current working directory: {os.getcwd()}")

# -- Project information -----------------------------------------------------
project = 'PYORPS'
copyright = '2025, Martin Hofmann'
author = 'Martin Hofmann'
release = '0.2.1'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx_autodoc_typehints',
    'myst_parser',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
]

# Autosummary configuration
autosummary_generate = True
autosummary_imported_members = False  # Avoid duplicates

myst_enable_extensions = [
    "colon_fence",
    "linkify",
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# Autodoc configuration
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'private-members': False,
    'special-members': '__init__',
    'show-inheritance': True,
    'inherited-members': False,  # Avoid duplicates
    'member-order': 'bysource',
}

autodoc_typehints = 'description'
autodoc_inherit_docstrings = False  # Prevent matplotlib docstring inheritance issues

# Mock imports for optional dependencies
autodoc_mock_imports = []

# Check and mock optional dependencies
optional_libs = ['rustworkx', 'igraph', 'networkx', 'networkit']

for lib in optional_libs:
    try:
        __import__(lib)
        print(f"✓ {lib} available")
    except ImportError:
        autodoc_mock_imports.append(lib)
        print(f"✗ {lib} not available - will be mocked")

# Detect if on ReadTheDocs
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'

if on_rtd:
    print("Building on ReadTheDocs")
    import subprocess

    try:
        result = subprocess.run(
            ['python', 'setup.py', 'build_ext', '--inplace'],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=300
        )
        if result.returncode == 0:
            print("✓ Cython extensions built successfully")
        else:
            print(f"✗ Failed to build Cython extensions: {result.stderr}")
    except Exception as e:
        print(f"✗ Could not build Cython extensions: {e}")

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

# Intersphinx mapping - Include matplotlib for role definitions
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'geopandas': ('https://geopandas.org/en/stable/', None),
}

templates_path = ['_templates']
exclude_patterns = []

# HTML output configuration
html_theme = "sphinx_rtd_theme"

html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "includehidden": True,
    "titles_only": False,
}

# Create _static directory if needed
static_dir = Path(__file__).parent / '_static'
static_dir.mkdir(exist_ok=True)
html_static_path = ['_static'] if static_dir.exists() else []

# Suppress specific warnings
suppress_warnings = ['autodoc.import_error', 'ref.footnote']
nitpicky = False  # Disable nitpicky mode to avoid matplotlib role warnings


# Custom setup to handle matplotlib roles
def setup(app):
    """Custom setup for documentation."""
    # Register matplotlib-specific roles if they're not available
    from docutils.parsers.rst import roles

    # Define dummy role functions for matplotlib-specific roles
    def dummy_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
        """Dummy role that just returns the text as a literal."""
        from docutils import nodes
        return [nodes.literal(text=text)], []

    # Register matplotlib-specific roles
    for role_name in ['mpltype', 'rc']:
        if role_name not in roles._roles:
            roles.register_local_role(role_name, dummy_role)

    print("Registered matplotlib-specific roles")
