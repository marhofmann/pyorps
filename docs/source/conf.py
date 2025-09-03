import os
import sys
from pathlib import Path

# Add paths to ensure modules can be found
project_root = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'pyorps'))

print(f"Python path includes: {sys.path[:3]}")

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
autosummary_generate_overwrite = True
autosummary_imported_members = True

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
    'imported-members': True,
    'show-inheritance': True,
    'inherited-members': True,
}

autodoc_typehints = 'description'
autodoc_type_aliases = {}

# Start with empty mock imports
autodoc_mock_imports = []

# Optional dependencies to check
optional_deps = {
    'rustworkx': 'rustworkx',
    'igraph': 'igraph',
    'networkx': 'networkx',
    'networkit': 'networkit',
}

# Check each optional dependency
for import_name, module_name in optional_deps.items():
    try:
        __import__(import_name)
        print(f"✓ {module_name} available")
    except ImportError:
        print(f"✗ {module_name} not available - will be mocked")
        autodoc_mock_imports.append(import_name)

# Detect if on ReadTheDocs
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'

if on_rtd:
    import subprocess

    try:
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
    except Exception as e:
        print(f"Warning: Could not build Cython extensions: {e}")


# Try to discover all modules
def discover_modules():
    """Discover all modules in the package."""
    try:
        import pyorps
        import pkgutil

        discovered = []
        for importer, modname, ispkg in pkgutil.walk_packages(
                path=pyorps.__path__,
                prefix=pyorps.__name__ + '.',
                onerror=lambda x: None
        ):
            if 'test' not in modname:
                discovered.append(modname)
                try:
                    __import__(modname)
                    print(f"✓ Imported {modname}")
                except ImportError as e:
                    print(f"✗ Could not import {modname}: {e}")

        return discovered
    except ImportError:
        print("Warning: Could not import pyorps for module discovery")
        return []


# Discover modules at configuration time
discovered_modules = discover_modules()

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
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
suppress_warnings = ['autodoc.import_error']


# Custom setup
def setup(app):
    """Ensure all modules are available for documentation."""
    print(f"Discovered {len(discovered_modules)} modules for documentation")
    print(f"Mock imports: {autodoc_mock_imports}")
