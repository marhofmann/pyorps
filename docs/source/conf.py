# Configuration file for the Sphinx documentation builder.

import os
import sys
from datetime import datetime

# -- Path setup --------------------------------------------------------------
# Add the parent directory to the Python path so sphinx can find the modules
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------
project = 'PYORPS'
copyright = f'{datetime.now().year}, Martin Hofmann'
author = 'Martin Hofmann'

# Import version from the package itself
try:
    from pyorps import __version__

    release = __version__
    version = __version__
except ImportError:
    print("Warning: Could not import pyorps.__version__, using fallback")
    release = '0.2.1'
    version = '0.2.1'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here
extensions = [
    'sphinx.ext.autodoc',  # Core autodoc functionality
    'sphinx.ext.autosummary',  # Generate summary tables
    'sphinx.ext.napoleon',  # Support for NumPy and Google style docstrings
    'sphinx.ext.viewcode',  # Add links to source code
    'sphinx.ext.intersphinx',  # Link to other projects' documentation
    'sphinx.ext.coverage',  # Check documentation coverage
    'sphinx.ext.mathjax',  # Render math via JavaScript
    'sphinx.ext.ifconfig',  # Include content based on configuration
    'sphinx.ext.githubpages',  # Create .nojekyll file for GitHub Pages
    'sphinx.ext.todo',  # Support for todo items
]

# Try to add MyST parser if available
try:
    import myst_parser

    extensions.append('myst_parser')

    # Configure MyST - removing linkify as it needs additional package
    myst_enable_extensions = [
        "deflist",
        "tasklist",
        "html_image",
        "colon_fence",
        "smartquotes",
        "replacements",
        "strikethrough",
        # "linkify" removed - requires linkify-it-py package
    ]
    myst_heading_anchors = 3
    myst_fence_as_directive = set()

except ImportError:
    print("Warning: myst_parser not installed. Markdown files will not be parsed.")
    print("Install with: pip install myst-parser")

# Add any paths that contain templates here, relative to this directory
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'tests', 'examples',
                    'case_studies']

# The suffix(es) of source filenames
source_suffix = {
    '.rst': 'restructuredtext',
}

# Add markdown support if MyST is available
try:
    import myst_parser

    source_suffix['.md'] = 'markdown'
except ImportError:
    pass

# The master toctree document
master_doc = 'index'

# -- Options for autodoc -----------------------------------------------------

# Automatically extract typehints when specified
autodoc_typehints = 'description'

# Sort members by source order
autodoc_member_order = 'bysource'

# Include both class and __init__ docstrings
autoclass_content = 'both'

# Include private members (those starting with _)
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__',
    'show-inheritance': True,
    'inherited-members': False,
    'noindex': False,
}

# Mock imports for packages that might not be available during doc build
autodoc_mock_imports = [
    'numba',
    'numpy',
    'pandas',
    'geopandas',
    'shapely',
    'rasterio',
    'scipy',
    'matplotlib',
    'rustworkx',
    'igraph',
    'networkx',
    'networkit',
    'requests',
    'defusedxml',
    'psutil',
    'fiona',
    'pyproj',
    'pandapower',
    'contextily',
    'openpyxl',
    'notebook',
]

# -- Options for autosummary -------------------------------------------------

autosummary_generate = True
autosummary_generate_overwrite = False
autosummary_imported_members = False

# -- Options for napoleon ----------------------------------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# -- Options for intersphinx -------------------------------------------------

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'geopandas': ('https://geopandas.org/en/stable/', None),
    'rasterio': ('https://rasterio.readthedocs.io/en/stable/', None),
    'shapely': ('https://shapely.readthedocs.io/en/stable/', None),
    'networkx': ('https://networkx.org/documentation/stable/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages
try:
    import sphinx_rtd_theme

    html_theme = 'sphinx_rtd_theme'
    html_theme_options = {
        'navigation_depth': 4,
        'collapse_navigation': False,
        'sticky_navigation': True,
        'includehidden': True,
        'titles_only': False,
        'display_version': True,
        'prev_next_buttons_location': 'bottom',
        'style_external_links': True,
    }
except ImportError:
    print("Warning: sphinx_rtd_theme not installed. Using default theme.")
    print("Install with: pip install sphinx-rtd-theme")
    html_theme = 'alabaster'
    html_theme_options = {}

# Add any paths that contain custom static files (such as style sheets)
# Create _static directory if it doesn't exist
static_dir = os.path.join(os.path.dirname(__file__), '_static')
if not os.path.exists(static_dir):
    os.makedirs(static_dir)
html_static_path = ['_static']

# Custom sidebar templates
html_sidebars = {
    '**': [
        'relations.html',
        'searchbox.html',
        'globaltoc.html',
    ]
}

# Output file base name for HTML help builder
htmlhelp_basename = 'PYORPSdoc'

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    'papersize': 'a4paper',
    'pointsize': '10pt',
}

latex_documents = [
    (master_doc, 'PYORPS.tex', 'PYORPS Documentation',
     'Martin Hofmann', 'manual'),
]

# -- Options for manual page output ------------------------------------------

man_pages = [
    (master_doc, 'pyorps', 'PYORPS Documentation',
     [author], 1)
]

# -- Options for Texinfo output ----------------------------------------------

texinfo_documents = [
    (master_doc, 'PYORPS', 'PYORPS Documentation',
     author, 'PYORPS', 'Python for Optimal Routes in Power Systems.',
     'Miscellaneous'),
]

# -- Options for todo extension ----------------------------------------------

todo_include_todos = True

# -- Options for coverage extension ------------------------------------------

coverage_show_missing_items = True

# -- Suppress specific warnings -----------------------------------------------

suppress_warnings = [
    'autodoc.import_object',
    'ref.python',
]


# -- Cython specific configuration -------------------------------------------

def setup(app):
    """Setup function for custom Sphinx configuration."""

    # Check if Cython extensions are available
    try:
        from pyorps.utils import path_algorithms
        print("✓ Cython extensions detected and available for documentation")
    except ImportError:
        print("""
⚠️  Cython extensions not available for documentation.

To include Cython modules in the documentation:
1. Build the extensions first: python setup.py build_ext --inplace
2. Then regenerate the documentation: make clean && make html

Note: Documentation will still be generated without Cython modules.
        """)

    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
