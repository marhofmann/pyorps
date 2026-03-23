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
    release = '0.3.2'
    version = '0.3.2'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.githubpages',
    'sphinx.ext.todo',
    'myst_parser',
    'sphinx_copybutton',
    'sphinx_design',
]

# -- MyST configuration ------------------------------------------------------

myst_enable_extensions = [
    "deflist",
    "tasklist",
    "html_image",
    "colon_fence",
    "smartquotes",
    "replacements",
    "strikethrough",
]
myst_heading_anchors = 3
myst_fence_as_directive = set()

# Add any paths that contain templates here, relative to this directory
templates_path = ['_templates']

# List of patterns to ignore when looking for source files
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'tests', 'examples',
                    'case_studies']

# The suffix(es) of source filenames
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# The master toctree document
master_doc = 'index'

# -- Options for autodoc -----------------------------------------------------

autodoc_typehints = 'description'
autodoc_member_order = 'bysource'
autoclass_content = 'both'

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
    'cupy',
    'cugraph',
    'cudf',
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

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'includehidden': True,
    'titles_only': False,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': True,
}

html_logo = '_static/images/pyorps_planning_results_21_targets_22_5deg_1mxm.png'

# Add any paths that contain custom static files
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
    app.add_css_file('css/custom.css')

    # Check if Cython extensions are available
    try:
        from pyorps.utils import path_algorithms
        print("Cython extensions detected and available for documentation")
    except ImportError:
        print("Cython extensions not available. Build with: "
              "python setup.py build_ext --inplace")

    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
