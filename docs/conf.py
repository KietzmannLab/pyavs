# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'pyAVS'
copyright = '2024, P. Sulewski'
author = 'P. Sulewski'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.githubpages',
    'myst_parser',
]

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_keyword = True
napoleon_custom_sections = None
napoleon_attr_annotations = True

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Autosummary settings
autosummary_generate = True
autosummary_imported_members = True

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Source file extensions
source_suffix = {
    '.rst': None,
    '.md': None,
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Theme options
html_theme_options = {
    'canonical_url': '',
    'analytics_id': '',
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    'style_nav_header_background': '#2980B9',
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

# Custom sidebar
html_sidebars = {
    '**': [
        'relations.html',  # needs 'show_related': True theme option to display
        'searchbox.html',
    ]
}

# HTML context
html_context = {
    'display_github': True,
    'github_user': 'your-org',
    'github_repo': 'pyavs',
    'github_version': 'main',
    'conf_py_path': '/docs/',
}

# -- Options for intersphinx extension ---------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions.html#extension-sphinx-ext-intersphinx

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'mne': ('https://mne.tools/stable/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
}

# -- Options for MyST parser ------------------------------------------------
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]

# -- Mock imports for ReadTheDocs -------------------------------------------
autodoc_mock_imports = [
    'mne',
    'sklearn',
    'matplotlib',
    'PIL',
    'cv2',
    'pycocotools',
    'h5py',
]

# Mock modules that may not be available during doc build
import unittest.mock as mock

MOCK_MODULES = [
    'mne', 'mne.io', 'mne.preprocessing', 'mne.minimum_norm', 'mne.beamformer',
    'mne.transforms', 'mne.bem', 'mne.time_frequency', 'mne.io.constants',
    'sklearn', 'sklearn.preprocessing', 'sklearn.decomposition',
    'matplotlib', 'matplotlib.pyplot',
    'PIL', 'PIL.Image',
    'cv2',
    'pycocotools', 'pycocotools.coco', 'pycocotools.mask',
    'h5py',
]

# for mod_name in MOCK_MODULES:
#     sys.modules[mod_name] = mock.MagicMock()