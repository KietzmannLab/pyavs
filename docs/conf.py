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
    'sphinx_design',
    'sphinx_copybutton',
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
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
html_css_files = ['custom.css']
html_logo = '_static/klab-logo.png'
html_favicon = '_static/favicon.ico'

# Theme options
html_theme_options = {
    'sidebar_hide_name': False,
    'announcement': (
        '<a href="https://www.kietzmannlab.uni-osnabrueck.de/">'
        '&larr; Kietzmann Lab</a> &middot; pyAVS documentation'
    ),
    # Colors matched to the live kietzmannlab.uni-osnabrueck.de/avs/ page
    # (--bg/--text/--muted/--border/--link/--link-hover/--accent tokens in its
    # BaseLayout CSS bundle, checked 2026-08-17). That page has no dark variant;
    # dark_css_variables below is our own extrapolation from the same accent hues,
    # not sourced from the live site.
    'light_css_variables': {
        'color-brand-primary': '#245b7a',
        'color-brand-content': '#245b7a',
        'color-background-primary': '#fbfaf7',
        'color-background-secondary': '#f3f0ea',
        'color-background-hover': '#e7f0f4',
        'color-foreground-primary': '#172033',
        'color-foreground-secondary': '#647084',
        'color-foreground-muted': '#647084',
        'color-foreground-border': '#ddd7cb',
        'color-background-border': '#ddd7cb',
        'color-link': '#245b7a',
        'color-link--hover': '#a3432f',
    },
    'dark_css_variables': {
        'color-brand-primary': '#6fa8c4',
        'color-brand-content': '#6fa8c4',
        'color-background-primary': '#141a26',
        'color-background-secondary': '#1b2230',
        'color-background-hover': '#232b3c',
        'color-foreground-primary': '#e7e9ee',
        'color-foreground-secondary': '#a8b0bf',
        'color-foreground-muted': '#a8b0bf',
        'color-foreground-border': '#2e3646',
        'color-background-border': '#2e3646',
        'color-link': '#6fa8c4',
        'color-link--hover': '#d98a72',
    },
    'footer_icons': [
        {
            'name': 'GitHub',
            'url': 'https://github.com/KietzmannLab/pyavs',
            'html': (
                '<svg stroke="currentColor" fill="currentColor" stroke-width="0" '
                'viewBox="0 0 16 16"><path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 '
                '8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-'
                '2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 '
                '1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-'
                '.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82'
                '.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 '
                '1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 '
                '1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-'
                '3.58-8-8-8z" clip-rule="evenodd" fill-rule="evenodd"></path></svg>'
            ),
            'class': '',
        },
    ],
}

# HTML context
html_context = {
    'display_github': True,
    'github_user': 'KietzmannLab',
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
# matplotlib/PIL/h5py are genuinely installed via docs/requirements.txt, so they
# are intentionally NOT mocked here (mocking a package that's actually available
# would throw away real type hints/docstrings in the rendered API docs).
autodoc_mock_imports = [
    'mne',
    'sklearn',
    'cv2',
    'pycocotools',
    'torch',
]