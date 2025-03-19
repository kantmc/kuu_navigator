# noqa: INP001
"""
Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full
list see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
from pathlib import Path
import sys

# sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, str(Path('../..').resolve()))
# settings.configure()

# -- Project information -----------------------------------------------------

project = 'Kuu Navigator'
copyright = '2025 TOYOTA MOTOR CORPORATION'  # noqa: A001
author = 'Kan Torii'

# The full version, including alpha/beta/rc tags
release = 'α0.2'  # noqa: RUF001


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.coverage',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
    # 'sphinx.ext.graphviz',
    'sphinx.ext.inheritance_diagram',
    # 'sphinxcontrib.blockdiag',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns: list[str] = []


# -- autoapi configuration ---------------------------------------------------
extensions.append('autoapi.extension')

autoapi_type = 'python'
autoapi_dirs = ['../../app']
# autoapi_ignore = ['*/test_*.py']

autoapi_options = [
    'members',
    'special-members',
    'private-members',
    # 'undoc-members',
    'show-inheritance-diagram',
    'show-module-summary',
]

autoapi_python_class_content = 'class'


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


# -- Intersphinx configuration -----------------------------------------------
extensions.append('sphinx.ext.intersphinx')

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('http://docs.scipy.org/doc/numpy', None),
    'panda3d': ('https://docs.panda3d.org/1.10', None),
    'scikit-learn': ('https://scikit-learn.org/stable', None),
    'scipy': ('http://docs.scipy.org/doc/scipy/reference', None),
    'pyqoolloop': (
        '../../../pyqoolloop/docs/build/html/',
        '../../pyqoolloop/docs/build/html/objects.inv',
    ),
}

html_use_index=False  # because formatting is corrupt
