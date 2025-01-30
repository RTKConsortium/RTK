# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from datetime import date

import subprocess

# -- Build setup -------------------------------------------------------------
def setup(app):
  # Fetch documentation images
  subprocess.check_call("cmake -P documentation/docs/ExternalData/FetchExternalData.cmake", stderr=subprocess.STDOUT, shell=True)

# -- Project information -----------------------------------------------------
project = 'RTK'
copyright = f'{date.today().year}, RTK Consortium'
author = 'RTK Consortium'

# The full version, including alpha/beta/rc tags
# release = '2.6.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'myst_parser',
    'sphinx_copybutton',
]

myst_enable_extensions = [
    "attrs_inline", # inline image attributes
    "colon_fence",
    "dollarmath",  # Support syntax for inline and block math using `$...$` and `$$...$$`
                   # (see https://myst-parser.readthedocs.io/en/latest/syntax/optional.html#dollar-delimited-math)
    "fieldlist",
    "linkify",  # convert bare links to hyperlinks
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'furo'

# Furo options
html_theme_options = {
    "top_of_page_button": "edit",
    "source_repository": "https://github.com/RTKConsortium/RTK/",
    "source_branch": "master",
    "source_directory": "",
}

# Add any paths that contain custom static files (such as style sheets or icons)
# here, relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []
html_logo = "https://www.openrtk.org/opensourcelogos/rtk75.png"
html_title = f"{project}'s documentation"
html_favicon = "https://www.openrtk.org/RTK/img/rtk_favicon.ico"

# -- Master document -------------------------------------------------
master_doc = 'index'
