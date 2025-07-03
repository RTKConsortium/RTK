# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from datetime import date

import subprocess
import os

# -- Build setup -------------------------------------------------------------
def setup(app):
    # Fetch documentation images
    cwd = os.getcwd()
    subprocess.check_call(
        f"cmake -DRTK_SOURCE_DIR:PATH={cwd}"
        f"      -DRTK_DOC_OUTPUT_DIR:PATH={cwd}"
        "      -P documentation/docs/copy_and_fetch_sphinx_doc_files.cmake",
        stderr=subprocess.STDOUT,
        shell=True,
    )


# -- Project information -----------------------------------------------------
project = "RTK"
copyright = f"{date.today().year}, RTK Consortium"
author = "RTK Consortium"

# The full version, including alpha/beta/rc tags
# release = '2.6.0'

# -- General configuration ---------------------------------------------------
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx.ext.graphviz",
]

myst_enable_extensions = [
    "attrs_inline",  # inline image attributes
    "colon_fence",
    "dollarmath",  # Support syntax for inline and block math using `$...$` and `$$...$$`
    # (see https://myst-parser.readthedocs.io/en/latest/syntax/optional.html#dollar-delimited-math)
    "fieldlist",
    "linkify",  # convert bare links to hyperlinks
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The output format for Graphviz when building HTML files. This must be either 'png' or 'svg'; the default is 'png'.
graphviz_output_format = "svg"

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"

# Furo options
html_theme_options = {
    "top_of_page_button": "edit",
    "source_repository": "https://github.com/RTKConsortium/RTK/",
    "source_branch": "main",
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
master_doc = "index"
