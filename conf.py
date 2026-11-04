# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from datetime import date

import importlib.util
import subprocess
import os
import sys
import types

# Directory of this file, i.e., the documentation source directory, so that the
# build does not depend on the directory sphinx-build is run from
_conf_dir = os.path.dirname(os.path.abspath(__file__))


# -- Lightweight itk.RTK for sphinx-argparse ---------------------------------
# The application parsers only need the pure Python argument parser and option
# groups of applications/. Load them from the source tree into a stub itk
# package so that the documented options match the documented commit without
# requiring an RTK wheel.
def _install_itk_stub():
    itk = types.ModuleType("itk")
    itk.__path__ = []
    sys.modules["itk"] = itk

    # Only used for the --version output, which is not documented
    rtk_config = types.ModuleType("itk.rtkConfig")
    rtk_config.RTK_GLOBAL_VERSION_STRING = ""
    itk.rtkConfig = sys.modules["itk.rtkConfig"] = rtk_config

    rtk = types.ModuleType("itk.RTK")
    itk.RTK = sys.modules["itk.RTK"] = rtk

    # Same submodules as wrapping/__init_rtk__.py, except rtkExtras which
    # needs the compiled module
    for mod_name in [
        "rtkargumentparser",
        "rtkinputprojections_group",
        "rtk3Doutputimage_group",
        "rtk4Doutputimage_group",
        "rtkprojectors_group",
        "rtkiterations_group",
        "rtknoise_group",
    ]:
        spec = importlib.util.spec_from_file_location(
            f"itk.{mod_name}", os.path.join(_conf_dir, "applications", f"{mod_name}.py")
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        for a in dir(mod):
            if a[0] != "_":
                setattr(rtk, a, getattr(mod, a))


_install_itk_stub()


# -- Build setup -------------------------------------------------------------
def setup(app):
    # Fetch documentation images
    subprocess.check_call(
        f"cmake -DRTK_SOURCE_DIR:PATH={_conf_dir}"
        f"      -DRTK_DOC_OUTPUT_DIR:PATH={_conf_dir}"
        f"      -P {_conf_dir}/documentation/docs/copy_and_fetch_sphinx_doc_files.cmake",
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
    "sphinxarg.ext",
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
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_logo = "https://www.openrtk.org/opensourcelogos/rtk75.png"
html_title = f"{project}'s documentation"
html_favicon = "https://www.openrtk.org/RTK/img/rtk_favicon.ico"

# -- Master document -------------------------------------------------
master_doc = "index"
