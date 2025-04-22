# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath("../../"))


# -- Project information -----------------------------------------------------

project = "bdi-kit"
copyright = "2024, NYU"
author = "NYU"
master_doc = "index"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "nbsphinx",
    "nbsphinx_link",
    "sphinxemoji.sphinxemoji",
    "sphinx.ext.extlinks",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# These paths are either relative to html_static_path
# or fully qualified paths (eg. https://...)
html_css_files = [
    "css/custom.css",
]

# JavaScript files to be included in the HTML output
html_js_files = [
    ("javascript/readthedocs.js", {"defer": "defer"}),
]

html_show_sourcelink = False

# html_logo = "images/logo.png"
html_theme_options = {
    # "logo_only": True,
    "version_selector": False,
}


# -- Options for autodoc -------------------------------------------------
autodoc_member_order = "bysource"

autoclass_content = "both"

add_module_names = False

autodoc_mock_imports = [
    "sklearn",
    "pandas",
    "numpy",
    "IPython",
    "torch",
    "transformers",
    "matplotlib",
    "openai",
    "polyfuzz",
    "flair",
    "autofj",
    "Levenshtein",
    "valentine",
    "altair",
    "panel",
    "tqdm",
    "rapidfuzz",
]

autodoc_type_aliases = {"MappingSpecLike": "MappingSpecLike"}


# -- Version retrieval -------------------------------------------------
def read_version():
    module_path = os.path.join("../../bdikit/__init__.py")
    with open(module_path) as file:
        for line in file:
            parts = line.strip().split(" ")
            if parts and parts[0] == "__version__":
                return parts[-1].strip("'").strip('"')

    raise KeyError("Version not found in {0}".format(module_path))


version = read_version()
version_link = version

if "dev" in version:
    version_link = "devel"

# Create links pointing to the current version
extlinks = {
    "example": (
        "https://github.com/VIDA-NYU/bdi-kit/blob/" + version_link + "/examples/%s",
        None,
    ),
}
