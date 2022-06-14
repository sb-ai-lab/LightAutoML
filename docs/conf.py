# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import datetime

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys


CURR_PATH = os.path.abspath(os.path.dirname(__file__))
LIB_PATH = os.path.join(CURR_PATH, os.path.pardir)
sys.path.insert(0, LIB_PATH)

project = "LightAutoML"
copyright = "%s, Sber AI Lab" % str(datetime.datetime.now().year)
author = "Sber AI Lab"

os.environ["DOCUMENTATION_ENV"] = "True"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",  # will be used for tables
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",  # structure
    "sphinx.ext.viewcode",  # for [source] button
    "nbsphinx",
    # 'nbsphinx_link',
    "sphinx_autodoc_typehints",
]

exclude_patterns = [
    "_build/*",
]

# Delete external references
autosummary_mock_imports = [
    "numpy",
    "pandas",
    "catboost",
    "scipy",
    "sklearn",
    "torch",
    "lightgbm",
    "networkx",
    "holidays",
    "joblib",
    "yaml",
    "gensim",
    "optuna",
    "PIL",
    "cv2",
    "albumentations",
    "efficientnet_pytorch",
    "tqdm",
    "nltk",
    "transformers",
    "autowoe",
    "matplotlib",
    "seaborn",
    "json2html",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# code style
pygments_style = "default"

nbsphinx_execute = "never"

# autodoc
# function names that will not be included in documentation
EXCLUDED_MEMBERS = ",".join(
    [
        "get_own_record_history_wrapper",
        "get_record_history_wrapper",
        "record_history_omit",
        "record_history_only",
    ]
)

autodoc_default_options = {
    "ignore-module-all": True,
    "show-inheritance": True,
    "exclude-members": EXCLUDED_MEMBERS,
}

# order of members in docs, usefully for methods in class
autodoc_member_order = "bysource"

# typing, use in signature
autodoc_typehints = "none"

# to omit some __init__ methods in classes where it not defined
autoclass_content = "class"

# all warnings will be produced as errors
autodoc_warningiserror = True

# when there is a link to function not use parentheses
add_function_parentheses = False

# napoleon
# in this docs google docstring format used
napoleon_google_docstring = True
napoleon_numpy_docstring = False

napoleon_include_init_with_doc = True

# to omit private members
napoleon_include_private_with_doc = False

# use spectial members
napoleon_include_special_with_doc = False

napoleon_use_param = True

# True to use a :keyword: role for each function keyword argument
napoleon_use_keyword = True

# True to use the .. admonition:: directive for References sections instead .. rubric::
napoleon_use_admonition_for_examples = True

# Autosummary true if you want to generate it from very beginning
autosummary_generate = True

set_type_checking_flag = True

always_document_param_types = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "PIL": ("https://pillow.readthedocs.io/en/stable/", None),
}

autodoc_type_aliases = {
    "RoleType": "lightautoml.dataset.roles.ColumnRole",
    "NpDataset": "lightautoml.text.utils.NpDataset",
}


def skip_member(app, what, name, obj, skip, options):
    if obj.__doc__ is None:
        return True
    return None


def setup(app):
    app.add_css_file("style.css")  # customizing default theme
    app.connect("autodoc-skip-member", skip_member)
