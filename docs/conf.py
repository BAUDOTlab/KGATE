# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'KGATE'
copyright = '2025, Benjamin Loire, Galadriel Brière'
author = 'Benjamin Loire, Galadriel Brière'
release = '0.1.13'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc",
              "sphinx.ext.doctest",
              "sphinx.ext.napoleon",
              "sphinx.ext.apidoc",
              "sphinx.ext.coverage",
              "sphinx.ext.napoleon",
              "sphinx.ext.autosummary",
              "myst_parser"]

myst_enable_extensions = [
   "colon_fence",
    "substitution",
    "replacements",
    "deflist",
    "tasklist",
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autosummary_generate = True  # Enable autosummary to generate pages
autodoc_default_flags = ['members']  # Automatically document class members
epub_show_urls = "footnote"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
