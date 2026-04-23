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

extensions = ["autodoc2",
              "sphinx.ext.apidoc", # yeet
              "sphinx.ext.autosummary", # yeet?
              "sphinx.ext.coverage",
              "sphinx.ext.doctest",
              "sphinx.ext.mathjax",
              "sphinx.ext.napoleon",
              "myst_parser"]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "replacements",
    "substitution",
    "tasklist",
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


autodoc2_packages = [
    {
        "path": "../src/kgate",
        "auto_mode": True,  # Enable manual mode, to manually specify which objects to document
    },
]
autodoc2_render_plugin = "myst" # Create all files with the “.md” extension, and thus docstrings will be interpreted as MyST by default

autodoc2_docstring_parser_regexes = [
    # this will render all docstrings as Markdown
    (r".*", "docstrings_parser"),
]

 # yeet ??
autosummary_generate = True  # Enable autosummary to generate pages
#autodoc_default_flags = ['members']  # Automatically document class members
epub_show_urls = "footnote"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
