# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
from datetime import datetime
from importlib.metadata import version as get_version
from pathlib import Path
import re


# Convert GitHub-style admonitions in README.md to MyST format 
def replace_admonitions(match):
    admonition_type = match.group(1).lower()
    content = match.group(2)
    content = re.sub(r'^> ', '', content, flags=re.MULTILINE)
    content = content.strip()
    return f'```{{{admonition_type}}}\n{content}\n```\n'

with open('../README.md', 'r', encoding='utf-8') as f:
    content = f.read()

pattern = r'> \[!(WARNING|NOTE|TIP|IMPORTANT|CAUTION)\]\s*\n((?:> .*\n?)*)'
converted = re.sub(pattern, replace_admonitions, content, flags=re.MULTILINE)

Path('_build').mkdir(exist_ok=True)
with open('_build/README_myst.md', 'w', encoding='utf-8') as f:
    f.write(converted)


project = 'pespace'
year = datetime.now().year
copyright = f'{year}, Rui Niu'
author = 'Rui Niu'
version = get_version("pespace")
release = ".".join(version.split('.')[:3])

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    'sphinx.ext.intersphinx',
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
    '.ipynb': 'myst-nb',
}

# Do not execute notebooks during doc build by default
nb_execution_mode = "off"

autodoc_default_options = {
    "members": True,
    "private-members": True,
    "undoc-members": True,
    "member-order": "bysource",
    "show-inheritance": True,
}
autodoc_inherit_docstrings = False
autodoc_typehints = "description"
autodoc_typehints_description_target = "all"
autodoc_typehints_format = "short"
autodoc_member_order = 'bysource'

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
html_last_updated_fmt = '%Y-%m-%d %H:%M:%S'
