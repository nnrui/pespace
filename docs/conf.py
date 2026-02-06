# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
from datetime import datetime
from importlib.metadata import version as get_version
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
    "sphinx.ext.napoleon",
]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
    '.ipynb': 'myst-nb',
}
autodoc_typehints = "description"

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Do not execute notebooks during doc build by default
nb_execution_mode = "off"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
