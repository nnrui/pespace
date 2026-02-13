.. pespace documentation master file, created by
   sphinx-quickstart on Sun Jan 25 23:09:32 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. Add your content using ``reStructuredText`` syntax. See the
.. `reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_
.. documentation for details.

.. include:: _build/README_myst.md
   :parser: myst_parser.sphinx_

.. toctree::
   :includehidden:
   :hidden:
   :maxdepth: 2
   :caption: Examples:

   _examples/basic.ipynb
   _examples/conventions_difference.ipynb
   _examples/using_f32.ipynb
   _examples/autodiff_forward.ipynb
   _examples/autodiff_backward.ipynb
   pe_scripts.rst


.. toctree::
   :includehidden:
   :hidden:
   :maxdepth: 2
   :caption: API Reference:

   api.rst



