Full PE scripts
===========================

The scripts on this page are full parameter estimation examples for a massive black hole binary merger.

.. note::

   We have not yet successfully obtained reliable posteriors sampling with ``flowMC`` and ``jaxns`` in our tests.
   The configurations provided in the example scripts should be regarded as preliminary and have not been fully validated.


1. Using the sampler interface of `bilby` and the likelihood computation of `taichi-lang` backend
---------------------------------------------------------------------------------------------------

1.1 A single LISA detector with `PhenomXAS` waveform
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: _examples/pe_scripts/full_pe_ti_lisa_xas.py
   :language: python
   :linenos:
   :caption: full_pe_ti_lisa_xas.py


1.2 The LISA-Taiji-Tianqin network with `PhenomXHM` waveform
-----------------------------------------------------------------

.. literalinclude:: _examples/pe_scripts/full_pe_ti_network_xhm.py
   :language: python
   :linenos:
   :caption: full_pe_ti_network_xhm.py


.. _jax-pe-scripts:

2. Using samplers with `jax`
------------------------------

`jax` is not installed by default, please install dependencies additionally:

.. code-block:: bash

   pip install "pespace[jax]"
   pip install "pespace[samplers]"
   pip install "rippleGW @ git+https://github.com/nnrui/ripple"

2.1 Adaptive tempering SMC with NUTS and Hessian
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: _examples/pe_scripts/full_pe_jax_smc.py
   :language: python
   :linenos:
   :caption: full_pe_jax_smc.py


2.2 jaxns
~~~~~~~~~

.. literalinclude:: _examples/pe_scripts/full_pe_jax_jaxns.py
   :language: python
   :linenos:
   :caption: full_pe_jax_jaxns.py


2.3 flowMC
~~~~~~~~~~

.. literalinclude:: _examples/pe_scripts/full_pe_jax_flowmc.py
   :language: python
   :linenos:
   :caption: full_pe_jax_flowmc.py
