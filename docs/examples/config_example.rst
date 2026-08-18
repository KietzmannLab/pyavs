Configuration System
========================

``examples/config_example.py`` demonstrates pyAVS's configuration system
(:class:`~pyavs.config.config.PyAVSConfig`, :class:`~pyavs.config.manager.ConfigManager`):
getting/modifying/validating an analysis configuration, saving and reloading it from a JSON
file, deriving function-specific keyword-argument dictionaries (for
:class:`~pyavs.preprocessing.composer.AVSComposer`, filter computation, population code
saving), and generating the parameter-signature string pyAVS uses to organize derivative
outputs on disk (see :func:`pyavs.utils.derivatives.generate_parameter_signature`).

This is the same configuration object driving the real, config-based analysis script in
:doc:`source_reconstruction_examples`.

.. literalinclude:: ../../examples/config_example.py
   :language: python

See Also
--------

- :doc:`reproduce_analysis` -- reproducing an analysis from a configuration saved alongside
  earlier population codes
