Reproducing a Saved Analysis
=================================

Every time pyAVS saves population codes via :func:`pyavs.save_population_codes_h5`, the
:class:`~pyavs.config.config.PyAVSConfig` used to produce them is saved alongside the data.
``examples/reproduce_analysis_example.py`` demonstrates using that saved configuration to
discover, inspect, and reproduce (or extend) a prior analysis, via
:func:`pyavs.list_available_parameter_sets` and the reproducibility helpers in
:mod:`pyavs.io`.

.. literalinclude:: ../../examples/reproduce_analysis_example.py
   :language: python

See Also
--------

- :doc:`config_example` -- the configuration system this builds on
- :doc:`source_reconstruction_examples` -- an analysis that produces the population codes
  being reproduced here
