Cross-Session Beamformer Filters
====================================

``examples/compute_cross_session_filters.py`` computes LCMV beamformer filters that stay
consistent across a subject's sessions, using the strategy described in
:doc:`../tutorials/source_reconstruction_population_codes`: a shared data covariance
estimated across sessions, combined with per-session noise covariance from empty-room
recordings.

.. code-block:: bash

   python examples/compute_cross_session_filters.py --subject-id 1 --event-type fixation \
       --sessions 1 2 3 4 5 6 7 8 9 10 --verbose

Full script:

.. literalinclude:: ../../examples/compute_cross_session_filters.py
   :language: python

See Also
--------

- :doc:`source_reconstruction_examples` -- the full pipeline this feeds into
- :doc:`../api/source` -- :func:`~pyavs.source.filters.compute_cross_session_data_covariance`
  and :func:`~pyavs.source.filters.compute_per_session_lcmv_filters` API reference
