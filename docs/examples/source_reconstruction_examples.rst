Source Reconstruction Examples
==============================

pyAVS provides source reconstruction via LCMV beamforming: forward modeling, per-session
filter computation, ROI-based data extraction, and population code computation for encoding
analyses. This page walks through the two real, runnable example scripts that demonstrate
the full pipeline, plus the lower-level functions they're built from.

Quickstart: Synthetic Data
---------------------------

``examples/simple_source_reconstruction.py`` is the fastest way to see the pipeline run
end to end. It fabricates synthetic MEG epochs and a synthetic forward model, so it needs no
real AVS data and runs standalone:

.. literalinclude:: ../../examples/simple_source_reconstruction.py
   :language: python
   :lines: 1-23

The core reconstruction step is a two-call sequence -- compute beamformer filters, then
apply them:

.. literalinclude:: ../../examples/simple_source_reconstruction.py
   :language: python
   :start-after: Performing source reconstruction
   :end-before: Step 3: Create population codes
   :dedent:

See the full script (``examples/simple_source_reconstruction.py``) for the surrounding
synthetic-data setup and how results are saved via
:func:`pyavs.io.write.save_population_codes_h5`.

Real-Data, Config-Driven Workflow
------------------------------------

``examples/compute_population_codes_example.py`` mirrors the full AVS-machine-room
population-code pipeline against real subject data: MEG loading and ICA via
:class:`~pyavs.preprocessing.composer.AVSComposer`, eye-tracking event epoching, per-session
LCMV filter loading/computation, ROI extraction, and HDF5 storage. It uses the
:class:`~pyavs.config.config.PyAVSConfig` system (see :doc:`config_example`) to drive all
analysis parameters rather than hardcoding them.

The source-level reconstruction step loads or computes per-session LCMV filters and applies
them to the current session's epochs:

.. literalinclude:: ../../examples/compute_population_codes_example.py
   :language: python
   :pyobject: compute_source_population_codes

Note that this uses :func:`~pyavs.source.filters.load_or_compute_lcmv_filters` (which wraps
:func:`~pyavs.source.filters.compute_cross_session_data_covariance` and
:func:`~pyavs.source.filters.compute_per_session_lcmv_filters` -- see
:doc:`compute_cross_session_filters` for computing those filters as a standalone step) and
:func:`~pyavs.source.filters.apply_lcmv_to_epochs`, not the lower-level
:func:`~pyavs.source.reconstruction.apply_beamformer` shown above -- the per-session-filter
route is what the real pipeline uses when filters need to be shared/reused across a session,
while ``apply_beamformer`` is the more direct call for a one-off epochs/forward/filters triple.

ROI-Based Extraction
----------------------

Once you have source-space data (an array of shape ``(n_epochs, n_sources, n_times)``), use
:func:`pyavs.extract_roi_data` to average or select within named regions:

.. code-block:: python

   import pyavs

   # Glasser atlas ROI names for one area category at a time
   # (area is a single string: 'all', 'high_visual', 'early_visual', or 'intermediate_visual')
   visual_rois = pyavs.get_glasser_roi_labels(area='early_visual')

   roi_data = pyavs.extract_roi_data(
       source_data,          # np.ndarray, shape (n_epochs, n_sources, n_times)
       forward_model['src'],
       roi_labels=visual_rois,
       method='mean',         # average activity within each ROI
       verbose=True,
   )

   for roi_name, data in roi_data.items():
       print(f"{roi_name}: {data.shape}")  # (n_epochs, n_times)

Population Codes
------------------

:func:`pyavs.compute_population_codes` turns source-space data plus per-epoch metadata into
condition-averaged population codes:

.. code-block:: python

   population_codes = pyavs.compute_population_codes(
       source_data,             # np.ndarray, shape (n_epochs, n_sources, n_times)
       events_metadata=epochs.metadata,
       conditions=['scene_id'],  # column(s) in metadata defining conditions
       time_window=(0.0, 0.3),
       times=epochs.times,
   )

See Also
--------

- :doc:`../tutorials/source_reconstruction_population_codes` -- narrative tutorial covering
  this same pipeline end to end
- :doc:`../api/source` -- full API documentation for forward modeling, beamforming, and ROI
  extraction
- :doc:`compute_cross_session_filters` -- computing shared LCMV filters across a subject's
  sessions
- :doc:`../package/composer_guide` -- AVSComposer, the recommended entry point for
  MEG + eye-tracking loading used by ``compute_population_codes_example.py``
