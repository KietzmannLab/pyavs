Source Reconstruction and Population Codes
=============================================

This tutorial explains the concepts behind pyAVS's source reconstruction pipeline and
where to find the runnable code for each step. For copy-pasteable code, see
:doc:`../examples/source_reconstruction_examples` -- this page focuses on *why* each step
exists and how the pieces fit together.

Why Beamforming?
-------------------

pyAVS's source reconstruction is built around **LCMV beamforming** (Linearly Constrained
Minimum Variance): a spatial filter, computed per subject/session from a forward model and
a data/noise covariance estimate, that projects sensor-space MEG data onto estimated
cortical source activity. This is the same family of method used for the source-projected
fixation ERFs described in :doc:`../analyses/index` and in the Methods section
(:doc:`../methods/source_reconstruction`).

The pipeline has three stages:

1. **Forward modeling** -- :func:`pyavs.create_bem_model`, :func:`pyavs.create_source_space`,
   :func:`pyavs.setup_coregistration`, :func:`pyavs.load_forward_model` combine a subject's
   structural MRI (FreeSurfer reconstruction) and MEG sensor geometry into a forward solution:
   how activity at each cortical location would project onto each sensor.
2. **Filter computation** -- :func:`pyavs.compute_beamformer_filters` (or, for filters shared
   across a session, :func:`~pyavs.source.filters.compute_per_session_lcmv_filters`) inverts
   that relationship given a data covariance estimate, producing a spatial filter per source
   location.
3. **Application and summarization** -- the filters are applied to epoched data
   (:func:`pyavs.apply_source_reconstruction` or
   :func:`~pyavs.source.reconstruction.apply_beamformer`) to get per-epoch source-space data,
   which can then be summarized within regions of interest
   (:func:`pyavs.extract_roi_data`) and averaged into condition-level population codes
   (:func:`pyavs.compute_population_codes`).

Why Per-Session Filters?
----------------------------

Because AVS sessions are recorded on different days (median 2 days apart, up to 93 days --
see :doc:`../methods/participants`), head position relative to the MEG sensors drifts between
sessions despite the individualized head-stabilizing casts. Computing beamformer filters
independently per session, but from a **shared cross-session data covariance** estimate
(:func:`~pyavs.source.filters.compute_cross_session_data_covariance`), keeps the source
estimates comparable across sessions without needing millimeter-perfect repositioning. See
:doc:`../examples/compute_cross_session_filters` for the standalone script that computes
this, and :doc:`../examples/source_reconstruction_examples` for how the filters are then
applied within the full population-code pipeline.

From Source Data to Population Codes
----------------------------------------

Raw source-reconstructed data has shape ``(n_epochs, n_sources, n_times)`` -- one value per
cortical source location, per timepoint, per epoch. Two further steps make this usable for
encoding/RSA-style analyses:

- **ROI extraction** (:func:`pyavs.extract_roi_data`) reduces the source dimension by
  averaging (or otherwise summarizing) within named regions -- e.g. early visual cortex,
  via the Glasser atlas (:func:`pyavs.get_glasser_roi_labels`) or the FreeSurfer-based
  scheme (:func:`pyavs.get_roi_labels`).
- **Population codes** (:func:`pyavs.compute_population_codes`) further average within a
  time window and group epochs by experimental condition (e.g. fixated object category),
  producing the condition-level representations used in analyses like the dRSA pipeline
  in :doc:`../analyses/index`.

Results are saved via :func:`pyavs.save_population_codes_h5` in a standardized HDF5 format,
and can be rediscovered later with :func:`pyavs.find_population_codes_files` and
:func:`pyavs.list_available_parameter_sets` -- see :doc:`reproduce_analysis
<../examples/reproduce_analysis>` for reproducing an analysis from a saved configuration.

Next Steps
-------------

- :doc:`../examples/source_reconstruction_examples` for the full, runnable code
- :doc:`../api/source` for the complete forward-modeling and beamforming API
- :doc:`../methods/source_reconstruction` for the acquisition/reconstruction parameters used
  in the AVS dataset paper itself
