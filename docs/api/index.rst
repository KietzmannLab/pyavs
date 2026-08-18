API Reference
=================

This page indexes the full ``pyavs`` top-level API (everything importable as
``pyavs.<name>``). For richer, docstring-heavy documentation organized by topic, see the
per-submodule pages below. ``AVSComposer`` and ``EyeTrackingPlotter`` are documented in full
on their dedicated pages (:doc:`preprocessing`, :doc:`visualization`) rather than duplicated
below.

Top-Level API
-----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   pyavs.set_data_path
   pyavs.get_data_path
   pyavs.setup_data_directory
   pyavs.configure
   pyavs.check_data_availability
   pyavs.load_eye_events
   pyavs.load_experiment_log
   pyavs.load_anatomical
   pyavs.load_scenes
   pyavs.load_and_enrich_eye_events
   pyavs.add_fixation_sequence_position
   pyavs.add_cross_event_information
   pyavs.load_and_preprocess_eye_tracking
   pyavs.load_meg_raw
   pyavs.load_meg_preprocessed
   pyavs.load_meg_session
   pyavs.load_and_preprocess_meg_run
   pyavs.get_fixated_objects
   pyavs.preprocess_eye_events
   pyavs.apply_maxwell_filter
   pyavs.filter_meg
   pyavs.resample_meg
   pyavs.preprocess_meg_block
   pyavs.apply_precomputed_ica
   pyavs.compute_ica
   pyavs.apply_ica
   pyavs.find_cardiac_components
   pyavs.find_eye_components_xy_correlation
   pyavs.run_ica_et_pipeline
   pyavs.build_et_raw_from_samples
   pyavs.align_et_to_meg
   pyavs.create_et_event_epochs
   pyavs.get_meg_trigger_mapping
   pyavs.repair_meg_trigger_events
   pyavs.get_meg_trigger_dict
   pyavs.get_avs_blocks
   pyavs.get_meg_timestamp
   pyavs.add_fix_event_trigger
   pyavs.attach_scene_ids_to_samples
   pyavs.load_samples_with_scenes
   pyavs.validate_samples_scene_assignment
   pyavs.create_forward_model
   pyavs.create_bem_model
   pyavs.setup_coregistration
   pyavs.load_forward_model
   pyavs.apply_source_reconstruction
   pyavs.compute_beamformer_filters
   pyavs.compute_population_codes
   pyavs.extract_roi_data
   pyavs.save_source_data
   pyavs.save_annotated_raw
   pyavs.save_population_codes_h5
   pyavs.find_population_codes_files
   pyavs.list_available_parameter_sets
   pyavs.load_source_data
   pyavs.create_source_space
   pyavs.get_roi_labels
   pyavs.get_glasser_roi_labels
   pyavs.plot_evoked_joint
   pyavs.plot_median_erf
   pyavs.plot_sensor_space_overview
   pyavs.load_pilot_events
   pyavs.load_pilot_samples
   pyavs.add_sample_scene_coordinates
   pyavs.load_and_preprocess
   pyavs.get_epochs

Browse by Topic
-------------------

.. toctree::
   :maxdepth: 1

   dataloader
   preprocessing
   source
   scenes
   captions
   utils
   config
   visualization
   io
   pilot
   cli

``AVSComposer`` (:class:`pyavs.preprocessing.composer.AVSComposer`, documented in full on
:doc:`preprocessing`) is the recommended high-level entry point for most analyses; see
:doc:`../package/composer_guide` for a guided walkthrough. ``pyavs.MEGETComposer`` is a
backward-compatibility alias for the same class.
