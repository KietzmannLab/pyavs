AVSComposer Guide
======================

:class:`~pyavs.preprocessing.composer.AVSComposer` is the recommended entry point for
MEG + eye-tracking data fusion in pyAVS: it loads MEG blocks, applies ICA artifact removal
and filtering, concatenates blocks per session, finds MEG trigger events, and aligns
eye-tracking events (fixations, saccades, blinks, or whole-scene trials) to build epoched
MEG data with rich per-epoch metadata.

Full Worked Example
-----------------------

``examples/avs_composer_example.py`` runs the complete pipeline end to end -- MEG loading,
filtering, ICA, event finding, eye-tracking annotation and epoching for every event type,
and a simple median-ERF visualization:

.. literalinclude:: ../../examples/avs_composer_example.py
   :language: python

Advanced Initialization
----------------------------

The composer accepts many optional parameters beyond the basics shown above -- data paths,
block range, ICA source (precomputed vs. computed on the fly), and filter/resample settings:

.. code-block:: python

   composer = pyavs.AVSComposer(
       subject=1,
       session_num=1,

       # Data paths (default to the globally configured data path if omitted)
       data_path="/path/to/avs/dataset",
       output_path="/path/to/outputs",
       et_path="/path/to/eye_tracking",

       # Processing options
       preprocessed=True,
       recompute_prepro=False,      # set True to recompute preprocessing
       max_block=10,                # process blocks 1-10
       min_block=1,

       # Bad channel handling
       interpolate_bad_channels=True,

       # ICA artifact removal
       apply_ica=False,             # compute ICA on the fly
       use_precomputed_ica=True,    # use existing ICA solutions
       ica_solutions_path="/path/to/ica",
       ica_exclusions_file="/path/to/exclusions.json",

       # Filtering
       l_freq=0.2,                  # high-pass frequency
       h_freq=200.0,                # low-pass frequency
       causal_filter=True,          # causal filtering preserves timing

       # Resampling
       resample_freq=500.0,

       # Misc
       n_jobs=4,
       random_state=42,
       verbose=True,
       write_output=True,
   )

Working with Empty-Room Recordings
--------------------------------------

The composer detects and separates empty-room recordings automatically during
:meth:`~pyavs.preprocessing.composer.AVSComposer.load_meg_data`:

.. code-block:: python

   composer = pyavs.AVSComposer(subject=1, session_num=1)
   composer.load_meg_data()

   if composer.empty_room_available:
       print("Empty room recordings found!")
       print(f"Empty room blocks: {list(composer.raws_dict_empty_room.keys())}")
       composer.concatenate_raws_per_session()
       print(f"Empty room duration: {composer.raws_concatenated_empty_room.times[-1]:.1f}s")
   else:
       print("No empty room recordings available")

Data Summary
----------------

:meth:`~pyavs.preprocessing.composer.AVSComposer.get_data_summary` returns a dict
summarizing what's been loaded/processed so far:

.. code-block:: python

   summary = composer.get_data_summary()
   print(f"Subject: {summary['subject']}, Session: {summary['session']}")
   print(f"Blocks loaded: {summary['blocks_loaded']}")
   print(f"MEG channels: {summary['meg_channels']}, duration: {summary['meg_duration']:.1f}s")
   print(f"Eye events: {summary['eye_events']}, epochs created: {summary['epochs_created']}")
   print(f"Empty room available: {summary['empty_room_available']}")

Integration with Source Reconstruction
-------------------------------------------

Composer epochs feed directly into source reconstruction:

.. code-block:: python

   composer = pyavs.AVSComposer(subject=1, session_num=1)
   composer.load_meg_data()
   composer.apply_ica_to_blocks()
   composer.concatenate_raws_per_session()
   composer.find_events_in_raw()
   composer.get_et_annotations(event_type="fixation")
   composer.make_et_event_epochs(tmin=-0.2, tmax=0.5, event_type="fixation")

   epochs = composer.et_epochs

   forward_model = pyavs.load_forward_model(subject_id=1, session=1)
   source_data = pyavs.apply_source_reconstruction(
       epochs, forward_model, method='beamformer'
   )
   roi_data = pyavs.extract_roi_data(
       source_data, forward_model['src'],
       roi_labels=pyavs.get_glasser_roi_labels(area='early_visual'),
   )

   print(f"Source data: {source_data.shape} (epochs, sources, timepoints)")

See :doc:`../tutorials/source_reconstruction_population_codes` and
:doc:`../examples/source_reconstruction_examples` for the full pipeline, including
population code computation and storage.

See Also
--------

- :doc:`../tutorials/meg_eye_workflow` -- narrative tutorial covering this same pipeline
- :doc:`../api/preprocessing` -- full :class:`~pyavs.preprocessing.composer.AVSComposer` API
  reference
