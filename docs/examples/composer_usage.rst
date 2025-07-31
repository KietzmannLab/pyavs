AVS Composer Usage Examples
============================

The AVSComposer class is the core component of pyAVS for MEG-eye tracking data fusion. 
It provides a comprehensive pipeline that replicates and improves upon the original AVS-machine-room composer functionality.

Basic Composer Workflow
------------------------

Here's the essential workflow using the AVSComposer:

.. code-block:: python

   import pyavs
   
   # Set up data path
   pyavs.set_data_path('/path/to/avs/dataset')
   
   # Initialize the AVS Composer
   composer = pyavs.AVSComposer(
       subject=4,
       session_num=1,
       preprocessed=True,           # Use preprocessed MEG data
       interpolate_bad_channels=True,
       use_precomputed_ica=True,    # Use precomputed ICA solutions
       l_freq=0.2,                  # Low-pass filter frequency
       h_freq=200.0,                # High-pass filter frequency
       resample_freq=500.0,         # Target sampling frequency
       causal_filter=True,          # Preserve temporal order
       verbose=True
   )
   
   # Load MEG data
   composer.load_meg_data()
   
   # Apply ICA artifact removal
   composer.apply_ica_to_blocks()
   
   # Filter MEG data (if not already filtered during preprocessing)
   composer.filter_meg_data()
   
   # Concatenate blocks
   composer.concatenate_raws_per_session()
   
   # Find MEG trigger events
   composer.find_events_in_raw()
   
   # Process eye tracking data and create epochs
   composer.get_et_annotations(event_type="fixation", recording="scene")
   composer.make_et_event_epochs(
       tmin=-0.2, tmax=0.8, 
       event_type="fixation",
       get_metadata=True
   )
   
   print(f"Created {len(composer.et_epochs)} fixation epochs")

Advanced Composer Configuration
-------------------------------

The composer offers extensive configuration options:

.. code-block:: python

   # Advanced configuration
   composer = pyavs.AVSComposer(
       subject=1,
       session_num=1,
       
       # Data paths
       data_path="/path/to/avs/dataset",
       output_path="/path/to/outputs",
       et_path="/path/to/eye_tracking",
       
       # Processing options
       preprocessed=True,
       recompute_prepro=False,      # Set True to recompute preprocessing
       max_block=10,                # Process blocks 1-10
       min_block=1,
       
       # Bad channel handling
       interpolate_bad_channels=True,
       
       # ICA artifact removal
       apply_ica=False,             # Compute ICA on-the-fly
       use_precomputed_ica=True,    # Use existing ICA solutions
       ica_solutions_path="/path/to/ica",
       ica_exclusions_file="/path/to/exclusions.json",
       
       # Filtering parameters
       l_freq=0.2,                  # High-pass filter
       h_freq=200.0,                # Low-pass filter  
       causal_filter=True,          # Causal filtering preserves timing
       
       # Resampling
       resample_freq=500.0,         # Target sampling rate
       
       # Processing
       n_jobs=4,                    # Parallel processing
       random_state=42,             # Reproducible results
       
       # Output
       verbose=True,
       write_output=True
   )

Multi-Event Type Processing
---------------------------

Process different eye tracking event types systematically:

.. code-block:: python

   # Initialize composer (once)
   composer = pyavs.AVSComposer(subject=1, session_num=1, verbose=True)
   composer.load_meg_data()
   composer.apply_ica_to_blocks()
   composer.concatenate_raws_per_session()
   composer.find_events_in_raw()
   
   # Process multiple event types
   event_types = ["fixation", "saccade", "blink"]
   epochs_results = {}
   
   for event_type in event_types:
       print(f"\\nProcessing {event_type} events...")
       
       # Get annotations for this event type
       composer.get_et_annotations(
           event_type=event_type,
           recording="scene",
           exclude_last_fixation=True,
           add_cross_event_info=True
       )
       
       # Create epochs
       composer.make_et_event_epochs(
           tmin=-0.2, tmax=0.8,
           event_type=event_type,
           get_metadata=True,
           baseline=None
       )
       
       # Store results
       epochs_results[event_type] = composer.et_epochs.copy()
       print(f"Created {len(composer.et_epochs)} {event_type} epochs")
   
   # Now you have epochs for all event types
   print(f"Total epochs: {sum(len(epochs) for epochs in epochs_results.values())}")

Composer with Custom Preprocessing
-----------------------------------

Use the composer to recompute preprocessing with custom parameters:

.. code-block:: python

   # Force recomputation of preprocessing
   composer = pyavs.AVSComposer(
       subject=1,
       session_num=1,
       preprocessed=True,
       recompute_prepro=True,       # Force recompute
       
       # Custom preprocessing parameters
       l_freq=1.0,                  # Stricter high-pass filter
       h_freq=100.0,                # Lower low-pass filter
       resample_freq=250.0,         # Lower sampling rate
       causal_filter=False,         # Non-causal filtering
       
       # ICA parameters
       apply_ica=True,              # Compute new ICA
       use_precomputed_ica=False,
       
       verbose=True
   )
   
   # Load data (will trigger recomputation)
   composer.load_meg_data(compute_missing_prepro=True)
   
   # Continue with standard workflow
   composer.concatenate_raws_per_session()
   # ... rest of pipeline

Working with Empty Room Recordings
-----------------------------------

The composer automatically handles empty room recordings when available:

.. code-block:: python

   composer = pyavs.AVSComposer(subject=1, session_num=1)
   composer.load_meg_data()
   
   # Check if empty room data is available
   if composer.empty_room_available:
       print("Empty room recordings found!")
       print(f"Empty room blocks: {list(composer.raws_dict_empty_room.keys())}")
       
       # The empty room data is automatically processed and stored separately
       # Access via composer.raws_concatenated_empty_room (after concatenation)
       composer.concatenate_raws_per_session()
       
       print(f"Empty room duration: {composer.raws_concatenated_empty_room.times[-1]:.1f}s")
   else:
       print("No empty room recordings available")

Data Quality and Summary
------------------------

Get comprehensive information about your processed data:

.. code-block:: python

   # After running the complete pipeline
   summary = composer.get_data_summary()
   
   print("Data Summary:")
   print(f"  Subject: {summary['subject']}")
   print(f"  Session: {summary['session']}")  
   print(f"  Blocks loaded: {summary['blocks_loaded']}")
   print(f"  MEG channels: {summary['meg_channels']}")
   print(f"  MEG duration: {summary['meg_duration']:.1f} seconds")
   print(f"  Eye events: {summary['eye_events']}")
   print(f"  Epochs created: {summary['epochs_created']}")
   print(f"  Annotations: {summary['annotations']}")
   print(f"  Empty room available: {summary['empty_room_available']}")

Integration with Source Reconstruction
--------------------------------------

Use composer results for source-level analysis:

.. code-block:: python

   # Complete composer pipeline
   composer = pyavs.AVSComposer(subject=1, session_num=1)
   composer.load_meg_data()
   composer.apply_ica_to_blocks()
   composer.concatenate_raws_per_session()
   composer.find_events_in_raw()
   composer.get_et_annotations(event_type="fixation")
   composer.make_et_event_epochs(tmin=-0.2, tmax=0.5, event_type="fixation")
   
   # Use epochs for source reconstruction
   epochs = composer.et_epochs
   
   # Load forward model
   forward_model = pyavs.load_forward_model(subject=1, session=1)
   
   # Apply source reconstruction
   source_estimates = pyavs.apply_source_reconstruction(
       epochs, forward_model, method='beamformer'
   )
   
   # Extract ROI data
   roi_data = pyavs.extract_roi_data(
       source_estimates, forward_model['src'], 
       roi_labels=['V1', 'V4', 'IT']
   )
   
   print(f"Source reconstruction complete: {len(source_estimates)} source estimates")

Batch Processing with Composer
-------------------------------

Process multiple subjects/sessions efficiently:

.. code-block:: python

   def process_subject_session(subject_id, session_num, data_path):
       """Process one subject-session with composer."""
       
       try:
           composer = pyavs.AVSComposer(
               subject=subject_id,
               session_num=session_num,
               data_path=data_path,
               verbose=False  # Reduce output for batch processing
           )
           
           # Run pipeline
           composer.load_meg_data()
           composer.apply_ica_to_blocks()
           composer.concatenate_raws_per_session()
           composer.find_events_in_raw()
           composer.get_et_annotations(event_type="fixation")
           composer.make_et_event_epochs(
               tmin=-0.2, tmax=0.5, 
               event_type="fixation"
           )
           
           return composer.et_epochs
           
       except Exception as e:
           print(f"Error processing subject {subject_id}, session {session_num}: {e}")
           return None
   
   # Batch process multiple subjects
   subjects = [1, 2, 3, 4]
   sessions = [1, 2]
   data_path = "/path/to/avs/dataset"
   
   results = {}
   for subject_id in subjects:
       for session_num in sessions:
           key = f"sub-{subject_id:02d}_ses-{session_num:02d}"
           epochs = process_subject_session(subject_id, session_num, data_path)
           if epochs is not None:
               results[key] = epochs
               print(f"✓ {key}: {len(epochs)} epochs")
   
   print(f"\\nBatch processing complete: {len(results)} datasets processed")

Error Handling and Debugging
-----------------------------

The composer provides comprehensive error handling and debugging information:

.. code-block:: python

   import logging
   
   # Enable detailed logging
   logging.basicConfig(level=logging.DEBUG)
   
   composer = pyavs.AVSComposer(
       subject=1, 
       session_num=1,
       verbose=True  # Enable verbose output
   )
   
   try:
       # Each step provides detailed feedback
       composer.load_meg_data(compute_missing_prepro=True)
       
       # Check what was loaded
       print(f"Loaded blocks: {list(composer.raws_dict.keys())}")
       print(f"Empty room available: {composer.empty_room_available}")
       
       # Apply ICA with error handling
       composer.apply_ica_to_blocks()
       
   except FileNotFoundError as e:
       print(f"Data not found: {e}")
   except ValueError as e:
       print(f"Configuration error: {e}")
   except Exception as e:
       print(f"Unexpected error: {e}")
       # Continue with fallback processing if needed

See Also
--------

- :doc:`../tutorials/meg_eye_workflow` - Complete MEG-ET workflow tutorial
- :doc:`../api/preprocessing` - Full API documentation for AVSComposer
- :doc:`../examples/meg_preprocessing` - MEG-specific preprocessing examples
- Example scripts in ``examples/avs_composer_example.py``