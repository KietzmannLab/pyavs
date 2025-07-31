MEG + Eye Tracking Workflow
===========================

This tutorial demonstrates the complete workflow for analyzing MEG and eye tracking data using pyAVS, 
focusing primarily on the **AVSComposer** - the recommended tool for MEG-ET data fusion.

Overview
--------

The pyAVS MEG + eye tracking workflow provides two main approaches:

**Recommended: AVSComposer Approach**
   The AVSComposer class provides a complete, tested pipeline that replicates and improves upon 
   the original AVS-machine-room composer functionality. This is the **recommended approach** 
   for most users.

**Alternative: Functional API Approach**
   Individual functions for custom workflows requiring fine-grained control.

The complete workflow follows these main steps:

1. **Composer Initialization**: Configure MEG-ET processing parameters
2. **MEG Data Loading**: Load and preprocess MEG data with Maxwell filtering, ICA
3. **Eye Tracking Integration**: Load ET data and create trigger-based alignment
4. **Event-based Epoching**: Create MEG epochs around eye tracking events
5. **Source Reconstruction**: Transform sensor data to source space (optional)
6. **Analysis**: Population codes, encoding models, statistics

Prerequisites
-------------

Before starting this workflow, ensure you have:

- The AVS dataset downloaded and properly structured
- pyAVS installed with all dependencies
- Data path configured using ``pyavs.set_data_path()``

Method 1: AVSComposer Approach (Recommended)
----------------------------------------------

The AVSComposer provides a streamlined, robust pipeline for MEG-ET data processing:

.. code-block:: python

   import pyavs
   
   # Set up data path
   pyavs.set_data_path('/path/to/avs/dataset')
   
   # Initialize AVS Composer with optimal settings
   composer = pyavs.AVSComposer(
       subject=1,
       session_num=1,
       preprocessed=True,              # Use preprocessed MEG data
       interpolate_bad_channels=True,  # Handle bad channels
       use_precomputed_ica=True,       # Use existing ICA solutions
       l_freq=0.2,                     # High-pass filter
       h_freq=200.0,                   # Low-pass filter  
       resample_freq=500.0,            # Target sampling rate
       causal_filter=True,             # Preserve temporal order
       verbose=True
   )
   
   # Load MEG data (automatic preprocessing)
   composer.load_meg_data()
   print(f"Loaded MEG blocks: {list(composer.raws_dict.keys())}")
   
   # Apply ICA artifact removal
   composer.apply_ica_to_blocks()
   
   # Filter data if needed
   composer.filter_meg_data()  # Uses configured filter parameters
   
   # Concatenate MEG blocks
   composer.concatenate_raws_per_session()
   
   # Find MEG trigger events
   composer.find_events_in_raw()
   print(f"Found {len(composer.meg_trigger_events)} MEG trigger events")

Eye Tracking Integration and Epoching with Composer
---------------------------------------------------

Now integrate eye tracking data and create epochs for different event types:

.. code-block:: python

   # Process different eye tracking event types
   event_types = ["fixation", "saccade", "blink"]
   epochs_results = {}
   
   for event_type in event_types:
       print(f"Processing {event_type} events...")
       
       # Get eye tracking annotations for this event type
       composer.get_et_annotations(
           event_type=event_type,
           recording="scene",              # Focus on scene viewing
           exclude_last_fixation=True,     # Exclude incomplete fixations
           add_cross_event_info=True,      # Add contextual information
           preprocessed=True               # Use preprocessed ET data
       )
       
       print(f"Loaded {len(composer.et_events)} {event_type} events")
       
       # Create MEG epochs around eye tracking events
       composer.make_et_event_epochs(
           tmin=-0.2,              # 200ms before event
           tmax=0.8,               # 800ms after event
           event_type=event_type,
           recording="scene",
           get_metadata=True,      # Include rich metadata
           baseline=None           # No baseline correction (recommended for AVS)
       )
       
       # Store results for later analysis
       epochs_results[event_type] = composer.et_epochs.copy()
       print(f"Created {len(composer.et_epochs)} {event_type} epochs")
       
       # Display some metadata information
       if composer.et_epochs.metadata is not None:
           metadata_cols = list(composer.et_epochs.metadata.columns)[:5]
           print(f"Metadata columns (first 5): {metadata_cols}")

Alternative Method 2: Functional API Approach
----------------------------------------------

For users requiring fine-grained control, pyAVS also provides individual functions:

.. code-block:: python

   # This approach gives you more control but requires more setup
   
   # Load MEG data
   meg_raw = pyavs.load_meg_raw(subject_id=1, session=1)
   
   # Load eye tracking data
   explog, eye_events = pyavs.load_and_enrich_eye_events([1], [1])
   
   # Apply MEG preprocessing
   meg_clean = pyavs.preprocess_meg_block(
       meg_raw, 
       l_freq=0.2, h_freq=200.0,
       apply_ica=True
   )
   
   # Create MEG-ET composer for alignment
   meg_et_composer = pyavs.MEGETComposer(1, 1, data_path, data_path)
   
   # Create epochs from eye tracking events
   epochs = pyavs.create_et_event_epochs(
       meg_clean, eye_events,
       event_type='fixation',
       tmin=-0.2, tmax=0.5
   )
   
   print(f"Created {len(epochs)} fixation epochs")

**Note**: The AVSComposer approach is recommended for most users as it handles edge cases, 
provides better error handling, and ensures compatibility with the AVS dataset structure.

Source Reconstruction with Composer
------------------------------------

Transform sensor data to source space using the composer epochs:

.. code-block:: python

   # Continue with the composer epochs from previous steps
   # epochs_results contains epochs for different event types
   
   # Use fixation epochs for source reconstruction
   fixation_epochs = epochs_results['fixation']
   
   # Load forward model
   try:
       forward_model = pyavs.load_forward_model(subject_id=1, session=1)
       print("✓ Forward model loaded")
   except FileNotFoundError:
       print("⚠ Forward model not found - creating for demonstration")
       # In practice, you need to create this using FreeSurfer and coregistration
       forward_model = create_demo_forward_model(fixation_epochs.info)
   
   # Setup source reconstruction
   source_setup = pyavs.setup_source_reconstruction(
       subject_id=1,
       session=1,
       method='beamformer',
       reg=0.05,
       weight_norm='unit-noise-gain'
   )
   
   # Compute beamformer filters
   print("Computing LCMV beamformer filters...")
   filters = pyavs.compute_beamformer_filters(
       epochs=fixation_epochs,
       forward=forward_model,
       reg=0.05,                    # Regularization parameter
       weight_norm='unit-noise-gain', # Normalization method
       pick_ori='max-power',        # Orientation selection
       verbose=True
   )
   
   # Apply source reconstruction
   print("Applying beamformer to epochs...")
   source_estimates = pyavs.apply_source_reconstruction(
       fixation_epochs,
       forward_model,
       method='beamformer',
       filters=filters
   )
   
   print(f"✓ Created {len(source_estimates)} source estimates")
   print(f"Source space: {source_estimates[0].data.shape} (sources × timepoints)")

ROI Analysis and Population Codes
----------------------------------

Extract activity from regions of interest:

.. code-block:: python

   # Define regions of interest
   visual_rois = ['V1', 'V2', 'V4', 'MT', 'LOC']
   
   # Extract ROI data
   print("Extracting ROI data...")
   roi_data = pyavs.extract_roi_data(
       source_estimates,
       forward_model['src'],
       roi_labels=visual_rois,
       method='mean',  # Average within each ROI
       verbose=True
   )
   
   print(f"✓ Extracted data from {len(roi_data)} ROIs")
   for roi_name, data in roi_data.items():
       print(f"  {roi_name}: {data.shape} (epochs × timepoints)")
   
   # Compute population codes for experimental conditions
   print("Computing population codes...")
   population_codes = pyavs.compute_population_codes(
       source_estimates,
       events_metadata=fixation_epochs.metadata,
       time_window=(0.0, 0.3),  # Analysis window: 0-300ms post-fixation
       method='mean_amplitude',
       conditions=['scene_id', 'trial_type'],
       verbose=True
   )
   
   print(f"✓ Population codes computed")
   print(f"Available data: {list(population_codes.keys())}")
   
   # Save population codes for further analysis
   h5_path = pyavs.save_population_codes_h5(
       population_codes=population_codes,
       metadata=fixation_epochs.metadata,
       subject_id=1,
       session=1,
       event_type='fixation',
       sampling_rate=int(fixation_epochs.info['sfreq']),
       rois=list(roi_data.keys()),
       times=fixation_epochs.times,
       filter_params={'l_freq': 0.2, 'h_freq': 200.0}
   )
   
   print(f"✓ Population codes saved: {h5_path}")

Alternative: Source Reconstruction with Functional API
-------------------------------------------------------

For comparison, here's the functional API approach:

.. code-block:: python

   # Load MEG and eye tracking data separately
   meg_raw = pyavs.load_meg_raw(subject_id=1, session=1)
   explog, eye_events = pyavs.load_and_enrich_eye_events([1], [1])
   
   # Apply MEG preprocessing
   meg_clean = pyavs.preprocess_meg_block(
       meg_raw, 
       l_freq=0.2, h_freq=200.0,
       apply_ica=True
   )
   
   # Create epochs from eye tracking events
   epochs = pyavs.create_et_event_epochs(
       meg_clean, eye_events,
       event_type='fixation',
       tmin=-0.2, tmax=0.5
   )
   
   # Load forward model and apply source reconstruction
   forward_model = pyavs.load_forward_model(subject_id=1, session=1)
   source_estimates = pyavs.apply_source_reconstruction(
       epochs, forward_model, method='beamformer'
   )
   
   print(f"Functional API: {len(source_estimates)} source estimates")

Step 7: Analysis
----------------

Perform analysis on source-reconstructed data:

.. code-block:: python

   from pyavs.source import filters
   
   # Extract data for specific ROIs
   roi_labels = ['V1', 'V4', 'IT', 'PFC']  # Example ROIs
   roi_data = filters.extract_roi_data(
       stc,
       roi_labels,
       atlas='glasser'
   )
   
   # Compute population codes
   pop_codes = filters.compute_population_codes(
       roi_data,
       events=fixation_events,
       time_window=(0.0, 0.3),  # Analysis window
       method='mean_amplitude'
   )
   
   print(f"Population codes shape: {pop_codes.shape}")
   print(f"ROIs analyzed: {list(roi_data.keys())}")

Complete Workflow Function
--------------------------

Here's a complete function that runs the entire workflow:

.. code-block:: python

   def meg_eye_workflow(subject_id, session, data_path):
       """
       Complete MEG + eye tracking analysis workflow.
       
       Parameters
       ----------
       subject_id : int
           Subject identifier
       session : int
           Session number
       data_path : str
           Path to AVS dataset
           
       Returns
       -------
       results : dict
           Dictionary containing all analysis results
       """
       import pyavs
       
       # Set up
       pyavs.set_data_path(data_path)
       
       # Load data
       meg_raw = pyavs.dataloader.load_meg_raw(subject_id, session)
       eye_data = pyavs.dataloader.load_eye_data(subject_id, session)
       
       # Preprocess MEG
       meg_clean = pyavs.preprocessing.meg.preprocess_meg(
           meg_raw,
           apply_maxwell=True,
           bandpass=(0.1, 40),
           apply_ica=True
       )
       
       # Preprocess eye tracking
       eye_clean, events = pyavs.preprocessing.eye.preprocess_eye(
           eye_data,
           detect_events=True
       )
       
       # Synchronize
       meg_eye_sync = pyavs.preprocessing.alignment.synchronize_meg_eye(
           meg_clean, eye_clean
       )
       
       # Create epochs
       epochs = pyavs.preprocessing.create_epochs_from_eye_events(
           meg_eye_sync['meg'], events, 
           event_type='fixation',
           tmin=-0.2, tmax=0.5
       )
       
       # Source reconstruction
       stc = pyavs.source.reconstruction.apply_source_reconstruction(
           epochs, subject_id, session, method='beamformer'
       )
       
       # Analysis
       results = {
           'epochs': epochs,
           'source_data': stc,
           'eye_events': events,
           'sync_info': meg_eye_sync
       }
       
       return results

Next Steps
----------

After completing this workflow, you can:

- Perform statistical analysis on the population codes
- Create encoding models relating brain activity to visual features
- Analyze temporal dynamics of visual processing
- Compare activity across different experimental conditions

See the :doc:`../examples/index` for more specific analysis examples.

Troubleshooting
---------------

Common issues and solutions:

**Synchronization Problems**
   - Check trigger channels are properly recorded
   - Verify eye tracker and MEG system clocks
   - Use cross-correlation method if triggers are unreliable

**ICA Convergence Issues**
   - Reduce number of components (try 15-20)
   - Filter data more aggressively (e.g., 1-30 Hz)
   - Check for bad channels before ICA

**Memory Issues**
   - Process data in smaller chunks
   - Use lower source space resolution
   - Apply decimation to reduce sampling rate

For more help, see the :doc:`../troubleshooting` guide.