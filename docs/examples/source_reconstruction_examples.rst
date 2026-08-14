Source Reconstruction Examples
==============================

pyAVS provides comprehensive source reconstruction capabilities including forward modeling, 
beamforming, minimum norm estimation, and population code analysis. This guide demonstrates 
how to use these features with both the AVSComposer and functional API approaches.

Complete Source Reconstruction Workflow
----------------------------------------

Here's the full pipeline from MEG epochs to population codes:

.. code-block:: python

   import pyavs
   import numpy as np
   
   # Set up data path
   pyavs.set_data_path('/path/to/avs/dataset')
   
   # Step 1: Create epochs using AVSComposer
   composer = pyavs.AVSComposer(
       subject=1,
       session_num=1,
       preprocessed=True,
       use_precomputed_ica=True,
       verbose=True
   )
   
   # Load and process MEG data
   composer.load_meg_data()
   composer.apply_ica_to_blocks()
   composer.concatenate_raws_per_session()
   composer.find_events_in_raw()
   
   # Create fixation epochs
   composer.get_et_annotations(event_type="fixation", recording="scene")
   composer.make_et_event_epochs(
       tmin=-0.2, tmax=0.5,
       event_type="fixation",
       get_metadata=True
   )
   
   epochs = composer.et_epochs
   print(f"Created {len(epochs)} epochs for source reconstruction")
   
   # Step 2: Load forward model
   forward_model = pyavs.load_forward_model(subject_id=1, session=1)
   print(f"Forward model: {len(forward_model['src'])} source spaces")
   
   # Step 3: Setup source reconstruction
   source_setup = pyavs.setup_source_reconstruction(
       subject_id=1,
       session=1,
       method='beamformer',
       reg=0.05,
       weight_norm='unit-noise-gain'
   )
   
   # Step 4: Compute beamformer filters
   filters = pyavs.compute_beamformer_filters(
       epochs=epochs,
       forward=forward_model,
       reg=0.05,
       weight_norm='unit-noise-gain',
       pick_ori='max-power'
   )
   
   # Step 5: Apply source reconstruction
   source_estimates = pyavs.apply_source_reconstruction(
       epochs, forward_model, 
       method='beamformer',
       filters=filters
   )
   
   print(f"Source estimates: {len(source_estimates)} epochs")
   print(f"Source space: {source_estimates[0].data.shape} (sources x timepoints)")

ROI-Based Analysis
------------------

Extract activity from specific regions of interest:

.. code-block:: python

   # Define ROIs using different atlases
   visual_rois = ['V1', 'V2', 'V4', 'MT', 'LOC']  # Visual areas
   glasser_rois = pyavs.get_glasser_roi_labels(['visual', 'temporal'])  # Glasser atlas
   
   # Extract ROI data from source estimates
   roi_data = pyavs.extract_roi_data(
       source_estimates,
       forward_model['src'],
       roi_labels=visual_rois,
       method='mean',  # Average activity within each ROI
       verbose=True
   )
   
   print(f"Extracted data from {len(roi_data)} ROIs:")
   for roi_name, data in roi_data.items():
       print(f"  {roi_name}: {data.shape} (epochs x timepoints)")
   
   # Create summary statistics per ROI
   roi_summary = {}
   for roi_name, data in roi_data.items():
       roi_summary[roi_name] = {
           'mean_activity': np.mean(data, axis=0),  # Average across epochs
           'peak_time': epochs.times[np.argmax(np.mean(data, axis=0))],
           'peak_amplitude': np.max(np.mean(data, axis=0))
       }
       print(f"{roi_name}: Peak at {roi_summary[roi_name]['peak_time']:.3f}s")

Population Code Analysis
------------------------

Compute population codes for encoding analysis:

.. code-block:: python

   # Compute population codes for different experimental conditions
   population_codes = pyavs.compute_population_codes(
       source_estimates,
       events_metadata=epochs.metadata,
       time_window=(0.0, 0.3),  # Analysis window
       method='mean_amplitude',
       conditions=['scene_id', 'trial_type'],
       verbose=True
   )
   
   print(f"Population codes shape: {population_codes['stc'].shape}")
   print(f"Conditions analyzed: {list(population_codes.keys())}")
   
   # Save population codes for later analysis
   h5_path = pyavs.save_population_codes_h5(
       population_codes=population_codes,
       metadata=epochs.metadata,
       subject_id=1,
       session=1,
       event_type='fixation',
       sampling_rate=int(epochs.info['sfreq']),
       rois=list(roi_data.keys()),
       times=epochs.times,
       filter_params={'l_freq': 0.2, 'h_freq': 200.0}
   )
   
   print(f"Population codes saved: {h5_path}")

Beamformer Configuration Options
--------------------------------

Customize beamformer parameters for different analysis needs:

.. code-block:: python

   # Standard LCMV beamformer
   standard_filters = pyavs.compute_beamformer_filters(
       epochs=epochs,
       forward=forward_model,
       reg=0.05,                    # Regularization strength
       weight_norm='unit-noise-gain', # Normalization method
       pick_ori='max-power',        # Orientation selection
       reduce_rank=False            # Keep full rank
   )
   
   # High-resolution beamformer (less regularization)
   highres_filters = pyavs.compute_beamformer_filters(
       epochs=epochs,
       forward=forward_model,
       reg=0.01,                    # Lower regularization
       weight_norm='nai',           # Neural activity index
       pick_ori='vector',           # Keep vector orientation
       reduce_rank=True             # Reduce rank for stability
   )
   
   # Noise-normalized beamformer
   noise_norm_filters = pyavs.compute_beamformer_filters(
       epochs=epochs,
       forward=forward_model,
       reg=0.05,
       weight_norm='unit-noise-gain-invariant',
       pick_ori='max-power'
   )
   
   # Apply different beamformers
   for name, filters in [('standard', standard_filters), 
                        ('highres', highres_filters),
                        ('noise_norm', noise_norm_filters)]:
       
       stcs = pyavs.apply_source_reconstruction(
           epochs, forward_model,
           method='beamformer',
           filters=filters
       )
       
       print(f"{name} beamformer: {len(stcs)} source estimates")

Cross-Session Analysis
----------------------

Analyze data across multiple sessions with consistent filters:

.. code-block:: python

   # Load data from multiple sessions
   sessions = [1, 2]
   all_epochs = {}
   
   for session in sessions:
       composer = pyavs.AVSComposer(
           subject=1, session_num=session,
           preprocessed=True, verbose=False
       )
       
       # Complete pipeline
       composer.load_meg_data()
       composer.apply_ica_to_blocks()
       composer.concatenate_raws_per_session()
       composer.find_events_in_raw()
       composer.get_et_annotations(event_type="fixation")
       composer.make_et_event_epochs(tmin=-0.2, tmax=0.5, event_type="fixation")
       
       all_epochs[session] = composer.et_epochs
       print(f"Session {session}: {len(composer.et_epochs)} epochs")
   
   # Compute cross-session covariance for consistent filters
   cross_session_cov = pyavs.compute_cross_session_data_covariance(
       epochs_list=list(all_epochs.values()),
       method='empirical',
       verbose=True
   )
   
   # Compute consistent LCMV filters across sessions
   consistent_filters = pyavs.compute_per_session_lcmv_filters(
       epochs_list=list(all_epochs.values()),
       forward=forward_model,
       data_cov=cross_session_cov,
       reg=0.05
   )
   
   # Apply consistent filters to each session
   session_source_data = {}
   for session, epochs in all_epochs.items():
       stcs = pyavs.apply_lcmv_to_epochs(
           epochs, consistent_filters, verbose=True
       )
       session_source_data[session] = stcs
       print(f"Session {session}: {len(stcs)} source estimates")

Advanced Source Space Setup
----------------------------

Create custom source spaces and forward models:

.. code-block:: python

   # Create surface source space
   surface_src = pyavs.create_source_space(
       subject='fsaverage',  # FreeSurfer subject
       spacing='oct6',       # Source spacing (6th order octahedron)
       surface='white',      # Surface type
       subjects_dir='/path/to/freesurfer/subjects'
   )
   
   # Create volume source space for subcortical regions
   volume_src = pyavs.setup_volume_source_space(
       subject='fsaverage',
       pos=5.0,  # 5mm spacing
       mri='T1.mgz',
       subjects_dir='/path/to/freesurfer/subjects'
   )
   
   # Create BEM model for forward calculation
   bem_model = pyavs.create_bem_model(
       subject='fsaverage',
       ico=4,  # BEM mesh density
       conductivity=[0.3],  # Single-layer BEM
       subjects_dir='/path/to/freesurfer/subjects'
   )
   
   # Create forward model
   custom_fwd = pyavs.create_forward_model(
       epochs.info,
       trans='fsaverage-trans.fif',  # Coregistration
       src=surface_src,
       bem=bem_model,
       meg=True,
       eeg=False
   )
   
   print(f"Custom forward model: {custom_fwd['nsource']} sources")

Data Management and File Discovery
----------------------------------

Efficiently manage and discover source reconstruction results:

.. code-block:: python

   # Find existing population codes files
   existing_files = pyavs.find_population_codes_files(
       subject_id=1,
       session=1,
       event_type='fixation',
       sampling_rate=500
   )
   
   print(f"Found {len(existing_files)} existing population codes files")
   
   # List all available parameter sets
   param_sets = pyavs.list_available_parameter_sets()
   print(f"Available parameter sets: {len(param_sets)}")
   
   # Load existing source data
   if existing_files:
       latest_file = existing_files[0]  # Most recent
       loaded_data = pyavs.load_source_data(latest_file)
       print(f"Loaded data shape: {loaded_data.shape}")
   
   # Save source estimates in standard format
   source_path = pyavs.save_source_data(
       source_estimates,
       subject_id=1,
       session=1,
       data_type='fixation_source_estimates',
       method='beamformer'
   )
   
   print(f"Source estimates saved: {source_path}")

Integration with Composer Workflow
-----------------------------------

Use source reconstruction within the AVSComposer pipeline:

.. code-block:: python

   def complete_meg_et_source_workflow(subject_id, session, rois_of_interest):
       """Complete MEG-ET source reconstruction workflow."""
       
       # Initialize composer
       composer = pyavs.AVSComposer(
           subject=subject_id,
           session_num=session,
           preprocessed=True,
           use_precomputed_ica=True
       )
       
       # MEG preprocessing pipeline
       composer.load_meg_data()
       composer.apply_ica_to_blocks()
       composer.concatenate_raws_per_session()
       composer.find_events_in_raw()
       
       # Eye tracking integration
       composer.get_et_annotations(event_type="fixation", recording="scene")
       composer.make_et_event_epochs(
           tmin=-0.2, tmax=0.5,
           event_type="fixation",
           get_metadata=True
       )
       
       # Source reconstruction
       forward_model = pyavs.load_forward_model(subject_id, session)
       
       # Compute beamformer
       filters = pyavs.compute_beamformer_filters(
           composer.et_epochs, forward_model,
           reg=0.05, weight_norm='unit-noise-gain'
       )
       
       # Apply to epochs
       source_estimates = pyavs.apply_source_reconstruction(
           composer.et_epochs, forward_model,
           method='beamformer', filters=filters
       )
       
       # Extract ROI data
       roi_data = pyavs.extract_roi_data(
           source_estimates, forward_model['src'],
           roi_labels=rois_of_interest
       )
       
       # Compute population codes
       population_codes = pyavs.compute_population_codes(
           source_estimates,
           events_metadata=composer.et_epochs.metadata,
           time_window=(0.0, 0.3)
       )
       
       # Save results
       h5_path = pyavs.save_population_codes_h5(
           population_codes, composer.et_epochs.metadata,
           subject_id, session, 'fixation',
           int(composer.et_epochs.info['sfreq'])
       )
       
       return {
           'epochs': composer.et_epochs,
           'source_estimates': source_estimates,
           'roi_data': roi_data,
           'population_codes': population_codes,
           'saved_path': h5_path
       }
   
   # Run complete workflow
   results = complete_meg_et_source_workflow(
       subject_id=1, session=1,
       rois_of_interest=['V1', 'V4', 'IT', 'PFC']
   )
   
   print(f"Workflow complete: {len(results['source_estimates'])} source estimates")
   print(f"ROI data: {list(results['roi_data'].keys())}")
   print(f"Results saved: {results['saved_path']}")

Troubleshooting Source Reconstruction
-------------------------------------

Common issues and solutions:

**Forward Model Issues**

.. code-block:: python

   try:
       forward_model = pyavs.load_forward_model(subject_id, session)
   except FileNotFoundError:
       print("Forward model not found. Please ensure:")
       print("1. FreeSurfer reconstruction completed")
       print("2. MEG-MRI coregistration performed")
       print("3. Forward model computed and saved")
       
       # Create forward model if needed
       forward_model = pyavs.create_forward_model(
           epochs.info, trans_file, src, bem_model
       )

**Memory Issues with Large Datasets**

.. code-block:: python

   # Process epochs in smaller chunks
   chunk_size = 50
   all_source_estimates = []
   
   for i in range(0, len(epochs), chunk_size):
       chunk_epochs = epochs[i:i+chunk_size]
       chunk_stcs = pyavs.apply_source_reconstruction(
           chunk_epochs, forward_model, method='beamformer'
       )
       all_source_estimates.extend(chunk_stcs)
       print(f"Processed chunk {i//chunk_size + 1}/{(len(epochs)-1)//chunk_size + 1}")

**Beamformer Rank Issues**

.. code-block:: python

   # Handle rank deficiency
   filters = pyavs.compute_beamformer_filters(
       epochs, forward_model,
       reg=0.1,           # Increase regularization
       reduce_rank=True,  # Reduce rank automatically
       verbose=True
   )

See Also
--------

- :doc:`../tutorials/meg_eye_workflow` - Complete MEG-ET workflow tutorial
- :doc:`../api/source` - Full API documentation for source reconstruction
- :doc:`composer_usage` - AVSComposer integration examples
- Example scripts: ``examples/simple_source_reconstruction.py`` (synthetic-data quickstart),
  ``examples/compute_population_codes_example.py`` (real-data, config-driven workflow)