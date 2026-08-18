Quick Start Guide
=================

This guide will get you started with pyAVS in just a few minutes.

Setup
-----

First, install pyAVS and set up your data path:

.. code-block:: python

   import pyavs
   
   # Set path to your AVS dataset
   pyavs.set_data_path('/path/to/avs/dataset')
   
   # Check data availability
   availability = pyavs.check_data_availability(subject_id=1, session=1)
   print(availability)

Basic Workflow
--------------

Eye Tracking Only
~~~~~~~~~~~~~~~~~

Start with eye tracking analysis:

.. code-block:: python

   # Load eye tracking data for multiple subjects
   subjects = [1, 2, 3]
   sessions = [1, 2]
   
   explog, events = pyavs.load_and_preprocess_eye_tracking(
       subjects=subjects,
       sessions=sessions,
       preprocessed=True,
       fix_multi_saccades=True
   )
   
   # Add sequence information
   events = pyavs.add_fixation_sequence_position(events)
   events = pyavs.add_cross_event_information(events)
   
   # Map fixations to objects
   events_with_objects = pyavs.get_fixated_objects(events)
   
   print(f"Processed {len(events_with_objects)} events")

MEG + Eye Tracking
~~~~~~~~~~~~~~~~~~

Complete MEG and eye tracking workflow:

.. code-block:: python

   # Load and preprocess both modalities
   subject_data = pyavs.load_and_preprocess(
       subject_id=1,
       session=1,
       include_meg=True,
       include_eye=True,
       preprocess_meg=True,
       apply_ica=True,  # Remove artifacts
       blocks=[1, 2, 3]  # Specific blocks
   )
   
   # Create MEG epochs locked to eye tracking events
   epochs, events = pyavs.get_epochs(
       subject_data,
       event_type='fixation',
       sensor_type='meg',
       tmin=-0.2,
       tmax=0.5,
       baseline=(-0.2, 0)
   )
   
   print(f"Created {len(epochs)} fixation-locked epochs")

Source Reconstruction
~~~~~~~~~~~~~~~~~~~~~

Perform source-level analysis:

.. code-block:: python

   # Load forward model (must be computed beforehand)
   try:
       forward_model = pyavs.load_forward_model(subject_id=1, session=1)
       
       # Apply beamformer source reconstruction
       source_data = pyavs.apply_source_reconstruction(
           epochs, 
           forward_model, 
           method='beamformer'
       )
       
       # Extract data from visual ROIs
       roi_labels = pyavs.get_glasser_roi_labels('high_visual')
       roi_data = pyavs.extract_roi_data(
           source_data,
           forward_model['src'],
           roi_labels
       )
       
       # Compute population codes for different conditions
       pop_codes = pyavs.compute_population_codes(
           source_data,
           events_metadata=events,
           conditions=['object_category'],
           time_window=(0.1, 0.3),
           times=epochs.times,
       )
       
       print(f"Population codes computed for {len(pop_codes)} conditions")
       
   except FileNotFoundError:
       print("Forward model not found - create one first")

Command Line Interface
----------------------

pyAVS provides a powerful CLI for common tasks:

Data Availability
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Check what data is available
   pyavs check-data --subject 1 --session 1 --data-path /path/to/data

Preprocessing
~~~~~~~~~~~~~

.. code-block:: bash

   # Preprocess MEG + eye tracking data
   pyavs preprocess --subject 1 --session 1 --blocks 1 2 3 --apply-ica

Epoch Creation
~~~~~~~~~~~~~~

.. code-block:: bash

   # Create fixation-locked MEG epochs
   pyavs create-epochs --subject 1 --session 1 \
       --event-type fixation --sensor-type meg \
       --tmin -0.2 --tmax 0.5 --save

Source Reconstruction
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Run beamformer source reconstruction
   pyavs source-reconstruction --subject 1 --session 1 \
       --method beamformer --save-source-data

Batch Processing
~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Process multiple subjects in parallel
   pyavs batch --subjects 1 2 3 4 5 --sessions 1 2 \
       --workflow preprocess --parallel --n-jobs 4

Configuration
~~~~~~~~~~~~~

.. code-block:: bash

   # Set up configuration
   pyavs setup --data-path /path/to/avs/dataset \
       --freesurfer-dir /usr/local/freesurfer/subjects \
       --create-config

Common Patterns
---------------

Single Subject Analysis
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Complete single-subject pipeline
   def analyze_subject(subject_id, session):
       # Load data
       data = pyavs.load_and_preprocess(
           subject_id, session,
           include_meg=True,
           include_eye=True,
           apply_ica=True
       )
       
       # Create epochs
       epochs, events = pyavs.get_epochs(
           data, 'fixation', 'meg'
       )
       
       # Analyze evoked responses
       evoked = epochs.average()
       
       return evoked, events
   
   # Run analysis
   evoked, events = analyze_subject(1, 1)
   print(f"Evoked response computed from {len(events)} fixations")

Multi-Subject Analysis
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Analyze multiple subjects
   subjects = [1, 2, 3, 4, 5]
   sessions = [1, 2]
   
   all_evoked = []
   all_events = []
   
   for subject in subjects:
       for session in sessions:
           try:
               evoked, events = analyze_subject(subject, session)
               all_evoked.append(evoked)
               all_events.append(events)
               print(f"✓ Subject {subject}, Session {session}")
           except Exception as e:
               print(f"✗ Subject {subject}, Session {session}: {e}")
   
   print(f"Successfully processed {len(all_evoked)} datasets")

Advanced MEG-ET Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Load raw MEG data for one run/block
   meg_raw = pyavs.load_meg_raw(subject_id=1, session=1, run=1)

   # Load eye tracking events
   eye_events = subject_data['eye_events']

   # Create precisely time-locked epochs -- returns (epochs, epochs_dataframe)
   aligned_epochs, aligned_epochs_df = pyavs.create_et_event_epochs(
       meg_raw,
       eye_events,
       event_type='fixation',
       tmin=-0.2,
       tmax=0.8,
       baseline=(-0.2, 0)
   )

   print(f"Created {len(aligned_epochs)} precisely aligned epochs")

Visualization
~~~~~~~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt
   
   # Plot evoked responses
   evoked.plot(spatial_colors=True, gfp=True)
   plt.title('Fixation-locked MEG Response')
   
   # Plot topography at specific times
   evoked.plot_topomap(
       times=[0.1, 0.15, 0.2, 0.25, 0.3],
       ch_type='mag'
   )
   
   # Plot eye tracking events
   fixations = events[events['type'] == 'fixation']
   plt.figure(figsize=(12, 6))
   plt.scatter(fixations['start_time'], fixations['pos_x'], 
              s=fixations['duration']*10, alpha=0.6)
   plt.xlabel('Time (s)')
   plt.ylabel('X Position')
   plt.title('Fixation Patterns')
   plt.show()

Next Steps
----------

- Explore the :doc:`tutorials/index` for detailed walkthroughs
- Check out :doc:`examples/index` for complete analysis scripts
- Read the :doc:`api/index` for detailed function documentation
- See :doc:`reference/faq` for answers to common questions

Getting Help
------------

- **Documentation**: https://pyavs.readthedocs.io/
- **GitHub Issues**: https://github.com/KietzmannLab/pyavs/issues
- **Email Support**: psulewski@uni-osnabrueck.de