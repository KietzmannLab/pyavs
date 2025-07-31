pyAVS Documentation
===================

.. image:: https://badge.fury.io/py/pyavs.svg
   :target: https://badge.fury.io/py/pyavs
   :alt: PyPI version

.. image:: https://readthedocs.org/projects/pyavs/badge/?version=latest
   :target: https://pyavs.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

Welcome to pyAVS, a streamlined Python package for loading and preprocessing MEG + eye-tracking data from the Active Visual Semantics (AVS) BIDS dataset.

Features
--------

**Core Functionality**

- **MEG Data Processing**: Complete pipeline from raw data to source reconstruction
- **Eye Tracking Analysis**: Advanced fixation and saccade detection with scene integration  
- **MEG-ET Synchronization**: Temporal alignment and synchronized analysis
- **BIDS Compatibility**: Support for BIDS-formatted datasets with legacy compatibility
- **Source Reconstruction**: Beamformer and minimum norm estimation methods
- **Population Code Analysis**: Advanced encoding analysis for experimental conditions

**Key Capabilities**

- ✅ **Automated Preprocessing**: Maxwell filtering, bandpass filtering, ICA artifact removal
- ✅ **Object Detection Integration**: MSCOCO object mapping for visual scenes
- ✅ **ROI-based Analysis**: FreeSurfer and Glasser atlas integration  
- ✅ **Command Line Interface**: Easy-to-use CLI for common workflows
- ✅ **Parallel Processing**: Batch processing for multiple subjects/sessions
- ✅ **Comprehensive Examples**: Full workflow demonstrations and tutorials

Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   pip install pyavs

Basic Usage with AVSComposer (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pyavs

   # Set up data path
   pyavs.set_data_path('/path/to/avs/dataset')

   # Initialize AVS Composer - the main tool for MEG-ET data fusion
   composer = pyavs.AVSComposer(
       subject=1,
       session_num=1,
       preprocessed=True,           # Use preprocessed MEG data
       use_precomputed_ica=True,    # Apply ICA artifact removal
       verbose=True
   )

   # Complete MEG-ET processing pipeline
   composer.load_meg_data()                    # Load MEG blocks
   composer.apply_ica_to_blocks()              # Remove artifacts
   composer.concatenate_raws_per_session()     # Combine blocks
   composer.find_events_in_raw()               # Find MEG triggers

   # Process eye tracking events and create epochs
   composer.get_et_annotations(event_type="fixation", recording="scene")
   composer.make_et_event_epochs(
       tmin=-0.2, tmax=0.8, 
       event_type="fixation",
       get_metadata=True
   )

   print(f"Created {len(composer.et_epochs)} fixation epochs")
   print(f"Metadata columns: {list(composer.et_epochs.metadata.columns)[:5]}")

Alternative Functional API
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For custom workflows requiring fine-grained control:

.. code-block:: python

   # Load and preprocess data
   subject_data = pyavs.load_and_preprocess(
       subject_id=1, session=1,
       include_meg=True, include_eye=True
   )

   # Create epochs from eye tracking events
   epochs, events = pyavs.get_epochs(
       subject_data, 'fixation', 'meg',
       tmin=-0.2, tmax=0.5
   )

   # Source reconstruction
   forward_model = pyavs.load_forward_model(subject_id=1, session=1)
   source_data = pyavs.apply_source_reconstruction(
       epochs, forward_model, method='beamformer'
   )

   print(f"Created {len(epochs)} epochs")

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   tutorials/index
   examples/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/dataloader
   api/preprocessing  
   api/scenes
   api/source
   api/utils
   api/cli

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   changelog
   license

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`