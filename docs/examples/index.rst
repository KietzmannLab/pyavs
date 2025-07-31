Examples
========

This section contains practical examples demonstrating key pyAVS functionality, with a focus on the **AVSComposer** - the core component for MEG-eye tracking data fusion.

.. toctree::
   :maxdepth: 2
   :caption: Available Examples

   composer_usage
   basic_data_loading
   meg_preprocessing
   eye_tracking_analysis
   meg_eye_synchronization
   source_reconstruction_examples

AVS Composer Usage (Recommended)
---------------------------------

The :doc:`composer_usage` guide shows how to use the **AVSComposer class** - the main tool for MEG-eye tracking data processing in pyAVS. This is the recommended approach for most analyses as it provides:

- Complete MEG preprocessing pipeline with ICA artifact removal
- Eye tracking data integration with multiple event types
- Trigger-based MEG-ET alignment and synchronization
- Flexible epoch creation with rich metadata
- Built-in quality control and error handling

Basic Data Loading
------------------

Learn fundamental data loading concepts with the :doc:`basic_data_loading` guide.

MEG Preprocessing
-----------------

Detailed MEG preprocessing options including filtering, ICA, and artifact removal in :doc:`meg_preprocessing`.

Eye Tracking Analysis  
---------------------

Fixation detection, saccade analysis, and scene integration in :doc:`eye_tracking_analysis`.

MEG-Eye Synchronization
-----------------------

Advanced temporal alignment and synchronized analysis workflows in :doc:`meg_eye_synchronization`.

Source Reconstruction
---------------------

From forward modeling to source localization in :doc:`source_reconstruction_examples`.

Complete Example Scripts
------------------------

You can find all working example scripts in the ``examples/`` directory:

- ``avs_composer_example.py`` - Comprehensive AVSComposer demonstration
- ``meg_et_workflow.py`` - Complete MEG-ET analysis pipeline  
- ``meg_et_workflow_simple.py`` - Minimal working example
- ``source_reconstruction_example.py`` - Source space analysis
- ``batch_preprocessing.py`` - Batch processing multiple subjects

**Getting Started**: We recommend starting with ``avs_composer_example.py`` to understand the core pyAVS workflow.