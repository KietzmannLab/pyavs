Examples
========

This section contains focused examples demonstrating specific pyAVS functionality. Each
example page shows the real, runnable script from the ``examples/`` directory alongside
explanatory prose.

For the core **AVSComposer** workflow (the recommended entry point for most analyses), see
the dedicated :doc:`../package/composer_guide` instead of this section.

.. toctree::
   :maxdepth: 2
   :caption: Available Examples

   source_reconstruction_examples
   compute_cross_session_filters
   config_example
   cocostuff_object_detection
   real_data_object_detection
   reproduce_analysis

Source Reconstruction
----------------------

From forward modeling to source localization and population code extraction, see
:doc:`source_reconstruction_examples`.

Cross-Session Beamformer Filters
-----------------------------------

Computing a single set of LCMV beamformer filters shared across a subject's sessions, see
:doc:`compute_cross_session_filters`.

Configuration
-------------

Setting up and inspecting the pyAVS data path / configuration system, see
:doc:`config_example`.

Object Detection on Scenes
----------------------------

Mapping fixations to MS-COCO / COCO-Stuff object categories, both on synthetic
(:doc:`cocostuff_object_detection`) and real (:doc:`real_data_object_detection`) data.

Reproducing a Saved Analysis
-------------------------------

Re-running an analysis from a configuration saved alongside earlier population codes, see
:doc:`reproduce_analysis`.

Complete Example Scripts
--------------------------

All working example scripts live in the ``examples/`` directory at the repository root.
We recommend starting with ``avs_composer_example.py`` (covered in
:doc:`../package/composer_guide`) to understand the core pyAVS workflow.
