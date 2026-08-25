Scene Analysis (pyavs.scenes)
=============================

The scenes module provides functions for analyzing visual scenes and mapping eye movements to objects.

Object Detection and Mapping
-----------------------------

.. automodule:: pyavs.scenes.objects
   :members:
   :undoc-members:
   :show-inheritance:

Image Cropping
--------------

.. automodule:: pyavs.scenes.crops
   :members:
   :undoc-members:
   :show-inheritance:

COCO-Stuff Category Definitions
-------------------------------------

.. automodule:: pyavs.scenes.cocostuff_classes
   :members:
   :undoc-members:
   :show-inheritance:

License-Filtered Image Subsets
------------------------------------

.. automodule:: pyavs.scenes.coco_licenses
   :members:
   :undoc-members:
   :show-inheritance:

Scene Annotation Transformation
-------------------------------------

.. automodule:: pyavs.scenes.transform_scene_annotations
   :members:
   :undoc-members:
   :show-inheritance:

ANN Embeddings for Fixation Crops
-----------------------------------------

.. note::

   Unlike the rest of ``pyavs.scenes``, this submodule and :mod:`pyavs.captions` are
   intentionally **not** re-exported from top-level ``pyavs`` -- they're submodule-only
   utilities used directly by the scripts that need them (e.g.
   ``compute_fixation_embeddings.py``), not part of the core top-level API surface.

.. automodule:: pyavs.scenes.embeddings
   :members:
   :undoc-members:
   :show-inheritance: