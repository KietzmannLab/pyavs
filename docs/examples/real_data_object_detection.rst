Object Detection on Real Data
==================================

``examples/real_data_object_detection_example.py`` loads real eye-tracking data for a
subject/session, applies fixation-to-object-category mapping using pre-transformed AVS
scene annotations (via :func:`pyavs.get_fixated_objects`), and overlays fixations and object
labels on the scene images -- producing figures like the fixation-object-annotation panels
in the AVS dataset paper (see :doc:`../methods/object_labeling`).

Requires pre-transformed scene annotations (``transform_scene_annotations.py``) and
processed scene images to be available locally.

.. literalinclude:: ../../examples/real_data_object_detection_example.py
   :language: python

See Also
--------

- :doc:`cocostuff_object_detection` -- the synthetic/demo-data version of this example
- :doc:`../api/scenes` -- full scenes API reference
