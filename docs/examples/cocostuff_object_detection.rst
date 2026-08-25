COCO-Stuff Object Detection
================================

``examples/cocostuff_object_detection_example.py`` demonstrates mapping fixations onto
MS-COCO / COCO-Stuff object category segmentations (see :doc:`../methods/object_labeling`
for how this is used in the AVS dataset itself). It compares detection coverage between
standard MS-COCO (80 "thing" classes) and COCO-Stuff (172 classes: 80 things + 91 "stuff"
categories like sky, grass, or building), and analyzes thing-vs-stuff fixation patterns.

.. code-block:: bash

   python examples/cocostuff_object_detection_example.py --subjects 1 2 3 --sessions 1 2

For a walkthrough against real (rather than synthetic/demo) eye-tracking data, see
:doc:`real_data_object_detection`.

.. literalinclude:: ../../examples/cocostuff_object_detection_example.py
   :language: python

See Also
--------

- :doc:`../api/scenes` -- :mod:`pyavs.scenes.objects`, :mod:`pyavs.scenes.cocostuff_classes`
  API reference
