Fixation Object Labeling
=============================

To link fixation behavior to the visual and semantic content being viewed, each fixation was
assigned an object category label by mapping its gaze position (in image coordinates) onto
MS-COCO and COCO-Stuff instance segmentations (Caesar et al., 2018; Lin et al., 2014).
Where multiple segmentation masks overlapped at a fixation location, the category with the
smallest mask area was assigned; fixations falling outside all segmented regions were
labeled ``"None"``.

Coverage
------------

Of 203,356 total fixations during scene viewing, 97.0% received an object label, spanning
171 unique categories. "Person" was the most frequently fixated category, consistent with
the well-established priority of person/face content in natural scene viewing; the remaining
most-fixated categories spanned buildings, animals, vehicles, and food items.

Using This With pyAVS
--------------------------

:func:`pyavs.get_fixated_objects` performs this mapping for a set of eye-tracking events.
The underlying category-management utilities (COCO-Stuff class definitions, thing/stuff
distinctions, license-filtered image subsets for reuse) live in :mod:`pyavs.scenes` -- see
:doc:`../api/scenes` and :doc:`../examples/cocostuff_object_detection` /
:doc:`../examples/real_data_object_detection` for runnable examples.
