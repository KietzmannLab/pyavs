Methods
===========

.. important::

   These pages summarize the Methods section of the AVS dataset manuscript, currently in
   preparation: Sulewski, Amme, König, Hebart & Kietzmann, *"Active Visual Semantics: A
   large-scale MEG and eye-tracking dataset for understanding visual intelligence in
   action"* (in prep.). Figures cited here are taken directly from that manuscript. They will
   be updated (and a DOI added, see :doc:`../reference/citation`) once the manuscript is
   published -- treat this as a working summary, not a substitute for the published paper.

.. toctree::
   :maxdepth: 1
   :hidden:

   participants
   stimuli
   meg_acquisition
   eye_tracking
   meg_preprocessing
   object_labeling
   source_reconstruction
   semantic_captioning

The AVS dataset was collected to study active vision: brain activity during self-directed
scene exploration, rather than passive viewing with enforced central fixation. These pages
cover, in order:

1. :doc:`participants` -- who was recorded, and the session schedule
2. :doc:`stimuli` -- the natural scene stimulus set and how it was selected
3. :doc:`meg_acquisition` -- the MEG system and head-stabilization setup
4. :doc:`eye_tracking` -- the eye-tracking system, calibration, and event detection
5. :doc:`meg_preprocessing` -- filtering, ICA artifact removal, and epoching
6. :doc:`object_labeling` -- mapping fixations to MS-COCO / COCO-Stuff object categories
7. :doc:`source_reconstruction` -- forward modeling and beamforming
8. :doc:`semantic_captioning` -- the verbal scene-description task

For how to run the corresponding processing steps with pyAVS, see
:doc:`../package/composer_guide` and :doc:`../tutorials/index`.
