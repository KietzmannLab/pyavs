Dataset at a Glance
========================

The Active Visual Semantics (AVS) dataset combines MEG, eye tracking, and structural MRI,
recorded while participants freely explored natural scenes -- active vision, rather than the
passive, fixation-enforced viewing used in most existing neuroimaging datasets.

.. list-table::
   :widths: 30 70

   * - Participants
     - 5 (see :doc:`methods/participants`)
   * - Sessions
     - Up to 10 MEG + eye-tracking sessions per participant, plus one anatomical session
   * - Stimuli
     - 4,080 natural scenes from the Natural Scenes Dataset (NSD), semantically balanced
       (see :doc:`methods/stimuli`)
   * - Task
     - Free viewing (4 s/scene) with a verbal scene-captioning task on 25% of trials
       (see :doc:`methods/semantic_captioning`)
   * - MEG
     - 306-channel Elekta Neuromag TRIUX, 1000 Hz (see :doc:`methods/meg_acquisition`)
   * - Eye tracking
     - EyeLink 1000, 1000 Hz (see :doc:`methods/eye_tracking`)
   * - Fixation epochs
     - More than 200,000 across the dataset
   * - Object labels
     - Per-fixation MS-COCO / COCO-Stuff category labels, 171 categories
       (see :doc:`methods/object_labeling`)

For structural/BIDS layout details, see :doc:`dataset/overview`; for known data-quality
caveats, see :doc:`dataset/known_issues`; for how to get the data, see :doc:`data_access`.

What Do I Need?
--------------------

.. list-table::
   :header-rows: 1

   * - Goal
     - Start here
   * - Load MEG + eye-tracking data for an analysis
     - :doc:`package/composer_guide`, :doc:`quickstart`
   * - Understand the acquisition/preprocessing pipeline
     - :doc:`methods/index`
   * - Source reconstruction / population codes
     - :doc:`tutorials/source_reconstruction_population_codes`
   * - Map fixations to scene objects
     - :doc:`methods/object_labeling`, :doc:`examples/cocostuff_object_detection`
   * - See example analyses (RSA, encoding, source ERFs)
     - :doc:`analyses/index`
   * - Full function/class reference
     - :doc:`api/index`
