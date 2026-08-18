Dataset Structure
======================

The AVS dataset is organized as a `BIDS <https://bids.neuroimaging.io/>`_ dataset, following
the MEG-BIDS extension for the MEG data and the eye-tracking BIDS extension for gaze data.

.. note::

   This page describes the naming and layout conventions pyAVS's loading functions expect
   and produce. **TODO (pending public release):** the exact top-level path/URL where the
   public BIDS dataset will be hosted is not yet finalized -- see :doc:`../data_access`.

Subject and Session Naming
-------------------------------

pyAVS's internal analysis code (and this documentation) uses two parallel naming schemes:

.. list-table::
   :header-rows: 1

   * - Internal/analysis code
     - BIDS
     - Meaning
   * - ``as01`` -- ``as05``
     - ``sub-01`` -- ``sub-05``
     - The 5 participants
   * - session letter ``a`` -- ``j``
     - ``ses-01`` -- ``ses-10``
     - The 10 MEG + eye-tracking recording sessions per participant
   * - (n/a)
     - ``ses-anat``
     - The structural MRI session (no MEG/eye-tracking data)
   * - ``run-XX`` / block
     - ``run-XX``
     - A single scene-viewing block within a session

pyAVS's :mod:`pyavs.utils.derivatives` module encodes these conventions for locating and
naming derivative (processed) outputs; see :doc:`../api/utils`.

Data Completeness
---------------------

All five participants (as01-as05 / sub-01 -- sub-05) completed all 10 sessions plus the
anatomical session. See :doc:`known_issues` for session-level data-quality caveats within
individual recorded sessions.

What's Included
--------------------

Per the dataset paper (:doc:`../reference/citation`), the released dataset includes:

- Raw and preprocessed MEG data (BIDS MEG format)
- Eye-tracking data with fixation, saccade, and blink annotations
- Per-fixation object category labels (MS-COCO / COCO-Stuff, see
  :doc:`../methods/object_labeling`)
- Transcribed participant scene captions (see :doc:`../methods/semantic_captioning`)
- Stimulus presentation logs with timing information
- Individual anatomical MRIs (defaced)

Modality-Specific Loading
------------------------------

pyAVS's :mod:`pyavs.dataloader` submodule provides the loading functions for each modality
(:func:`pyavs.load_meg_raw`, :func:`pyavs.load_eye_events`, :func:`pyavs.load_anatomical`),
and :class:`pyavs.AVSComposer` (see :doc:`../package/composer_guide`) combines them into a
single aligned MEG + eye-tracking workflow. See :doc:`../quickstart` to get started.
