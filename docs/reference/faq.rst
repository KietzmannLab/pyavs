Frequently Asked Questions
================================

**What's the difference between AVSComposer and load_and_preprocess?**

:class:`pyavs.AVSComposer` (see :doc:`../package/composer_guide`) is the actively developed,
recommended high-level interface for MEG + eye-tracking data fusion. The functional API
(:func:`pyavs.load_and_preprocess` / :func:`pyavs.get_epochs`) is an older interface that
still works but receives less active development -- prefer ``AVSComposer`` for new work.

**Where can I download the AVS dataset?**

Not yet -- the release is assembled and verified but not published. It is being deposited on
GRO.data (Dataverse), and the DOI and download route will be listed on :doc:`../data_access`
once it is live.

**Do I have to download all 743 GiB?**

No. The tree is organized so you can take only what you need, and the epoch *metadata* tables
are small and separate from the epoch data -- so you can filter epochs by scene, fixated
object or event kinematics before downloading any MEG. See :doc:`../data_access` for a
per-goal breakdown.

**Is the dataset BIDS?**

BIDS-*inspired*, not BIDS-valid. It uses BIDS-style ``sub-<label>/ses-<label>/<datatype>/``
directories and a ``derivatives/`` tree, but raw files keep their original acquisition
filenames and no validator sidecars are shipped. See :doc:`../dataset/overview`.

**Can I do source reconstruction with the released data?**

Yes -- ``derivatives/freesurfer/`` is a directly usable MNE ``SUBJECTS_DIR`` with each
participant's source space, single-shell BEM, forward solution, coregistration and
parcellations. Scalp/head surfaces are withheld for privacy, so you cannot recompute the
coregistration or a multi-shell BEM from scratch; use the shipped ones. See
:doc:`../dataset/known_issues`.

**Are the released epochs locked to scene onset?**

No -- despite the ``_scene_`` in their filenames, which means "during the scene-viewing task".
``fixation_scene`` and ``saccade_scene`` epochs are locked to eye movements. Stimulus-locked
epochs can be built from the released SSS runs and scene annotations with
:class:`pyavs.AVSComposer`.

**How do I cite this dataset or pyAVS?**

See :doc:`citation`.

**My eye-tracking/MEG alignment or fixation counts look off -- is this a known issue?**

Check :doc:`../dataset/known_issues` first -- several session-specific data-quality
caveats and the automatic corrections pyAVS applies are documented there.

**Does pyAVS work without MEG data, e.g. for eye-tracking-only analyses?**

Yes -- the eye-tracking loading and preprocessing functions
(:func:`pyavs.load_eye_events`, :func:`pyavs.load_and_enrich_eye_events`,
:func:`pyavs.preprocess_eye_events`) don't require MEG data. See the "Eye Tracking Only"
example in :doc:`../quickstart`.

**Something else?**

Open an issue on `GitHub <https://github.com/KietzmannLab/pyavs/issues>`_.
