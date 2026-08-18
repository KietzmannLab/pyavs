Frequently Asked Questions
================================

**What's the difference between AVSComposer and load_and_preprocess?**

:class:`pyavs.AVSComposer` (see :doc:`../package/composer_guide`) is the actively developed,
recommended high-level interface for MEG + eye-tracking data fusion. The functional API
(:func:`pyavs.load_and_preprocess` / :func:`pyavs.get_epochs`) is an older interface that
still works but receives less active development -- prefer ``AVSComposer`` for new work.

**Where can I download the AVS dataset?**

**TODO (pending publication):** the dataset's public hosting location is not yet finalized.
See :doc:`../data_access` for what's known so far.

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
