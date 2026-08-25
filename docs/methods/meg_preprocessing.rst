MEG Preprocessing
======================

MEG data were processed using MNE-Python (Gramfort et al., 2013). Temporal signal space
separation (tSSS) with movement compensation was applied to suppress external interference
(MaxFilter, Elekta Oy). Data were bandpass filtered between 0.2 and 200 Hz and resampled to
500 Hz.

Eye-Movement Artifact Removal
-----------------------------------

Independent component analysis (ICA; FastICA) was applied per session to 80 components
extracted from a 1-40 Hz bandpass-filtered copy of the concatenated MEG data. Eye-movement
artifact components were identified by computing the Pearson correlation between each
component's source time series and the continuous horizontal (gx) and vertical (gy) gaze
position signals, epoched around scene onsets. Components within the top 5% of absolute
correlation with either gaze axis were excluded (union criterion), removing on average 7.8
+/- 0.4 ocular components per session (mean absolute r = 0.16 for gx, 0.15 for gy; range 7-8
per session across 50 sessions). ICA was applied to the broadband data prior to epoching.

pyAVS's ET-gaze-correlation ICA pipeline (:func:`pyavs.find_eye_components_xy_correlation`,
:func:`pyavs.run_ica_et_pipeline`) implements this approach; see :doc:`../api/preprocessing`.

Epoching
------------

MEG data were epoched around fixation onsets (-500 to +800 ms), based on eye-tracking event
markers aligned at each scene onset. No baseline correction was applied.

Running This With pyAVS
----------------------------

:class:`pyavs.AVSComposer` (see :doc:`../package/composer_guide`) wraps loading, filtering,
ICA, and event-locked epoching into a single workflow that mirrors this pipeline; the
functional building blocks (:func:`pyavs.apply_maxwell_filter`, :func:`pyavs.filter_meg`,
:func:`pyavs.resample_meg`, :func:`pyavs.compute_ica`, :func:`pyavs.apply_ica`) are also
available individually for custom pipelines.
