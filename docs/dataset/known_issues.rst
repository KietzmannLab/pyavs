Known Issues
================

This page collects known data-quality caveats and the fixes pyAVS applies automatically
when you use its loading/preprocessing functions. It is derived from the AVS lab's internal
data-quality notes made during collection; if you use pyAVS's loading pipelines
(:class:`pyavs.AVSComposer`, :func:`pyavs.load_and_enrich_eye_events`,
:func:`pyavs.repair_meg_trigger_events`), the items below marked *handled automatically* do
not require any action on your part.

Session-Level Issues
-------------------------

- **sub-03, ses-02**: three blocks of eye-tracking data were lost due to an eye-tracker
  hardware failure during recording.
- **sub-04, ses-04**: this session was re-recorded after an MEG acquisition server crash
  interrupted the original recording. As a result, sub-04 viewed the session-4 stimulus set
  twice (once in the interrupted recording, once in the repeat) -- analyses that assume each
  scene is viewed once per subject should account for this.

Fixed in pyAVS (Handled Automatically)
-------------------------------------------

- **Trial-numbering offset for sessions after the first**: the original trial-indexing code
  had an off-by-30 bug for sessions beyond the first. This is corrected internally by
  pyAVS's data-loading code and does not require manual correction when using
  :func:`pyavs.load_experiment_log` or :class:`pyavs.AVSComposer`.
- **"Wandering" MEG block trigger values**: MEG block-identifying trigger codes could
  overflow past their intended range during long recordings. :func:`pyavs.repair_meg_trigger_events`
  detects and corrects this; :class:`pyavs.AVSComposer`'s
  :meth:`~pyavs.preprocessing.composer.AVSComposer.find_events_in_raw` applies it
  automatically.
- **Systematic ~20 ms MEG trigger delay**: a fixed hardware/software delay between the
  eye-tracker's scene-onset trigger and the corresponding MEG trigger is corrected during
  eye-tracking/MEG alignment (:func:`pyavs.load_and_enrich_eye_events`'s
  ``offset_scene_triggers_ms`` parameter, and equivalently within
  :func:`pyavs.align_et_to_meg` / :func:`pyavs.create_et_event_epochs`).
- **Double/multi-saccade artifacts**: occasional spurious multi-saccade sequences in the
  raw eye-tracking event stream are cleaned up when ``fix_multi_saccades=True`` (the
  default) is passed to :func:`pyavs.load_and_enrich_eye_events` or used implicitly by
  :meth:`~pyavs.preprocessing.composer.AVSComposer.get_et_annotations`.

If You Hit Something Not Listed Here
------------------------------------------

Please open a `GitHub issue <https://github.com/KietzmannLab/pyavs/issues>`_ with the
subject/session and a description of what you observed.
