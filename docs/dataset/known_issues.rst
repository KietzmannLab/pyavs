Known Issues
================

This page collects known data-quality caveats, quirks of the released tree, and the fixes
pyAVS applies automatically when you use its loading/preprocessing functions. It is derived
from the AVS lab's internal data-quality notes made during collection plus checks run against
the release tree itself; if you use pyAVS's loading pipelines (:class:`pyavs.AVSComposer`,
:func:`pyavs.load_and_enrich_eye_events`, :func:`pyavs.repair_meg_trigger_events`), the items
below marked *handled automatically* do not require any action on your part.

Session-Level Issues
-------------------------

- **sub-03, ses-02**: eye tracking was lost for the first blocks of the session due to an
  eye-tracker hardware failure. The session's only recording segment is ``as3_2_5``; MEG for
  the session is complete (14 runs), so MEG runs early in this session have no corresponding
  gaze data.
- **sub-04, ses-04**: this session was re-recorded after an MEG acquisition server crash
  interrupted the original recording. Eye tracking therefore ships as **two** segments,
  ``as4_4_0`` and ``as4_4_12``. As a result sub-04 viewed the session-4 stimulus set twice
  (once in the interrupted recording, once in the repeat) -- analyses that assume each scene
  is viewed once per subject should account for this.
- **sub-05, ses-01**: the *after* empty-room recording is missing (only ``as05ab.fif``, the
  before-session recording, exists). Noise covariance for this session must be estimated from
  the before-session recording alone.

Quirks of the Released Tree
--------------------------------

Not data-quality problems, but things that surprise people:

- **``_scene_`` in epoch filenames does not mean scene-onset-locked.** It means "recorded
  during the scene-viewing task". ``fixation_scene`` and ``saccade_scene`` epochs are locked
  to eye movements. No stimulus-onset-locked epochs are shipped -- build them from the SSS
  runs and the scene annotations if you need them.
- **ICA is shipped fitted but not applied.** The released SSS files still contain ocular
  components. Apply the shipped solution with :func:`pyavs.apply_precomputed_ica`, or review
  ``..._ica-exclusions.json`` and ``..._ica-et-scores.parquet`` and choose your own exclusions.
- **Epochs have no baseline correction** and span -0.5 to 0.8 s at 500 Hz. During free viewing
  there is no neutral pre-fixation baseline, so none was imposed; apply your own if your
  analysis needs one.
- **Session-level concatenated raws are not shipped**, only per-run SSS files plus separate
  annotation FIFs. :class:`pyavs.AVSComposer` reassembles them.
- **Run counts differ between sessions**: 10 task runs in ses-01, 14 in ses-02 -- ses-10.
  Code that hardcodes a run count will silently miss data.

Source Reconstruction Caveats
----------------------------------

- **sub-05's forward solution has 8,195 sources, not 8,196** like the other four
  participants. One source was dropped by the 5 mm minimum-distance criterion when the
  forward model was originally computed. This is a property of the data, not a packaging
  artifact -- code that assumes an identical source count across participants (e.g.
  pre-allocating a subject x source array) will break on sub-05.
- **No scalp or head surfaces are released.** They reconstruct facial geometry and are
  therefore identifying. Consequences: you cannot recompute the coregistration from scratch,
  rebuild a multi-shell BEM, or plot a head/scalp surface. Use the shipped
  ``sub-0X-trans.fif``, ``sub-0X-bem-sol.fif`` (single-shell, inner skull) and
  ``sub-0X-fwd.fif`` instead -- everything pyAVS's source pipeline needs is present. See
  :doc:`../methods/source_reconstruction`.
- **pyAVS does not yet resolve the released FreeSurfer layout on its own.** The released
  ``derivatives/freesurfer/`` tree is correct and works directly with MNE -- pass it as
  ``subjects_dir`` and :func:`mne.read_labels_from_annot`, :func:`mne.compute_source_morph`
  and friends behave normally. What does not work yet is pyAVS's *own* path resolution: it
  was written against the lab's internal tree and builds ``as0X/src/...`` rather than the
  released ``sub-0X/bem/...`` (MNE's convention), in both library helpers
  (``pyavs.source.forward``, ``pyavs.source.filters``, ``pyavs.utils.paths``) and several
  scripts under ``scripts/source/`` and ``scripts/meg_viz/``. Until this is parameterised,
  pass forward-model and ``subjects_dir`` paths explicitly rather than relying on pyAVS's
  defaults. Making the package layout-agnostic is actively tracked work.

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
  :meth:`~pyavs.preprocessing.composer.AVSComposer.get_et_annotations`. Consecutive saccades
  within a trial are merged into the first one, whose ``duration`` is extended to cover the
  sequence; the epoch metadata's ``multi_saccade`` column marks these events (``'first'``,
  versus ``'no'`` for untouched events), so you can identify or exclude them after the fact.

If You Hit Something Not Listed Here
------------------------------------------

Please open a `GitHub issue <https://github.com/KietzmannLab/pyavs/issues>`_ with the
subject/session and a description of what you observed.
