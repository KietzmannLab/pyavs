Dataset Structure
======================

This page describes the layout of the public AVS release -- what each directory holds, how
files are named, and what is deliberately not included. For how to obtain the data, see
:doc:`../data_access`; for data-quality caveats, see :doc:`known_issues`.

.. note::

   **The public release is being prepared.** The layout, file names and file formats below
   are those of the release tree as built and verified internally. One packaging detail is
   still open -- whether small files are grouped into archives for the hosting repository --
   and is marked as such below.

BIDS-Inspired, Not BIDS-Valid
----------------------------------

The release borrows BIDS's directory grammar -- ``sub-<label>/ses-<label>/<datatype>/`` with a
``derivatives/`` tree -- but it is **not a BIDS-validated dataset** and does not ship the
sidecar JSON/TSV files a validator would require. Do not point a BIDS tool at it and expect it
to parse.

The main consequence for users is in the file names: **raw files keep the names they were
written with by the acquisition systems** (``as01b03.fif``, ``as1_1_0.EDF``), while everything
pyAVS produced uses BIDS-style entity names (``sub-01_ses-02_task-avs_run-03_raw-sss.fif``).
Directories are reorganized; filenames were deliberately left alone so that released files stay
byte-identical to -- and traceable to -- the originals. BIDS-conformant renaming was considered
and deferred.

Top-Level Layout
---------------------

.. code-block:: text

   avs-public/
   ├── manifest.tsv                       # every file: category, path, size
   ├── stimuli/
   │   ├── images/                        # 4,080 scene JPEGs
   │   ├── annotations/
   │   │   ├── coco_objects/              # MS-COCO instance masks, per scene
   │   │   └── cocostuff/                 # COCO-Stuff masks, per scene
   │   └── avs_scenes_all_licenses.parquet # per-image licence/attribution table
   ├── sub-01/ .. sub-05/
   │   ├── anat/                          # T1.mgz (defaced)
   │   └── ses-01/ .. ses-10/
   │       ├── meg/                       # raw MEG, native names
   │       ├── eyetrack/                  # raw EyeLink EDF + Parquet exports
   │       └── beh/                       # experiment logs, transcriptions
   └── derivatives/
       ├── pyavs/sub-0X/ses-0Y/
       │   ├── meg/                       # per-run SSS, annotations, ICA
       │   ├── epochs/                    # fixation/saccade epochs + metadata
       │   └── eyetrack/                  # preprocessed gaze events and samples
       └── freesurfer/sub-0X/             # a directly usable MNE SUBJECTS_DIR
           ├── bem/  label/  surf/  mri/transforms/

Subject, Session and Run Naming
------------------------------------

Two naming schemes coexist: the internal one used by the acquisition systems (and still
visible in raw filenames), and the BIDS-style one used for directories and derivatives.

.. list-table::
   :header-rows: 1

   * - Internal / raw filenames
     - Release / derivatives
     - Meaning
   * - ``as01`` -- ``as05``
     - ``sub-01`` -- ``sub-05``
     - The 5 participants
   * - session letter ``a`` -- ``j``
     - ``ses-01`` -- ``ses-10``
     - The 10 MEG + eye-tracking sessions per participant
   * - (n/a)
     - ``anat/``
     - The structural MRI (acquired in a separate session)
   * - ``as01a03.fif``
     - ``run-03``
     - One scene-viewing block within a session
   * - ``as01ab.fif`` / ``as01ad.fif``
     - ``task-noise_recording-b`` / ``-d``
     - Empty-room recordings, **b**\ efore and **d**\ anach (= after) the session

So ``as03b07.fif`` is sub-03, ses-02, run-07, and its Maxwell-filtered counterpart is
``sub-03_ses-02_task-avs_run-07_raw-sss.fif``.

pyAVS's :mod:`pyavs.utils.derivatives` module encodes the derivative naming conventions; see
:doc:`../api/utils`.

Session Composition
------------------------

Runs per session are **not constant**:

- **ses-01: 10 task runs.** ses-02 -- ses-10: **14 task runs** each.
- Each run is a block of **30 trials** (one scene per trial), so a participant completes
  ``30 x (10 + 9 x 14) = 4,080`` trials -- one per scene in the stimulus set.
- Every session additionally has **2 empty-room recordings**, one before and one after the
  task (``...b.fif`` / ``...d.fif``). One is missing -- see :doc:`known_issues`.

Raw Data (``sub-0X/``)
---------------------------

``anat/``
   ``T1.mgz`` -- the individual structural MRI, **defaced**. One per participant.

``ses-0Y/meg/``
   Unprocessed MEG as written by the Elekta Neuromag TRIUX, one FIF per run
   (``as{SS}{letter}{run:02d}.fif``) plus the two empty-room files. 306 channels
   (204 gradiometers, 102 magnetometers) at 1000 Hz. See :doc:`../methods/meg_acquisition`.

``ses-0Y/eyetrack/``
   The native EyeLink ``.EDF`` plus its three standard exports -- ``*_events.parquet``
   (fixations/saccades/blinks), ``*_messages.parquet`` (experiment messages and triggers) and
   ``*_samples.parquet`` (the 1000 Hz gaze/pupil sample stream). Named
   ``as{subject}_{session}_{index}``, where the trailing index identifies the recording
   segment within the session: it is ``0`` when eye tracking ran from the session's first
   block. A session where the tracker was restarted mid-session carries more than one segment
   (sub-04, ses-04), and one where early blocks were lost starts at a non-zero index
   (sub-03, ses-02) -- both are documented in :doc:`known_issues`.

``ses-0Y/beh/``
   Experiment logs written by the presentation script: ``as_exp_data_*.parquet`` (with
   ``_archive_S`` / ``_archive_T`` companions), the transcribed caption logs
   (``explog_transcribed_*.parquet``, including a COCO-category-coded variant), and
   ``transcription_data/*.json``, one JSON per transcribed caption trial. **Speech audio
   (``.wav``) is not released** -- only the transcriptions.

Derivatives (``derivatives/pyavs/``)
-----------------------------------------

Produced by pyAVS; regenerable from the raw data with the package.

``ses-0Y/meg/``
   - ``..._run-XX_raw-sss.fif`` -- per-run Maxwell-filtered (SSS) MEG, including the two
     empty-room recordings (``task-noise_recording-{b,d}_raw-sss.fif``).
   - ``..._annotations-{scene,microphone,caption}.fif`` -- annotation-only FIFs carrying the
     event structure for the three recording types, small and separable from the data.
   - ``..._ica.fif``, ``..._ica-exclusions.json``, ``..._ica-et-scores.parquet`` -- the fitted ICA
     solution, the components marked for exclusion, and the gaze-correlation scores that
     identified the ocular components. **ICA is shipped fitted but not applied**, so you can
     inspect or override the exclusions; :func:`pyavs.apply_precomputed_ica` applies them.

``ses-0Y/epochs/``
   Fixation- and saccade-locked sensor-space epochs, as HDF5, plus their metadata tables:

   .. code-block:: text

      sub-01_ses-01_task-avs_fixation_scene_epochs.h5
      sub-01_ses-01_task-avs_saccade_scene_epochs.h5
      sub-01_ses-01_fixation_metadata.parquet
      sub-01_ses-01_saccade_metadata.parquet

   ``_scene_`` means "recorded during the scene-viewing task" -- **not** scene-onset-locked.
   These epochs are locked to eye movements, not to stimulus onset. No scene-onset-locked
   epochs are shipped; you can build them from the SSS runs and the scene annotations with
   :class:`pyavs.AVSComposer`.

   Inside each HDF5 file, one group per channel type (``grad``, ``mag``), each holding an
   ``onset`` dataset of shape ``(n_epochs, n_channels, n_times)`` in float32, gzip-compressed
   and **chunked one epoch per chunk** so that reading a scattered subset of epochs costs
   roughly one chunk read per epoch. Acquisition and processing parameters are stored as file
   attributes (``subject``, ``session``, ``event_type``, ``blocks``, ``times``, ``rois``,
   ``hz``, ``filter``). Epochs span -0.5 to 0.8 s around the event at 500 Hz (651 samples),
   band-pass filtered 0.2--200 Hz, with **no baseline correction** (AVS practice: there is no
   neutral pre-fixation baseline during free viewing).

   The metadata table has one row per epoch, in file order, and is what you filter on. Columns
   cover event kinematics (``duration``, ``amplitude``, ``peak_velocity``, ``mean_gx`` /
   ``mean_gy``, ``start_*`` / ``end_*``, ``rms``, ``sd``), sequence position (``fix_sequence``,
   ``fix_sequence_from_last``, ``sac_sequence``, ``sac_sequence_from_last``, and the
   ``*_pre`` / ``*_post`` neighbour columns), trial bookkeeping (``subject``, ``session``,
   ``trial``, ``block``, ``trial_per_block``, ``sceneID``, ``time_in_trial``, ``caption_task``,
   ``recording``), and the fixated object (``object_label``, ``object_id``) derived by
   intersecting gaze position with the COCO-Stuff masks in ``stimuli/annotations/`` -- see
   :doc:`../methods/object_labeling`.

   Because ``sceneID`` and ``object_label`` are plain columns, cross-subject queries ("every
   fixation on scene X", "every fixation on a *person*") reduce to concatenating the 50
   per-session metadata tables and filtering -- no MEG data needs to be touched until you know
   which epochs you want.

``ses-0Y/eyetrack/``
   Preprocessed gaze, four files per session: ``as_s{N}_el_events.parquet`` (fixations,
   saccades and blinks with scene/trial assignment), ``as_s{N}_el_msgs.parquet``,
   ``as_s{N}_el_samples.parquet`` and ``as_s{N}_el_cleaned_samples.parquet`` (the sample stream, and
   the drift-corrected/cleaned version). See :doc:`../methods/eye_tracking`.

Anatomy (``derivatives/freesurfer/``)
------------------------------------------

``derivatives/freesurfer/`` is a **directly usable MNE** ``SUBJECTS_DIR``. Point
``subjects_dir`` at it and MNE will find each participant as ``sub-01`` ... ``sub-05``:

.. code-block:: python

   import mne

   subjects_dir = "avs-public/derivatives/freesurfer"
   labels = mne.read_labels_from_annot("sub-01", parc="aparc", subjects_dir=subjects_dir)
   morph = mne.compute_source_morph(src, subject_to="fsaverage", subjects_dir=subjects_dir)

Per participant it contains:

- ``bem/sub-0X_oct6-src.fif`` -- the ``oct6`` cortical source space (8,196 sources).
- ``bem/sub-0X-bem-sol.fif`` -- single-shell (inner skull) BEM solution.
- ``bem/sub-0X-fwd.fif`` -- the precomputed forward solution.
- ``mri/transforms/sub-0X-trans.fif`` -- the MEG--MRI coregistration.
- ``surf/{lh,rh}.{white,pial,inflated,sphere.reg,curv,sulc}`` -- cortical surfaces;
  ``sphere.reg`` is what drives morphing to ``fsaverage``.
- ``label/`` -- 457 files: the FreeSurfer parcellations (``aparc``, ``a2009s``, ``DKTatlas``,
  ``BA_exvivo``) as ``.annot``/``.ctab`` plus the individual ``.label`` files.

You supply ``fsaverage`` yourself; MNE ships one (:func:`mne.datasets.fetch_fsaverage`).

.. note::

   Only cortex is released. **Scalp and head surfaces are withheld** -- they reconstruct facial
   geometry and are therefore identifying. This is a privacy decision, not an oversight; see
   :doc:`known_issues` for what it means in practice.

Stimuli (``stimuli/``)
---------------------------

- ``images/`` -- the 4,080 scene JPEGs (see :doc:`../methods/stimuli`).
- ``annotations/coco_objects/`` and ``annotations/cocostuff/`` -- per-scene segmentation masks
  (RLE-encoded JSON, one file per scene) used to assign a category to each fixation.
  :func:`pyavs.get_fixated_objects` reads these.
- ``avs_scenes_all_licenses.parquet`` -- one row per image with its licence and attribution.
  Check this before reproducing any scene in a figure: the images come from MS-COCO via the
  Natural Scenes Dataset and carry heterogeneous Flickr licences.

Size and Composition
-------------------------

The release tree as built is **22,488 files, 663.1 GiB**. Five categories account for over
99% of the volume:

.. list-table::
   :header-rows: 1

   * - Category
     - Size
   * - Fixation/saccade epochs (``derivatives/pyavs/.../epochs``)
     - 270.9 GiB
   * - Raw MEG (``sub-0X/ses-0Y/meg``)
     - 254.2 GiB
   * - Maxwell-filtered MEG (``derivatives/pyavs/.../meg``, SSS)
     - 129.1 GiB
   * - Raw eye tracking (``sub-0X/ses-0Y/eyetrack``)
     - 5.3 GiB
   * - Preprocessed eye tracking (``derivatives/pyavs/.../eyetrack``)
     - 1.7 GiB

Everything else -- stimuli, annotations, behavioural logs, ICA solutions, the anatomy tier --
together comes to roughly 1.8 GiB. If you do not need raw MEG or the precomputed epochs, a
usable working subset is small.

The two eye-tracking categories are small because the gaze tables ship as Parquet rather than
CSV (see below); as CSV they were 62.4 GiB and 23.6 GiB respectively.

``manifest.tsv`` at the tree root lists every released file with its category, path within the
release, and size in bytes; use it to check a download is complete. Per-file checksums are
planned for the published version.

File Formats
-----------------

.. list-table::
   :header-rows: 1

   * - Data
     - Format
   * - MEG (raw, SSS, annotations, ICA)
     - FIF (:mod:`mne`)
   * - Epochs
     - HDF5 (:mod:`h5py`; see the epochs section above for the internal layout)
   * - Anatomy
     - FIF + FreeSurfer binary surface/label formats
   * - Eye tracking (native)
     - EyeLink ``.EDF``
   * - Tabular data
     - Parquet -- gaze events/samples, epoch metadata, behavioural logs, licence table
   * - Scene annotations
     - JSON (RLE-encoded masks)
   * - Scenes
     - JPEG

**All tabular data ships as Parquet, not CSV.** Every table in the release was converted:
gaze sample streams and events, epoch metadata, behavioural and transcription logs, ICA
gaze-correlation scores, and the image licence table. ``pandas.read_parquet`` reads them with
the same call shape as ``read_csv``.

The reason is the gaze sample streams: dense numeric time series at 1000 Hz are the worst case
for row-oriented text and the best case for columnar storage. They account for nearly all of
the ~80 GiB the conversion saved -- the two eye-tracking categories dropped from 86.0 GiB as
CSV to 7.0 GiB as Parquet. Beyond size, Parquet carries an embedded schema, so dtypes no longer
have to be re-guessed on every read, and it supports reading one column or one row group
without pulling the whole file.

Two consequences worth stating plainly:

- **Small tables can be slightly larger as Parquet than as CSV.** Parquet has fixed per-file
  footer and schema overhead, which dominates for files of a few hundred bytes (some
  behavioural logs). This was accepted for format consistency across the release.
- **The behavioural logs are raw experiment output**, not a derived export the way the gaze
  tables are of the untouched ``.EDF``. Converting them means that tier no longer ships
  bit-identical to its original on-disk format. The ``.EDF`` files themselves ship untouched,
  so the native eye-tracking format is preserved.

.. note::

   **Still open:** the release may be published with small files grouped into per-subject or
   per-session ZIP archives, to keep the file count manageable for the hosting repository.
   Large files (MEG, epochs, gaze samples) would stay individually addressable so that
   selective download remains possible. This affects packaging, not content. See
   :doc:`../data_access`.

What Is Not Included
-------------------------

Deliberate exclusions, with the reasoning:

- **Session-level concatenated/annotated MEG raws.** Only per-run SSS files ship. The
  session-level ``*_raw-concatenated.fif`` / ``*_raw-annotated.fif`` are reconstructible from
  the per-run SSS files plus the small ``annotations-*.fif`` files, which do ship -- at a
  saving of roughly 80 GB. :class:`pyavs.AVSComposer` rebuilds them.
- **Microphone-locked (``mic_on``) epochs.** Only ``fixation_scene`` and ``saccade_scene``
  epochs ship. Other event-locked epochs can be computed from the released SSS runs and
  annotations.
- **Speech audio (``.wav``).** Voice is identifying; the transcriptions ship instead.
- **Scalp/head surfaces** (``*-head*.fif``, ``*.seghead``) and non-defaced or additional MRI
  volumes -- facial geometry.
- **Volumetric source spaces and forward solutions**, FreeSurfer ``stats/``/``scripts/``, and
  intermediate reconstruction artifacts not used by any pyAVS pipeline.
- **Higher-tier derivatives** -- pupil dynamics, UMAP embeddings, ANN embeddings, fixation
  crops. These are analysis products, reproducible from what ships; several are demonstrated
  in :doc:`../analyses/index`.
- **Non-release participants** -- pilot and test recordings.

Modality-Specific Loading
------------------------------

pyAVS's :mod:`pyavs.dataloader` submodule provides the loading functions for each modality
(:func:`pyavs.load_meg_raw`, :func:`pyavs.load_eye_events`, :func:`pyavs.load_anatomical`),
and :class:`pyavs.AVSComposer` (see :doc:`../package/composer_guide`) combines them into a
single aligned MEG + eye-tracking workflow. See :doc:`../quickstart` to get started.
