Data Access
===============

.. warning::

   **The AVS dataset is not yet publicly downloadable.** The release tree has been assembled
   and verified, and upload to the hosting repository is in preparation. This page describes
   what will be released and how it will be distributed; the concrete download link and DOI
   will be added here once the dataset is published. Until then, nothing on this page should
   be treated as a live access route.

Where It Will Be Hosted
----------------------------

The dataset is being deposited on **GRO.data** (`Göttingen Research Online / Data
<https://data.goettingen-research-online.de/dataverse/avs>`_), the GWDG-operated Dataverse
repository, in the existing **AVS collection** -- the same collection that already holds the
published derivative dataset for Sulewski et al., *Fixation duration on natural scenes is
explained by memory encoding not processing demand*.

What that means for you as a user:

- **A citable DOI**, minted per dataset version. Cite the version you actually used.
- **Versioning is repository-native.** Published versions are immutable; corrections and
  additions appear as new versions with their own DOIs, and files shared between versions are
  not duplicated. There is no separate version-control scheme to learn.
- **Programmatic download.** Dataverse exposes a REST API, so the whole tree -- or any subset
  -- can be fetched from a script rather than clicked through a web UI. The shipped
  ``manifest.tsv`` (see :doc:`dataset/overview`) maps 1:1 onto the file paths in the deposit,
  which makes it a convenient driver for a scripted, resumable download.

Additional mirroring for direct cloud object access is being explored; if it materializes it
will be listed here alongside the primary deposit.

Planning Your Download
---------------------------

The full release is **663.1 GiB**. Most users will not want all of it, and the layout is
designed so you do not have to take it:

.. list-table::
   :header-rows: 1

   * - If you want to...
     - Download
   * - Work with fixation/saccade-locked MEG epochs
     - ``derivatives/pyavs/sub-0X/ses-0Y/epochs/`` (270.9 GiB total; per-session files)
   * - Filter epochs *before* downloading any MEG
     - The ``*_metadata.parquet`` files only -- a few hundred KB each, and they carry ``sceneID``,
       ``object_label`` and all event kinematics
   * - Do your own MEG preprocessing from a clean starting point
     - ``derivatives/pyavs/.../meg/`` (SSS + annotations + ICA, 129.1 GiB)
   * - Start from genuinely unprocessed data
     - ``sub-0X/ses-0Y/meg/`` (254.2 GiB)
   * - Do eye-tracking-only analyses
     - ``derivatives/pyavs/.../eyetrack/`` (1.7 GiB) plus ``stimuli/``
   * - Source-reconstruct
     - ``derivatives/freesurfer/`` (< 1 GiB) plus whichever sensor data you need
   * - Work on stimuli/annotations alone
     - ``stimuli/`` (a few hundred MB)

Because the epoch metadata tables are tiny and separate from the epoch data itself, the
practical workflow is: download metadata, work out which sessions and epochs you need, then
fetch only those files.

.. note::

   Small files may be published grouped into ZIP archives (per subject/session) to keep the
   file count manageable for the repository, while large files stay individually addressable.
   This affects how you fetch things, not what you get. See :doc:`dataset/overview` for the
   current state of that decision.

Verifying a Download
-------------------------

``manifest.tsv`` at the root of the release lists every file with its path and size in bytes,
which is enough to confirm a download is complete and untruncated. Per-file checksums are
planned for the published version; the hosting repository additionally records a checksum per
file and exposes it through its API.

What Is Released
---------------------

In short: raw and Maxwell-filtered MEG, raw and preprocessed eye tracking, fixation- and
saccade-locked epochs with per-fixation object labels, behavioural and transcribed-caption
logs, the 4,080 scene images with their COCO/COCO-Stuff annotations and licence table,
defaced structural MRIs, and a ready-to-use FreeSurfer ``SUBJECTS_DIR`` for source
reconstruction.

:doc:`dataset/overview` documents each of these in detail, including the deliberate
exclusions (speech audio, scalp/head surfaces, session-level concatenated raws) and why they
were made.

Terms of Use
-----------------

Terms for the released data are still being finalized -- see :doc:`reference/terms_of_use`.
Note separately that the scene images carry their own upstream licences; see
``stimuli/avs_scenes_all_licenses.parquet`` before reproducing any of them.

Code
--------

The pyAVS package and the analysis code used in the dataset paper are already public:

.. code-block:: bash

   pip install pyavs
   # or, for the latest development version:
   git clone https://github.com/KietzmannLab/pyavs.git

See :doc:`installation` and :doc:`quickstart` to get started once you have data access.

Staying Updated
--------------------

Watch or star the `pyavs GitHub repository <https://github.com/KietzmannLab/pyavs>`_ for
updates, or check back on this page.
