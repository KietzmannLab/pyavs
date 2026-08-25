Data Access
===============

.. note::

   **The full AVS dataset -- all 5 subjects, all 10 sessions each -- is publicly available
   today**, on demand, via :func:`pyavs.open_remote` -- no manual staging, no AWS account or
   credentials required, and no request or registration needed. See
   :ref:`data-access-available-now` below for a working example.

.. _data-access-available-now:

On-Demand Access via AWS
---------------------------------------------

The release bucket ``kietzmannlab-avs`` (region ``us-west-2``) is public-read on Amazon S3, as
part of the `AWS Open Data Sponsorship Program <https://aws.amazon.com/opendata/>`_ (its
Registry of Open Data listing is pending review; the bucket itself is fully live and usable
regardless of that listing's status). It holds the complete released dataset -- all 5 subjects,
all 10 sessions each -- raw MEG, Maxwell-filtered MEG, raw and preprocessed eye tracking,
fixation- and saccade-locked epochs, and the experiment/behavioural logs, plus the scene-image
licence table and the full epoch catalog.

pyAVS's built-in remote client, :func:`pyavs.open_remote` (:class:`pyavs.remote.AVSRemote`),
fetches each file a call needs directly from the bucket the first time it's needed, caching it
locally (default ``~/.cache/pyavs/kietzmannlab-avs``) so repeat calls and reruns skip the
network entirely:

.. code-block:: python

   import pyavs

   avs = pyavs.open_remote()

   # Same call signatures as the local loaders -- just no set_data_path() needed first.
   raw = avs.load_meg_raw(subject_id=1, session=1, run=1, preload=True)
   explog = avs.load_experiment_log(subject_id=1, session=1)
   epochs = avs.load_epochs(subject_id=1, session=1, event_type='fixation_scene')

   # Once populated, the cache is itself a valid (partial) data_path:
   pyavs.set_data_path(avs.data_path)

For fixation/saccade-locked epochs specifically, :meth:`~pyavs.remote.AVSRemote.epochs` opens a
content-indexed query over the small pre-built catalog -- filtering by fixated object, scene,
or kinematics happens locally, and only the *matching* epochs are range-read from S3, without
downloading whole session files:

.. code-block:: python

   dogs = avs.epochs(event_type='fixation_scene', subject_id=1).where("object_label == 'dog'")
   dog_epochs = dogs.load()  # range-reads only the matching epochs

A query for a subject/session/object combination that doesn't exist in the released data raises
:class:`~pyavs.remote.RemoteFileNotFoundError` on ``.load()``. See :doc:`quickstart` for a
fuller walkthrough, the `Colab quickstart notebook
<https://colab.research.google.com/github/KietzmannLab/pyavs/blob/main/examples/pyavs_colab_quickstart.ipynb>`__
for an interactive, no-install version, and :doc:`api/remote` for the full API.

Where the Dataset Is Hosted
-------------------------------------------

**AWS Open Data is the sole hosting track for the AVS release.** The complete dataset lives on
the public ``kietzmannlab-avs`` S3 bucket described above -- no separate repository, no
registration, and no full-tree download required, since :func:`pyavs.open_remote` already
supports fetching exactly the subjects/sessions/files you need.

The dataset is part of the `AWS Open Data Sponsorship Program
<https://aws.amazon.com/opendata/>`_; its listing on the `Registry of Open Data on AWS
<https://registry.opendata.aws/>`_ is pending review and will become the citable reference
point for the dataset going forward once merged (alongside the dataset paper itself -- see
:doc:`reference/citation`).

The shipped ``manifest.tsv`` (see :doc:`dataset/overview`) maps 1:1 onto the S3 object keys,
which makes it a convenient driver for a scripted, resumable ``aws s3 cp``/``sync`` download of
the parts you need, in addition to the on-demand :func:`pyavs.open_remote` route.

Planning Your Download
---------------------------

The full release is **663.1 GiB** across all 5 subjects. Most users will not want all of it,
and the layout (see :ref:`data-access-available-now`) is designed so you do not have to take
it:

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

See :doc:`reference/terms_of_use` for the license (CC BY 4.0), attribution requirements, and
ethical-use terms. Note separately that the scene images carry their own upstream licences; see
``stimuli/avs_scenes_all_licenses.parquet`` before reproducing any of them.

Code
--------

The pyAVS package and the analysis code used in the dataset paper are already public:

.. code-block:: bash

   pip install pyavs
   # or, for the latest development version:
   git clone https://github.com/KietzmannLab/pyavs.git

See :doc:`installation` and :doc:`quickstart` to get started.

Staying Updated
--------------------

Watch or star the `pyavs GitHub repository <https://github.com/KietzmannLab/pyavs>`_ for
updates, or check back on this page.
