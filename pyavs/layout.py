"""
Filesystem layout of the public AVS dataset (``avs-public``).

This module is the single place in pyAVS that knows how a
``(subject, session, datatype, kind)`` tuple maps onto a path. Loaders,
preprocessing and analysis code call :class:`Layout` methods instead of
building path strings themselves.

pyAVS addresses the **public release layout only**. The lab-internal tree
(``rawdir/as01a/``, ``results/as01_01/``, ``input/``, ``AVS-UTILS/``) is no
longer supported; there is no auto-detection and no fallback. Point pyAVS at an
``avs-public`` root.

The layout is BIDS-*inspired*, not BIDS-valid: directories are reorganised but
filenames are preserved, so raw files keep their native scanner/EyeLink names
(``as01b01.fif``, ``as1_1_0.EDF``) while derivatives keep pyAVS'
``sub-XX_ses-YY_*`` names. ``derivatives/freesurfer/`` is the one place that is
renamed (``as0X`` → ``sub-0X``), because it has to work as an MNE
``SUBJECTS_DIR``.

::

    avs-public/
    ├── manifest.tsv
    ├── stimuli/{images,annotations/{coco_objects,cocostuff},avs_scenes_all_licenses.parquet}
    ├── sub-0X/{anat,ses-0Y/{meg,eyetrack,beh}}
    └── derivatives/
        ├── pyavs/sub-0X/ses-0Y/{meg,epochs,eyetrack}
        └── freesurfer/sub-0X/{bem,label,mri/transforms,surf}

Nothing here touches the filesystem: every method is pure string construction
and returns a :class:`~pathlib.Path` whether or not the file exists. The one
exception is :meth:`Layout.ensure_fsaverage`, which downloads ``fsaverage`` on
demand (the release ships ``sub-01``…``sub-05`` but no template subject).

Examples
--------
>>> from pyavs.layout import Layout
>>> layout = Layout('/data/avs-public')
>>> layout.meg_raw(1, 1, 1).relative_to(layout.root)
PosixPath('sub-01/ses-01/meg/as01a01.fif')
>>> layout.forward(1).relative_to(layout.root)
PosixPath('derivatives/freesurfer/sub-01/bem/sub-01-fwd.fif')
"""

import os
from pathlib import Path
from typing import Optional, Union

__all__ = [
    'sub', 'ses', 'session_letter', 'letter_to_session', 'sub_sess_id',
    'bids_stem', 'Layout', 'get_layout',
]

_ALPHABET = 'abcdefghijklmnopqrstuvwxyz'

# Subject/session pairs whose experiment log has a non-standard run suffix.
# Format: (subject_id, session) -> suffix string used in the filename.
_EXPLOG_SUFFIX = {
    (60, 3): '3_11',
    (60, 7): '3_10',
}


def sub(subject_id: int) -> str:
    """Format a subject ID as a BIDS subject label.

    Parameters
    ----------
    subject_id : int
        Subject ID (1-based).

    Returns
    -------
    str
        Subject label, e.g. ``'sub-01'``.
    """
    return f"sub-{subject_id:02d}"


def ses(session: int) -> str:
    """Format a session number as a BIDS session label.

    Parameters
    ----------
    session : int
        Session number (1-based).

    Returns
    -------
    str
        Session label, e.g. ``'ses-01'``.
    """
    return f"ses-{session:02d}"


def session_letter(session: int) -> str:
    """Convert a session number to its MEG-filename letter.

    The MEG naming convention at MPI represents sessions as letters
    (1, 2, 3 → a, b, c).

    Parameters
    ----------
    session : int
        Session number (1-based).

    Returns
    -------
    str
        Session letter.

    Raises
    ------
    ValueError
        If ``session`` is outside 1-26.
    """
    if session < 1 or session > len(_ALPHABET):
        raise ValueError(f"Session number {session} out of range (1-{len(_ALPHABET)})")
    return _ALPHABET[session - 1]


def letter_to_session(letter: str) -> int:
    """Convert a session letter back to a session number.

    Parameters
    ----------
    letter : str
        Session letter.

    Returns
    -------
    int
        Session number (1-based).

    Raises
    ------
    ValueError
        If ``letter`` is not a lowercase ASCII letter.
    """
    if letter not in _ALPHABET:
        raise ValueError(f"Session letter {letter!r} not recognized")
    return _ALPHABET.index(letter) + 1


def sub_sess_id(subject_id: int, session: int, prefix: str = 'as') -> str:
    """Build the native subject-session ID used in raw MEG/ET filenames.

    Parameters
    ----------
    subject_id : int
        Subject ID.
    session : int
        Session number (1-based).
    prefix : str, optional
        Filename prefix (default: ``'as'``).

    Returns
    -------
    str
        e.g. ``'as01a'`` for subject 1, session 1; ``'as01j'`` for session 10.
    """
    return f"{prefix}{subject_id:02d}{session_letter(session)}"


def bids_stem(subject_id: int, session: int, task: Optional[str] = 'avs',
              run: Optional[int] = None, recording: Optional[str] = None) -> str:
    """Build a BIDS-style filename stem for a derivative.

    Parameters
    ----------
    subject_id : int
        Subject ID.
    session : int
        Session number.
    task : str or None, optional
        Task label (default: ``'avs'``). Pass ``None`` to omit the entity.
    run : int or None, optional
        Run (block) number. Omitted when ``None``.
    recording : str or None, optional
        Recording label, used by empty-room files. Omitted when ``None``.

    Returns
    -------
    str
        e.g. ``'sub-01_ses-01_task-avs_run-01'``.
    """
    parts = [sub(subject_id), ses(session)]
    if task:
        parts.append(f"task-{task}")
    if run is not None:
        parts.append(f"run-{run:02d}")
    if recording is not None:
        parts.append(f"recording-{recording}")
    return "_".join(parts)


class Layout:
    """Path resolver for an ``avs-public`` dataset root.

    Parameters
    ----------
    root : str or Path
        Path to the ``avs-public`` dataset root — the directory containing
        ``manifest.tsv``, ``stimuli/``, ``sub-0X/`` and ``derivatives/``.
    derivatives_root : str or Path, optional
        Where pyAVS *writes* its derivatives. Defaults to
        ``root/derivatives/pyavs``. Set this when the dataset copy is
        read-only, so newly computed outputs land somewhere writable while
        shipped derivatives are still read from ``root``.

    Attributes
    ----------
    root : Path
        The dataset root.
    derivatives_root : Path
        Write root for pyAVS derivatives.

    Notes
    -----
    Read methods for shipped derivatives (``meg_sss``, ``ica``,
    ``eye_preprocessed``, ``epochs``, …) resolve under ``derivatives_root``, so
    pointing ``derivatives_root`` elsewhere redirects both reads and writes
    consistently for those products.
    """

    def __init__(self, root: Union[str, Path],
                 derivatives_root: Optional[Union[str, Path]] = None):
        self.root = Path(root)
        self.derivatives_root = (
            Path(derivatives_root) if derivatives_root is not None
            else self.root / 'derivatives' / 'pyavs'
        )

    def __repr__(self) -> str:
        return f"Layout(root={str(self.root)!r}, derivatives_root={str(self.derivatives_root)!r})"

    # === raw data: sub-0X/ses-0Y/ ===========================================

    def subject_dir(self, subject_id: int) -> Path:
        """Subject directory, ``sub-01/``."""
        return self.root / sub(subject_id)

    def session_dir(self, subject_id: int, session: int) -> Path:
        """Session directory, ``sub-01/ses-01/``."""
        return self.subject_dir(subject_id) / ses(session)

    def meg_dir(self, subject_id: int, session: int) -> Path:
        """Raw MEG directory, ``sub-01/ses-01/meg/``."""
        return self.session_dir(subject_id, session) / 'meg'

    def meg_raw(self, subject_id: int, session: int, run: int) -> Path:
        """Raw MEG run, ``sub-01/ses-01/meg/as01a01.fif``.

        Parameters
        ----------
        subject_id : int
            Subject ID.
        session : int
            Session number.
        run : int
            Run (block) number, 1-based.
        """
        return self.meg_dir(subject_id, session) / f"{sub_sess_id(subject_id, session)}{run:02d}.fif"

    def meg_empty_room(self, subject_id: int, session: int, recording: str) -> Path:
        """Raw empty-room recording, ``sub-01/ses-01/meg/as01ab.fif``.

        Parameters
        ----------
        subject_id : int
            Subject ID.
        session : int
            Session number.
        recording : {'b', 'd'}
            ``'b'`` = before (*bevor*), ``'d'`` = after (*danach*) the session.
        """
        return self.meg_dir(subject_id, session) / f"{sub_sess_id(subject_id, session)}{recording}.fif"

    def eyetrack_dir(self, subject_id: int, session: int) -> Path:
        """Raw eye-tracking directory, ``sub-01/ses-01/eyetrack/``."""
        return self.session_dir(subject_id, session) / 'eyetrack'

    def _eye_stem(self, subject_id: int, session: int, prefix: str = 'as') -> str:
        """Native EyeLink stem, ``as1_1_0`` — subject/session are unpadded."""
        return f"{prefix}{subject_id}_{session}_0"

    def eye_edf(self, subject_id: int, session: int, prefix: str = 'as') -> Path:
        """Raw EyeLink EDF, ``sub-01/ses-01/eyetrack/as1_1_0.EDF``."""
        return self.eyetrack_dir(subject_id, session) / f"{self._eye_stem(subject_id, session, prefix)}.EDF"

    def eye_raw(self, subject_id: int, session: int, kind: str,
                prefix: str = 'as') -> Path:
        """Raw EyeLink export table.

        Parameters
        ----------
        subject_id : int
            Subject ID.
        session : int
            Session number.
        kind : {'events', 'messages', 'samples'}
            Which export to address.
        prefix : str, optional
            Filename prefix (default: ``'as'``).

        Returns
        -------
        Path
            e.g. ``sub-01/ses-01/eyetrack/as1_1_0_events.parquet``.
        """
        stem = self._eye_stem(subject_id, session, prefix)
        return self.eyetrack_dir(subject_id, session) / f"{stem}_{kind}.parquet"

    def beh_dir(self, subject_id: int, session: int) -> Path:
        """Behavioural directory, ``sub-01/ses-01/beh/``."""
        return self.session_dir(subject_id, session) / 'beh'

    def explog(self, subject_id: int, session: int, prefix: str = 'as') -> Path:
        """Experiment log, ``sub-01/ses-01/beh/as_exp_data_1_1_3_0.parquet``."""
        suffix = _EXPLOG_SUFFIX.get((subject_id, session), '3_0')
        filename = f"{prefix}_exp_data_{subject_id}_{session}_{suffix}.parquet"
        return self.beh_dir(subject_id, session) / filename

    def explog_transcribed(self, subject_id: int, session: int,
                           variant: str = 'corrected') -> Path:
        """Transcribed experiment log.

        Parameters
        ----------
        subject_id : int
            Subject ID.
        session : int
            Session number.
        variant : str, optional
            Transcription variant (default: ``'corrected'``).

        Returns
        -------
        Path
            e.g. ``sub-01/ses-01/beh/explog_transcribed_corrected_01_01.parquet``.
        """
        filename = f"explog_transcribed_{variant}_{subject_id:02d}_{session:02d}.parquet"
        return self.beh_dir(subject_id, session) / filename

    def anat_dir(self, subject_id: int) -> Path:
        """Anatomy directory, ``sub-01/anat/``."""
        return self.subject_dir(subject_id) / 'anat'

    def anat_t1(self, subject_id: int) -> Path:
        """Defaced T1, ``sub-01/anat/T1.mgz``."""
        return self.anat_dir(subject_id) / 'T1.mgz'

    # === pyAVS derivatives: derivatives/pyavs/sub-0X/ses-0Y/ ================

    def deriv_dir(self, subject_id: int, session: Optional[int] = None,
                  datatype: Optional[str] = None) -> Path:
        """Directory under the pyAVS derivatives root.

        Parameters
        ----------
        subject_id : int
            Subject ID.
        session : int, optional
            Session number. Omitted from the path when ``None``.
        datatype : str, optional
            Datatype directory (``'meg'``, ``'epochs'``, ``'eyetrack'``, …).
            Omitted when ``None``.

        Returns
        -------
        Path
            e.g. ``derivatives/pyavs/sub-01/ses-01/meg``.

        Notes
        -----
        The shape is always ``sub-XX/ses-XX/{datatype}``. pyAVS previously also
        used ``{datatype}/sub-XX/ses-XX`` in places, which disagreed with its
        own readers; the public release uses this shape.
        """
        path = self.derivatives_root / sub(subject_id)
        if session is not None:
            path = path / ses(session)
        if datatype is not None:
            path = path / datatype
        return path

    def deriv_meg_dir(self, subject_id: int, session: int) -> Path:
        """Derived MEG directory, ``derivatives/pyavs/sub-01/ses-01/meg/``."""
        return self.deriv_dir(subject_id, session, 'meg')

    def meg_sss(self, subject_id: int, session: int, run: int) -> Path:
        """Maxwell-filtered run.

        Returns
        -------
        Path
            ``derivatives/pyavs/sub-01/ses-01/meg/sub-01_ses-01_task-avs_run-01_raw-sss.fif``.
        """
        stem = bids_stem(subject_id, session, task='avs', run=run)
        return self.deriv_meg_dir(subject_id, session) / f"{stem}_raw-sss.fif"

    def meg_sss_empty_room(self, subject_id: int, session: int, recording: str) -> Path:
        """Maxwell-filtered empty-room recording.

        Parameters
        ----------
        subject_id : int
            Subject ID.
        session : int
            Session number.
        recording : {'b', 'd'}
            Before / after the session.

        Returns
        -------
        Path
            ``derivatives/pyavs/sub-01/ses-01/meg/sub-01_ses-01_task-noise_recording-b_raw-sss.fif``.
        """
        stem = bids_stem(subject_id, session, task='noise', recording=recording)
        return self.deriv_meg_dir(subject_id, session) / f"{stem}_raw-sss.fif"

    def meg_annotations(self, subject_id: int, session: int,
                        recording_type: str = 'scene') -> Path:
        """Session annotations FIF.

        Parameters
        ----------
        subject_id : int
            Subject ID.
        session : int
            Session number.
        recording_type : {'scene', 'microphone', 'caption'}, optional
            Which annotation set (default: ``'scene'``).

        Returns
        -------
        Path
            ``derivatives/pyavs/sub-01/ses-01/meg/sub-01_ses-01_task-avs_annotations-scene.fif``.

        Notes
        -----
        The release does not ship the session-level ``*_raw-annotated.fif`` or
        ``*_raw-concatenated.fif`` files — both are rebuildable from the
        per-run SSS files plus these annotations.
        """
        stem = bids_stem(subject_id, session, task='avs')
        return self.deriv_meg_dir(subject_id, session) / f"{stem}_annotations-{recording_type}.fif"

    def ica(self, subject_id: int, session: int) -> Path:
        """Precomputed ICA solution, ``…/meg/sub-01_ses-01_task-avs_ica.fif``."""
        stem = bids_stem(subject_id, session, task='avs')
        return self.deriv_meg_dir(subject_id, session) / f"{stem}_ica.fif"

    def ica_exclusions(self, subject_id: int, session: int) -> Path:
        """ICA exclusions, ``…/meg/sub-01_ses-01_task-avs_ica-exclusions.json``."""
        stem = bids_stem(subject_id, session, task='avs')
        return self.deriv_meg_dir(subject_id, session) / f"{stem}_ica-exclusions.json"

    def ica_scores(self, subject_id: int, session: int) -> Path:
        """ICA gaze-correlation scores, ``…/meg/sub-01_ses-01_task-avs_ica-et-scores.parquet``."""
        stem = bids_stem(subject_id, session, task='avs')
        return self.deriv_meg_dir(subject_id, session) / f"{stem}_ica-et-scores.parquet"

    def deriv_eyetrack_dir(self, subject_id: int, session: int) -> Path:
        """Preprocessed ET directory, ``derivatives/pyavs/sub-01/ses-01/eyetrack/``."""
        return self.deriv_dir(subject_id, session, 'eyetrack')

    def eye_preprocessed(self, subject_id: int, session: int, kind: str,
                         prefix: str = 'as') -> Path:
        """Preprocessed eye-tracking table.

        Parameters
        ----------
        subject_id : int
            Subject ID.
        session : int
            Session number.
        kind : {'events', 'msgs', 'samples', 'cleaned_samples'}
            Which table to address.
        prefix : str, optional
            Filename prefix (default: ``'as'``).

        Returns
        -------
        Path
            e.g. ``derivatives/pyavs/sub-01/ses-01/eyetrack/as_s1_el_events.parquet``.

        Notes
        -----
        The filename carries only the subject, not the session — the session is
        expressed by the directory.
        """
        filename = f"{prefix}_s{subject_id}_el_{kind}.parquet"
        return self.deriv_eyetrack_dir(subject_id, session) / filename

    def epochs_dir(self, subject_id: int, session: int) -> Path:
        """Epochs directory, ``derivatives/pyavs/sub-01/ses-01/epochs/``."""
        return self.deriv_dir(subject_id, session, 'epochs')

    def epochs(self, subject_id: int, session: int, event_type: str) -> Path:
        """Epoched MEG data.

        Parameters
        ----------
        subject_id : int
            Subject ID.
        session : int
            Session number.
        event_type : str
            Event type, e.g. ``'fixation_scene'`` or ``'saccade_scene'``.
            ``_scene`` means "during the scene task", not scene-onset-locked.

        Returns
        -------
        Path
            ``derivatives/pyavs/sub-01/ses-01/epochs/sub-01_ses-01_task-avs_fixation_scene_epochs.h5``.
        """
        stem = bids_stem(subject_id, session, task='avs')
        return self.epochs_dir(subject_id, session) / f"{stem}_{event_type}_epochs.h5"

    def epochs_metadata(self, subject_id: int, session: int, event_type: str) -> Path:
        """Per-epoch metadata table.

        Parameters
        ----------
        subject_id : int
            Subject ID.
        session : int
            Session number.
        event_type : str
            Event type; a trailing ``_scene`` is stripped for the filename.

        Returns
        -------
        Path
            ``derivatives/pyavs/sub-01/ses-01/epochs/sub-01_ses-01_fixation_metadata.parquet``.
        """
        base_event_type = event_type.replace('_scene', '')
        filename = f"{sub(subject_id)}_{ses(session)}_{base_event_type}_metadata.parquet"
        return self.epochs_dir(subject_id, session) / filename

    # === FreeSurfer: derivatives/freesurfer/ ================================

    @property
    def subjects_dir(self) -> Path:
        """FreeSurfer ``SUBJECTS_DIR``, ``derivatives/freesurfer/``.

        Directly usable as an MNE ``subjects_dir``.
        """
        return self.root / 'derivatives' / 'freesurfer'

    def fs_subject(self, subject_id: int) -> str:
        """FreeSurfer subject name for ``subject_id``, e.g. ``'sub-01'``.

        The release renames the internal ``as0X`` FreeSurfer directories to
        ``sub-0X``, rewriting the ``subject_his_id`` baked into the src/fwd
        FIFs to match.
        """
        return sub(subject_id)

    def fs_dir(self, subject_id: int) -> Path:
        """FreeSurfer subject directory, ``derivatives/freesurfer/sub-01/``."""
        return self.subjects_dir / self.fs_subject(subject_id)

    def bem_dir(self, subject_id: int) -> Path:
        """BEM directory, ``derivatives/freesurfer/sub-01/bem/``."""
        return self.fs_dir(subject_id) / 'bem'

    def forward(self, subject_id: int) -> Path:
        """Forward solution, ``…/sub-01/bem/sub-01-fwd.fif``."""
        return self.bem_dir(subject_id) / f"{self.fs_subject(subject_id)}-fwd.fif"

    def src(self, subject_id: int, spacing: str = 'oct6') -> Path:
        """Source space, ``…/sub-01/bem/sub-01_oct6-src.fif``.

        Parameters
        ----------
        subject_id : int
            Subject ID.
        spacing : str, optional
            Source-space spacing (default: ``'oct6'``, the only one shipped).
        """
        return self.bem_dir(subject_id) / f"{self.fs_subject(subject_id)}_{spacing}-src.fif"

    def bem_sol(self, subject_id: int) -> Path:
        """BEM solution, ``…/sub-01/bem/sub-01-bem-sol.fif``.

        Single-shell (inner skull) — the scalp surfaces are withheld from the
        release because they carry facial detail.
        """
        return self.bem_dir(subject_id) / f"{self.fs_subject(subject_id)}-bem-sol.fif"

    def trans(self, subject_id: int) -> Path:
        """MEG↔MRI transform, ``…/sub-01/mri/transforms/sub-01-trans.fif``."""
        return (self.fs_dir(subject_id) / 'mri' / 'transforms'
                / f"{self.fs_subject(subject_id)}-trans.fif")

    def label_dir(self, subject_id: int) -> Path:
        """FreeSurfer label directory, ``…/sub-01/label/``."""
        return self.fs_dir(subject_id) / 'label'

    def surf_dir(self, subject_id: int) -> Path:
        """FreeSurfer surface directory, ``…/sub-01/surf/``."""
        return self.fs_dir(subject_id) / 'surf'

    def morph_map(self, subject_id: int, template: str = 'fsaverage') -> Path:
        """Precomputed morph map, ``derivatives/freesurfer/morph-maps/fsaverage-sub-01-morph.fif``.

        Parameters
        ----------
        subject_id : int
            Subject ID.
        template : str, optional
            Template subject (default: ``'fsaverage'``).
        """
        return (self.subjects_dir / 'morph-maps'
                / f"{template}-{self.fs_subject(subject_id)}-morph.fif")

    def ensure_fsaverage(self, verbose: Optional[bool] = None) -> Path:
        """Make sure ``fsaverage`` exists in :attr:`subjects_dir`, fetching it if not.

        The release ships ``sub-01``…``sub-05`` but no template subject, while
        the shipped morph maps assume ``fsaverage`` is present in
        ``SUBJECTS_DIR``. MNE ships its own copy; this downloads it on first
        use.

        Parameters
        ----------
        verbose : bool, optional
            Passed through to MNE.

        Returns
        -------
        Path
            Path to the ``fsaverage`` directory.
        """
        fsaverage_dir = self.subjects_dir / 'fsaverage'
        if fsaverage_dir.exists():
            return fsaverage_dir

        import mne
        self.subjects_dir.mkdir(parents=True, exist_ok=True)
        return Path(mne.datasets.fetch_fsaverage(
            subjects_dir=str(self.subjects_dir), verbose=verbose))

    # === stimuli ============================================================

    @property
    def stimuli_dir(self) -> Path:
        """Stimuli root, ``stimuli/``."""
        return self.root / 'stimuli'

    @property
    def scenes_dir(self) -> Path:
        """Scene images, ``stimuli/images/`` (4,080 MEG-size JPEGs)."""
        return self.stimuli_dir / 'images'

    def scene_image(self, scene_id: Union[int, str]) -> Path:
        """Scene image for a COCO image ID.

        Parameters
        ----------
        scene_id : int or str
            COCO image ID. Integers are zero-padded to 12 digits; strings are
            used as given, so an already-padded ID also works.

        Returns
        -------
        Path
            e.g. ``stimuli/images/000000000151_MEG_size.jpg``.
        """
        stem = f"{int(scene_id):012d}" if not isinstance(scene_id, str) else scene_id
        return self.scenes_dir / f"{stem}_MEG_size.jpg"

    def annotations_dir(self, kind: str = 'cocostuff') -> Path:
        """Transformed scene annotations.

        Parameters
        ----------
        kind : {'cocostuff', 'coco_objects'}, optional
            Annotation set (default: ``'cocostuff'``).

        Returns
        -------
        Path
            e.g. ``stimuli/annotations/cocostuff``.
        """
        return self.stimuli_dir / 'annotations' / kind

    def scene_annotation(self, scene_id: Union[int, str],
                         kind: str = 'cocostuff') -> Path:
        """Transformed annotation JSON for one scene.

        Parameters
        ----------
        scene_id : int or str
            COCO image ID. Unlike :meth:`scene_image`, annotation filenames use
            the *unpadded* ID.
        kind : {'cocostuff', 'coco_objects'}, optional
            Annotation set (default: ``'cocostuff'``).

        Returns
        -------
        Path
            e.g. ``stimuli/annotations/cocostuff/151_transformed.json``.
        """
        stem = str(int(scene_id)) if not isinstance(scene_id, str) else scene_id
        return self.annotations_dir(kind) / f"{stem}_transformed.json"

    def scene_licenses(self) -> Path:
        """Per-image license table, ``stimuli/avs_scenes_all_licenses.parquet``."""
        return self.stimuli_dir / 'avs_scenes_all_licenses.parquet'

    def manifest(self) -> Path:
        """Release manifest, ``manifest.tsv``."""
        return self.root / 'manifest.tsv'


def get_layout(data_path: Optional[Union[str, Path]] = None,
               derivatives_root: Optional[Union[str, Path]] = None) -> Layout:
    """Build a :class:`Layout`, falling back to the configured data path.

    Parameters
    ----------
    data_path : str or Path, optional
        ``avs-public`` root. If ``None``, taken from the global pyAVS config
        (set via :func:`pyavs.configure` / :func:`pyavs.set_data_path`, or the
        ``PYAVS_DATA_PATH`` environment variable).
    derivatives_root : str or Path, optional
        Write root for pyAVS derivatives. If ``None``, taken from the global
        config's ``derivatives_path`` (env ``PYAVS_DERIVATIVES_PATH``), and
        otherwise defaults to ``<root>/derivatives/pyavs``.

    Returns
    -------
    Layout

    Raises
    ------
    ValueError
        If no data path is given and none is configured.
    """
    if data_path is not None:
        # An explicit root is self-contained: never mix in a derivatives root
        # configured for some *other* dataset.
        return Layout(data_path,
                      derivatives_root or os.environ.get('PYAVS_DERIVATIVES_PATH'))

    # Deferred import: config imports utils, which imports layout.
    from .config.manager import get_config

    cfg = get_config().config

    if cfg.data_path is None:
        raise ValueError(
            "No data path configured. Point pyAVS at an avs-public root via "
            "pyavs.set_data_path('/path/to/avs-public'), pyavs.configure(), or "
            "the PYAVS_DATA_PATH environment variable."
        )

    if derivatives_root is None:
        derivatives_root = os.environ.get('PYAVS_DERIVATIVES_PATH') or cfg.derivatives_path

    return Layout(cfg.data_path, derivatives_root)
