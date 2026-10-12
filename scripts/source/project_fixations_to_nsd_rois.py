#!/usr/bin/env python3
"""
Project fixation-locked MEG epochs onto the NSD cortical stream ROIs.

Applies the per-subject LCMV beamformer filter (built from the pre-scene-baseline
noise covariance, see ``compute_scene_onset_noise_cov.py`` ->
``compute_beamformer_filters.py``) to fixation-locked epochs, and reduces each
single-fixation source estimate to one time course per NSD stream ROI.

The seven NSD streams (Allen et al., 2022) are read from each subject's own
FreeSurfer label directory as bilateral pairs::

    {lh,rh}.{L,R}_<stream>_ROI.label
    stream in {early, midventral, ventral, midlateral, lateral,
               midparietal, parietal}

Note the NSD scheme has no "dorsal" stream -- the dorsal pathway is covered by
``parietal`` / ``midparietal``.

Sign ambiguity
--------------
The filter uses ``pick_ori='max-power'``, so each vertex's time-course sign is
arbitrary. A plain signed mean across vertices therefore cancels. ``--roi-mode``
selects how vertices are combined:

``abs_mean`` (default)
    Mean of ``|source|`` -- the "source amplitude" convention used for the
    source-projected fixation ERFs in the dataset paper. Positively biased, so
    the pre-fixation baseline is not zero.
``rms``
    Root-mean-square across vertices. Same sign-insensitivity, different
    weighting of outlying vertices.
``pca_flip``
    MNE's sign-aligned first principal component
    (:func:`mne.extract_label_time_course`). Preserves a signed, zero-baseline
    time course; requires the subject's source space file.
``mean``
    Plain signed mean. Included only for completeness -- expect cancellation.

Pre-requisites (must already exist):
  - Forward model:     compute_forward_model.py  (or the shipped release one)
  - Noise covariance:  compute_scene_onset_noise_cov.py
  - LCMV filter:       compute_beamformer_filters.py

Epochs are rebuilt from the preprocessed raws via ``AVSComposer`` rather than
read from the stored epoch HDF5 files: those store bare grad/mag arrays with
synthetic channel names and no sensor geometry, which a beamformer cannot use.

Usage:
    python project_fixations_to_nsd_rois.py \\
        --subjects 1 --sessions 1 \\
        --data-path /share/klab/datasets/avs/ \\
        --subjects-dir /share/klab/datasets/avs/avs-public/derivatives/freesurfer \\
        --output-dir /share/klab/psulewski/psulewski/pyavs/nsd_roi_timecourses
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import mne
import numpy as np

# Allow running from the scripts/source directory directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from pyavs.layout import sub as fs_subject_name
from pyavs.preprocessing.composer import AVSComposer
from pyavs.utils.logging import get_logger

from compute_beamformer_filters import get_lcmv_filter_path

logger = get_logger('scripts.source.project_fixations_to_nsd_rois')


# The NSD streams atlas, coarse-to-fine along each pathway. No "dorsal" —
# the dorsal pathway is parietal/midparietal.
NSD_STREAMS = [
    'early',
    'midventral',
    'ventral',
    'midlateral',
    'lateral',
    'midparietal',
    'parietal',
]

ROI_MODES = ('abs_mean', 'rms', 'pca_flip', 'mean')


# ============================================================================
# Output path
# ============================================================================

def get_roi_timecourse_path(
    subject_id: int,
    session: int,
    output_dir: str,
    roi_mode: str,
) -> Path:
    """Return the save path for one session's NSD-stream ROI time courses.

    Parameters
    ----------
    subject_id : int
        Subject number (1-based).
    session : int
        Session number (1-based).
    output_dir : str
        Root directory for outputs.
    roi_mode : str
        Vertex-combination mode, recorded in the filename so runs with
        different reductions cannot overwrite each other.

    Returns
    -------
    pathlib.Path
        Full path to the .h5 file.
    """
    stem = (
        f"sub-{subject_id:02d}_ses-{session:02d}_task-avs"
        f"_desc-nsdstreams_mode-{roi_mode.replace('_', '')}_fixation-roitc"
    )
    return Path(output_dir) / f"sub-{subject_id:02d}" / f"ses-{session:02d}" / f"{stem}.h5"


# ============================================================================
# NSD stream labels
# ============================================================================

def load_nsd_stream_labels(
    subject_id: int,
    subjects_dir: str,
    streams: List[str] = NSD_STREAMS,
) -> Dict[str, mne.BiHemiLabel]:
    """Load the bilateral NSD stream labels for one subject.

    Parameters
    ----------
    subject_id : int
        Subject number (1-based).
    subjects_dir : str
        FreeSurfer ``SUBJECTS_DIR``.
    streams : list of str, optional
        Stream names to load (default: all seven in :data:`NSD_STREAMS`).

    Returns
    -------
    dict of str to mne.BiHemiLabel
        One bilateral label per stream, in the order given.
    """
    fs_sub = fs_subject_name(subject_id)
    label_dir = Path(subjects_dir) / fs_sub / 'label'

    labels = {}
    for stream in streams:
        lh = mne.read_label(
            str(label_dir / f'lh.L_{stream}_ROI.label'), subject=fs_sub
        )
        rh = mne.read_label(
            str(label_dir / f'rh.R_{stream}_ROI.label'), subject=fs_sub
        )
        labels[stream] = lh + rh

    logger.info(f"{fs_sub}: loaded {len(labels)} NSD stream labels from {label_dir}")
    return labels


def build_roi_row_indices(
    vertices: List[np.ndarray],
    labels: Dict[str, mne.BiHemiLabel],
) -> Dict[str, np.ndarray]:
    """Map each ROI to its row indices into a source estimate's data array.

    A :class:`~mne.SourceEstimate` stacks hemispheres along axis 0 (all of
    ``vertices[0]`` then all of ``vertices[1]``), so an ROI's rows are the
    positions of its vertices within those arrays — not the vertex numbers
    themselves. Computed once and reused for every epoch.

    Parameters
    ----------
    vertices : list of np.ndarray
        ``[lh_vertno, rh_vertno]`` of the source space the filter was built on.
    labels : dict of str to mne.BiHemiLabel
        Bilateral labels, as returned by :func:`load_nsd_stream_labels`.

    Returns
    -------
    dict of str to np.ndarray
        Row indices per ROI. Raises if any ROI has no vertex in the source
        space, since a silently empty ROI would yield all-zero time courses.
    """
    lh_vertno, rh_vertno = vertices
    n_lh = len(lh_vertno)

    indices = {}
    for name, label in labels.items():
        lh_rows = np.flatnonzero(np.isin(lh_vertno, label.lh.vertices))
        rh_rows = np.flatnonzero(np.isin(rh_vertno, label.rh.vertices))
        rows = np.concatenate([lh_rows, n_lh + rh_rows])

        if rows.size == 0:
            raise ValueError(
                f"ROI '{name}' contains no vertex of the filter's source space "
                f"— label and source space are mismatched."
            )

        indices[name] = rows
        logger.info(
            f"  {name:12s}: {rows.size:5d} source vertices "
            f"({lh_rows.size} lh / {rh_rows.size} rh)"
        )

    return indices


def reduce_to_rois(
    stc_data: np.ndarray,
    roi_indices: Dict[str, np.ndarray],
    roi_names: List[str],
    roi_mode: str,
) -> np.ndarray:
    """Collapse one epoch's source data to one time course per ROI.

    Parameters
    ----------
    stc_data : np.ndarray
        Source data for one epoch, shape ``(n_sources, n_times)``.
    roi_indices : dict of str to np.ndarray
        Row indices per ROI, from :func:`build_roi_row_indices`.
    roi_names : list of str
        ROI order of the output rows.
    roi_mode : str
        One of ``'abs_mean'``, ``'rms'``, ``'mean'``. ``'pca_flip'`` is handled
        by MNE and never reaches this function.

    Returns
    -------
    np.ndarray
        Shape ``(n_rois, n_times)``.
    """
    out = np.empty((len(roi_names), stc_data.shape[1]), dtype=np.float64)

    for i, name in enumerate(roi_names):
        roi = stc_data[roi_indices[name]]
        if roi_mode == 'abs_mean':
            out[i] = np.abs(roi).mean(axis=0)
        elif roi_mode == 'rms':
            out[i] = np.sqrt((roi ** 2).mean(axis=0))
        elif roi_mode == 'mean':
            out[i] = roi.mean(axis=0)
        else:
            raise ValueError(f"Unknown roi_mode for direct reduction: {roi_mode}")

    return out


# ============================================================================
# Epoch loading
# ============================================================================

def load_session_fixation_epochs(
    subject: int,
    session: int,
    data_path: str,
    tmin: float,
    tmax: float,
    verbose: bool = False,
) -> mne.Epochs:
    """Rebuild fixation-locked epochs for one session via ``AVSComposer``.

    Uses the same preprocessing recipe as ``compute_source_erp.py`` and
    ``compute_scene_onset_noise_cov.py`` so the epochs are commensurate with
    the covariance the filter was built from.

    Parameters
    ----------
    subject : int
        Subject ID.
    session : int
        Session number.
    data_path : str
        AVS data root.
    tmin, tmax : float
        Epoch window relative to fixation onset [s].
    verbose : bool
        Enable verbose MNE/composer output.

    Returns
    -------
    mne.Epochs
        Preloaded fixation epochs with eye-tracking metadata attached.
    """
    logger.info(f"sub-{subject:02d} ses-{session:02d}: rebuilding fixation epochs")

    composer = AVSComposer(
        subject=subject,
        session_num=session,
        data_path=data_path,
        output_path=data_path,
        et_path=data_path,
        preprocessed=True,
        recompute_prepro=False,
        verbose=verbose,
        interpolate_bad_channels=True,
        use_precomputed_ica=True,
        apply_ica=False,
        l_freq=0.2,
        h_freq=200,
        causal_filter=False,
        resample_freq=500.0,
    )

    composer.load_meg_data(compute_missing_prepro=False)
    composer.filter_meg_data(ignore_existing_filter=True)
    composer.apply_ica_to_blocks()
    composer.concatenate_raws_per_session()
    composer.find_events_in_raw()
    composer.get_et_annotations(
        event_type='fixation',
        recording='scene',
        exclude_last_fixation=True,
        add_cross_event_info=True,
        preprocessed=True,
        onset_offset='onset',
    )
    composer.make_et_event_epochs(
        tmin=tmin,
        tmax=tmax,
        event_type='fixation',
        recording='scene',
        get_metadata=True,
        baseline=None,
    )

    epochs = composer.et_epochs
    logger.info(f"sub-{subject:02d} ses-{session:02d}: {len(epochs)} fixation epochs")
    return epochs


# ============================================================================
# Per-session projection
# ============================================================================

def project_session(
    subject: int,
    session: int,
    data_path: str,
    subjects_dir: str,
    output_dir: str,
    roi_mode: str,
    tmin: float,
    tmax: float,
    streams: List[str],
    lcmv_root: Optional[str] = None,
    max_epochs: Optional[int] = None,
    overwrite: bool = False,
    verbose: bool = False,
) -> Optional[Path]:
    """Project one session's fixation epochs onto the NSD stream ROIs.

    Parameters
    ----------
    subject : int
        Subject ID.
    session : int
        Session number.
    data_path : str
        AVS data root holding the preprocessed raws the epochs are rebuilt
        from (the ``avs-public`` layout).
    subjects_dir : str
        FreeSurfer ``SUBJECTS_DIR`` holding the NSD stream labels.
    output_dir : str
        Root directory for outputs.
    roi_mode : str
        Vertex-combination mode, one of :data:`ROI_MODES`.
    tmin, tmax : float
        Epoch window relative to fixation onset [s].
    streams : list of str
        NSD streams to extract.
    lcmv_root : str, optional
        Root under which ``derivatives/pyavs/sub-XX/source/`` holds the LCMV
        filter. Defaults to ``data_path``. Needed because the filters and the
        preprocessed raws do not necessarily live under the same root — on the
        lab cluster the filters sit in the internal tree while the raws and
        FreeSurfer labels sit in ``avs-public``.
    max_epochs : int, optional
        Truncate to the first N epochs. For quick validation runs only.
    overwrite : bool
        If False, skip when the output already exists.
    verbose : bool
        Enable verbose MNE output.

    Returns
    -------
    pathlib.Path or None
        Path written, or None if the run was skipped.
    """
    out_path = get_roi_timecourse_path(subject, session, output_dir, roi_mode)

    if out_path.exists() and not overwrite:
        logger.info(f"sub-{subject:02d} ses-{session:02d}: exists, skipping ({out_path})")
        return None

    # --- Filter ---
    filter_path = get_lcmv_filter_path(subject, lcmv_root or data_path)
    if not filter_path.exists():
        raise FileNotFoundError(
            f"LCMV filter not found: {filter_path}\n"
            f"Run compute_beamformer_filters.py for sub-{subject:02d} first."
        )
    filters = mne.beamformer.read_beamformer(str(filter_path))
    logger.info(f"sub-{subject:02d}: loaded filter {filter_path.name} — {filters}")

    # --- Labels and their row indices into the source estimate ---
    labels = load_nsd_stream_labels(subject, subjects_dir, streams)
    roi_names = list(labels.keys())
    roi_indices = build_roi_row_indices(filters['vertices'], labels)
    roi_sizes = np.array([roi_indices[n].size for n in roi_names], dtype=np.int32)

    # --- Epochs ---
    epochs = load_session_fixation_epochs(
        subject, session, data_path, tmin, tmax, verbose=verbose
    )
    if max_epochs is not None:
        logger.warning(f"Truncating to the first {max_epochs} epochs (validation run)")
        epochs = epochs[:max_epochs]

    # Force the filter's exact channel set and order. apply_lcmv_epochs matches
    # on ch_names, so a stray bad-channel difference would otherwise raise deep
    # inside MNE with a less obvious message.
    epochs.pick(filters['ch_names'])

    n_epochs = len(epochs)
    n_times = len(epochs.times)
    roi_timecourses = np.empty((n_epochs, len(roi_names), n_times), dtype=np.float32)

    # --- Beamformer, streamed one epoch at a time ---
    # The full source estimate for a session is ~8196 x 651 x n_epochs floats
    # (tens of GB), so reduce each epoch to ROIs before the next is produced.
    if roi_mode == 'pca_flip':
        # Resolve the source space next to the labels, not via the data root:
        # the internal tree holds the epochs/filters but no derivatives/freesurfer.
        fs_sub = fs_subject_name(subject)
        src_path = Path(subjects_dir) / fs_sub / 'bem' / f'{fs_sub}_oct6-src.fif'
        src = mne.read_source_spaces(str(src_path))

    stcs = mne.beamformer.apply_lcmv_epochs(epochs, filters, return_generator=True)

    for i, stc in enumerate(stcs):
        if roi_mode == 'pca_flip':
            stc.subject = fs_sub
            tc = mne.extract_label_time_course(
                stc, [labels[n] for n in roi_names], src,
                mode='pca_flip', allow_empty=False, verbose=False,
            )
        else:
            tc = reduce_to_rois(stc.data, roi_indices, roi_names, roi_mode)

        roi_timecourses[i] = tc.astype(np.float32)

        if (i + 1) % 500 == 0:
            logger.info(f"  projected {i + 1}/{n_epochs} epochs")

    logger.info(f"sub-{subject:02d} ses-{session:02d}: projected {n_epochs} epochs")

    # --- Save ---
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(out_path, 'w') as f:
        f.create_dataset(
            'roi_timecourses', data=roi_timecourses,
            chunks=(1, len(roi_names), n_times), compression='gzip', compression_opts=4,
        )
        f.create_dataset('times', data=epochs.times.astype(np.float64))
        f.create_dataset('roi_vertex_counts', data=roi_sizes)
        f.attrs['roi_names'] = np.array(roi_names, dtype=h5py.string_dtype())
        f.attrs['roi_mode'] = roi_mode
        f.attrs['subject'] = subject
        f.attrs['session'] = session
        f.attrs['fs_subject'] = fs_subject_name(subject)
        f.attrs['n_epochs'] = n_epochs
        f.attrs['sfreq'] = float(epochs.info['sfreq'])
        f.attrs['tmin'] = float(epochs.times[0])
        f.attrs['tmax'] = float(epochs.times[-1])
        f.attrs['atlas'] = 'nsd_streams'
        f.attrs['inverse'] = 'lcmv'
        f.attrs['lcmv_filter'] = str(filter_path)
        f.attrs['noise_cov'] = 'scene-onset pre-stimulus baseline (-0.4 to 0.0 s), OAS'
        f.attrs['pick_ori'] = str(filters['pick_ori'])
        f.attrs['weight_norm'] = str(filters['weight_norm'])
        f.attrs['filter_subject'] = str(filters['subject'])

    meta_path = out_path.with_name(out_path.stem + '_metadata.csv')
    epochs.metadata.to_csv(meta_path, index=False)

    size_gb = out_path.stat().st_size / 1024 ** 3
    logger.info(
        f"sub-{subject:02d} ses-{session:02d}: wrote {out_path} "
        f"({roi_timecourses.shape}, {size_gb:.2f} GiB) + {meta_path.name}"
    )
    return out_path


# ============================================================================
# CLI
# ============================================================================

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--subjects', '-s', type=int, nargs='+', default=[1],
                   help='Subject IDs (default: 1)')
    p.add_argument('--sessions', '-sess', type=int, nargs='+', default=[1],
                   help='Session numbers (default: 1)')
    p.add_argument('--data-path', '-d', type=str, required=True,
                   help='AVS data root (avs-public layout) holding the preprocessed '
                        'raws the epochs are rebuilt from')
    p.add_argument('--lcmv-root', type=str, default=None,
                   help='Root under which derivatives/pyavs/sub-XX/source/ holds the '
                        'LCMV filter (default: --data-path). Set this when the filters '
                        'live in a different tree than the raws')
    p.add_argument('--subjects-dir', type=str, default=None,
                   help='FreeSurfer SUBJECTS_DIR holding the NSD stream labels '
                        '(default: resolved from SUBJECTS_DIR / the pyAVS layout)')
    p.add_argument('--output-dir', '-o', type=str, required=True,
                   help='Output root directory for the ROI time courses')
    p.add_argument('--roi-mode', type=str, default='abs_mean', choices=ROI_MODES,
                   help='How to combine vertices within an ROI (default: abs_mean)')
    p.add_argument('--streams', type=str, nargs='+', default=NSD_STREAMS,
                   choices=NSD_STREAMS,
                   help='NSD streams to extract (default: all seven)')
    p.add_argument('--tmin', type=float, default=-0.5,
                   help='Epoch start relative to fixation onset [s] (default: -0.5)')
    p.add_argument('--tmax', type=float, default=0.8,
                   help='Epoch end relative to fixation onset [s] (default: 0.8)')
    p.add_argument('--max-epochs', type=int, default=None,
                   help='Truncate to the first N epochs (validation runs only)')
    p.add_argument('--overwrite', action='store_true',
                   help='Overwrite existing outputs')
    p.add_argument('--verbose', '-v', action='store_true',
                   help='Verbose MNE output')
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point."""
    args = parse_args(argv)

    subjects_dir = args.subjects_dir
    if subjects_dir is None:
        from pyavs.utils.paths import get_default_subjects_dir
        subjects_dir = get_default_subjects_dir()
    logger.info(f"SUBJECTS_DIR: {subjects_dir}")
    logger.info(f"ROI mode: {args.roi_mode} | streams: {args.streams}")

    written = []
    for subject in args.subjects:
        for session in args.sessions:
            path = project_session(
                subject=subject,
                session=session,
                data_path=args.data_path,
                subjects_dir=subjects_dir,
                output_dir=args.output_dir,
                roi_mode=args.roi_mode,
                tmin=args.tmin,
                tmax=args.tmax,
                streams=args.streams,
                lcmv_root=args.lcmv_root,
                max_epochs=args.max_epochs,
                overwrite=args.overwrite,
                verbose=args.verbose,
            )
            if path is not None:
                written.append(path)

    logger.info(f"Done: {len(written)} file(s) written")
    return 0


if __name__ == '__main__':
    sys.exit(main())
