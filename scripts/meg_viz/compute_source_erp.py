#!/usr/bin/env python3
"""
Compute source-reconstructed fixation/saccade ERP for MEG data.

This script runs on the HPC cluster. It:
  1. Loads preprocessed MEG epochs using AVSComposer
  2. Computes the sensor-level evoked response (ERP) per subject
  3. Applies pre-computed LCMV beamformer filters to project to source space
  4. Morphs each subject's source estimate to fsaverage (or a chosen target)
  5. Saves the morphed .stc files to disk

The resulting .stc files can be loaded locally (via SSH mount) for brain
visualization using plot_source_erp_brain.ipynb.

Usage:
    python compute_source_erp.py \\
        --subjects 1 2 3 4 5 \\
        --sessions 1 2 3 4 5 6 7 8 9 10 \\
        --event-type fixation \\
        --filters-dir /share/klab/datasets/avs/derivatives/pyavs/filters/fixation_HASH/ \\
        --subjects-dir /share/klab/datasets/avs/rawdir/ \\
        --output-dir /share/klab/psulewski/psulewski/pyavs/source_erp/

Author: P. Sulewski (psulewski@uos.de)
"""

import argparse
import os
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import numpy as np
import mne
import logging
from joblib import Parallel, delayed

import pyavs
from pyavs.preprocessing.composer import AVSComposer
from pyavs.utils.logging import get_logger

logger = get_logger('scripts.meg_viz.compute_source_erp')

# Subject ID to FreeSurfer subject name
SUBJECT_FS_MAPPING = {i: f'as{i:02d}' for i in range(1, 20)}


# ============================================================================
# Epoch loading (mirrors plot_event_erps.py pattern)
# ============================================================================

def load_session_epochs(
    subject: int,
    session: int,
    event_type: str,
    data_path: str,
    tmin: float = -0.2,
    tmax: float = 0.5,
    use_offset: bool = False,
    verbose: bool = False,
) -> Optional[mne.Epochs]:
    """
    Load MEG epochs for one subject/session using AVSComposer.

    Parameters
    ----------
    subject : int
        Subject ID
    session : int
        Session number
    event_type : str
        Event type: 'fixation', 'saccade', 'blink', or 'scene'
    data_path : str
        Path to AVS data directory
    tmin : float
        Epoch start time [s] (default: -0.2)
    tmax : float
        Epoch end time [s] (default: 0.5)
    use_offset : bool
        Use event offset timing instead of onset (default: False)
    verbose : bool
        Enable verbose output

    Returns
    -------
    mne.Epochs or None
    """
    try:
        logger.info(
            f"Loading sub-{subject:02d} ses-{session:02d} "
            f"event={event_type} offset={use_offset}"
        )

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
            event_type=event_type,
            recording='scene',
            exclude_last_fixation=True,
            add_cross_event_info=True,
            preprocessed=True,
            onset_offset='offset' if use_offset else 'onset',
        )
        composer.make_et_event_epochs(
            tmin=tmin,
            tmax=tmax,
            event_type=event_type,
            recording='scene',
            get_metadata=True,
            baseline=None,
        )

        n = len(composer.et_epochs)
        logger.info(f"  Created {n} epochs")
        return composer.et_epochs

    except Exception as e:
        logger.error(
            f"Error loading sub-{subject:02d} ses-{session:02d}: {e}"
        )
        return None


def _load_session_wrapper(
    subject: int, session: int, event_type: str, data_path: str,
    tmin: float, tmax: float, use_offset: bool, verbose: bool,
) -> Tuple[int, int, Optional[mne.Epochs]]:
    epochs = load_session_epochs(
        subject=subject, session=session, event_type=event_type,
        data_path=data_path, tmin=tmin, tmax=tmax,
        use_offset=use_offset, verbose=verbose,
    )
    return subject, session, epochs


# ============================================================================
# Source reconstruction helpers
# ============================================================================

def load_lcmv_filter(
    subject_id: int,
    filter_session: int,
    filters_dir: str,
) -> mne.beamformer.Beamformer:
    """
    Load a pre-computed LCMV beamformer filter.

    Parameters
    ----------
    subject_id : int
        Subject ID
    filter_session : int
        Session whose filter to load (1-indexed)
    filters_dir : str
        Root of the filters directory (event-type + hash subdirectory)

    Returns
    -------
    mne.beamformer.Beamformer
    """
    filter_path = (
        Path(filters_dir)
        / f"sub-{subject_id:02d}"
        / f"lcmv_filters_ses-{filter_session:02d}-lcmv.h5"
    )

    if not filter_path.exists():
        raise FileNotFoundError(
            f"LCMV filter not found: {filter_path}\n"
            f"  Check --filters-dir and --filter-session."
        )

    logger.info(f"  Loading filter: {filter_path.name}")
    lcmv_filter = mne.beamformer.read_beamformer(str(filter_path))
    logger.info(f"  Filter has {lcmv_filter['n_sources']} sources")
    return lcmv_filter


def compute_subject_source_erp(
    subject: int,
    sessions: List[int],
    event_type: str,
    data_path: str,
    filters_dir: str,
    filter_session: int,
    tmin: float,
    tmax: float,
    baseline: Optional[Tuple[float, float]],
    use_offset: bool,
    verbose: bool,
    n_jobs: int,
) -> Optional[mne.SourceEstimate]:
    """
    Compute sensor-level ERP and project to source space for one subject.

    Parameters
    ----------
    subject : int
        Subject ID
    sessions : list of int
        Sessions to include
    event_type : str
        Event type
    data_path : str
        AVS data path
    filters_dir : str
        Root of pre-computed LCMV filters
    filter_session : int
        Which session's filter to use
    tmin, tmax : float
        Epoch time window [s]
    baseline : tuple or None
        Baseline correction window (start, end) in seconds. None = no correction.
    use_offset : bool
        Use event offset timing
    verbose : bool
        Verbose logging
    n_jobs : int
        Parallel jobs for epoch loading

    Returns
    -------
    mne.SourceEstimate or None
        Source-level ERP (not yet morphed)
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing subject {subject}")
    logger.info(f"{'='*60}")

    # --- 1. Load epochs across all sessions ---
    pairs = [(subject, sess) for sess in sessions]
    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_load_session_wrapper)(
            subject=s, session=sess, event_type=event_type,
            data_path=data_path, tmin=tmin, tmax=tmax,
            use_offset=use_offset, verbose=verbose,
        )
        for s, sess in pairs
    )

    all_epochs = [ep for _, _, ep in results if ep is not None and len(ep) > 0]

    if not all_epochs:
        logger.warning(f"No epochs for subject {subject}")
        return None

    # Concatenate across sessions
    if len(all_epochs) > 1:
        combined = mne.concatenate_epochs(all_epochs, on_mismatch='warn')
    else:
        combined = all_epochs[0]

    n_total = len(combined)
    logger.info(f"Total epochs: {n_total}")

    # --- 2. Compute evoked (mean) ---
    evoked = combined.average(method='mean')
    evoked.nave = n_total

    # --- 3. Baseline correction ---
    if baseline is not None:
        evoked.apply_baseline(baseline)
        logger.info(f"Applied baseline: {baseline}")

    # --- 4. Load LCMV filter and project to source space ---
    try:
        lcmv_filter = load_lcmv_filter(subject, filter_session, filters_dir)
    except FileNotFoundError as e:
        logger.error(str(e))
        return None

    try:
        stc = mne.beamformer.apply_lcmv(evoked, lcmv_filter)
        logger.info(
            f"Source ERP: {stc.data.shape[0]} vertices "
            f"× {stc.data.shape[1]} time points"
        )
    except Exception as e:
        logger.error(f"Error applying beamformer for subject {subject}: {e}")
        return None

    return stc


# ============================================================================
# Morphing and saving
# ============================================================================

def morph_and_save_subject_stc(
    subject_id: int,
    stc: mne.SourceEstimate,
    subjects_dir: str,
    morph_to: str,
    output_dir: str,
    event_type: str,
    timing: str,
    smooth: int = 5,
) -> Optional[str]:
    """
    Morph a source estimate to a common space and save as .stc.

    Parameters
    ----------
    subject_id : int
        Subject ID
    stc : mne.SourceEstimate
        Individual-subject source estimate
    subjects_dir : str
        FreeSurfer subjects directory
    morph_to : str
        Target subject name (e.g., 'fsaverage')
    output_dir : str
        Directory in which to save the morphed .stc
    event_type : str
        Event type (for filename)
    timing : str
        'onset' or 'offset' (for filename)
    smooth : int
        Smoothing steps during morphing

    Returns
    -------
    str or None
        Path (without -lh.stc/-rh.stc extension) of saved file, or None on failure
    """
    subject_from = SUBJECT_FS_MAPPING.get(subject_id, f'as{subject_id:02d}')

    logger.info(f"Morphing sub-{subject_id:02d} ({subject_from}) → {morph_to}")

    try:
        morph = mne.compute_source_morph(
            stc,
            subject_from=subject_from,
            subject_to=morph_to,
            subjects_dir=subjects_dir,
            smooth=smooth,
            verbose=False,
        )
        stc_morphed = morph.apply(stc)
        logger.info(
            f"  Morphed: {stc_morphed.data.shape[0]} vertices"
        )
    except Exception as e:
        logger.error(f"Morphing failed for subject {subject_id}: {e}")
        return None

    # Save
    os.makedirs(output_dir, exist_ok=True)
    fname_stem = (
        f"sub-{subject_id:02d}_task-avs_{event_type}_{timing}_erp"
    )
    out_path = os.path.join(output_dir, fname_stem)
    stc_morphed.save(out_path, overwrite=True)
    logger.info(f"  Saved: {fname_stem}-lh.stc / -rh.stc")

    return out_path


# ============================================================================
# Main orchestration
# ============================================================================

def run(
    subjects: List[int],
    sessions: List[int],
    event_type: str,
    timing: str,
    data_path: str,
    filters_dir: str,
    subjects_dir: str,
    output_dir: str,
    filter_session: int = 1,
    tmin: float = -0.2,
    tmax: float = 0.5,
    baseline: Optional[Tuple[float, float]] = (-0.2, 0.0),
    morph_to: str = 'fsaverage',
    smooth: int = 5,
    verbose: bool = False,
    n_jobs: int = 1,
) -> None:
    """Run source ERP computation for all subjects."""
    logger.info("=" * 70)
    logger.info("Source ERP Computation")
    logger.info("=" * 70)
    logger.info(f"Subjects:       {subjects}")
    logger.info(f"Sessions:       {sessions}")
    logger.info(f"Event type:     {event_type}")
    logger.info(f"Timing:         {timing}")
    logger.info(f"Filter session: {filter_session}")
    logger.info(f"Morph target:   {morph_to}")
    logger.info(f"Output:         {output_dir}")

    pyavs.set_data_path(data_path)

    use_offset = (timing == 'offset')
    erp_output_dir = os.path.join(output_dir, event_type, timing)

    n_success = 0
    for subject in subjects:
        stc = compute_subject_source_erp(
            subject=subject,
            sessions=sessions,
            event_type=event_type,
            data_path=data_path,
            filters_dir=filters_dir,
            filter_session=filter_session,
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            use_offset=use_offset,
            verbose=verbose,
            n_jobs=n_jobs,
        )

        if stc is None:
            continue

        saved = morph_and_save_subject_stc(
            subject_id=subject,
            stc=stc,
            subjects_dir=subjects_dir,
            morph_to=morph_to,
            output_dir=erp_output_dir,
            event_type=event_type,
            timing=timing,
            smooth=smooth,
        )

        if saved is not None:
            n_success += 1

    logger.info("=" * 70)
    logger.info(f"Done: {n_success}/{len(subjects)} subjects saved")
    logger.info(f"Output: {erp_output_dir}")
    logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Compute source-reconstructed ERP and morph to fsaverage"
    )

    parser.add_argument(
        '--data-path', '-d',
        type=str,
        default='/share/klab/datasets/avs/',
        help='AVS data directory',
    )
    parser.add_argument(
        '--filters-dir',
        type=str,
        required=True,
        help=(
            'Path to LCMV filter directory for this event type '
            '(e.g., .../derivatives/pyavs/filters/fixation_HASH/)'
        ),
    )
    parser.add_argument(
        '--subjects-dir',
        type=str,
        default='/share/klab/datasets/avs/rawdir/',
        help='FreeSurfer subjects directory',
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='/share/klab/psulewski/psulewski/pyavs/source_erp/',
        help='Output directory for morphed .stc files',
    )
    parser.add_argument(
        '--subjects', '-s',
        nargs='+', type=int, default=[1, 2, 3, 4, 5],
        help='Subject IDs to process',
    )
    parser.add_argument(
        '--sessions', '-sess',
        nargs='+', type=int, default=list(range(1, 11)),
        help='Session numbers to include',
    )
    parser.add_argument(
        '--event-type', '-e',
        type=str, default='fixation',
        choices=['fixation', 'saccade', 'blink', 'scene'],
        help='Event type',
    )
    parser.add_argument(
        '--timing', '-t',
        type=str, default='onset',
        choices=['onset', 'offset'],
        help='Timing mode: onset or offset',
    )
    parser.add_argument(
        '--filter-session',
        type=int, default=1,
        help='Which session\'s LCMV filter to use (default: 1)',
    )
    parser.add_argument(
        '--tmin',
        type=float, default=-0.2,
        help='Epoch start time [s] (default: -0.2)',
    )
    parser.add_argument(
        '--tmax',
        type=float, default=0.5,
        help='Epoch end time [s] (default: 0.5)',
    )
    parser.add_argument(
        '--baseline',
        nargs=2, type=float, default=[-0.2, 0.0],
        metavar=('BMIN', 'BMAX'),
        help='Baseline window [s] (default: -0.2 0.0)',
    )
    parser.add_argument(
        '--morph-to',
        type=str, default='fsaverage',
        help='Target subject for morphing (default: fsaverage)',
    )
    parser.add_argument(
        '--smooth',
        type=int, default=5,
        help='Smoothing steps during morphing (default: 5)',
    )
    parser.add_argument(
        '--n-jobs', '-j',
        type=int, default=1,
        help='Parallel jobs for session loading per subject (default: 1)',
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output',
    )

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')

    run(
        subjects=args.subjects,
        sessions=args.sessions,
        event_type=args.event_type,
        timing=args.timing,
        data_path=args.data_path,
        filters_dir=args.filters_dir,
        subjects_dir=args.subjects_dir,
        output_dir=args.output_dir,
        filter_session=args.filter_session,
        tmin=args.tmin,
        tmax=args.tmax,
        baseline=tuple(args.baseline),
        morph_to=args.morph_to,
        smooth=args.smooth,
        n_jobs=args.n_jobs,
        verbose=args.verbose,
    )


if __name__ == '__main__':
    main()
