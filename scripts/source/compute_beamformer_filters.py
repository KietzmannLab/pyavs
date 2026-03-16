#!/usr/bin/env python3
"""
Compute LCMV beamformer filters for source reconstruction.

This script runs on the HPC cluster. It:
  1. Loads the pre-computed forward model (BIDS derivatives)
  2. Loads the scene-onset noise covariance (pooled across sessions, per subject)
  3. Computes data covariance from a cross-session random sample of fixation epochs
  4. Computes an LCMV beamformer filter and saves it to BIDS derivatives:
       {data_path}/derivatives/pyavs/sub-{id:02d}/source/
       sub-{id:02d}_task-avs_desc-sceneonset_lcmv.h5

One filter is computed per subject (not per session), consistent with the
already-pooled scene-onset noise covariance.

Pre-requisites (must exist):
  - Forward model:    compute_forward_model.py
  - Noise covariance: compute_scene_onset_noise_cov.py

Usage:
    python compute_beamformer_filters.py \\
        --subjects 1 2 3 4 5 \\
        --sessions 1 2 3 4 5 6 7 8 9 10 \\
        --data-path /share/klab/datasets/avs/ \\
        --fwd-session 1

Author: P. Sulewski (psulewski@uos.de)
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

import mne

# Allow running from the scripts/source directory directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from pyavs.source.filters import compute_cross_session_data_covariance
from pyavs.source.forward import load_forward_model
from pyavs.utils.logging import get_logger
from compute_scene_onset_noise_cov import get_noise_cov_path

logger = get_logger('scripts.source.compute_beamformer_filters')


# ============================================================================
# Helpers
# ============================================================================

def get_lcmv_filter_path(subject_id: int, data_path: str) -> Path:
    """Return the canonical save path for the LCMV beamformer filter.

    Parameters
    ----------
    subject_id : int
        Subject number (1-based).
    data_path : str
        AVS data root directory.

    Returns
    -------
    pathlib.Path
        Full path to the .h5 filter file.
    """
    return (
        Path(data_path) / 'derivatives' / 'pyavs'
        / f'sub-{subject_id:02d}' / 'source'
        / f'sub-{subject_id:02d}_task-avs_desc-sceneonset_lcmv.h5'
    )


# ============================================================================
# Per-subject computation
# ============================================================================

def compute_subject_lcmv_filter(
    subject_id: int,
    sessions: List[int],
    data_path: str,
    fwd_session: int = 1,
    n_epochs_per_session: int = 350,
    pick_ori: str = 'max-power',
    reg: float = 0.05,
    weight_norm: Optional[str] = 'unit-noise-gain',
    n_jobs: int = 1,
    overwrite: bool = False,
    verbose: bool = False,
) -> bool:
    """
    Compute and save the LCMV beamformer filter for one subject.

    Parameters
    ----------
    subject_id : int
        Subject ID.
    sessions : list of int
        Sessions to include in the cross-session data covariance.
    data_path : str
        AVS BIDS data root.
    fwd_session : int
        Session tag of the forward model to load.
    n_epochs_per_session : int
        Number of fixation epochs randomly sampled per session for the
        data covariance estimate.
    pick_ori : str
        Source orientation for the beamformer ('max-power' or 'normal').
    reg : float
        Regularization coefficient applied to the data covariance.
    weight_norm : str or None
        Weight normalization strategy ('unit-noise-gain', 'nai', or None).
    n_jobs : int
        Parallel workers used when loading epochs across sessions.
    overwrite : bool
        If False, skip if output already exists.
    verbose : bool
        Enable verbose MNE output.

    Returns
    -------
    bool
        True on success, False on any failure.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Subject {subject_id:02d}")
    logger.info(f"{'='*60}")

    out_path = get_lcmv_filter_path(subject_id, data_path)

    if out_path.exists() and not overwrite:
        logger.info(f"  Output exists, skipping ({out_path})")
        return True

    # --- Forward model ---
    logger.info(f"  Loading forward model (ses-{fwd_session:02d}) ...")
    try:
        fwd = load_forward_model(subject_id, fwd_session, data_path,
                                 verbose=verbose)
        logger.info(
            f"  Forward: {fwd['nsource']} sources, {fwd['nchan']} channels"
        )
    except Exception as e:
        logger.error(f"  Failed to load forward model: {e}")
        return False

    # --- Noise covariance ---
    noise_cov_path = get_noise_cov_path(subject_id, data_path)
    if not noise_cov_path.exists():
        logger.error(f"  Noise covariance not found: {noise_cov_path}")
        return False

    logger.info(f"  Loading noise covariance: {noise_cov_path.name}")
    try:
        noise_cov = mne.read_cov(str(noise_cov_path), verbose=False)
    except Exception as e:
        logger.error(f"  Failed to load noise covariance: {e}")
        return False

    # --- Cross-session data covariance ---
    logger.info(
        f"  Computing cross-session data covariance "
        f"({len(sessions)} sessions × {n_epochs_per_session} epochs) ..."
    )
    try:
        epochs = compute_cross_session_data_covariance(
            data_path=data_path,
            subject_id=subject_id,
            sessions=sessions,
            event_type='fixation',
            n_epochs_per_session=n_epochs_per_session,
            overwrite=overwrite,
        )
        logger.info(f"  Cross-session epochs: {len(epochs)}")
    except Exception as e:
        logger.error(f"  Failed to compute cross-session epochs: {e}")
        return False

    try:
        data_cov = mne.compute_covariance(
            epochs,
            method='empirical',
            rank='info',
            verbose=verbose,
        )
    except Exception as e:
        logger.error(f"  Failed to compute data covariance: {e}")
        return False

    # --- LCMV filter ---
    logger.info("  Computing LCMV filter ...")
    try:
        filters = mne.beamformer.make_lcmv(
            epochs.info,
            fwd,
            data_cov,
            noise_cov=noise_cov,
            pick_ori=pick_ori,
            reg=reg,
            weight_norm=weight_norm,
            rank='info',
            verbose=verbose,
        )
    except Exception as e:
        logger.error(f"  make_lcmv failed: {e}")
        return False

    # --- Save ---
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        filters.save(str(out_path), overwrite=overwrite)
        logger.info(f"  Saved: {out_path}")
    except Exception as e:
        logger.error(f"  Failed to save filter: {e}")
        return False

    return True


# ============================================================================
# Orchestration
# ============================================================================

def run(
    subjects: List[int],
    sessions: List[int],
    data_path: str,
    fwd_session: int = 1,
    n_epochs_per_session: int = 350,
    pick_ori: str = 'max-power',
    reg: float = 0.05,
    weight_norm: Optional[str] = 'unit-noise-gain',
    n_jobs: int = 1,
    overwrite: bool = False,
    verbose: bool = False,
) -> None:
    """Compute LCMV beamformer filters for all subjects."""
    logger.info("=" * 70)
    logger.info("LCMV Beamformer Filter Computation")
    logger.info("=" * 70)
    logger.info(f"Subjects:            {subjects}")
    logger.info(f"Sessions:            {sessions}")
    logger.info(f"Data path:           {data_path}")
    logger.info(f"Fwd session:         {fwd_session}")
    logger.info(f"Epochs/session:      {n_epochs_per_session}")
    logger.info(f"pick_ori:            {pick_ori}")
    logger.info(f"reg:                 {reg}")
    logger.info(f"weight_norm:         {weight_norm}")

    n_success = 0
    for subject in subjects:
        ok = compute_subject_lcmv_filter(
            subject_id=subject,
            sessions=sessions,
            data_path=data_path,
            fwd_session=fwd_session,
            n_epochs_per_session=n_epochs_per_session,
            pick_ori=pick_ori,
            reg=reg,
            weight_norm=weight_norm,
            n_jobs=n_jobs,
            overwrite=overwrite,
            verbose=verbose,
        )
        if ok:
            n_success += 1

    logger.info("=" * 70)
    logger.info(f"Done: {n_success}/{len(subjects)} subjects")
    logger.info("=" * 70)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute LCMV beamformer filters per subject and save to BIDS "
            "derivatives. Requires pre-computed forward model and scene-onset "
            "noise covariance."
        )
    )
    parser.add_argument(
        '--subjects', '-s',
        nargs='+', type=int, default=list(range(1, 6)),
        help='Subject IDs to process (default: 1-5)',
    )
    parser.add_argument(
        '--sessions',
        nargs='+', type=int, default=list(range(1, 11)),
        help='Sessions for cross-session data covariance (default: 1-10)',
    )
    parser.add_argument(
        '--data-path', '-d',
        type=str, default='/share/klab/datasets/avs/',
        help='AVS BIDS data root directory',
    )
    parser.add_argument(
        '--fwd-session',
        type=int, default=1,
        help='Session tag of the forward model to load (default: 1)',
    )
    parser.add_argument(
        '--n-epochs-per-session',
        type=int, default=350,
        help='Fixation epochs sampled per session for data covariance (default: 350)',
    )
    parser.add_argument(
        '--pick-ori',
        type=str, default='max-power',
        choices=['max-power', 'normal'],
        help='Beamformer source orientation (default: max-power)',
    )
    parser.add_argument(
        '--reg',
        type=float, default=0.05,
        help='Regularization coefficient (default: 0.05)',
    )
    parser.add_argument(
        '--weight-norm',
        type=str, default='unit-noise-gain',
        help=(
            'Weight normalization: "unit-noise-gain", "nai", or "none" '
            '(default: unit-noise-gain)'
        ),
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing output files',
    )
    parser.add_argument(
        '--n-jobs', '-j',
        type=int, default=1,
        help='Parallel workers for cross-session epoch loading (default: 1)',
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output',
    )

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')

    # Treat the string "none" as Python None for weight_norm
    weight_norm = None if args.weight_norm.lower() == 'none' else args.weight_norm

    run(
        subjects=args.subjects,
        sessions=args.sessions,
        data_path=args.data_path,
        fwd_session=args.fwd_session,
        n_epochs_per_session=args.n_epochs_per_session,
        pick_ori=args.pick_ori,
        reg=args.reg,
        weight_norm=weight_norm,
        n_jobs=args.n_jobs,
        overwrite=args.overwrite,
        verbose=args.verbose,
    )


if __name__ == '__main__':
    main()
