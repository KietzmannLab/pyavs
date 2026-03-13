#!/usr/bin/env python3
"""
Compute scene-onset noise covariance for source projection.

Estimates a data-driven noise covariance from the 400 ms pre-stimulus baseline
of scene-onset (trigger 100) epochs, pooled across all sessions for each subject.
Robust outlier handling: epoch rejection + OAS shrinkage.

The resulting .fif file is shared across all source-projection scripts:
  compute_source_rsa.py, compute_source_erp.py, source_project_encoding.py

Usage:
    python compute_scene_onset_noise_cov.py \\
        --subjects 1 2 3 4 5 \\
        --sessions 1 2 3 4 5 6 7 8 9 10 \\
        --data-path /share/klab/datasets/avs/
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

try:
    import mne
except ImportError as e:
    print(f"Missing dependency: {e}")
    sys.exit(1)

from joblib import Parallel, delayed

# Project imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from pyavs.preprocessing.composer import AVSComposer
from pyavs.utils.logging import get_logger

logger = get_logger('scripts.compute_scene_onset_noise_cov')

# Rejection thresholds for SQUID jumps, eye blinks, muscle bursts
REJECT = {'grad': 4000e-13, 'mag': 4e-12}


# ============================================================================
# Save-path helper (imported by projection scripts)
# ============================================================================

def get_noise_cov_path(subject_id: int, data_path: str) -> Path:
    """Return the canonical save path for the scene-onset noise covariance.

    Parameters
    ----------
    subject_id : int
        Subject number (1-based).
    data_path : str
        AVS data root directory.

    Returns
    -------
    pathlib.Path
        Full path to the .fif covariance file.
    """
    return (
        Path(data_path) / 'derivatives' / 'pyavs'
        / f'sub-{subject_id:02d}' / 'source'
        / f'sub-{subject_id:02d}_task-avs_desc-sceneonset_cov.fif'
    )


# ============================================================================
# Per-session epoch loading
# ============================================================================

def _load_session_epochs(
    subject: int,
    session: int,
    data_path: str,
) -> Optional[mne.Epochs]:
    """Extract scene-onset epochs for one session via AVSComposer."""
    try:
        composer = AVSComposer(
            subject=subject,
            session_num=session,
            data_path=data_path,
            output_path=data_path,
            et_path=data_path,
            preprocessed=True,
            recompute_prepro=False,
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
        epochs = composer.make_trigger_locked_epochs(
            trigger_name='scene_on',
            tmin=-0.4,
            tmax=0.1,
            baseline=None,
            preload=True,
        )
    except Exception as e:
        logger.warning(f"sub-{subject:02d} ses-{session:02d}: failed: {e}")
        return None

    if epochs is None or len(epochs) == 0:
        logger.warning(f"sub-{subject:02d} ses-{session:02d}: 0 epochs created")
        return None

    logger.info(
        f"sub-{subject:02d} ses-{session:02d}: {len(epochs)} scene-onset epochs"
    )
    return epochs


# ============================================================================
# Per-subject covariance computation
# ============================================================================

def compute_subject_noise_cov(
    subject: int,
    sessions: List[int],
    data_path: str,
    overwrite: bool = False,
    n_jobs: int = 1,
) -> None:
    """Compute and save scene-onset noise covariance for one subject.

    Parameters
    ----------
    subject : int
        Subject ID.
    sessions : list of int
        Session numbers to pool.
    data_path : str
        AVS data root.
    overwrite : bool
        If False, skip if output already exists.
    n_jobs : int
        Parallel workers for session loading.
    """
    out_path = get_noise_cov_path(subject, data_path)

    if out_path.exists() and not overwrite:
        logger.info(f"sub-{subject:02d}: output exists, skipping ({out_path})")
        return

    logger.info(f"\n{'='*60}")
    logger.info(f"sub-{subject:02d}: computing scene-onset noise covariance")
    logger.info(f"{'='*60}")

    # Load epochs per session (optionally in parallel)
    session_epochs = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_load_session_epochs)(subject, sess, data_path)
        for sess in sessions
    )

    valid = [ep for ep in session_epochs if ep is not None]
    if not valid:
        logger.error(f"sub-{subject:02d}: no valid sessions, cannot compute covariance")
        return

    # Pool across sessions
    if len(valid) > 1:
        epochs = mne.concatenate_epochs(valid, on_mismatch='warn')
    else:
        epochs = valid[0]

    n_before = len(epochs)
    logger.info(f"sub-{subject:02d}: {n_before} epochs before rejection")

    # Reject gross artifacts
    epochs.drop_bad(reject=REJECT)
    n_after = len(epochs)
    logger.info(
        f"sub-{subject:02d}: {n_after} epochs after rejection "
        f"({n_before - n_after} dropped)"
    )

    if n_after == 0:
        logger.error(f"sub-{subject:02d}: all epochs rejected, cannot compute covariance")
        return

    # Compute covariance from pre-stimulus window with OAS shrinkage
    noise_cov = mne.compute_covariance(
        epochs,
        tmin=-0.4,
        tmax=0.0,
        method='oas',
        rank='info',
        verbose=True,
    )

    # Log condition number as sanity check
    import numpy as np
    cov_data = noise_cov.data
    eigvals = np.linalg.eigvalsh(cov_data)
    positive = eigvals[eigvals > 0]
    if len(positive) > 1:
        cond = positive[-1] / positive[0]
        logger.info(f"sub-{subject:02d}: covariance condition number = {cond:.2e}")

    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    noise_cov.save(str(out_path), overwrite=overwrite)
    logger.info(f"sub-{subject:02d}: saved covariance to {out_path}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compute scene-onset noise covariance for source projection"
    )
    parser.add_argument(
        '--subjects', nargs='+', type=int, default=list(range(1, 6)),
        help='Subject IDs (default: 1-5)',
    )
    parser.add_argument(
        '--sessions', nargs='+', type=int, default=list(range(1, 11)),
        help='Session numbers (default: 1-10)',
    )
    parser.add_argument(
        '--data-path', '-d', type=str, default='/share/klab/datasets/avs/',
        help='AVS data root directory',
    )
    parser.add_argument(
        '--overwrite', action='store_true',
        help='Overwrite existing output files',
    )
    parser.add_argument(
        '--n-jobs', type=int, default=-1,
        help='Parallel workers for session loading within each subject',
    )

    args = parser.parse_args()

    logger.info(f"Subjects:  {args.subjects}")
    logger.info(f"Sessions:  {args.sessions}")
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Overwrite: {args.overwrite}")
    logger.info(f"n_jobs:    {args.n_jobs}")

    for subject in args.subjects:
        compute_subject_noise_cov(
            subject=subject,
            sessions=args.sessions,
            data_path=args.data_path,
            overwrite=args.overwrite,
            n_jobs=args.n_jobs,
        )

    logger.info("Done.")


if __name__ == '__main__':
    main()
