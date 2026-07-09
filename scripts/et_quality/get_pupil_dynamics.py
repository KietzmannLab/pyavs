#!/usr/bin/env python3
"""
Extract per-fixation pupil area (pa) timecourse epochs for all subjects/sessions.

For each fixation event, a 1000 ms epoch of raw pupil area samples is extracted
starting at fixation onset.  Outputs are saved per subject-session as:
  - sub-{N:02d}_ses-{S:02d}_pupil_epochs.npy      (n_fixations x 1000, float64)
  - sub-{N:02d}_ses-{S:02d}_pupil_epochs_events.csv  (enriched fixation metadata, row-aligned)

Usage:
    python get_pupil_dynamics.py --data-path /path/to/avs [--subjects 1 2 3] \\
        [--sessions 1 2 3] [--prefix as] [--verbose]

Author: P. Sulewski (psulewski@uos.de)
"""

import argparse
import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# Allow running from scripts/et_quality/ directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from pyavs.dataloader.eye import load_and_enrich_eye_events, extract_pupil_epochs
from pyavs.dataloader.loaders import load_eye_samples
from pyavs.utils.logging import get_logger
from extract_calibration_quality import discover_subject_sessions

logger = get_logger('scripts.et_quality.pupil_dynamics')


def process_session(subject_id: int,
                    session: int,
                    data_path: str,
                    prefix: str,
                    verbose: bool) -> Tuple[Optional[np.ndarray], Optional[pd.DataFrame], Optional[np.ndarray]]:
    """
    Extract pupil epochs for a single subject-session.

    Parameters
    ----------
    subject_id : int
        Subject ID.
    session : int
        Session number.
    data_path : str
        Base data directory path.
    prefix : str
        File prefix (e.g. 'as').
    verbose : bool
        Enable verbose logging.

    Returns
    -------
    epochs : np.ndarray, shape (n_fixations, 1000) or None
    fix_events : pd.DataFrame or None
    times : np.ndarray, shape (1000,) or None
    """
    logger.info(f"  Loading events...")
    try:
        _, events_df = load_and_enrich_eye_events(
            subjects=[subject_id],
            sessions=[session],
            data_path=data_path,
            output_prefix=prefix,
            verbose=verbose,
        )
    except Exception as e:
        logger.warning(f"  Could not load events — {e}")
        return None, None

    if events_df is None or len(events_df) == 0:
        logger.warning(f"  Empty events, skipping")
        return None, None, None

    logger.info(f"  Loading samples...")
    try:
        samples_df = load_eye_samples(subject_id, session, data_path, prefix)
    except FileNotFoundError as e:
        logger.warning(f"  {e} — skipping")
        return None, None, None

    epochs, fix_events, times = extract_pupil_epochs(events_df, samples_df)

    if len(fix_events) == 0:
        logger.warning(f"  No fixation events found")
        return None, None, None

    logger.info(f"  {len(fix_events)} fixations extracted")
    return epochs, fix_events, times


def save_session_output(subject_id: int,
                        session: int,
                        epochs: np.ndarray,
                        fix_events: pd.DataFrame,
                        times: np.ndarray,
                        data_path: str) -> None:
    """Save per-session pupil epoch array, times vector, and companion CSV."""
    out_dir = os.path.join(
        data_path, 'derivatives', 'pyavs',
        f'sub-{subject_id:02d}', 'pupil_dynamics'
    )
    os.makedirs(out_dir, exist_ok=True)

    stem = f'sub-{subject_id:02d}_ses-{session:02d}'
    npy_path   = os.path.join(out_dir, f'{stem}_pupil_epochs.npy')
    times_path = os.path.join(out_dir, f'{stem}_pupil_times.npy')
    csv_path   = os.path.join(out_dir, f'{stem}_pupil_epochs_events.csv')

    np.save(npy_path, epochs)
    np.save(times_path, times)
    fix_events.to_csv(csv_path, index=False)

    logger.info(f"  Saved {epochs.shape} array → {npy_path}")
    logger.info(f"  Saved times vector ({len(times)},) → {times_path}")
    logger.info(f"  Saved {len(fix_events)} rows  → {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract per-fixation pupil area timecourse epochs"
    )
    parser.add_argument(
        '--data-path', '-d',
        type=str,
        default=None,
        help='Path to AVS data directory (default: /share/klab/datasets/avs/)'
    )
    parser.add_argument(
        '--subjects', '-s',
        type=int,
        nargs='+',
        default=None,
        help='Subject IDs to process (default: auto-discover all)'
    )
    parser.add_argument(
        '--sessions',
        type=int,
        nargs='+',
        default=None,
        help='Sessions to process (default: auto-discover per subject)'
    )
    parser.add_argument(
        '--prefix', '-p',
        type=str,
        default='as',
        help="File prefix (default: 'as')"
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    if args.data_path is None:
        from pyavs import get_data_path as _get_dp
        args.data_path = _get_dp()
    if args.data_path is None:
        parser.error(
            "No data path configured. Run: pyavs configure --data-path /path/to/data"
        )
    # Discover all subject-session pairs, then filter by --subjects / --sessions
    all_pairs = discover_subject_sessions(args.data_path, args.prefix)

    if args.subjects is not None:
        all_pairs = [(s, sess) for s, sess in all_pairs if s in args.subjects]
    if args.sessions is not None:
        all_pairs = [(s, sess) for s, sess in all_pairs if sess in args.sessions]

    if not all_pairs:
        logger.error("No subject-session combinations found. Check --data-path and --prefix.")
        sys.exit(1)

    logger.info(f"Processing {len(all_pairs)} subject-session combination(s)")
    logger.info("=" * 70)

    n_success = 0
    n_failed = 0

    for subject_id, session in all_pairs:
        logger.info(f"Subject {subject_id:02d}, session {session:02d}")

        epochs, fix_events, times = process_session(
            subject_id=subject_id,
            session=session,
            data_path=args.data_path,
            prefix=args.prefix,
            verbose=args.verbose,
        )

        if epochs is None:
            n_failed += 1
            continue

        save_session_output(subject_id, session, epochs, fix_events, times, args.data_path)
        n_success += 1

    logger.info("=" * 70)
    logger.info("DONE")
    logger.info(f"  Successful: {n_success} / {len(all_pairs)}")
    logger.info(f"  Failed:     {n_failed} / {len(all_pairs)}")


if __name__ == '__main__':
    main()
