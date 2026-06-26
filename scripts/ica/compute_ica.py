#!/usr/bin/env python3
"""
Compute and store ICA solutions with ET-based eye component detection.

Runs run_ica_et_pipeline for one or more subjects/sessions and writes three
artefacts per subject-session to the BIDS derivatives directory:

  derivatives/pyavs/sub-{id}/ses-{sess}/meg/
    sub-{id}_ses-{sess}_task-avs_ica.fif                 ICA solution
    sub-{id}_ses-{sess}_task-avs_ica-et-scores.csv       per-component ET correlations
    sub-{id}_ses-{sess}_task-avs_ica-exclusions.json     component exclusion lists

The exclusions JSON mirrors the format used by apply_ica_to_raws / AVSComposer:
  {"as01": {"1": [0, 3, 12, 15], "2": [1, 5, 22]}}

Usage:
    # Single subject/session
    python compute_ica.py --subject 1 --session 1 --data-path /share/klab/datasets/avs/

    # Multiple subjects and sessions
    python compute_ica.py --subjects 1 2 3 --sessions 1 2 --data-path /share/klab/datasets/avs/

    # Skip already processed, run 4 jobs in parallel
    python compute_ica.py --subjects 1 2 3 --sessions 1 2 --n-jobs 4 --skip-existing
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from joblib import Parallel, delayed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from pyavs.preprocessing.ica import run_ica_et_pipeline
from pyavs.utils.validation import validate_subject_id, validate_session
from pyavs.utils.logging import get_logger

logger = get_logger('scripts.compute_ica')


def _ica_fif_path(subject_id: int, session: int, data_path: str) -> Path:
    return (
        Path(data_path) / 'derivatives' / 'pyavs'
        / f'sub-{subject_id:02d}' / f'ses-{session:02d}' / 'meg'
        / f'sub-{subject_id:02d}_ses-{session:02d}_task-avs_ica.fif'
    )


def process_single(subject_id: int,
                   session: int,
                   data_path: str,
                   skip_existing: bool = False,
                   save_results: bool = True,
                   top_fraction: float = 0.05,
                   n_components: Optional[int] = None,
                   verbose: bool = True) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        'subject_id': subject_id,
        'session':    session,
        'status':     'failed',
        'eye_exclusions':     [],
        'cardiac_exclusions': [],
        'all_exclusions':     [],
        'ica_path':   None,
        'error':      None,
    }

    if skip_existing and _ica_fif_path(subject_id, session, data_path).exists():
        logger.info(f"sub-{subject_id:02d} ses-{session:02d}: ICA already exists — skipping")
        result['status'] = 'skipped'
        result['ica_path'] = str(_ica_fif_path(subject_id, session, data_path))
        return result

    logger.info(f"Processing sub-{subject_id:02d} ses-{session:02d}...")

    try:
        ica, eye_excl, cardiac_excl, _ = run_ica_et_pipeline(
            subject_id=subject_id,
            session=session,
            data_path=data_path,
            top_fraction=top_fraction,
            n_components=n_components,
            save_results=save_results,
            verbose=verbose,
        )
        result['status']              = 'success'
        result['eye_exclusions']      = eye_excl
        result['cardiac_exclusions']  = cardiac_excl
        result['all_exclusions']      = list(ica.exclude)
        result['ica_path']            = str(_ica_fif_path(subject_id, session, data_path))

    except Exception as exc:
        logger.error(
            f"sub-{subject_id:02d} ses-{session:02d} failed: {exc}", exc_info=True
        )
        result['error'] = str(exc)

    return result


def process_batch(subjects: List[int],
                  sessions: List[int],
                  data_path: str,
                  n_jobs: int = 1,
                  skip_existing: bool = False,
                  save_results: bool = True,
                  top_fraction: float = 0.05,
                  n_components: Optional[int] = None,
                  verbose: bool = True) -> List[Dict[str, Any]]:
    combinations = [(s, sess) for s in subjects for sess in sessions]
    logger.info(
        f"Processing {len(combinations)} subject-session combinations "
        f"({len(subjects)} subjects × {len(sessions)} sessions)"
    )

    if n_jobs == 1:
        return [
            process_single(
                sub, sess, data_path,
                skip_existing=skip_existing,
                save_results=save_results,
                top_fraction=top_fraction,
                n_components=n_components,
                verbose=verbose,
            )
            for sub, sess in combinations
        ]

    return Parallel(n_jobs=n_jobs)(
        delayed(process_single)(
            sub, sess, data_path,
            skip_existing=skip_existing,
            save_results=save_results,
            top_fraction=top_fraction,
            n_components=n_components,
            verbose=verbose,
        )
        for sub, sess in combinations
    )


def print_summary(results: List[Dict[str, Any]]) -> None:
    success  = [r for r in results if r['status'] == 'success']
    skipped  = [r for r in results if r['status'] == 'skipped']
    failed   = [r for r in results if r['status'] == 'failed']

    print(f"\n=== ICA computation summary ===")
    print(f"  Total   : {len(results)}")
    print(f"  Success : {len(success)}")
    print(f"  Skipped : {len(skipped)}")
    print(f"  Failed  : {len(failed)}")

    if success:
        print(f"\n--- Successful ---")
        for r in success:
            print(
                f"  sub-{r['subject_id']:02d} ses-{r['session']:02d}  "
                f"eye={r['eye_exclusions']}  cardiac={r['cardiac_exclusions']}"
            )

    if skipped:
        print(f"\n--- Skipped (ICA already exists) ---")
        for r in skipped:
            print(f"  sub-{r['subject_id']:02d} ses-{r['session']:02d}")

    if failed:
        print(f"\n--- Failed ---")
        for r in failed:
            print(f"  sub-{r['subject_id']:02d} ses-{r['session']:02d}: {r['error']}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Compute and store ICA solutions with ET-based eye component detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compute_ica.py --subject 1 --session 1 --data-path /share/klab/datasets/avs/
  python compute_ica.py --subjects 1 2 3 --sessions 1 2 --n-jobs 4 --skip-existing
        """
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--subject',  type=int,          help='Single subject ID')
    group.add_argument('--subjects', type=int, nargs='+', help='List of subject IDs')

    parser.add_argument('--session',  type=int,
                        help='Single session number (required with --subject)')
    parser.add_argument('--sessions', type=int, nargs='+', default=[1],
                        help='List of session numbers (default: [1])')

    parser.add_argument('--data-path', type=str, default='/share/klab/datasets/avs/',
                        help='Path to AVS data directory')
    parser.add_argument('--n-jobs', type=int, default=-1,
                        help='Parallel jobs for batch processing (default: 1)')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip subject-sessions where ICA .fif already exists')
    parser.add_argument('--top-fraction', type=float, default=0.04,
                        help='Fraction of components to flag as eye-related (default: 0.04)')
    parser.add_argument('--n-components', type=int, default=None,
                        help='Number of ICA components (default: min(80, n_meg_channels))')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not write outputs to disk (dry run)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    if args.subject is not None and args.session is None:
        parser.error('--session is required when using --subject')

    subjects = [args.subject] if args.subject is not None else args.subjects
    sessions = [args.session] if args.subject is not None else args.sessions

    if not os.path.exists(args.data_path):
        print(f"Error: data path does not exist: {args.data_path}")
        return 1

    for s in subjects:
        validate_subject_id(s)
    for s in sessions:
        validate_session(s)

    print(f"=== ICA pipeline ===")
    print(f"Subjects : {subjects}")
    print(f"Sessions : {sessions}")
    print(f"Data path: {args.data_path}")
    print(f"Top fraction (eye): {args.top_fraction}")
    print(f"n_components: {args.n_components or 'auto (min(80, n_meg))'}")
    print(f"Skip existing: {args.skip_existing}")
    print(f"Save results : {not args.no_save}")
    print(f"Parallel jobs: {args.n_jobs}")
    print()

    results = process_batch(
        subjects=subjects,
        sessions=sessions,
        data_path=args.data_path,
        n_jobs=args.n_jobs,
        skip_existing=args.skip_existing,
        save_results=not args.no_save,
        top_fraction=args.top_fraction,
        n_components=args.n_components,
        verbose=args.verbose,
    )

    print_summary(results)

    failed = [r for r in results if r['status'] == 'failed']
    return 1 if failed else 0


if __name__ == '__main__':
    exit(main())
