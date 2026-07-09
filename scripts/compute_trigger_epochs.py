#!/usr/bin/env python3
"""
Compute and store trigger-locked MEG epochs (e.g. mic_on, caption_on).

This script creates epochs locked to specific MEG trigger events such as
microphone onset or caption onset, with trial metadata from the experiment log.

Usage:
    python compute_trigger_epochs.py --subject 1 --session 1 --data-path /path/to/data
    python compute_trigger_epochs.py --subjects 1 2 3 --sessions 1 2 --trigger caption_on --data-path /path/to/data

Author: pyAVS development team
"""

import argparse
import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any
from joblib import Parallel, delayed

# Add pyavs to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pyavs.preprocessing.composer import AVSComposer
from pyavs.io.write import save_epochs, save_metadata_csv
from pyavs.utils.validation import validate_subject_id, validate_session
from pyavs.utils.logging import get_logger
from pyavs.config.config import PyAVSConfig


# Initialize logger
logger = get_logger('scripts.compute_trigger_epochs')

# Configuration
config = PyAVSConfig()

# Default epoch timing for trigger-locked epochs
DEFAULT_TMIN = -0.5
DEFAULT_TMAX = 9.0
DEFAULT_TRIGGER = 'mic_on'


def setup_output_directory(data_path: str) -> Path:
    """
    Set up output directory for epochs.

    Parameters
    ----------
    data_path : str
        Base data path

    Returns
    -------
    Path
        Path to epochs output directory
    """
    output_dir = Path(data_path) / 'derivatives' / 'epochs'
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def process_single_subject_session(subject_id: int, session: int,
                                   data_path: str, output_dir: Path,
                                   trigger_name: str = DEFAULT_TRIGGER,
                                   tmin: float = DEFAULT_TMIN,
                                   tmax: float = DEFAULT_TMAX) -> Dict[str, Any]:
    """
    Process trigger-locked epochs for a single subject and session.

    Parameters
    ----------
    subject_id : int
        Subject ID
    session : int
        Session number
    data_path : str
        Path to data directory
    output_dir : Path
        Output directory for epochs
    trigger_name : str
        Trigger name (e.g. 'mic_on', 'caption_on')
    tmin : float
        Epoch start time relative to trigger in seconds
    tmax : float
        Epoch end time relative to trigger in seconds

    Returns
    -------
    dict
        Processing results and statistics
    """
    results = {
        'subject_id': subject_id,
        'session': session,
        'trigger': trigger_name,
        'status': 'failed',
        'epochs_created': {},
        'errors': []
    }

    logger.info(f"Processing subject {subject_id}, session {session}, trigger {trigger_name}")

    try:
        # Initialize composer with config parameters
        composer_kwargs = config.get_composer_kwargs()
        composer_kwargs.update({
            'subject': subject_id,
            'session_num': session,
            'data_path': data_path,
            'output_path': str(output_dir)
        })
        composer = AVSComposer(**composer_kwargs)

        # Load and preprocess MEG data
        logger.info("Loading MEG data...")
        composer.load_meg_data()

        # Apply ICA if available
        composer.apply_ica_to_blocks(use_precomputed=True)
        logger.info("Applied ICA to MEG blocks")

        # Filter and concatenate MEG data
        composer.filter_meg_data(ignore_existing_filter=True)
        composer.concatenate_raws_per_session()

        # Resample if configured
        if config.resample_freq and config.resample_freq != composer.raws_concatenated.info['sfreq']:
            logger.info(f"Resampling from {composer.raws_concatenated.info['sfreq']} Hz to {config.resample_freq} Hz")
            composer.raws_concatenated.resample(config.resample_freq, n_jobs=config.n_jobs)

        composer.find_events_in_raw()
        logger.info(f"Found {len(composer.meg_trigger_events)} MEG trigger events")

        # Create trigger-locked epochs
        epochs = composer.make_trigger_locked_epochs(
            trigger_name=trigger_name,
            tmin=tmin,
            tmax=tmax,
            baseline=None
        )

        if epochs is None or len(epochs) == 0:
            logger.warning(f"No {trigger_name} epochs created")
            results['errors'].append(f"No {trigger_name} epochs found")
            return results

        logger.info(f"Created {len(epochs)} {trigger_name} epochs")

        # Save epochs
        epochs_path = save_epochs(
            epochs=epochs,
            subject_id=subject_id,
            session=session,
            event_type=trigger_name,
            data_path=data_path
        )

        results['epochs_created'][trigger_name] = {
            'n_epochs': len(epochs),
            'path': epochs_path,
            'time_range': f"{epochs.tmin:.3f} to {epochs.tmax:.3f} s",
            'sampling_rate': epochs.info['sfreq']
        }
        print(epochs.metadata.head())  # Print first few rows of metadata for verification
        print(epochs.metadata.columns)  # Print metadata columns for verification
        # Save metadata as CSV
        if epochs.metadata is not None and not epochs.metadata.empty:
            metadata_path = save_metadata_csv(
                metadata=epochs.metadata,
                subject_id=subject_id,
                session=session,
                event_type=trigger_name,
                data_path=data_path
            )

            results['epochs_created'][trigger_name]['metadata_path'] = metadata_path
            results['epochs_created'][trigger_name]['metadata_columns'] = list(epochs.metadata.columns)

        results['status'] = 'success'
        logger.info(f"Successfully processed subject {subject_id}, session {session}")

    except Exception as e:
        logger.error(f"Error processing subject {subject_id}, session {session}: {e}", exc_info=True)
        results['errors'].append(str(e))

    return results


def process_batch(subjects: List[int], sessions: List[int],
                  data_path: str, trigger_name: str = DEFAULT_TRIGGER,
                  tmin: float = DEFAULT_TMIN, tmax: float = DEFAULT_TMAX,
                  n_jobs: int = 1) -> List[Dict[str, Any]]:
    """
    Process multiple subjects and sessions in batch.

    Parameters
    ----------
    subjects : list of int
        List of subject IDs to process
    sessions : list of int
        List of session numbers to process
    data_path : str
        Path to data directory
    trigger_name : str
        Trigger name
    tmin : float
        Epoch start time
    tmax : float
        Epoch end time
    n_jobs : int
        Number of parallel jobs (default: 1)

    Returns
    -------
    list of dict
        Processing results for each subject-session combination
    """
    output_dir = setup_output_directory(data_path)

    combinations = [(s, sess) for s in subjects for sess in sessions]

    logger.info(f"Processing {len(combinations)} subject-session combinations")
    logger.info(f"Subjects: {subjects}")
    logger.info(f"Sessions: {sessions}")
    logger.info(f"Trigger: {trigger_name}")
    logger.info(f"Output directory: {output_dir}")

    if n_jobs == 1:
        results = []
        for subject_id, session in combinations:
            result = process_single_subject_session(
                subject_id, session, data_path, output_dir,
                trigger_name, tmin, tmax
            )
            results.append(result)
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_single_subject_session)(
                subject_id, session, data_path, output_dir,
                trigger_name, tmin, tmax
            )
            for subject_id, session in combinations
        )

    return results


def print_batch_summary(results: List[Dict[str, Any]]) -> None:
    """
    Print summary of batch processing results.

    Parameters
    ----------
    results : list of dict
        Processing results from batch processing
    """
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']

    print(f"\n=== Trigger Epochs Processing Summary ===")
    print(f"Total combinations processed: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if successful:
        print(f"\n--- Successful Processing ---")
        total_epochs = 0

        for result in successful:
            subj, sess = result['subject_id'], result['session']
            trigger = result['trigger']
            if trigger in result['epochs_created']:
                n_epochs = result['epochs_created'][trigger]['n_epochs']
                total_epochs += n_epochs
                print(f"  Subject {subj}, Session {sess}: {n_epochs} {trigger} epochs")

        print(f"\nTotal epochs created: {total_epochs}")

    if failed:
        print(f"\n--- Failed Processing ---")
        for result in failed:
            subj, sess = result['subject_id'], result['session']
            print(f"  Subject {subj}, Session {sess}:")
            for error in result['errors']:
                print(f"    Error: {error}")


def main():
    """Main function for command line execution."""
    parser = argparse.ArgumentParser(
        description='Compute and store trigger-locked MEG epochs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single subject and session (default: mic_on)
  python compute_trigger_epochs.py --subject 1 --session 1 --data-path /path/to/data

  # Process caption_on trigger
  python compute_trigger_epochs.py --subject 1 --session 1 --trigger caption_on --data-path /path/to/data

  # Custom epoch window
  python compute_trigger_epochs.py --subject 1 --session 1 --tmin -0.2 --tmax 2.0 --data-path /path/to/data

  # Batch processing
  python compute_trigger_epochs.py --subjects 1 2 3 --sessions 1 2 --data-path /path/to/data
        """
    )

    # Subject and session arguments
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--subject', type=int, help='Single subject ID to process')
    group.add_argument('--subjects', type=int, nargs='+', help='List of subject IDs to process')

    parser.add_argument('--session', type=int, help='Single session number (required if --subject used)')
    parser.add_argument('--sessions', type=int, nargs='+', default=[1],
                        help='List of session numbers to process (default: [1])')

    # Trigger selection
    parser.add_argument('--trigger', type=str, default=DEFAULT_TRIGGER,
                        choices=['mic_on', 'mic_off', 'caption_on', 'caption_off', 'scene_on', 'scene_off'],
                        help=f'Trigger to lock epochs to (default: {DEFAULT_TRIGGER})')

    # Epoch timing
    parser.add_argument('--tmin', type=float, default=DEFAULT_TMIN,
                        help=f'Epoch start time relative to trigger in seconds (default: {DEFAULT_TMIN})')
    parser.add_argument('--tmax', type=float, default=DEFAULT_TMAX,
                        help=f'Epoch end time relative to trigger in seconds (default: {DEFAULT_TMAX})')

    # Data path
    parser.add_argument('--data-path', type=str, default=None,
                        help='Path to AVS data directory')

    # Processing options
    parser.add_argument('--n-jobs', type=int, default=1,
                        help='Number of parallel jobs for batch processing (default: 1)')

    # Logging
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')

    args = parser.parse_args()
    if args.data_path is None:
        from pyavs import get_data_path as _get_dp
        args.data_path = _get_dp()
    if args.data_path is None:
        parser.error(
            "No data path configured. Run: pyavs configure --data-path /path/to/data"
        )
    # Set up logging
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    # Validate arguments
    if args.subject is not None and args.session is None:
        parser.error("--session is required when using --subject")

    # Determine subjects and sessions to process
    if args.subject is not None:
        subjects = [args.subject]
        sessions = [args.session]
    else:
        subjects = args.subjects
        sessions = args.sessions

    # Validate data path
    if not os.path.exists(args.data_path):
        print(f"Error: Data path does not exist: {args.data_path}")
        return 1

    # Validate subjects and sessions
    for subject in subjects:
        validate_subject_id(subject)
    for session in sessions:
        validate_session(session)

    # Print configuration
    print("=== Trigger-Locked Epochs Computation ===")
    print(f"Subjects: {subjects}")
    print(f"Sessions: {sessions}")
    print(f"Trigger: {args.trigger}")
    print(f"Data path: {args.data_path}")
    print(f"Epoch timing: {args.tmin} to {args.tmax} s")
    print(f"Filter settings: {config.filter_params['l_freq']}-{config.filter_params['h_freq']} Hz")
    print(f"Resampling: {config.resample_freq} Hz")
    print(f"Parallel jobs: {args.n_jobs}")
    print()

    # Process epochs
    results = process_batch(
        subjects=subjects,
        sessions=sessions,
        data_path=args.data_path,
        trigger_name=args.trigger,
        tmin=args.tmin,
        tmax=args.tmax,
        n_jobs=args.n_jobs
    )

    # Print summary
    print_batch_summary(results)

    # Return exit code based on results
    failed_results = [r for r in results if r['status'] == 'failed']
    if failed_results:
        print(f"\nWarning: {len(failed_results)} combinations failed")
        return 1
    else:
        print("\nAll processing completed successfully!")
        return 0


if __name__ == "__main__":
    exit(main())
