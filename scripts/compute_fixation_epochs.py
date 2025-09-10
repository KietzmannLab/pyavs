#!/usr/bin/env python3
"""
Compute and store fixation and saccade event epochs using AVSComposer.

This script processes MEG data to create epochs around fixation and saccade events,
with unified timing parameters and object label metadata. It supports batch processing
of multiple subjects and sessions.

Usage:
    python compute_fixation_epochs.py --subjects 1 2 3 --sessions 1 2 --data-path /path/to/data
    python compute_fixation_epochs.py --subject 1 --session 1 --data-path /path/to/data

Author: pyAVS development team
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import logging
from joblib import Parallel, delayed

# Add pyavs to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pyavs.preprocessing.composer import AVSComposer
from pyavs.io.write import save_epochs, save_metadata_csv
from pyavs.utils.validation import validate_subject_id, validate_session
from pyavs.utils.logging import get_logger
from pyavs.config.config import PyAVSConfig


# Initialize logger
logger = get_logger('scripts.compute_fixation_epochs')


# Configuration
config = PyAVSConfig()
EVENT_TYPES = ['fixation', 'saccade']
RECORDING_TYPE = 'scene'  # Only process scene recordings


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
                                 include_object_labels: bool = True) -> Dict[str, Any]:
    """
    Process epochs for a single subject and session.
    
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
    include_object_labels : bool
        Whether to include object labels in metadata
        
    Returns
    -------
    dict
        Processing results and statistics
    """
    results = {
        'subject_id': subject_id,
        'session': session,
        'status': 'failed',
        'epochs_created': {},
        'metadata_saved': {},
        'errors': []
    }
    
    logger.info(f"Processing subject {subject_id}, session {session}")
    
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
    
    # Process each event type
    for event_type in EVENT_TYPES:
        
        logger.info(f"Processing {event_type} events...")
        
        # Load eye tracking events with object labels
        composer.get_et_annotations(
            event_type=event_type,
            recording=RECORDING_TYPE,
            get_object_labels=include_object_labels
        )
        
        if not hasattr(composer, 'et_events') or len(composer.et_events) == 0:
            logger.warning(f"No {event_type} events found")
            continue
        
        logger.info(f"Loaded {len(composer.et_events)} {event_type} events")
        
        # Create epochs
        composer.make_et_event_epochs(
            tmin=config.tmin,
            tmax=config.tmax,
            event_type=event_type,
            recording=RECORDING_TYPE,
            save_epochs=False,  # We'll save manually for better control
            get_metadata=True,
            get_object_labels=include_object_labels,
            baseline=None  # No baseline correction (AVS practice)
        )
        
        if not hasattr(composer, 'et_epochs') or len(composer.et_epochs) == 0:
            logger.warning(f"No {event_type} epochs created")
            continue
        
        epochs = composer.et_epochs
        logger.info(f"Created {len(epochs)} {event_type} epochs")
        
        # Save epochs using pyAVS save_epochs function
        epochs_path = save_epochs(
            epochs=epochs,
            subject_id=subject_id,
            session=session,
            event_type=f"{event_type}_scene",
            data_path=data_path
        )
        
        results['epochs_created'][event_type] = {
            'n_epochs': len(epochs),
            'path': epochs_path,
            'time_range': f"{epochs.tmin:.3f} to {epochs.tmax:.3f} s",
            'sampling_rate': epochs.info['sfreq']
        }
        
        # Save metadata as CSV using pyAVS IO infrastructure
        # The truth value of a DataFrame is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().
    
        metadata_path = save_metadata_csv(
            metadata=epochs.metadata,
            subject_id=subject_id,
            session=session,
            event_type=event_type,
            data_path=data_path
        )
        
        results['metadata_saved'][event_type] = {
            'path': metadata_path,
            'n_columns': len(epochs.metadata.columns),
            'columns': list(epochs.metadata.columns)[:10]  # First 10 columns
        }
        
       
    # Mark as successful if any epochs were created
    if results['epochs_created']:
        results['status'] = 'success'
        logger.info(f"Successfully processed subject {subject_id}, session {session}")
    
    return results


def process_batch(subjects: List[int], sessions: List[int], 
                 data_path: str, n_jobs: int = 1,
                 include_object_labels: bool = True) -> List[Dict[str, Any]]:
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
    n_jobs : int
        Number of parallel jobs (default: 1)
    include_object_labels : bool
        Whether to include object labels in metadata
        
    Returns
    -------
    list of dict
        Processing results for each subject-session combination
    """
    output_dir = setup_output_directory(data_path)
    
    # Create list of all subject-session combinations
    combinations = [(s, sess) for s in subjects for sess in sessions]
    
    logger.info(f"Processing {len(combinations)} subject-session combinations")
    logger.info(f"Subjects: {subjects}")
    logger.info(f"Sessions: {sessions}")
    logger.info(f"Parallel jobs: {n_jobs}")
    logger.info(f"Output directory: {output_dir}")
    
    # Process in parallel if requested
    if n_jobs == 1:
        results = []
        for subject_id, session in combinations:
            result = process_single_subject_session(
                subject_id, session, data_path, output_dir, include_object_labels
            )
            results.append(result)
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_single_subject_session)(
                subject_id, session, data_path, output_dir, include_object_labels
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
    
    print(f"\n=== Batch Processing Summary ===")
    print(f"Total combinations processed: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        print(f"\n--- Successful Processing ---")
        total_fixation_epochs = 0
        total_saccade_epochs = 0
        
        for result in successful:
            subj, sess = result['subject_id'], result['session']
            print(f"Subject {subj}, Session {sess}:")
            
            for event_type in EVENT_TYPES:
                if event_type in result['epochs_created']:
                    n_epochs = result['epochs_created'][event_type]['n_epochs']
                    print(f"  {event_type}: {n_epochs} epochs")
                    
                    if event_type == 'fixation':
                        total_fixation_epochs += n_epochs
                    else:
                        total_saccade_epochs += n_epochs
        
        print(f"\nTotal epochs created:")
        print(f"  Fixation: {total_fixation_epochs}")
        print(f"  Saccade: {total_saccade_epochs}")
    
    if failed:
        print(f"\n--- Failed Processing ---")
        for result in failed:
            subj, sess = result['subject_id'], result['session']
            print(f"Subject {subj}, Session {sess}:")
            for error in result['errors']:
                print(f"  Error: {error}")


def main():
    """Main function for command line execution."""
    parser = argparse.ArgumentParser(
        description='Compute and store fixation and saccade event epochs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single subject and session
  python /home/student/p/psulewski/pyAVS/scripts/compute_fixation_epochs.py --subject 1 --session 1 --data-path /share/klab/datasets/avs/
  
  # Process multiple subjects and sessions
  python compute_fixation_epochs.py --subjects 1 2 3 --sessions 1 2 --data-path /path/to/data
  
  # Use parallel processing
  python compute_fixation_epochs.py --subjects 1 2 3 --sessions 1 2 --data-path /path/to/data --n-jobs 4
  
  # Skip object labels (faster processing)
  python compute_fixation_epochs.py --subjects 1 2 --sessions 1 --data-path /path/to/data --no-object-labels
        """
    )
    
    # Subject and session arguments
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--subject', type=int, help='Single subject ID to process')
    group.add_argument('--subjects', type=int, nargs='+', help='List of subject IDs to process')
    
    parser.add_argument('--session', type=int, help='Single session number (required if --subject used)')
    parser.add_argument('--sessions', type=int, nargs='+', default=[1], 
                       help='List of session numbers to process (default: [1])')
    
    # Data path
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to AVS data directory')
    
    # Processing options
    parser.add_argument('--n-jobs', type=int, default=1,
                       help='Number of parallel jobs for batch processing (default: 1)')
    parser.add_argument('--no-object-labels', action='store_true',
                       help='Skip object label computation (faster processing)')
    
    # Logging
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
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
    print("=== Fixation Event Epochs Computation ===")
    print(f"Subjects: {subjects}")
    print(f"Sessions: {sessions}")
    print(f"Data path: {args.data_path}")
    print(f"Epoch timing: {config.tmin} to {config.tmax} s")
    print(f"Recording type: {RECORDING_TYPE}")
    print(f"Object labels: {'No' if args.no_object_labels else 'Yes'}")
    print(f"Parallel jobs: {args.n_jobs}")
    print(f"Filter settings: {config.filter_params['l_freq']}-{config.filter_params['h_freq']} Hz")
    print(f"Resampling: {config.resample_freq} Hz")
    print(f"ICA application: {'Yes' if config.apply_ica or config.use_precomputed_ica else 'No'}")
    print(f"Bad channel interpolation: {'Yes' if config.interpolate_bad_channels else 'No'}")
    print()
    
    # Process epochs
   
    results = process_batch(
        subjects=subjects,
        sessions=sessions,
        data_path=args.data_path,
        n_jobs=args.n_jobs,
        include_object_labels=not args.no_object_labels
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