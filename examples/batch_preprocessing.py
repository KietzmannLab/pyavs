#!/usr/bin/env python3
"""
Batch preprocessing script for all AVS dataset subjects and sessions.

This script uses the AVS composer to run preprocessing (without filtering) 
across all subjects and sessions in the dataset. It provides:
- Parallel processing capabilities
- Progress tracking and logging
- Error handling and recovery
- Configurable preprocessing parameters
- Summary statistics and reports

Usage:
    python batch_preprocessing.py --data_path /path/to/avs/data
    python batch_preprocessing.py --config preprocessing_config.json
    python batch_preprocessing.py --subjects 1,2,3 --sessions 1,2,3 --n_jobs 4
"""

import sys
import os
import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import traceback

# Add pyavs to path if not installed
sys.path.insert(0, str(Path(__file__).parent.parent))

from pyavs.config import get_config, load_config
from pyavs.preprocessing.composer import AVSComposer
from pyavs.utils.logging import get_logger
from pyavs.utils.validation import validate_subject_id, validate_session
from pyavs.utils.paths import get_max_blocks

logger = get_logger(__name__)


class PreprocessingStats:
    """Track preprocessing statistics and results."""
    
    def __init__(self):
        self.total_jobs = 0
        self.completed_jobs = 0
        self.failed_jobs = 0
        self.skipped_jobs = 0
        self.start_time = None
        self.results = []
        self.errors = []
    
    def start(self, total_jobs: int):
        """Start timing and set total jobs."""
        self.total_jobs = total_jobs
        self.start_time = time.time()
        logger.info(f"Starting batch preprocessing of {total_jobs} jobs...")
    
    def add_result(self, result: Dict):
        """Add a successful result."""
        self.results.append(result)
        self.completed_jobs += 1
        self._log_progress()
    
    def add_error(self, error_info: Dict):
        """Add an error result."""
        self.errors.append(error_info)
        self.failed_jobs += 1
        self._log_progress()
    
    def add_skip(self, skip_info: Dict):
        """Add a skipped job."""
        self.results.append(skip_info)
        self.skipped_jobs += 1
        self._log_progress()
    
    def _log_progress(self):
        """Log current progress."""
        processed = self.completed_jobs + self.failed_jobs + self.skipped_jobs
        if processed % 10 == 0 or processed == self.total_jobs:
            elapsed = time.time() - self.start_time
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (self.total_jobs - processed) / rate if rate > 0 else 0
            
            logger.info(f"Progress: {processed}/{self.total_jobs} "
                       f"({processed/self.total_jobs*100:.1f}%) - "
                       f"Success: {self.completed_jobs}, Failed: {self.failed_jobs}, "
                       f"Skipped: {self.skipped_jobs} - ETA: {eta/60:.1f}m")
    
    def get_summary(self) -> Dict:
        """Get final summary statistics."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        return {
            'total_jobs': self.total_jobs,
            'completed_jobs': self.completed_jobs,
            'failed_jobs': self.failed_jobs,
            'skipped_jobs': self.skipped_jobs,
            'success_rate': self.completed_jobs / max(self.total_jobs, 1) * 100,
            'elapsed_time': elapsed,
            'average_time_per_job': elapsed / max(self.completed_jobs, 1),
            'total_errors': len(self.errors)
        }


def preprocess_subject_session(subject_id: int, 
                              session: int,
                              data_path: str,
                              preprocessing_params: Dict,
                              overwrite: bool = False) -> Dict:
    """
    Preprocess a single subject-session combination.
    
    Parameters
    ----------
    subject_id : int
        Subject ID
    session : int
        Session number
    data_path : str
        Path to AVS dataset
    preprocessing_params : dict
        Preprocessing parameters
    overwrite : bool
        Whether to overwrite existing files
        
    Returns
    -------
    dict
        Processing result with statistics
    """
    start_time = time.time()
    
    try:
        # Validate inputs
        validate_subject_id(subject_id)
        validate_session(session)
        
        # Determine max block for this session
        max_block = get_max_blocks(session)
        min_block = preprocessing_params.get('min_block', 1)
        
        # Check if already processed (unless overwrite=True)
        if not overwrite:
            from pyavs.utils.derivatives import get_derivatives_manager
            manager = get_derivatives_manager(data_path)
            prepro_dir = manager.get_preprocessed_path(subject_id, session)
            
            # Check if any preprocessed files exist
            existing_files = list(prepro_dir.glob(f"sub-{subject_id:02d}_ses-{session:02d}_*.fif"))
            if existing_files:
                return {
                    'subject_id': subject_id,
                    'session': session,
                    'status': 'skipped',
                    'reason': 'already_exists',
                    'n_files': len(existing_files),
                    'processing_time': 0,
                    'message': f'Found {len(existing_files)} existing files'
                }
        
        # Initialize composer
        composer = AVSComposer(
            data_path=data_path,
            subject=subject_id,
            session_num=session,
            min_block=min_block,
            max_block=max_block,
            preprocessed=False,  # Force recomputation
            recompute_prepro=True,
            verbose=preprocessing_params.get('verbose', False),
            n_jobs=1,  # Use single job per worker to avoid nested parallelism
            **{k: v for k, v in preprocessing_params.items() 
               if k in ['l_freq', 'h_freq', 'resample_freq', 'causal_filter',
                       'use_precomputed_ica', 'ica_solutions_path']}
        )
        
        # Load and preprocess MEG data
        composer.load_meg_data()
        
        # Get processing statistics
        n_blocks_processed = len([block for block, raw in composer.raws_dict.items() 
                                 if raw is not None])
        n_blocks_failed = len([block for block, raw in composer.raws_dict.items() 
                              if raw is None])
        
        # Check for empty room data availability
        empty_room_available = getattr(composer, 'empty_room_available', False)
        
        processing_time = time.time() - start_time
        
        return {
            'subject_id': subject_id,
            'session': session,
            'status': 'success',
            'n_blocks_processed': n_blocks_processed,
            'n_blocks_failed': n_blocks_failed,
            'empty_room_available': empty_room_available,
            'processing_time': processing_time,
            'total_blocks': max_block - min_block + 1,
            'message': f'Processed {n_blocks_processed}/{max_block-min_block+1} blocks'
        }
        
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = str(e)
        traceback_str = traceback.format_exc()
        
        return {
            'subject_id': subject_id,
            'session': session,
            'status': 'error',
            'error': error_msg,
            'traceback': traceback_str,
            'processing_time': processing_time,
            'message': f'Failed: {error_msg}'
        }


def get_all_subject_sessions(data_path: str, 
                           subjects: Optional[List[int]] = None,
                           sessions: Optional[List[int]] = None) -> List[Tuple[int, int]]:
    """
    Get all valid subject-session combinations in the dataset.
    
    Parameters
    ----------
    data_path : str
        Path to AVS dataset
    subjects : list of int, optional
        Specific subjects to process (default: auto-detect)
    sessions : list of int, optional
        Specific sessions to process (default: auto-detect)
        
    Returns
    -------
    list of tuple
        List of (subject_id, session) tuples
    """
    combinations = []
    
    # Auto-detect subjects if not specified
    if subjects is None:
        subjects = []
        rawdir = Path(data_path) / 'rawdir'
        if rawdir.exists():
            for subject_dir in rawdir.iterdir():
                if subject_dir.is_dir() and subject_dir.name.startswith('as'):
                    try:
                        subject_id = int(subject_dir.name[2:])
                        subjects.append(subject_id)
                    except ValueError:
                        continue
        subjects = sorted(subjects)
        logger.info(f"Auto-detected subjects: {subjects}")
    
    # Auto-detect sessions if not specified
    if sessions is None:
        sessions = list(range(1, 11))  # AVS has 10 sessions typically
        logger.info(f"Using default sessions: {sessions}")
    
    # Check which combinations actually exist
    for subject_id in subjects:
        for session in sessions:
            # Check if subject directory exists
            subject_dir = Path(data_path) / 'rawdir' / f'as{subject_id:02d}'
            session_dir = Path(f'{subject_dir}{chr(96+session)}')  # 01a, 01b, etc.
            print(f"Checking: {subject_dir}, {session_dir}")
            if session_dir.exists():
                combinations.append((subject_id, session))
            else:
                logger.debug(f"Skipping non-existent: subject {subject_id}, session {session}")
    
    logger.info(f"Found {len(combinations)} valid subject-session combinations")
    return combinations


def save_results_report(results: List[Dict], output_file: str):
    """Save detailed results report to JSON file."""
    report_data = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'results': results
    }
    
    with open(output_file, 'w') as f:
        json.dump(report_data, f, indent=2, default=str)
    
    logger.info(f"Detailed results saved to: {output_file}")


def main():
    """Main function for batch preprocessing."""
    parser = argparse.ArgumentParser(
        description="Batch preprocessing for all AVS subjects and sessions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process all subjects and sessions
    python batch_preprocessing.py --data_path /path/to/avs/data
    
    # Process specific subjects and sessions
    python batch_preprocessing.py --data_path /path/to/avs/data \\
        --subjects 1,2,3 --sessions 1,2,3
    
    # Use parallel processing
    python batch_preprocessing.py --data_path /path/to/avs/data --n_jobs 8
    
    # Use configuration file
    python batch_preprocessing.py --config preprocessing_config.json
    
    # Overwrite existing files
    python batch_preprocessing.py --data_path /path/to/avs/data \\
        --overwrite --subjects 1 --sessions 1
        """
    )
    
    # Configuration options
    parser.add_argument('--config', type=str,
                       help='Path to configuration JSON file')
    parser.add_argument('--data_path', type=str,
                       help='Path to AVS dataset directory')
    
    # Subject/session selection
    parser.add_argument('--subjects', type=str,
                       help='Comma-separated list of subject IDs (e.g., "1,2,3")')
    parser.add_argument('--sessions', type=str,
                       help='Comma-separated list of session numbers (e.g., "1,2,3")')
    
    # Processing parameters
    parser.add_argument('--min_block', type=int, default=1,
                       help='Minimum block number (default: 1)')
    parser.add_argument('--resample_freq', type=int, default=500,
                       help='Resampling frequency in Hz (default: 500)')
    parser.add_argument('--n_jobs', type=int,
                       help='Number of parallel jobs (default: CPU count - 1)')
    
    # Options
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing preprocessed files')
    parser.add_argument('--dry_run', action='store_true',
                       help='Show what would be processed without actually doing it')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging for individual jobs')
    parser.add_argument('--output_dir', type=str,
                       help='Directory to save results report (default: current dir)')
    
    args = parser.parse_args()
    
    try:
        # Load configuration if provided
        if args.config:
            logger.info(f"Loading configuration from: {args.config}")
            config = load_config(args.config)
            data_path = args.data_path or config.paths.data_path
            preprocessing_params = {
                'resample_freq': args.resample_freq or config.processing.resample_freq,
                'min_block': args.min_block,
                'verbose': args.verbose,
            }
        else:
            # Use command line arguments or defaults
            if not args.data_path:
                config = get_config()
                data_path = config.paths.data_path
                if not data_path:
                    parser.error("--data_path is required when not using --config")
            else:
                data_path = args.data_path
            
            preprocessing_params = {
                'resample_freq': args.resample_freq,
                'min_block': args.min_block,
                'verbose': args.verbose,
            }
        
        # Parse subject and session lists
        subjects = None
        if args.subjects:
            subjects = [int(s.strip()) for s in args.subjects.split(',')]
        
        sessions = None
        if args.sessions:
            sessions = [int(s.strip()) for s in args.sessions.split(',')]
        
        # Determine number of parallel jobs
        n_jobs = args.n_jobs or max(1, cpu_count() - 1)
        logger.info(f"Using {n_jobs} parallel jobs")
        
        # Get all subject-session combinations to process
        combinations = get_all_subject_sessions(data_path, subjects, sessions)
        
        if not combinations:
            logger.error("No valid subject-session combinations found!")
            sys.exit(1)
        
        # Dry run mode
        if args.dry_run:
            logger.info("DRY RUN MODE - showing what would be processed:")
            for i, (subject_id, session) in enumerate(combinations):
                logger.info(f"  {i+1:3d}. Subject {subject_id:02d}, Session {session}")
            logger.info(f"Total: {len(combinations)} combinations")
            return
        
        # Initialize statistics tracking
        stats = PreprocessingStats()
        stats.start(len(combinations))
        
        # Process in parallel
        results = []
        
        if n_jobs == 1:
            # Sequential processing
            logger.info("Running sequential processing...")
            for subject_id, session in combinations:
                result = preprocess_subject_session(
                    subject_id, session, data_path, 
                    preprocessing_params, args.overwrite
                )
                results.append(result)
                
                if result['status'] == 'success':
                    stats.add_result(result)
                elif result['status'] == 'skipped':
                    stats.add_skip(result)
                else:
                    stats.add_error(result)
        else:
            # Parallel processing
            logger.info(f"Running parallel processing with {n_jobs} workers...")
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                # Submit all jobs
                futures = {
                    executor.submit(
                        preprocess_subject_session,
                        subject_id, session, data_path,
                        preprocessing_params, args.overwrite
                    ): (subject_id, session)
                    for subject_id, session in combinations
                }
                
                # Collect results as they complete
                for future in as_completed(futures):
                    subject_id, session = futures[future]
                    try:
                        result = future.result()
                        results.append(result)
                        
                        if result['status'] == 'success':
                            stats.add_result(result)
                        elif result['status'] == 'skipped':
                            stats.add_skip(result)
                        else:
                            stats.add_error(result)
                            
                    except Exception as e:
                        error_result = {
                            'subject_id': subject_id,
                            'session': session,
                            'status': 'error',
                            'error': str(e),
                            'message': f'Future failed: {str(e)}'
                        }
                        results.append(error_result)
                        stats.add_error(error_result)
        
        # Generate final summary
        summary = stats.get_summary()
        
        logger.info("="*80)
        logger.info("BATCH PREPROCESSING COMPLETED")
        logger.info("="*80)
        logger.info(f"Total jobs: {summary['total_jobs']}")
        logger.info(f"Successful: {summary['completed_jobs']}")
        logger.info(f"Failed: {summary['failed_jobs']}")
        logger.info(f"Skipped: {summary['skipped_jobs']}")
        logger.info(f"Success rate: {summary['success_rate']:.1f}%")
        logger.info(f"Total time: {summary['elapsed_time']/60:.1f} minutes")
        logger.info(f"Average time per job: {summary['average_time_per_job']:.1f} seconds")
        
        # Show failed jobs if any
        if stats.errors:
            logger.warning(f"\nFailed jobs ({len(stats.errors)}):")
            for error in stats.errors:
                logger.warning(f"  Subject {error['subject_id']:02d}, "
                             f"Session {error['session']}: {error.get('error', 'Unknown error')}")
        
        # Save detailed results report
        output_dir = Path(args.output_dir) if args.output_dir else Path('.')
        output_dir.mkdir(exist_ok=True)
        
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        results_file = output_dir / f'batch_preprocessing_results_{timestamp}.json'
        save_results_report(results, str(results_file))
        
        # Exit with error code if there were failures
        if summary['failed_jobs'] > 0:
            logger.error(f"Batch preprocessing completed with {summary['failed_jobs']} failures")
            sys.exit(1)
        else:
            logger.info("Batch preprocessing completed successfully!")
            
    except KeyboardInterrupt:
        logger.warning("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Batch preprocessing failed: {e}")
        logger.debug("Full traceback:", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()