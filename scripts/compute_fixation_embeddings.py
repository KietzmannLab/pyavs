#!/usr/bin/env python3
"""
Store fixation crops from scene images for later processing.

This script extracts and saves individual fixation crop images from scene presentations.
Crops are stored as individual PNG files following a systematic naming convention.

Usage:
    python compute_fixation_embeddings.py --subjects 1 2 3 --sessions 1 2 --data-path /path/to/data
    python compute_fixation_embeddings.py --subject 1 --session 1 --crop-size 112 112
    python compute_fixation_embeddings.py --all-subjects --all-sessions

Author: pyAVS development team
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from joblib import Parallel, delayed

# Add pyavs to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# No need for complex embedding dependencies - we're just storing crops

from pyavs.utils.validation import validate_subject_id, validate_session
from pyavs.utils.logging import get_logger
from PIL import Image
import pandas as pd
import numpy as np

# Initialize logger
logger = get_logger('scripts.compute_fixation_embeddings')

def setup_output_directory(data_path: str) -> Path:
    """
    Set up output directory for embeddings using BIDS structure.
    
    Parameters
    ----------
    data_path : str
        Base data path
        
    Returns
    -------
    Path
        Path to derivatives directory
    """
    derivatives_dir = Path(data_path) / 'derivatives' / 'pyavs'
    derivatives_dir.mkdir(parents=True, exist_ok=True)
    return derivatives_dir

def get_subject_sessions(data_path: str, subjects: Optional[List[int]] = None, 
                        sessions: Optional[List[int]] = None) -> List[tuple]:
    """
    Get all available subject-session combinations compatible with AVS Composer.
    
    Parameters
    ----------
    data_path : str
        Base data path
    subjects : list of int, optional
        Specific subjects to process
    sessions : list of int, optional
        Specific sessions to process
        
    Returns
    -------
    list of tuple
        List of (subject_id, session) pairs
    """
    from pyavs.utils.paths import get_legacy_paths
    
    # Check for legacy file structure that AVS Composer expects
    results_dir = Path(data_path) / 'results'
    
    if not results_dir.exists():
        logger.warning(f"Results directory not found: {results_dir}")
        return []
    
    combinations = []
    
    # Scan for subject-session directories in legacy format (as01_01, as02_01, etc.)
    for sub_sess_dir in results_dir.glob('as*_*'):
        if not sub_sess_dir.is_dir():
            continue
            
        try:
            # Parse directory name (e.g., "as01_01" -> subject_id=1, session=1)
            parts = sub_sess_dir.name.split('_')
            if len(parts) != 2:
                continue
                
            subject_part = parts[0]  # e.g., "as01"
            session_part = parts[1]  # e.g., "01"
            
            if not subject_part.startswith('as'):
                continue
                
            subject_id = int(subject_part[2:])  # Remove "as" prefix
            session = int(session_part)
            
        except (IndexError, ValueError):
            continue
            
        if subjects and subject_id not in subjects:
            continue
            
        if sessions and session not in sessions:
            continue
        
        # Check if the required eye tracking files exist for AVS Composer
        legacy_paths = get_legacy_paths(data_path, subject_id, session)
        events_file = Path(legacy_paths['events'])
        
        if events_file.exists():
            combinations.append((subject_id, session))
            logger.debug(f"Found valid data for subject {subject_id}, session {session}")
        else:
            logger.debug(f"Eye events file not found: {events_file}")
    
    return combinations


def store_fixation_crops(eye_events_df: pd.DataFrame, subject_id: int, session: int, 
                        data_path: str, crop_size: tuple) -> int:
    """
    Store fixation crops as individual image files.
    
    Parameters
    ----------
    eye_events_df : pd.DataFrame
        Eye tracking events dataframe with fixation locations
    subject_id : int
        Subject ID
    session : int
        Session number  
    data_path : str
        Base data path
    crop_size : tuple
        Size of crops in pixels (width, height)
        
    Returns
    -------
    int
        Number of crops created
    """
    from pyavs.utils.paths import get_legacy_paths
    
    # Set up output directory with crop size in path
    derivatives_dir = Path(data_path) / 'derivatives' / 'pyavs'
    subject_dir = derivatives_dir / f'sub-{subject_id:02d}' / f'ses-{session:02d}'
    crops_dir = subject_dir / 'fixation_crops' / f'{crop_size[0]}x{crop_size[1]}'
    crops_dir.mkdir(parents=True, exist_ok=True)
    
    # Screen parameters (from old codebase)
    screen_usage = 0.925
    stim_screen_size_xy = (1024, 768)
    screen_x_pix = stim_screen_size_xy[0]
    screen_y_pix = stim_screen_size_xy[1]
    
    # Scene path (from old codebase pattern)
    scene_prefix = "NSD_scenes_MEG_size_adjusted_"
    scene_suffix = str(screen_usage*100).replace(".", "")
    
    # Look for scene images in common locations
    potential_scene_paths = [
        os.path.join(data_path, "input", f"{scene_prefix}{scene_suffix}"),
        os.path.join(data_path, "stimuli", "scenes"),
        os.path.join(data_path, "input", "mscoco_scenes"),
        os.path.join(data_path, "mscoco_scenes")
    ]
    
    scene_path = None
    for path in potential_scene_paths:
        if os.path.exists(path):
            scene_path = path
            break
    
    if scene_path is None:
        raise FileNotFoundError("Could not find scene images directory")
    
    logger.info(f"Using scene path: {scene_path}")
    
    # Load experiment log to get scene filenames
    legacy_paths = get_legacy_paths(data_path, subject_id, session)
    exp_log_path = legacy_paths['experiment_log']
    
    if not os.path.exists(exp_log_path):
        raise FileNotFoundError(f"Experiment log not found: {exp_log_path}")
    
    exp_log = pd.read_csv(exp_log_path)
    
    total_crops = 0
    
    # Group by scene ID to process each scene
    for scene_id in eye_events_df['sceneID'].unique():
        scene_fixations = eye_events_df[eye_events_df['sceneID'] == scene_id]
        
        # Get scene filename from experiment log
        scene_fname_matches = exp_log.loc[
            (exp_log['subject'] == subject_id) & 
            (exp_log['trial'] >= 0) & 
            (exp_log['scene_ID'] == scene_id), 
            'scene_filename'
        ]
        
        if len(scene_fname_matches) == 0:
            logger.warning(f"No scene file found for scene {scene_id}")
            continue
            
        scene_fname = scene_fname_matches.values[0]
        scene_file_path = os.path.join(scene_path, scene_fname)
        
        if not os.path.exists(scene_file_path):
            logger.warning(f"Scene file not found: {scene_file_path}")
            continue
        
        # Load and resize scene image
        im = Image.open(scene_file_path)
        im_width = im.width
        im_height = im.height
        
        # Scale to presentation size
        im_scaler = (screen_y_pix * screen_usage) / im_height
        if np.round(im_scaler, 2) != 1:
            im_width_rescaled = int(im_width * im_scaler)
            im_height_rescaled = int(im_height * im_scaler)
            im_rescaled = im.resize((im_width_rescaled, im_height_rescaled))
        else:
            im_rescaled = im
            im_width_rescaled = im_width
            im_height_rescaled = im_height
        
        # Process each fixation
        for _, fixation in scene_fixations.iterrows():
            # Center coordinates around screen center
            x = fixation['mean_gx'] - screen_x_pix / 2
            y = fixation['mean_gy'] - screen_y_pix / 2
            
            # Calculate crop coordinates
            left = x + (im_width_rescaled / 2) - (crop_size[0] / 2)
            top = (im_height_rescaled / 2) - y - (crop_size[1] / 2)
            right = left + crop_size[0]
            bottom = top + crop_size[1]
            
            # Check boundaries
            if left < 0 or top < 0 or right > im_width_rescaled or bottom > im_height_rescaled:
                logger.debug(f"Crop goes beyond boundaries for scene {scene_id}, fixation {fixation['fix_sequence']}")
                continue
            
            # Create crop
            crop = im_rescaled.crop((left, top, right, bottom))
            
            # Create unique filename (following old codebase pattern)
            start_time = int(fixation['start_time'] * 1000)  # Convert to ms
            crop_identifier = f"{int(subject_id):02d}_{int(fixation['trial']):04d}_{int(fixation['fix_sequence']):02d}_{int(start_time):010d}_{int(scene_id):07d}"
            crop_filename = f"{crop_identifier}.png"
            crop_path = crops_dir / crop_filename
            
            # Save crop
            crop.save(crop_path)
            total_crops += 1
    
    logger.info(f"Stored {total_crops} fixation crops")
    return total_crops


def process_subject_session(subject_id: int, session: int, data_path: str, 
                          crop_size: tuple) -> Dict[str, Any]:
    """
    Process fixation crops for a single subject-session combination.
    
    Parameters
    ----------
    subject_id : int
        Subject ID
    session : int
        Session number
    data_path : str
        Base data path
    crop_size : tuple
        Size of crops in pixels (width, height)
        
    Returns
    -------
    dict
        Processing results and statistics
    """
    logger.info(f"Processing subject {subject_id}, session {session}")
    
    results = {
        'subject_id': subject_id,
        'session': session,
        'status': 'failed',
        'total_crops': 0,
        'error_message': None
    }
    
    try:
        # Validate inputs
        validate_subject_id(subject_id)
        validate_session(session)
        
        # Load eye tracking data using direct dataloader
        logger.info(f"Loading eye events for subject {subject_id}, session {session}")
        
        # Import the eye tracking dataloader function
        from pyavs.dataloader.eye import load_and_enrich_eye_events, add_cross_event_information, add_fixation_sequence_position
        
        # Load eye tracking events for fixations with recording='scene'
        _, eye_events_df = load_and_enrich_eye_events(
            subjects=[subject_id],
            sessions=[session],
            add_cross_event_info=True,
            data_path=data_path,
            preprocessed=True
        )
        eye_events_df = add_fixation_sequence_position(eye_events_df)
        # subsample the fixation events to only those with recording='scene'
        logger.info(f"removing non-scene recording events. Number of events removed: {len(eye_events_df) - len(eye_events_df[eye_events_df['recording'] == 'scene'])}")
        eye_events_df = eye_events_df[eye_events_df['recording'] == 'scene']
        logger.info(f"Removindg non-fixation events. Number of removed events: {len(eye_events_df) - len(eye_events_df[eye_events_df['type'] == 'fixation'])}")
        eye_events_df = eye_events_df[eye_events_df['type'] == 'fixation']
        
        if eye_events_df.empty:
            results['error_message'] = "No eye tracking data found"
            return results
            
        # Filter for fixation events and recording='scene' (already done by composer)
        fixations = eye_events_df
        if len(fixations) == 0:
            results['error_message'] = "No fixation events found"
            return results
            
        logger.info(f"Found {len(fixations)} fixation events for recording='scene'")
        
        # Store fixation crops
        logger.info("Storing fixation crops")
        total_crops = store_fixation_crops(
            eye_events_df=fixations,
            subject_id=subject_id,
            session=session,
            data_path=data_path,
            crop_size=crop_size
        )
        
        # Record results
        results['status'] = 'success'
        results['total_crops'] = total_crops
                
        logger.info(f"Successfully processed subject {subject_id}, session {session}")
        logger.info(f"Created {results['total_crops']} fixation crops")
        
    except Exception as e:
        results['error_message'] = str(e)
        logger.error(f"Error processing subject {subject_id}, session {session}: {e}")
    
    return results

def main():
    """Main processing function."""
    parser = argparse.ArgumentParser(
        description="Store fixation crops from scene images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process specific subjects and sessions
    python compute_fixation_embeddings.py --subjects 1 2 3 --sessions 1 2
    
    # Process single subject-session with custom crop size
    python compute_fixation_embeddings.py --subject 1 --session 1 --crop-size 224 224
    
    # Process all available data
    python compute_fixation_embeddings.py --all-subjects --all-sessions
        """
    )
    
    # Subject and session selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--subjects', type=int, nargs='+', 
                      help='Subject IDs to process')
    group.add_argument('--subject', type=int,
                      help='Single subject ID to process')
    group.add_argument('--all-subjects', action='store_true',
                      help='Process all available subjects')
    
    session_group = parser.add_mutually_exclusive_group(required=True)
    session_group.add_argument('--sessions', type=int, nargs='+',
                              help='Session numbers to process')
    session_group.add_argument('--session', type=int,
                              help='Single session number to process')
    session_group.add_argument('--all-sessions', action='store_true',
                              help='Process all available sessions')
    
    # Crop parameters
    parser.add_argument('--crop-size', type=int, nargs=2, default=[112, 112], metavar=('WIDTH', 'HEIGHT'),
                       help='Size of crops in pixels (default: 112 112)')
    
    # Data and processing
    parser.add_argument('--data-path', type=str,
                       help='Path to data directory (uses configured path if not specified)')
    parser.add_argument('--n-jobs', type=int, default=1,
                       help='Number of parallel jobs (default: 1, use -1 for all cores)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Increase verbosity')
    
    
    args = parser.parse_args()
    
    # Set up logging
    if args.verbose:
        logging.getLogger('pyavs').setLevel(logging.DEBUG)
    
    # Get data path
    if args.data_path:
        data_path = args.data_path
    else:
        from pyavs.utils.config import get_data_path
        data_path = get_data_path()
        if data_path is None:
            parser.error("No data path configured. Use --data-path to specify.")
    
    if not os.path.exists(data_path):
        parser.error(f"Data path does not exist: {data_path}")
    
    logger.info(f"Using data path: {data_path}")
    
    # Set up output directory
    setup_output_directory(data_path)
    
    # Parse subject and session arguments
    if args.subjects:
        subjects = args.subjects
    elif args.subject:
        subjects = [args.subject]
    else:  # all_subjects
        subjects = None
        
    if args.sessions:
        sessions = args.sessions
    elif args.session:
        sessions = [args.session]
    else:  # all_sessions
        sessions = None
    
    # Get subject-session combinations to process
    combinations = get_subject_sessions(data_path, subjects, sessions)
    
    if not combinations:
        logger.error("No subject-session combinations found to process")
        return 1
        
    logger.info(f"Found {len(combinations)} subject-session combinations to process")
    logger.info(f"Crop size: {args.crop_size[0]}x{args.crop_size[1]} pixels")
    
    # Process combinations
    if args.n_jobs == 1:
        # Sequential processing
        results = []
        for i, (subject_id, session) in enumerate(combinations, 1):
            logger.info(f"Processing {i}/{len(combinations)}: Subject {subject_id}, Session {session}")
            result = process_subject_session(
                subject_id=subject_id,
                session=session,
                data_path=data_path,
                crop_size=tuple(args.crop_size)
            )
            results.append(result)
    else:
        # Parallel processing
        logger.info(f"Using {args.n_jobs} parallel jobs")
        results = Parallel(n_jobs=args.n_jobs)(
            delayed(process_subject_session)(
                subject_id=subject_id,
                session=session,
                data_path=data_path,
                crop_size=tuple(args.crop_size)
            ) for subject_id, session in combinations
        )
    
    # Summary statistics
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']
    
    total_crops = sum(r['total_crops'] for r in successful)
    
    logger.info("="*60)
    logger.info("PROCESSING SUMMARY")
    logger.info("="*60)
    logger.info(f"Total combinations processed: {len(results)}")
    logger.info(f"Successful: {len(successful)}")
    logger.info(f"Failed: {len(failed)}")
    logger.info(f"Total fixation crops created: {total_crops}")
    
    if failed:
        logger.warning("\nFailed combinations:")
        for result in failed:
            logger.warning(f"  Subject {result['subject_id']}, Session {result['session']}: {result['error_message']}")
    
    if successful:
        logger.info(f"\nFixation crops saved to BIDS structure in: {data_path}/derivatives/pyavs/")
        logger.info("Files created:")
        logger.info("  - sub-XX/ses-YY/fixation_crops/{crop_size}/*.png")
    
    return 0 if not failed else 1

if __name__ == "__main__":
    sys.exit(main())