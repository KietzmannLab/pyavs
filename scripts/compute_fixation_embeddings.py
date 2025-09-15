#!/usr/bin/env python3
"""
Compute fixation patch embeddings using ResNet50-EcoSet.

This script extracts neural network embeddings from fixation-based crops of scene images.
Instead of saving thousands of crop images, it directly computes and saves compressed
embeddings for storage efficiency.

Usage:
    python compute_fixation_embeddings.py --subjects 1 2 3 --sessions 1 2 --data-path /path/to/data
    python compute_fixation_embeddings.py --subject 1 --session 1 --model resnet50_ecoset_crop
    python compute_fixation_embeddings.py --all-subjects --all-sessions --layers avgpool fc

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

# Try to import pyavs components, handle thingsvision import issues gracefully
try:
    from pyavs.scenes import extract_crop_embeddings, get_available_models, get_default_ecoset_path
    EMBEDDINGS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import embeddings functions: {e}")
    print("This is likely due to thingsvision/torchtyping compatibility issues.")
    print("Please check your environment or install compatible versions.")
    EMBEDDINGS_AVAILABLE = False

from pyavs.io import load_scene_images
from pyavs.config import PyAVSConfig
from pyavs.utils.validation import validate_subject_id, validate_session
from pyavs.utils.logging import get_logger
from pyavs import AVSComposer

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

def process_subject_session(subject_id: int, session: int, data_path: str, 
                          model_name: str, layers: List[str], crop_size: tuple,
                          batch_size: int, device: Optional[str], 
                          weights_path: Optional[str], center_on: str) -> Dict[str, Any]:
    """
    Process embeddings for a single subject-session combination.
    
    Parameters
    ----------
    subject_id : int
        Subject ID
    session : int
        Session number
    data_path : str
        Base data path
    model_name : str
        Model name for feature extraction
    layers : list of str
        Model layers to extract features from
    crop_size : tuple
        Size of crops in pixels (width, height)
    batch_size : int
        Batch size for processing
    device : str, optional
        Device to use ('cuda', 'cpu', 'mps')
    weights_path : str, optional
        Path to custom model weights
    center_on : str
        Coordinate type to center crops on
        
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
        'embeddings_created': {},
        'total_crops': 0,
        'error_message': None
    }
    
    try:
        # Validate inputs
        validate_subject_id(subject_id)
        validate_session(session)
        
        # Load eye tracking data using AVS Composer
        logger.info(f"Loading eye events for subject {subject_id}, session {session}")
        
        # Initialize AVS Composer for eye tracking data loading
        composer = AVSComposer(
            subject=subject_id,
            session_num=session,
            data_path=data_path,
            output_path=data_path,
            et_path=data_path,
            preprocessed=True,
            recompute_prepro=False,
            max_block=None,
            min_block=1,
            verbose=False,
            interpolate_bad_channels=False,
            use_precomputed_ica=False,
            apply_ica=False
        )
        
        # Get eye tracking annotations for fixations with recording='scene'
        composer.get_et_annotations(
            event_type="fixation",
            recording="scene",
            exclude_last_fixation=True,
            add_cross_event_info=True,
            preprocessed=True
        )
        
        eye_events_df = composer.et_events
        
        if eye_events_df.empty:
            results['error_message'] = "No eye tracking data found"
            return results
            
        # Filter for fixation events and recording='scene' (already done by composer)
        fixations = eye_events_df
        if len(fixations) == 0:
            results['error_message'] = "No fixation events found"
            return results
            
        logger.info(f"Found {len(fixations)} fixation events for recording='scene'")
        
        # Load scene images
        logger.info("Loading scene images")
        scene_images = load_scene_images(data_path=data_path)
        
        if not scene_images:
            results['error_message'] = "No scene images found"
            return results
            
        # Set up configuration
        config = PyAVSConfig()
        
        # Extract embeddings
        logger.info(f"Extracting embeddings using model {model_name}")
        embeddings = extract_crop_embeddings(
            eye_events_df=fixations,
            scene_images=scene_images,
            config=config,
            crop_size=crop_size,
            model_name=model_name,
            layers=layers,
            batch_size=batch_size,
            device=device,
            weights_path=weights_path,
            center_on=center_on,
            use_bids_structure=True,
            data_path=data_path
        )
        
        # Record results
        results['status'] = 'success'
        results['total_crops'] = len(list(embeddings.values())[0]) if embeddings else 0
        
        for layer in layers:
            if layer in embeddings:
                results['embeddings_created'][layer] = {
                    'n_embeddings': len(embeddings[layer]),
                    'embedding_shape': list(embeddings[layer].values())[0].shape if embeddings[layer] else None
                }
                
        logger.info(f"Successfully processed subject {subject_id}, session {session}")
        logger.info(f"Created {results['total_crops']} embeddings across {len(layers)} layers")
        
    except Exception as e:
        results['error_message'] = str(e)
        logger.error(f"Error processing subject {subject_id}, session {session}: {e}")
    
    return results

def main():
    """Main processing function."""
    parser = argparse.ArgumentParser(
        description="Compute fixation patch embeddings using neural networks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process specific subjects and sessions
    python compute_fixation_embeddings.py --subjects 1 2 3 --sessions 1 2
    
    # Process single subject-session with custom settings
    python compute_fixation_embeddings.py --subject 1 --session 1 --model resnet50_ecoset_crop --batch-size 32
    
    # Process all available data with multiple layers
    python compute_fixation_embeddings.py --all-subjects --all-sessions --layers avgpool fc layer4
    
    # Use custom model weights
    python compute_fixation_embeddings.py --subjects 1 --sessions 1 --weights-path /path/to/weights.pth
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
    
    # Model and extraction parameters
    parser.add_argument('--model', '--model-name', dest='model_name', default='resnet50_ecoset_crop',
                       help='Model name for feature extraction (default: resnet50_ecoset_crop)')
    parser.add_argument('--layers', type=str, nargs='+', default=['avgpool'],
                       help='Model layers to extract features from (default: avgpool)')
    parser.add_argument('--crop-size', type=int, nargs=2, default=[112, 112], metavar=('WIDTH', 'HEIGHT'),
                       help='Size of crops in pixels (default: 112 112)')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for processing (default: 64)')
    parser.add_argument('--device', choices=['cuda', 'cpu', 'mps'],
                       help='Device to use (auto-detected if not specified)')
    parser.add_argument('--weights-path', type=str,
                       help='Path to custom model weights (uses EcoSet default if not specified)')
    parser.add_argument('--center-on', choices=['mean', 'start', 'end'], default='mean',
                       help='Coordinate type to center crops on (default: mean)')
    
    # Data and processing
    parser.add_argument('--data-path', type=str,
                       help='Path to data directory (uses configured path if not specified)')
    parser.add_argument('--n-jobs', type=int, default=1,
                       help='Number of parallel jobs (default: 1, use -1 for all cores)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Increase verbosity')
    
    # Information commands
    parser.add_argument('--list-models', action='store_true',
                       help='List available models and exit')
    
    args = parser.parse_args()
    
    # Set up logging
    if args.verbose:
        logging.getLogger('pyavs').setLevel(logging.DEBUG)
    
    # Handle information commands
    if args.list_models:
        models = get_available_models()
        print("Available models:")
        print("\nVision models (standard pretrained):")
        for model in models['vision_models']:
            print(f"  - {model}")
        print(f"\nEcoSet models (custom weights):")
        for model in models['ecoset_models']:
            print(f"  - {model}")
        print(f"\nEcoSet weights path: {models['ecoset_weights_path']}")
        print(f"\nAvailable layers by model:")
        for model, layers in models['layers'].items():
            print(f"  {model}: {', '.join(layers)}")
        return
    
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
    
    # Show model information
    logger.info(f"Using model: {args.model_name}")
    logger.info(f"Extracting from layers: {args.layers}")
    
    if args.model_name == 'resnet50_ecoset_crop' and args.weights_path is None:
        ecoset_path = get_default_ecoset_path()
        if ecoset_path:
            logger.info(f"Using default EcoSet weights: {ecoset_path}")
        else:
            logger.warning("Default EcoSet weights not found")
    
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
                model_name=args.model_name,
                layers=args.layers,
                crop_size=tuple(args.crop_size),
                batch_size=args.batch_size,
                device=args.device,
                weights_path=args.weights_path,
                center_on=args.center_on
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
                model_name=args.model_name,
                layers=args.layers,
                crop_size=tuple(args.crop_size),
                batch_size=args.batch_size,
                device=args.device,
                weights_path=args.weights_path,
                center_on=args.center_on
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
    logger.info(f"Total embeddings created: {total_crops}")
    
    if failed:
        logger.warning("\nFailed combinations:")
        for result in failed:
            logger.warning(f"  Subject {result['subject_id']}, Session {result['session']}: {result['error_message']}")
    
    if successful:
        logger.info(f"\nEmbeddings saved to BIDS structure in: {data_path}/derivatives/pyavs/")
        logger.info("Files created:")
        logger.info("  - sub-XX/ses-YY/embeddings/{model_name}/{layer}/embeddings.npz")
        logger.info("  - sub-XX/ses-YY/embeddings/{model_name}/sub-XX_ses-YY_embeddings_metadata.csv")
    
    return 0 if not failed else 1

if __name__ == "__main__":
    sys.exit(main())