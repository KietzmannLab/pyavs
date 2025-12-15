#!/usr/bin/env python3
"""
Compute neural network embeddings from stored fixation crops.

This script processes previously stored fixation crop images and computes
neural network embeddings using various models. It follows the workflow pattern
from the old codebase, using thingsvision for efficient batch processing.

Usage:
    python compute_fixation_embeddings.py --subjects 1 2 3 --sessions 1 2 --crop-size 112 112
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

from pyavs.scenes.embeddings import extract_embeddings_from_crops, get_available_models, get_default_ecoset_path, create_bids_embeddings_path
    


from pyavs.utils.validation import validate_subject_id, validate_session
from pyavs.utils.logging import get_logger

# Initialize logger
logger = get_logger('scripts.compute_fixation_embeddings')


def find_crop_directories(data_path: str, subjects: Optional[List[int]] = None,
                         sessions: Optional[List[int]] = None,
                         crop_size: Optional[tuple] = None) -> List[Dict[str, Any]]:
    """
    Find available crop directories.
    
    Parameters
    ----------
    data_path : str
        Base data path
    subjects : list of int, optional
        Specific subjects to process
    sessions : list of int, optional
        Specific sessions to process
    crop_size : tuple, optional
        Specific crop size to process
        
    Returns
    -------
    list of dict
        List of dictionaries with subject, session, crop_size, and path info
    """
    derivatives_dir = Path(data_path) / 'derivatives' / 'pyavs'
    
    if not derivatives_dir.exists():
        logger.warning(f"Derivatives directory not found: {derivatives_dir}")
        return []
    
    combinations = []
    
    # Find available subject directories
    for sub_dir in derivatives_dir.glob('sub-*'):
        if not sub_dir.is_dir():
            continue
            
        try:
            subject_id = int(sub_dir.name.split('-')[1])
        except (IndexError, ValueError):
            continue
            
        if subjects and subject_id not in subjects:
            continue
            
        # Find available session directories
        for ses_dir in sub_dir.glob('ses-*'):
            if not ses_dir.is_dir():
                continue
                
            try:
                session = int(ses_dir.name.split('-')[1])
            except (IndexError, ValueError):
                continue
                
            if sessions and session not in sessions:
                continue
                
            # Find crop size directories
            crops_base_dir = ses_dir / 'fixation_crops'
            if not crops_base_dir.exists():
                continue
                
            for crop_size_dir in crops_base_dir.glob('*x*'):
                if not crop_size_dir.is_dir():
                    continue
                    
                try:
                    width, height = map(int, crop_size_dir.name.split('x'))
                    current_crop_size = (width, height)
                except ValueError:
                    continue
                    
                if crop_size and current_crop_size != crop_size:
                    continue
                    
                # Check if there are any PNG files
                png_files = list(crop_size_dir.glob('*.png'))
                if png_files:
                    combinations.append({
                        'subject_id': subject_id,
                        'session': session,
                        'crop_size': current_crop_size,
                        'crops_path': crop_size_dir,
                        'n_crops': len(png_files)
                    })
                    logger.debug(f"Found {len(png_files)} crops for subject {subject_id}, session {session}, size {current_crop_size}")
    
    return combinations


def compute_embeddings_for_subject(subject_id: int, session: int, crop_size: tuple,
                                 crops_path: Path, model_name: str, layers: List[str],
                                 batch_size: int, device: Optional[str],
                                 weights_path: Optional[str], data_path: str,
                                 overwrite: bool = False, verbose: bool = False) -> Dict[str, Any]:
    """
    Compute embeddings for all crops for a subject/session using thingsvision workflow.
    
    Parameters
    ----------
    subject_id : int
        Subject ID
    session : int
        Session number
    crop_size : tuple
        Crop size (width, height)
    crops_path : Path
        Path to directory containing crop PNG files
    model_name : str
        Model name for feature extraction
    layers : list of str
        Model layers to extract features from
    batch_size : int
        Batch size for processing
    device : str, optional
        Device to use ('cuda', 'cpu', 'mps')
    weights_path : str, optional
        Path to custom model weights
    data_path : str
        Base data path for saving outputs
    overwrite : bool, default False
        Whether to overwrite existing embeddings
    verbose : bool, default False
        Print verbose output
        
    Returns
    -------
    dict
        Processing results and statistics
    """
    
    logger.info(f"Processing crops for subject {subject_id}, session {session}, size {crop_size}")
    
    results = {
        'subject_id': subject_id,
        'session': session,
        'crop_size': crop_size,
        'status': 'failed',
        'layers_processed': [],
        'total_crops': 0,
        'error_message': None
    }
    
    try:
        # Validate inputs
        validate_subject_id(subject_id)
        validate_session(session)
        
        # Check if crops directory exists and has files
        if not crops_path.exists():
            results['error_message'] = f"Crops directory not found: {crops_path}"
            return results
            
        crop_files = list(crops_path.glob('*.png'))
        if not crop_files:
            results['error_message'] = "No crop files found"
            return results
            
        results['total_crops'] = len(crop_files)
        logger.info(f"Found {len(crop_files)} crop files")
        
        # Set up output directory
        output_dir = create_bids_embeddings_path(subject_id, session, data_path, model_name)
        
        # Extract embeddings using the cleaned embeddings module
        output_paths = extract_embeddings_from_crops(
            crops_dir=str(crops_path),
            output_dir=output_dir,
            model_name=model_name,
            layers=layers,
            batch_size=batch_size,
            device=device,
            weights_path=weights_path,
            overwrite=overwrite,
            verbose=verbose
        )
        
        # Record results
        results['status'] = 'success'
        results['layers_processed'] = list(output_paths.keys())
        
        logger.info(f"Successfully processed subject {subject_id}, session {session}")
        logger.info(f"Created embeddings for {results['total_crops']} crops across {len(results['layers_processed'])} layers")
        
    except Exception as e:
        results['error_message'] = str(e)
        logger.error(f"Error processing subject {subject_id}, session {session}: {e}")
    
    return results



def main():
    """Main processing function."""
    parser = argparse.ArgumentParser(
        description="Compute neural network embeddings from stored fixation crops",
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
    parser.add_argument('--layers', type=str, nargs='+', default=['layer1', "layer2", "layer3"], 
                       help='Model layers to extract features from (default: avgpool)')
    parser.add_argument('--crop-size', type=int, nargs=2, metavar=('WIDTH', 'HEIGHT'), default=(112,112),
                       help='Specific crop size to process (if not specified, processes all available sizes)')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for processing (default: 64)')
    parser.add_argument('--device', choices=['cuda', 'cpu', 'mps'],
                       help='Device to use (auto-detected if not specified)')
    parser.add_argument('--weights-path', type=str,
                       help='Path to custom model weights (uses EcoSet default if not specified)')
    
    # Processing options
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing embeddings')
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
        print("\\nVision models (standard pretrained):")
        for model in models['vision_models']:
            print(f"  - {model}")
        print(f"\\nEcoSet models (custom weights):")
        for model in models['ecoset_models']:
            print(f"  - {model}")
        print(f"\\nEcoSet weights path: {models['ecoset_weights_path']}")
        print(f"\\nAvailable layers by model:")
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
    
    # Get crop size filter
    crop_size = tuple(args.crop_size) if args.crop_size else None
    
    # Find available crop directories
    combinations = find_crop_directories(data_path, subjects, sessions, crop_size)
    
    if not combinations:
        logger.error("No crop directories found to process")
        return 1
        
    logger.info(f"Found {len(combinations)} crop directories to process")
    
    # Show model information
    logger.info(f"Using model: {args.model_name}")
    logger.info(f"Extracting from layers: {args.layers}")
    
    if args.model_name == 'resnet50_ecoset_crop' and args.weights_path is None:
        ecoset_path = get_default_ecoset_path()
        args.weights_path = ecoset_path
        if ecoset_path:
            logger.info(f"Using default EcoSet weights: {ecoset_path}")
        else:
            logger.warning("Default EcoSet weights not found")
    
    # Process combinations
    if args.n_jobs == 1:
        # Sequential processing
        results = []
        for i, combo in enumerate(combinations, 1):
            logger.info(f"Processing {i}/{len(combinations)}: Subject {combo['subject_id']}, Session {combo['session']}, Size {combo['crop_size']}")
            result = compute_embeddings_for_subject(
                subject_id=combo['subject_id'],
                session=combo['session'],
                crop_size=combo['crop_size'],
                crops_path=combo['crops_path'],
                model_name=args.model_name,
                layers=args.layers,
                batch_size=args.batch_size,
                device=args.device,
                weights_path=args.weights_path,
                data_path=data_path,
                overwrite=args.overwrite,
                verbose=args.verbose
            )
            results.append(result)
    else:
        # Parallel processing
        logger.info(f"Using {args.n_jobs} parallel jobs")
        results = Parallel(n_jobs=args.n_jobs)(
            delayed(compute_embeddings_for_subject)(
                subject_id=combo['subject_id'],
                session=combo['session'],
                crop_size=combo['crop_size'],
                crops_path=combo['crops_path'],
                model_name=args.model_name,
                layers=args.layers,
                batch_size=args.batch_size,
                device=args.device,
                weights_path=args.weights_path,
                data_path=data_path,
                overwrite=args.overwrite,
                verbose=args.verbose
            ) for combo in combinations
        )
    
    # Summary statistics
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']
    
    total_crops = sum(r['total_crops'] for r in successful)
    total_layers = sum(len(r['layers_processed']) for r in successful)
    
    logger.info("="*60)
    logger.info("PROCESSING SUMMARY")
    logger.info("="*60)
    logger.info(f"Total combinations processed: {len(results)}")
    logger.info(f"Successful: {len(successful)}")
    logger.info(f"Failed: {len(failed)}")
    logger.info(f"Total crops processed: {total_crops}")
    logger.info(f"Total layer embeddings created: {total_layers}")
    
    if failed:
        logger.warning("\\nFailed combinations:")
        for result in failed:
            logger.warning(f"  Subject {result['subject_id']}, Session {result['session']}: {result['error_message']}")
    
    if successful:
        logger.info(f"\\nEmbeddings saved to BIDS structure in: {data_path}/derivatives/pyavs/")
        logger.info("Files created:")
        logger.info(f"  - sub-XX/ses-YY/embeddings/{args.model_name}/{{layer}}/features.h5")
    
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())