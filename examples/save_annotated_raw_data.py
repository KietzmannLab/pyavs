#!/usr/bin/env python3
"""
Save annotated raw MEG data for any subject and session.

This script demonstrates how to load, preprocess, and save MEG data with 
eye-tracking annotations using the pyAVS package. The saved data includes:
- Preprocessed MEG signals (filtered, resampled)
- Eye-tracking event annotations (saccades, fixations)
- Bad channel information
- Proper BIDS-compliant file naming

Usage:
    python save_annotated_raw_data.py --subject 1 --session 1
    python save_annotated_raw_data.py --subject 2 --session 3 --resample_freq 1000
    python save_annotated_raw_data.py --config my_config.json
"""

import sys
import argparse
from pathlib import Path

# Add pyavs to path if not installed
sys.path.insert(0, str(Path(__file__).parent.parent))

from pyavs.config import get_config, load_config
from pyavs.preprocessing.composer import AVSComposer
from pyavs.utils.logging import get_logger
from pyavs.utils.derivatives import get_derivatives_manager
from pyavs.utils.validation import validate_subject_id, validate_session

logger = get_logger(__name__)


def save_annotated_raw_data(subject_id: int, 
                           session: int,
                           data_path: str,
                           resample_freq: int = 500,
                           filter_params: dict = None,
                           event_types: list = None,
                           min_block: int = 1,
                           max_block: int = None,
                           overwrite: bool = False,
                           ignore_existing_filter: bool = False,
                           verbose: bool = True):
    """
    Save annotated raw MEG data for a subject and session.
    
    Parameters
    ----------
    subject_id : int
        Subject ID (1-99)
    session : int
        Session number (1-10)
    data_path : str
        Path to the AVS dataset
    resample_freq : int, optional
        Target sampling frequency in Hz (default: 500)
    filter_params : dict, optional
        Filter parameters (default: {'l_freq': 0.2, 'h_freq': 200})
    event_types : list, optional
        Event types to annotate (default: ['saccade', 'fixation'])
    min_block : int, optional
        Minimum block number (default: 1)
    max_block : int, optional
        Maximum block number (default: determined from session)
    overwrite : bool, optional
        Whether to overwrite existing files (default: False)
    ignore_existing_filter : bool, optional
        If True, ignore existing filters and apply new ones anyway (default: False)
    verbose : bool, optional
        Whether to print detailed progress (default: True)
        
    Returns
    -------
    str
        Path to saved annotated raw data file
    """
    # Validate inputs
    validate_subject_id(subject_id)
    validate_session(session)
    
    # Set defaults
    if filter_params is None:
        filter_params = {'l_freq': 0.2, 'h_freq': 200, 'picks': None, 'causal': True}
    
    if event_types is None:
        event_types = ['saccade', 'fixation', 'blink']
    
    if max_block is None:
        # Determine max block based on session
        from pyavs.utils.paths import get_max_blocks
        max_block = get_max_blocks(session)
    
    if verbose:
        logger.info(f"Processing subject {subject_id}, session {session}")
        logger.info(f"Blocks: {min_block}-{max_block}")
        logger.info(f"Resample frequency: {resample_freq} Hz")
        logger.info(f"Filter parameters: {filter_params}")
        logger.info(f"Event types: {event_types}")
    
    # Set up derivatives manager for BIDS-compliant paths
    manager = get_derivatives_manager(data_path)
    
    # Create output directory
    output_dir = manager.get_preprocessed_path(subject_id, session)
    
    # Create output filename
    output_filename = manager.create_bids_filename(
        subject_id=subject_id,
        session=session,
        task='avs',
        suffix='raw-annotated',
        extension='.fif'
    )
    output_path = output_dir / output_filename
    
    # Check if file already exists
    if output_path.exists() and not overwrite:
        logger.info(f"File already exists: {output_path}")
        logger.info("Use --overwrite to recreate the file")
        return str(output_path)
    
    # Initialize composer
    if verbose:
        logger.info("Initializing MEG data composer...")
    
    composer = AVSComposer(
        data_path=data_path,
        subject=subject_id,
        session_num=session,
        min_block=min_block,
        max_block=max_block,
        verbose=verbose
    )
    
    try:
        # Load MEG data
        if verbose:
            logger.info("Loading MEG data...")
        composer.load_meg_data()
        
        # Concatenate raw data per session
        if verbose:
            logger.info("Concatenating raw data...")
        composer.concatenate_raws_per_session()
        
        # Apply resampling if specified
        if resample_freq and resample_freq != composer.raw_concatenated.info['sfreq']:
            if verbose:
                logger.info(f"Resampling to {resample_freq} Hz...")
            composer.resample_meg_data(target_sfreq=resample_freq)
        
        # Apply filtering
        if verbose:
            logger.info("Applying filters...")
        composer.filter_meg_data(**filter_params, ignore_existing_filter=True)
        
        # Add eye-tracking annotations for each event type
        for event_type in event_types:
            if verbose:
                logger.info(f"Adding {event_type} annotations...")
            composer.get_et_annotations(event_type=event_type)
        
        # Get the annotated raw data
        annotated_raw = composer.raw_concatenated.copy()
        
        # Add processing information to info
        annotated_raw.info['description'] = f'pyAVS annotated raw data - Subject {subject_id}, Session {session}'
        
        # Add custom info about processing
        processing_info = {
            'pyavs_version': 'development',
            'subject_id': subject_id,
            'session': session,
            'resample_freq': resample_freq,
            'filter_params': filter_params,
            'event_types': event_types,
            'blocks_processed': f'{min_block}-{max_block}',
            'n_annotations': len(annotated_raw.annotations)
        }
        
        # Store processing info in the MNE info dict
        annotated_raw.info['proc_history'] = [processing_info]
        
        # Save the annotated raw data
        if verbose:
            logger.info(f"Saving annotated raw data to: {output_path}")
        
        annotated_raw.save(output_path, overwrite=overwrite, verbose=verbose)
        
        # Print summary
        if verbose:
            logger.info("="*60)
            logger.info("ANNOTATED RAW DATA SUMMARY")
            logger.info("="*60)
            logger.info(f"Subject: {subject_id}")
            logger.info(f"Session: {session}")
            logger.info(f"Sampling frequency: {annotated_raw.info['sfreq']:.1f} Hz")
            logger.info(f"Duration: {annotated_raw.times[-1]:.1f} seconds")
            logger.info(f"Channels: {len(annotated_raw.ch_names)} ({len(annotated_raw.info['bads'])} bad)")
            logger.info(f"Annotations: {len(annotated_raw.annotations)}")
            
            # Show annotation breakdown
            if len(annotated_raw.annotations) > 0:
                from collections import Counter
                annotation_counts = Counter(annotated_raw.annotations.description)
                for desc, count in annotation_counts.most_common():
                    logger.info(f"  {desc}: {count}")
            
            logger.info(f"File size: {output_path.stat().st_size / (1024*1024):.1f} MB")
            logger.info(f"Saved to: {output_path}")
            logger.info("="*60)
        
        return str(output_path)
        
    except Exception as e:
        logger.error(f"Error processing data: {e}")
        raise
    
    finally:
        # Clean up
        if hasattr(composer, 'raw_concatenated'):
            del composer.raw_concatenated
        del composer


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Save annotated raw MEG data for any subject and session",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python save_annotated_raw_data.py --subject 1 --session 1
    
    # With custom parameters
    python save_annotated_raw_data.py --subject 2 --session 3 \\
        --resample_freq 1000 --l_freq 1.0 --h_freq 100
    
    # Using configuration file
    python save_annotated_raw_data.py --config my_analysis_config.json
    
    # Process specific blocks only
    python save_annotated_raw_data.py --subject 1 --session 1 \\
        --min_block 5 --max_block 8
        """
    )
    
    # Configuration option
    parser.add_argument('--config', type=str,
                       help='Path to configuration JSON file')
    
    # Basic parameters
    parser.add_argument('--subject', type=int,
                       help='Subject ID (1-99)')
    parser.add_argument('--session', type=int,
                       help='Session number (1-10)')
    parser.add_argument('--data_path', type=str,
                       help='Path to AVS dataset directory')
    
    # Processing parameters
    parser.add_argument('--resample_freq', type=int, default=500,
                       help='Target sampling frequency in Hz (default: 500)')
    parser.add_argument('--l_freq', type=float, default=0.2,
                       help='Low-pass filter frequency (default: 0.2)')
    parser.add_argument('--h_freq', type=float, default=200,
                       help='High-pass filter frequency (default: 200)')
    parser.add_argument('--event_types', nargs='+', 
                       default=['saccade', 'fixation'],
                       help='Event types to annotate (default: saccade fixation)')
    
    # Block selection
    parser.add_argument('--min_block', type=int, default=1,
                       help='Minimum block number (default: 1)')
    parser.add_argument('--max_block', type=int,
                       help='Maximum block number (default: auto-detect)')
    
    # Options
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing files')
    parser.add_argument('--ignore_existing_filter', action='store_true',
                       help='Ignore existing filters and apply new ones anyway')
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce output verbosity')
    
    args = parser.parse_args()
    
    try:
        # Load configuration if provided
        if args.config:
            logger.info(f"Loading configuration from: {args.config}")
            config = load_config(args.config)
            
            # Override with command line arguments
            if args.subject:
                config.analysis.subject_id = args.subject
            if args.session:
                config.analysis.sessions = [args.session]
            if args.data_path:
                config.paths.data_path = args.data_path
            
            # Use config values
            subject_id = config.analysis.subject_id
            session = config.analysis.sessions[0] if config.analysis.sessions else 1
            data_path = config.paths.data_path
            resample_freq = args.resample_freq or config.processing.resample_freq
            filter_params = {
                'l_freq': args.l_freq or config.processing.filter_params.get('l_freq', 0.2),
                'h_freq': args.h_freq or config.processing.filter_params.get('h_freq', 200),
                'picks': None,
                'causal': True
            }
            
        else:
            # Use command line arguments or defaults
            if not args.subject or not args.session:
                parser.error("--subject and --session are required when not using --config")
            
            # Get default configuration for data_path if not provided
            if not args.data_path:
                config = get_config()
                data_path = config.paths.data_path
                if not data_path:
                    parser.error("--data_path is required or must be set in configuration")
            else:
                data_path = args.data_path
            
            subject_id = args.subject
            session = args.session
            resample_freq = args.resample_freq
            filter_params = {
                'l_freq': args.l_freq,
                'h_freq': args.h_freq,
                'picks': None,
                'causal': True
            }
        
        # Save annotated raw data
        output_path = save_annotated_raw_data(
            subject_id=subject_id,
            session=session,
            data_path=data_path,
            resample_freq=resample_freq,
            filter_params=filter_params,
            event_types=args.event_types,
            min_block=args.min_block,
            max_block=args.max_block,
            overwrite=args.overwrite,
            ignore_existing_filter=args.ignore_existing_filter,
            verbose=not args.quiet
        )
        
        print(f"\nSuccess! Annotated raw data saved to:\n{output_path}")
        
    except Exception as e:
        logger.error(f"Script failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()