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
    list of str
        Paths to saved annotated raw data files (one per recording type and event type combination)
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
    
    # Create output directory for annotated raws under pyavs derivatives
    output_dir = Path(data_path) / 'derivatives' / 'pyavs' / 'annotated_raws' / f'sub-{subject_id:02d}' / f'ses-{session:02d}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    # Load MEG data
    if verbose:
        logger.info("Loading MEG data...")
    composer.load_meg_data()
    
    # Concatenate raw data per session
    if verbose:
        logger.info("Concatenating raw data...")
    composer.concatenate_raws_per_session()
    
    # Apply resampling if specified
    if resample_freq and resample_freq != composer.raws_concatenated.info['sfreq']:
        if verbose:
            logger.info(f"Resampling to {resample_freq} Hz...")
        composer.resample_meg_data(target_sfreq=resample_freq)
    
    # Apply filtering
    if verbose:
        logger.info("Applying filters...")
    composer.filter_meg_data(**filter_params, ignore_existing_filter=True)
    
    # First, save the concatenated raw data (without annotations)
    if verbose:
        logger.info("Saving concatenated raw data...")
    
    # Create base raw filename
    base_raw_filename = f'sub-{subject_id:02d}_ses-{session:02d}_task-avs_raw-concatenated.fif'
    base_raw_path = output_dir / base_raw_filename
    
    # Add processing information to the base raw
    base_raw = composer.raws_concatenated.copy()
    base_raw.info['description'] = f'pyAVS concatenated raw data - Subject {subject_id}, Session {session}'
    
    # Add custom info about processing
    base_processing_info = {
        'pyavs_version': 'development',
        'subject_id': subject_id,
        'session': session,
        'resample_freq': resample_freq,
        'filter_params': filter_params,
        'blocks_processed': f'{min_block}-{max_block}',
        'n_channels': len(base_raw.ch_names),
        'n_bad_channels': len(base_raw.info['bads'])
    }
    
    # Make this a string sep with newlines
    base_processing_info_str = "\n".join(f"{k}: {v}" for k, v in base_processing_info.items())
    base_raw.info['description'] += f"\nProcessing info:\n{base_processing_info_str}"
    base_raw.save(base_raw_path, overwrite=overwrite, verbose=verbose)
    
    saved_files = [str(base_raw_path)]
    
    # Now create and save annotation files for each recording type
    recording_types = ['scene', 'microphone', 'caption']
    annotation_files = []
    event_counts = {}  # Track event counts per recording type
    
    for recording_type in recording_types:
        if verbose:
            logger.info(f"Processing {recording_type} recordings with all event types...")
        
        # Collect all annotations for this recording type
        combined_annotations = None
        recording_event_counts = {}  # Track counts per event type for this recording
        
        # Add annotations for all event types for this recording type
        for event_type in event_types:
            if verbose:
                logger.info(f"  Adding {event_type} annotations...")
            
            # Get annotations for this specific event type and recording type
            composer.get_et_annotations(event_type=event_type, recording=recording_type)
            
            # Collect the annotations from this event type
            if hasattr(composer, 'raws_annotated') and len(composer.raws_annotated.annotations) > 0:
                new_annotations = composer.raws_annotated.annotations
                
                # Count events for this event type
                event_type_count = len(new_annotations)
                recording_event_counts[event_type] = event_type_count
                
                if verbose:
                    logger.info(f"    Found {event_type_count} {event_type} events")
                
                if combined_annotations is None:
                    combined_annotations = new_annotations
                else:
                    combined_annotations = combined_annotations + new_annotations
            else:
                recording_event_counts[event_type] = 0
                if verbose:
                    logger.info(f"    Found 0 {event_type} events")
        
        # Store event counts for this recording type
        event_counts[recording_type] = recording_event_counts
        
        # Save the annotations as a separate MNE Annotations .fif file
        if combined_annotations is not None and len(combined_annotations) > 0:
            annotation_filename = f'sub-{subject_id:02d}_ses-{session:02d}_task-avs_annotations-{recording_type}.fif'
            annotation_path = output_dir / annotation_filename
            
            # Save using MNE's native Annotations save method
            combined_annotations.save(str(annotation_path), overwrite=overwrite)
            annotation_files.append(str(annotation_path))
            
            total_events = len(combined_annotations)
            if verbose:
                logger.info(f"Saved {total_events} total annotations for {recording_type} to: {annotation_path}")
        else:
            if verbose:
                logger.warning(f"No annotations found for {recording_type}")
    
    saved_files.extend(annotation_files)
    
    # Print summary
    if verbose:
        logger.info("="*60)
        logger.info("ANNOTATED RAW DATA SUMMARY")
        logger.info("="*60)
        logger.info(f"Subject: {subject_id}")
        logger.info(f"Session: {session}")
        logger.info(f"Files created: {len(saved_files)} (1 raw + {len(annotation_files)} annotation files)")
        logger.info(f"Recording types: {recording_types}")
        logger.info(f"Event types: {event_types}")
        
        # Show event counts per recording type
        logger.info("\nEvent counts per recording type:")
        total_events_all = 0
        for recording_type in recording_types:
            if recording_type in event_counts:
                counts = event_counts[recording_type]
                total_for_recording = sum(counts.values())
                total_events_all += total_for_recording
                
                logger.info(f"  {recording_type.capitalize()}:")
                for event_type in event_types:
                    count = counts.get(event_type, 0)
                    logger.info(f"    {event_type}: {count}")
                logger.info(f"    Total: {total_for_recording}")
            else:
                logger.info(f"  {recording_type.capitalize()}: No events found")
        
        logger.info(f"\nGrand total events: {total_events_all}")
        
        total_size = sum(Path(f).stat().st_size for f in saved_files) / (1024*1024)
        logger.info(f"Total file size: {total_size:.1f} MB")
        
        # Show files created
        logger.info("\nFiles created:")
        logger.info("Raw data file:")
        logger.info(f"  {Path(saved_files[0]).name}: {Path(saved_files[0]).stat().st_size / (1024*1024):.1f} MB")
        
        if annotation_files:
            logger.info("Annotation files:")
            for file_path in annotation_files:
                file_size = Path(file_path).stat().st_size / (1024)  # KB for .fif annotation files
                logger.info(f"  {Path(file_path).name}: {file_size:.1f} KB")
        
        logger.info("="*60)
        
    return saved_files


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
    output_paths = save_annotated_raw_data(
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
    
    print(f"\nSuccess! Annotated raw data saved to {len(output_paths)} files:")
    for path in output_paths:
        print(f"  {path}")


if __name__ == '__main__':
    main()