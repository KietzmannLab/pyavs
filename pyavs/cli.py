"""
Command-line interface for pyAVS package.

This module provides command-line tools for common pyAVS workflows,
making it easy to process data without writing custom scripts.
"""

import argparse
import sys
import os
import json
from pathlib import Path
from typing import Optional, List

import pyavs
from pyavs.utils.logging import get_logger, configure_logging, set_log_level

# Module logger
logger = get_logger('cli')


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='pyAVS: Python package for Active Visual Semantics dataset processing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check data availability
  pyavs check-data --subject 1 --session 1 --data-path /path/to/data
  
  # Preprocess MEG + eye tracking data
  pyavs preprocess --subject 1 --session 1 --blocks 1 2 3 --apply-ica
  
  # Create epochs from eye tracking events
  pyavs create-epochs --subject 1 --session 1 --event-type fixation --sensor-type meg
  
  # Run source reconstruction
  pyavs source-reconstruction --subject 1 --session 1 --method beamformer
        """
    )
    
    parser.add_argument('--data-path', type=str, help='Path to AVS dataset')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Logging level (default: INFO)')
    parser.add_argument('--log-file', type=str, help='Log to file')
    parser.add_argument('--no-colors', action='store_true', help='Disable colored output')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Check data availability
    check_parser = subparsers.add_parser('check-data', help='Check data availability')
    check_parser.add_argument('--subject', type=int, required=True, help='Subject ID')
    check_parser.add_argument('--session', type=int, required=True, help='Session number')
    
    # Preprocessing
    preprocess_parser = subparsers.add_parser('preprocess', help='Preprocess MEG + eye tracking data')
    preprocess_parser.add_argument('--subject', type=int, required=True, help='Subject ID')
    preprocess_parser.add_argument('--session', type=int, required=True, help='Session number')
    preprocess_parser.add_argument('--blocks', type=int, nargs='+', help='Block numbers to process')
    preprocess_parser.add_argument('--include-meg', action='store_true', default=True, help='Include MEG data')
    preprocess_parser.add_argument('--include-eye', action='store_true', default=True, help='Include eye tracking data')
    preprocess_parser.add_argument('--apply-ica', action='store_true', help='Apply ICA for artifact removal')
    preprocess_parser.add_argument('--output-dir', type=str, help='Output directory for processed data')
    
    # Create epochs
    epochs_parser = subparsers.add_parser('create-epochs', help='Create epochs from eye tracking events')
    epochs_parser.add_argument('--subject', type=int, required=True, help='Subject ID')
    epochs_parser.add_argument('--session', type=int, required=True, help='Session number')
    epochs_parser.add_argument('--event-type', type=str, choices=['fixation', 'saccade', 'all'], 
                              default='fixation', help='Event type')
    epochs_parser.add_argument('--sensor-type', type=str, choices=['meg', 'eeg', 'eye'], 
                              default='meg', help='Sensor type')
    epochs_parser.add_argument('--tmin', type=float, default=-0.2, help='Start time (s)')
    epochs_parser.add_argument('--tmax', type=float, default=0.5, help='End time (s)')
    epochs_parser.add_argument('--baseline', type=float, nargs=2, help='Baseline correction window')
    epochs_parser.add_argument('--block', type=int, help='Specific block to use')
    epochs_parser.add_argument('--save', action='store_true', help='Save epochs to file')
    
    # Source reconstruction
    source_parser = subparsers.add_parser('source-reconstruction', help='Perform source reconstruction')
    source_parser.add_argument('--subject', type=int, required=True, help='Subject ID')
    source_parser.add_argument('--session', type=int, required=True, help='Session number')
    source_parser.add_argument('--method', type=str, choices=['beamformer', 'mne'], 
                              default='beamformer', help='Source reconstruction method')
    source_parser.add_argument('--event-type', type=str, default='fixation', help='Event type for epochs')
    source_parser.add_argument('--roi-labels', type=str, nargs='+', help='ROI labels to extract')
    source_parser.add_argument('--save-source-data', action='store_true', help='Save source data')
    
    # Batch processing
    batch_parser = subparsers.add_parser('batch', help='Batch process multiple subjects/sessions')
    batch_parser.add_argument('--subjects', type=int, nargs='+', help='Subject IDs')
    batch_parser.add_argument('--sessions', type=int, nargs='+', help='Session numbers')
    batch_parser.add_argument('--workflow', type=str, choices=['preprocess', 'epochs', 'source'], 
                             required=True, help='Workflow to run')
    batch_parser.add_argument('--parallel', action='store_true', help='Run in parallel')
    batch_parser.add_argument('--n-jobs', type=int, default=1, help='Number of parallel jobs')
    
    # Configure (one-time data path setup)
    configure_parser = subparsers.add_parser(
        'configure',
        help='Configure the AVS data path (run once per machine)',
    )
    configure_parser.add_argument(
        '--data-path', type=str,
        help='Absolute path to the AVS data directory',
    )
    configure_parser.add_argument(
        '--show', action='store_true',
        help='Print the currently configured data path and exit',
    )

    # Setup (legacy, kept for backward compatibility)
    setup_parser = subparsers.add_parser('setup', help='Set up pyAVS configuration (legacy; prefer configure)')
    setup_parser.add_argument('--data-path', type=str, required=True, help='Path to AVS dataset')
    setup_parser.add_argument('--freesurfer-dir', type=str, help='FreeSurfer subjects directory')
    setup_parser.add_argument('--create-config', action='store_true', help='Create configuration file')
    
    args = parser.parse_args()
    
    # Configure logging based on arguments
    configure_logging(
        level=args.log_level,
        console=True,
        file_path=args.log_file,
        use_colors=not args.no_colors
    )
    
    # If verbose flag is set, override to DEBUG level
    if args.verbose:
        set_log_level('DEBUG')
        logger.debug("Verbose mode enabled - log level set to DEBUG")
    
    logger.info(f"pyAVS CLI started with log level: {args.log_level}")
    
    # Load configuration if provided
    if args.config:
        load_config(args.config)
    
    # Set data path
    if args.data_path:
        pyavs.set_data_path(args.data_path)
        logger.info(f"Data path set to: {args.data_path}")
    
    # Execute command
    if args.command == 'configure':
        configure_command(args)
    elif args.command == 'check-data':
        check_data_command(args)
    elif args.command == 'preprocess':
        preprocess_command(args)
    elif args.command == 'create-epochs':
        create_epochs_command(args)
    elif args.command == 'source-reconstruction':
        source_reconstruction_command(args)
    elif args.command == 'batch':
        batch_command(args)
    elif args.command == 'setup':
        setup_command(args)
    else:
        parser.print_help()


def configure_command(args):
    """Configure the data path (one-time per machine)."""
    if args.show:
        path = pyavs.get_data_path()
        if path is None:
            logger.error('No data path configured.')
            logger.error('Run: pyavs configure --data-path /path/to/data')
            sys.exit(1)
        print(path)
    elif args.data_path is None:
        logger.error('Provide --data-path or use --show to print the current path.')
        sys.exit(1)
    else:
        pyavs.configure(args.data_path)
        logger.info(f'Data path configured: {args.data_path}')
        logger.info('Config written to: ~/.config/pyavs/config.json')


def check_data_command(args):
    """Check data availability command."""
    logger.info(f"Checking data availability for subject {args.subject}, session {args.session}")
    
    try:
        availability = pyavs.check_data_availability(args.subject, args.session)

        logger.info("Data availability:")
        for data_type, available in availability['available'].items():
            status = "✓" if available else "✗"
            logger.info(f"  {status} {data_type}")

        if availability['missing']:
            logger.info("Missing files:")
            for missing_path in availability['missing']:
                logger.info(f"  {missing_path}")

    except Exception as e:
        logger.error(f"Error checking data: {e}")
        sys.exit(1)


def preprocess_command(args):
    """Preprocessing command."""
    logger.info(f"Preprocessing subject {args.subject}, session {args.session}")
    
    if args.blocks:
        logger.info(f"Processing blocks: {args.blocks}")
    
    try:
        subject_data = pyavs.load_and_preprocess(
            args.subject, args.session,
            blocks=args.blocks,
            include_meg=args.include_meg,
            include_eye=args.include_eye,
            preprocess_meg=True,
            apply_ica=args.apply_ica
        )
        
        logger.info("Preprocessing completed successfully!")
        logger.info(f"MEG data: {'✓' if subject_data['meg_data'] is not None else '✗'}")
        logger.info(f"Eye data: {'✓' if subject_data['eye_events'] is not None else '✗'}")
        
        if args.output_dir:
            # Save preprocessed data
            output_path = Path(args.output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            filename = f"sub-{args.subject:02d}_ses-{args.session:02d}_preprocessed.json"
            output_file = output_path / filename
            
            # Convert to serializable format
            save_data = {
                'subject_id': subject_data['subject_id'],
                'session': subject_data['session'],
                'preprocessing_info': subject_data['preprocessing_info'],
                'eye_events_count': len(subject_data['eye_events']) if subject_data['eye_events'] is not None else 0,
                'meg_blocks': list(subject_data['meg_data'].keys()) if subject_data['meg_data'] is not None else []
            }
            
            with open(output_file, 'w') as f:
                json.dump(save_data, f, indent=2)
            
            logger.info(f"Preprocessing info saved to: {output_file}")
            
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        sys.exit(1)


def create_epochs_command(args):
    """Create epochs command."""
    logger.info(f"Creating {args.event_type} epochs for subject {args.subject}, session {args.session}")
    
    try:
        # Load preprocessed data
        subject_data = pyavs.load_and_preprocess(args.subject, args.session)
        
        # Create epochs
        baseline = tuple(args.baseline) if args.baseline else None
        
        epochs, events = pyavs.get_epochs(
            subject_data,
            event_type=args.event_type,
            sensor_type=args.sensor_type,
            tmin=args.tmin,
            tmax=args.tmax,
            baseline=baseline,
            block=args.block
        )
        
        if epochs is not None:
            logger.info(f"Created {len(epochs)} epochs")
            logger.info(f"Epoch duration: {args.tmin} to {args.tmax} s")
            logger.info(f"Baseline: {baseline}")
            
            if args.save:
                # Save epochs
                filename = f"sub-{args.subject:02d}_ses-{args.session:02d}_{args.event_type}-epo.fif"
                epochs.save(filename, overwrite=True)
                logger.info(f"Epochs saved to: {filename}")
        else:
            logger.warning(f"No epochs created (sensor_type: {args.sensor_type})")
            
        logger.info(f"Events dataframe: {len(events)} events")
        
    except Exception as e:
        logger.error(f"Error creating epochs: {e}")
        sys.exit(1)


def source_reconstruction_command(args):
    """Source reconstruction command."""
    logger.info(f"Performing source reconstruction for subject {args.subject}, session {args.session}")
    logger.info(f"Method: {args.method}")
    
    try:
        # Load preprocessed data
        subject_data = pyavs.load_and_preprocess(args.subject, args.session)
        
        # Create epochs
        epochs, events = pyavs.get_epochs(
            subject_data,
            event_type=args.event_type,
            sensor_type='meg'
        )
        
        if len(epochs) == 0:
            logger.warning("No epochs available for source reconstruction")
            return
        
        # Load forward model
        forward_model = pyavs.load_forward_model(args.subject, args.session)
        
        # Apply source reconstruction
        source_data = pyavs.apply_source_reconstruction(
            epochs, forward_model, method=args.method
        )
        
        logger.info(f"Source reconstruction completed")
        logger.info(f"Source data shape: {source_data.shape}")
        
        # Extract ROI data if requested
        if args.roi_labels:
            roi_data = pyavs.extract_roi_data(
                source_data,
                forward_model['src'],
                args.roi_labels,
                subjects_dir=get_default_subjects_dir()
            )
            logger.info(f"Extracted data from {len(roi_data)} ROIs")
        
        # Save source data if requested
        if args.save_source_data:
            save_path = pyavs.save_source_data(
                source_data, events, args.subject, args.session,
                data_type=f'{args.method}_source_estimates'
            )
            logger.info(f"Source data saved to: {save_path}")
            
    except Exception as e:
        logger.error(f"Error during source reconstruction: {e}")
        sys.exit(1)


def batch_command(args):
    """Batch processing command."""
    logger.info(f"Running batch {args.workflow} workflow")
    logger.info(f"Subjects: {args.subjects}")
    logger.info(f"Sessions: {args.sessions}")
    
    # Create all subject/session combinations
    combinations = []
    for subject in args.subjects:
        for session in args.sessions:
            combinations.append((subject, session))
    
    logger.info(f"Total combinations: {len(combinations)}")
    
    if args.parallel and args.n_jobs > 1:
        logger.info(f"Running in parallel with {args.n_jobs} jobs")
        # TODO: Implement parallel processing
        logger.warning("Parallel processing not yet implemented - running sequentially")
    
    # Process each combination
    for i, (subject, session) in enumerate(combinations):
        logger.info(f"Processing {i+1}/{len(combinations)}: Subject {subject}, Session {session}")
        
        try:
            if args.workflow == 'preprocess':
                pyavs.load_and_preprocess(subject, session, preprocess_meg=True)
            elif args.workflow == 'epochs':
                subject_data = pyavs.load_and_preprocess(subject, session)
                epochs, events = pyavs.get_epochs(subject_data, 'fixation', 'meg')
                logger.info(f"  Created {len(epochs)} epochs")
            elif args.workflow == 'source':
                subject_data = pyavs.load_and_preprocess(subject, session)
                epochs, events = pyavs.get_epochs(subject_data, 'fixation', 'meg')
                forward_model = pyavs.load_forward_model(subject, session)
                source_data = pyavs.apply_source_reconstruction(epochs, forward_model)
                logger.info(f"  Source data shape: {source_data.shape}")
            
            logger.info(f"  ✓ Completed")
            
        except Exception as e:
            logger.error(f"  ✗ Error: {e}")
            continue
    
    logger.info("Batch processing completed!")


def setup_command(args):
    """Setup command (legacy — delegates to configure)."""
    logger.info("Setting up pyAVS configuration...")

    # Persist via the canonical configure path
    pyavs.configure(args.data_path)
    logger.info(f"Data path configured: {args.data_path}")

    if args.freesurfer_dir:
        os.environ['SUBJECTS_DIR'] = args.freesurfer_dir
        logger.info(f"FreeSurfer subjects directory set to: {args.freesurfer_dir}")

    logger.info("Setup completed!")


def load_config(config_path: str):
    """Load configuration from file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        if 'data_path' in config:
            pyavs.set_data_path(config['data_path'])
        
        if 'freesurfer_dir' in config:
            os.environ['SUBJECTS_DIR'] = config['freesurfer_dir']
        
        logger.info(f"Configuration loaded from: {config_path}")
        
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()