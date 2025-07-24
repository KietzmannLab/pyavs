#!/usr/bin/env python3
"""
Compute cross-session LCMV beamformer filters for MEG source reconstruction.

This script implements the strategy where:
1. Cross-session data covariance is computed from subsampled epochs across all sessions
2. Per-session noise covariance matrices are loaded from empty room recordings
3. LCMV beamformer filters are computed per session and stored with event-type specificity

The filters can then be used in population code computation for efficient source reconstruction.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import List, Optional

# Add pyavs to path if not installed
sys.path.insert(0, str(Path(__file__).parent.parent))

from pyavs.utils.logging import get_logger
from pyavs.utils.config import get_config
from pyavs.source.filters import (
    compute_cross_session_data_covariance,
    compute_per_session_lcmv_filters
)


def main():
    """Main function to compute cross-session LCMV filters."""
    
    parser = argparse.ArgumentParser(
        description='Compute cross-session LCMV beamformer filters using pyAVS config system',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration options
    parser.add_argument('--config', type=str,
                       help='Path to configuration file (JSON/YAML)')
    parser.add_argument('--subject-id', type=int,
                       help='Subject ID to process (overrides config)')
    parser.add_argument('--event-type', type=str,
                       choices=['saccade', 'fixation', 'blink', 'scene'],
                       help='Event type for filter computation (overrides config)')
    parser.add_argument('--sessions', nargs='+', type=int,
                       help='Sessions to include (overrides config)')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing files')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logger = get_logger(__name__)
    
    # Initialize configuration
    from pyavs.config import get_config, load_config
    
    if args.config:
        logger.info(f"Loading configuration from: {args.config}")
        config = load_config(args.config)
    else:
        config = get_config()
    
    # Override config with command line arguments
    if args.subject_id is not None:
        config.analysis.subject_id = args.subject_id
    if args.event_type is not None:
        config.analysis.event_type = args.event_type
    if args.sessions is not None:
        config.analysis.sessions = args.sessions
    
    # Validate configuration
    try:
        config.validate()
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return
    
    logger.info(f"Processing subject {config.analysis.subject_id}, "
                f"event type: {config.analysis.event_type}")
    logger.info(f"Sessions: {config.analysis.sessions}")
    logger.info(f"Data path: {config.paths.data_path}")
    
    try:
        # Step 1: Compute cross-session data covariance
        logger.info("Step 1: Computing cross-session data covariance...")
        
        # Get filter parameters from config
        filter_kwargs = config.get_filter_kwargs()
        
        cross_session_epochs = compute_cross_session_data_covariance(
            data_path=config.paths.data_path,
            subject_id=config.analysis.subject_id,
            sessions=config.analysis.sessions,
            event_type=config.analysis.event_type,
            overwrite=args.overwrite,
        )
        
        logger.info(f"Cross-session epochs shape: {cross_session_epochs.get_data().shape}")
        
        # Step 2: Compute per-session LCMV filters
        logger.info("Step 2: Computing per-session LCMV filters...")
        
        filters = compute_per_session_lcmv_filters(
            data_path=config.paths.data_path,
            subject_id=config.analysis.subject_id,
            sessions=config.analysis.sessions,
            event_type=config.analysis.event_type,
            cross_session_epochs=cross_session_epochs,
            overwrite=args.overwrite,
        )
        
        logger.info(f"Successfully computed filters for {len(filters)} sessions")
        
        # Summary - using parameter signature path
        from pyavs.source.filters import _generate_parameter_signature
        
        param_signature = _generate_parameter_signature(**config.get_parameter_signature_dict())
        
        param_dir = os.path.join(config.paths.data_path, 'derivatives', 'pyavs', 'population_codes', param_signature)
        subject_group = f"sub{((config.analysis.subject_id - 1) // 5) * 5 + 1:02d}-{min(((config.analysis.subject_id - 1) // 5 + 1) * 5, 99):02d}"
        filter_dir = os.path.join(param_dir, subject_group, 'beamformer_filters')
        
        logger.info("="*60)
        logger.info("FILTER COMPUTATION SUMMARY")
        logger.info("="*60)
        logger.info(f"Subject: {config.analysis.subject_id}")
        logger.info(f"Event type: {config.analysis.event_type}")
        logger.info(f"Sessions processed: {sorted(filters.keys())}")
        logger.info(f"Cross-session epochs: {len(cross_session_epochs)} epochs")
        logger.info(f"Parameter signature: {param_signature}")
        logger.info(f"Filter directory: {filter_dir}")
        logger.info("="*60)
        
        # Verification
        logger.info("Verifying saved filters...")
        for session in config.analysis.sessions:
            filter_file = os.path.join(filter_dir, f'lcmv_filters_sess{session:02d}.h5')
            if os.path.exists(filter_file):
                file_size = os.path.getsize(filter_file) / (1024*1024)  # MB
                logger.info(f"  Session {session}: {file_size:.1f} MB")
            else:
                logger.warning(f"  Session {session}: MISSING")
        
        logger.info("Filter computation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during filter computation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()