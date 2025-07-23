#!/usr/bin/env python3
"""
Compute cross-session LCMV beamformer filters for MEG source reconstruction.

This script implements the strategy from the AVS machine room where:
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

from pyavs.utils.logging import setup_logger
from pyavs.utils.config import get_config
from pyavs.source.filters import (
    compute_cross_session_data_covariance,
    compute_per_session_lcmv_filters
)


def main():
    """Main function to compute cross-session LCMV filters."""
    
    parser = argparse.ArgumentParser(
        description='Compute cross-session LCMV beamformer filters',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--subject-id', type=int, required=True,
                       help='Subject ID to process')
    parser.add_argument('--event-type', type=str, required=True,
                       choices=['saccade', 'fixation', 'button', 'stimulus'],
                       help='Event type for filter computation')
    
    # Optional arguments
    parser.add_argument('--data-path', type=str, 
                       help='Path to dataset (uses config default if not specified)')
    parser.add_argument('--sessions', nargs='+', type=int,
                       default=list(range(1, 11)),
                       help='Sessions to include in filter computation')
    parser.add_argument('--n-epochs-per-session', type=int, default=350,
                       help='Number of epochs to sample per session for data covariance')
    parser.add_argument('--tmin', type=float, default=-0.5,
                       help='Start time for epochs (seconds)')
    parser.add_argument('--tmax', type=float, default=0.8,
                       help='End time for epochs (seconds)')
    parser.add_argument('--resample-freq', type=int, default=500,
                       help='Resampling frequency (Hz)')
    parser.add_argument('--filter-low', type=float, default=0.2,
                       help='Low-pass filter frequency (Hz)')
    parser.add_argument('--filter-high', type=float, default=200,
                       help='High-pass filter frequency (Hz)')
    parser.add_argument('--pick-ori', type=str, default='normal',
                       choices=['normal', 'max-power', 'vector', None],
                       help='Orientation selection for beamformer')
    parser.add_argument('--block-selection', type=str, default='all',
                       choices=['all', '10_only'],
                       help='Block selection strategy')
    parser.add_argument('--reg', type=float, default=0.05,
                       help='Regularization parameter for beamformer')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing files')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logger = setup_logger(__name__, level=log_level)
    
    # Get data path
    if args.data_path is None:
        config = get_config()
        data_path = config.get('data_path', '/share/klab/datasets/avs')
    else:
        data_path = args.data_path
    
    logger.info(f"Processing subject {args.subject_id}, event type: {args.event_type}")
    logger.info(f"Sessions: {args.sessions}")
    logger.info(f"Data path: {data_path}")
    
    try:
        # Step 1: Compute cross-session data covariance
        logger.info("Step 1: Computing cross-session data covariance...")
        
        filter_params = {
            "l_freq": args.filter_low,
            "h_freq": args.filter_high,
            "picks": None,
            "causal": True
        }
        
        cross_session_epochs = compute_cross_session_data_covariance(
            data_path=data_path,
            subject_id=args.subject_id,
            sessions=args.sessions,
            event_type=args.event_type,
            n_epochs_per_session=args.n_epochs_per_session,
            tmin=args.tmin,
            tmax=args.tmax,
            filter_params=filter_params,
            resample_freq=args.resample_freq,
            rois=None,  # Could be an argument  
            blocks=None,  # Could be an argument
            hemi='both',  # Could be an argument
            block_selection=args.block_selection,
            overwrite=args.overwrite
        )
        
        logger.info(f"Cross-session epochs shape: {cross_session_epochs.get_data().shape}")
        
        # Step 2: Compute per-session LCMV filters
        logger.info("Step 2: Computing per-session LCMV filters...")
        
        filters = compute_per_session_lcmv_filters(
            data_path=data_path,
            subject_id=args.subject_id,
            sessions=args.sessions,
            event_type=args.event_type,
            tmin=args.tmin,
            tmax=args.tmax,
            filter_params=filter_params,
            resample_freq=args.resample_freq,
            rois=None,
            blocks=None,
            hemi='both',
            n_epochs_per_session=args.n_epochs_per_session,
            cross_session_epochs=cross_session_epochs,
            pick_ori=args.pick_ori,
            reg=args.reg,
            overwrite=args.overwrite
        )
        
        logger.info(f"Successfully computed filters for {len(filters)} sessions")
        
        # Summary - using parameter signature path
        from pyavs.source.filters import _generate_parameter_signature
        
        param_signature = _generate_parameter_signature(
            event_type=args.event_type,
            sampling_rate=args.resample_freq,
            filter_params=filter_params,
            hemi='both',
            rois=None,
            blocks=None,
            tmin=args.tmin,
            tmax=args.tmax,
            n_epochs_per_session=args.n_epochs_per_session
        )
        
        param_dir = os.path.join(data_path, 'derivatives', 'pyavs', 'population_codes', param_signature)
        subject_group = f"sub{((args.subject_id - 1) // 5) * 5 + 1:02d}-{min(((args.subject_id - 1) // 5 + 1) * 5, 99):02d}"
        filter_dir = os.path.join(param_dir, subject_group, 'beamformer_filters')
        
        logger.info("="*60)
        logger.info("FILTER COMPUTATION SUMMARY")
        logger.info("="*60)
        logger.info(f"Subject: {args.subject_id}")
        logger.info(f"Event type: {args.event_type}")
        logger.info(f"Sessions processed: {sorted(filters.keys())}")
        logger.info(f"Cross-session epochs: {len(cross_session_epochs)} epochs")
        logger.info(f"Parameter signature: {param_signature}")
        logger.info(f"Filter directory: {filter_dir}")
        logger.info("="*60)
        
        # Verification
        logger.info("Verifying saved filters...")
        for session in args.sessions:
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