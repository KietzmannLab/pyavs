#!/usr/bin/env python3
"""
Compute and save noise covariance matrices for MEG source reconstruction.

This script computes noise covariance matrices from empty room recordings
across all sessions for each subject and saves them for use in source
reconstruction pipelines.
"""

import os
import sys
import logging
import numpy as np
import mne
from pathlib import Path
from typing import List, Optional, Dict, Tuple

# Add pyavs to path if not installed
sys.path.insert(0, str(Path(__file__).parent.parent))

import pyavs
from pyavs.utils.logging import get_logger
from pyavs.utils.config import get_config
from pyavs.utils.paths import get_derivatives_path
from pyavs.dataloader.meg import load_raw_meg


def compute_empty_room_covariance(
    data_path: str,
    subject_id: int,
    sessions: Optional[List[int]] = None,
    tmin: float = 0.0,
    tmax: Optional[float] = None,
    method: str = 'empirical',
    save_individual: bool = True,
    overwrite: bool = False
) -> Tuple[mne.Covariance, Dict[int, mne.Covariance]]:
    """
    Compute noise covariance matrix from empty room recordings.
    
    Parameters
    ----------
    data_path : str
        Path to the dataset
    subject_id : int
        Subject ID
    sessions : list of int, optional
        Sessions to include. If None, uses all available sessions
    tmin : float
        Start time for covariance computation (default: 0.0)
    tmax : float, optional
        End time for covariance computation. If None, uses entire recording
    method : str
        Covariance estimation method ('empirical', 'diagonal_fixed', 'shrunk', 'oas', 'ledoit_wolf')
    save_individual : bool
        Whether to save individual session covariance matrices
    overwrite : bool
        Whether to overwrite existing files
        
    Returns
    -------
    combined_cov : mne.Covariance
        Combined covariance matrix across all sessions
    session_covs : dict
        Individual session covariance matrices
    """
    logger = logging.getLogger(__name__)
    
    # Create output directory
    cov_dir = os.path.join(get_derivatives_path(data_path, subject_id), 'source_reconstruction', 'noise_covariance')
    os.makedirs(cov_dir, exist_ok=True)
    
    # Combined covariance filename
    combined_cov_file = os.path.join(cov_dir, f'sub-{subject_id:02d}_task-avs_desc-emptyroom_cov.fif')
    
    if os.path.exists(combined_cov_file) and not overwrite:
        logger.info(f"Loading existing combined covariance: {combined_cov_file}")
        return mne.read_cov(combined_cov_file), {}
    
    if sessions is None:
        sessions = list(range(1, 11))  # Default to sessions 1-10
    
    logger.info(f"Computing noise covariance for subject {subject_id}, sessions {sessions}")
    
    empty_room_raws = []
    session_covs = {}
    
    for session in sessions:
        logger.info(f"Processing session {session}")
        
        # Individual session covariance filename
        session_cov_file = os.path.join(cov_dir, f'sub-{subject_id:02d}_ses-{session:02d}_task-avs_desc-emptyroom_cov.fif')
        
        if os.path.exists(session_cov_file) and not overwrite:
            logger.info(f"Loading existing session covariance: {session_cov_file}")
            session_cov = mne.read_cov(session_cov_file)
            session_covs[session] = session_cov
            continue
        
        try:
            # Try to load empty room recordings for this session
            # Look for both before ('b') and after ('d') empty room recordings
            er_raws = []
            
            for suffix in ['b', 'd']:  # before and after session
                try:
                    er_raw = load_raw_meg(
                        data_path=data_path,
                        subject_id=subject_id,
                        session=session,
                        recording_type='empty_room',
                        suffix=suffix,
                        preload=True
                    )
                    er_raws.append(er_raw)
                    logger.info(f"Loaded empty room recording: sub-{subject_id:02d}_ses-{session:02d}_{suffix}")
                except Exception as e:
                    logger.warning(f"Could not load empty room recording for session {session}, suffix {suffix}: {e}")
            
            if not er_raws:
                logger.warning(f"No empty room recordings found for session {session}")
                continue
            
            # Concatenate empty room recordings for this session
            if len(er_raws) > 1:
                session_raw = mne.concatenate_raws(er_raws)
            else:
                session_raw = er_raws[0]
            
            # Compute covariance for this session
            session_cov = mne.compute_raw_covariance(
                session_raw,
                tmin=tmin,
                tmax=tmax,
                method=method,
                rank='info'
            )
            
            session_covs[session] = session_cov
            
            # Save individual session covariance if requested
            if save_individual:
                session_cov.save(session_cov_file, overwrite=overwrite)
                logger.info(f"Saved session covariance: {session_cov_file}")
            
            # Add to list for combined covariance
            empty_room_raws.extend(er_raws)
            
        except Exception as e:
            logger.error(f"Error processing session {session}: {e}")
            continue
    
    if not empty_room_raws:
        raise ValueError(f"No empty room recordings found for subject {subject_id}")
    
    # Compute combined covariance across all sessions
    logger.info("Computing combined covariance matrix across all sessions")
    combined_raw = mne.concatenate_raws(empty_room_raws)
    
    combined_cov = mne.compute_raw_covariance(
        combined_raw,
        tmin=tmin,
        tmax=tmax,
        method=method,
        rank='info'
    )
    
    # Save combined covariance
    combined_cov.save(combined_cov_file, overwrite=overwrite)
    logger.info(f"Saved combined covariance: {combined_cov_file}")
    
    return combined_cov, session_covs


def diagnose_covariance_generalization(
    session_covs: Dict[int, mne.Covariance],
    combined_cov: mne.Covariance,
    subject_id: int,
    save_report: bool = True,
    output_dir: Optional[str] = None
) -> Dict[str, float]:
    """
    Diagnose how well noise covariance matrices generalize across sessions.
    
    Parameters
    ----------
    session_covs : dict
        Individual session covariance matrices
    combined_cov : mne.Covariance
        Combined covariance matrix
    subject_id : int
        Subject ID for reporting
    save_report : bool
        Whether to save diagnostic report
    output_dir : str, optional
        Output directory for report
        
    Returns
    -------
    diagnostics : dict
        Diagnostic metrics
    """
    logger = logging.getLogger(__name__)
    
    if len(session_covs) < 2:
        logger.warning("Need at least 2 sessions for generalization diagnostics")
        return {}
    
    logger.info(f"Running covariance generalization diagnostics for subject {subject_id}")
    
    diagnostics = {}
    
    # Compare each session covariance to the combined covariance
    session_similarities = []
    
    for session, session_cov in session_covs.items():
        # Compute Frobenius norm of difference
        diff_norm = np.linalg.norm(session_cov.data - combined_cov.data, 'fro')
        combined_norm = np.linalg.norm(combined_cov.data, 'fro')
        relative_diff = diff_norm / combined_norm
        
        session_similarities.append(relative_diff)
        diagnostics[f'session_{session}_relative_diff'] = relative_diff
        
        logger.info(f"Session {session} relative difference: {relative_diff:.4f}")
    
    # Overall statistics
    diagnostics['mean_relative_diff'] = np.mean(session_similarities)
    diagnostics['std_relative_diff'] = np.std(session_similarities)
    diagnostics['max_relative_diff'] = np.max(session_similarities)
    diagnostics['min_relative_diff'] = np.min(session_similarities)
    
    # Compute pairwise session similarities
    sessions = list(session_covs.keys())
    pairwise_diffs = []
    
    for i, sess1 in enumerate(sessions):
        for sess2 in sessions[i+1:]:
            diff_norm = np.linalg.norm(session_covs[sess1].data - session_covs[sess2].data, 'fro')
            norm1 = np.linalg.norm(session_covs[sess1].data, 'fro')
            relative_diff = diff_norm / norm1
            pairwise_diffs.append(relative_diff)
            
            diagnostics[f'pairwise_diff_ses{sess1}_ses{sess2}'] = relative_diff
    
    diagnostics['mean_pairwise_diff'] = np.mean(pairwise_diffs)
    diagnostics['std_pairwise_diff'] = np.std(pairwise_diffs)
    
    # Save diagnostic report
    if save_report and output_dir:
        report_file = os.path.join(output_dir, f'sub-{subject_id:02d}_covariance_diagnostics.txt')
        
        with open(report_file, 'w') as f:
            f.write(f"Noise Covariance Generalization Diagnostics\n")
            f.write(f"Subject: {subject_id}\n")
            f.write(f"Sessions analyzed: {list(session_covs.keys())}\n\n")
            
            f.write("Individual Session vs Combined Covariance:\n")
            for session in session_covs.keys():
                diff = diagnostics[f'session_{session}_relative_diff']
                f.write(f"  Session {session}: {diff:.4f}\n")
            
            f.write(f"\nSummary Statistics:\n")
            f.write(f"  Mean relative difference: {diagnostics['mean_relative_diff']:.4f}\n")
            f.write(f"  Std relative difference: {diagnostics['std_relative_diff']:.4f}\n")
            f.write(f"  Max relative difference: {diagnostics['max_relative_diff']:.4f}\n")
            f.write(f"  Min relative difference: {diagnostics['min_relative_diff']:.4f}\n")
            
            f.write(f"\nPairwise Session Comparisons:\n")
            f.write(f"  Mean pairwise difference: {diagnostics['mean_pairwise_diff']:.4f}\n")
            f.write(f"  Std pairwise difference: {diagnostics['std_pairwise_diff']:.4f}\n")
            
            # Interpretation
            f.write(f"\nInterpretation:\n")
            if diagnostics['mean_relative_diff'] < 0.1:
                f.write("  Excellent generalization - covariance matrices are very similar across sessions\n")
            elif diagnostics['mean_relative_diff'] < 0.2:
                f.write("  Good generalization - covariance matrices are reasonably similar across sessions\n")
            elif diagnostics['mean_relative_diff'] < 0.5:
                f.write("  Moderate generalization - some variation in covariance across sessions\n")
            else:
                f.write("  Poor generalization - significant variation in covariance across sessions\n")
        
        logger.info(f"Saved diagnostic report: {report_file}")
    
    return diagnostics


def main():
    """Main function to compute noise covariance matrices."""
    # Set up logging
    logger = get_logger(__name__)
    
    # Get configuration
    config = get_config()
    data_path = config.get('data_path') or pyavs.get_data_path()
    
    # Parse command line arguments (basic implementation)
    import argparse
    parser = argparse.ArgumentParser(description='Compute noise covariance matrices')
    parser.add_argument('--subject-id', type=int, required=True, help='Subject ID')
    parser.add_argument('--sessions', nargs='+', type=int, help='Sessions to include')
    parser.add_argument('--data-path', type=str, default=data_path, help='Path to dataset')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing files')
    parser.add_argument('--method', type=str, default='empirical', 
                       choices=['empirical', 'diagonal_fixed', 'shrunk', 'oas', 'ledoit_wolf'],
                       help='Covariance estimation method')
    
    args = parser.parse_args()
    
    if args.data_path is None:
        from pyavs import get_data_path as _get_dp
        args.data_path = _get_dp()
    if args.data_path is None:
        parser.error(
            "No data path configured. Run: pyavs configure --data-path /path/to/data"
        )
    try:
        # Compute covariance matrices
        combined_cov, session_covs = compute_empty_room_covariance(
            data_path=args.data_path,
            subject_id=args.subject_id,
            sessions=args.sessions,
            overwrite=args.overwrite,
            method=args.method
        )
        
        logger.info(f"Successfully computed covariance matrices for subject {args.subject_id}")
        
        # Run diagnostics if we have session covariances
        if session_covs:
            output_dir = os.path.join(
                get_derivatives_path(args.data_path, args.subject_id), 
                'source_reconstruction', 'noise_covariance'
            )
            
            diagnostics = diagnose_covariance_generalization(
                session_covs=session_covs,
                combined_cov=combined_cov,
                subject_id=args.subject_id,
                save_report=True,
                output_dir=output_dir
            )
            
            logger.info(f"Mean relative difference across sessions: {diagnostics.get('mean_relative_diff', 'N/A'):.4f}")
            
    except Exception as e:
        logger.error(f"Error computing covariance matrices: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()