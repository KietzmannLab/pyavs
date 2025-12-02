#!/usr/bin/env python3
"""
Compute head movement trajectories from MEG data using HPI coil tracking.

This script extracts head position data from raw (un-preprocessed) MEG recordings
and computes movement metrics (XYZ displacement, rotations) across sessions.
Results are saved as NPZ files for visualization and analysis.

Usage:
    python compute_stabilizer.py --subjects 1 2 3 --sessions 1 2 3

Author: pyAVS development team
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Dict
import logging

import numpy as np
import mne
from mne.chpi import compute_chpi_amplitudes, compute_chpi_locs, compute_head_pos
from joblib import Parallel, delayed

# Add pyavs to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from pyavs.dataloader.meg import load_meg_session, concatenate_meg_runs
from pyavs.utils.logging import get_logger

# Initialize logger
logger = get_logger('scripts.stabilizer_analysis.compute_stabilizer')


def extract_head_positions(raw: mne.io.Raw) -> Optional[np.ndarray]:
    """
    Extract head position data from raw MEG using HPI coils.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw MEG data with HPI coil information

    Returns
    -------
    head_pos : np.ndarray or None
        Head position array of shape (N_times, 10) containing:
        [time, q1, q2, q3, x, y, z, gof, vel_x, vel_y]
        Returns None if HPI extraction fails
    """
    try:
        logger.info("Extracting HPI coil amplitudes...")
        chpi_amplitudes = compute_chpi_amplitudes(raw, verbose=False, t_step_min=0.05)

        logger.info("Computing HPI coil locations...")
        chpi_locs = compute_chpi_locs(raw.info, chpi_amplitudes, verbose=False)

        logger.info("Computing head positions...")
        head_pos = compute_head_pos(raw.info, chpi_locs, verbose=False)

        logger.info(f"Extracted {len(head_pos)} head position samples")
        return head_pos

    except Exception as e:
        logger.error(f"Failed to extract head positions: {e}")
        return None


def compute_movement_metrics(head_pos: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute movement metrics from head position data.

    Parameters
    ----------
    head_pos : np.ndarray
        Head position array from compute_head_pos() with shape (N_times, 10)
        Columns: [time, q1, q2, q3, x, y, z, gof, vel_x, vel_y]

    Returns
    -------
    metrics : dict
        Dictionary containing:
        - 'times': Time vector (seconds)
        - 'positions': XYZ positions in meters (N_times, 3)
        - 'rotations': Rotation quaternions (N_times, 3)
        - 'rotations_euler': Euler angles in degrees (N_times, 3) - [pitch, roll, yaw]
        - 'displacement': XYZ displacement from start in mm (N_times, 3)
        - 'displacement_magnitude': Euclidean displacement in mm (N_times,)
        - 'goodness_of_fit': HPI fit quality (N_times,)
    """
    # Extract components
    times = head_pos[:, 0]
    rotations = head_pos[:, 1:4]  # Quaternions q1, q2, q3
    positions = head_pos[:, 4:7]  # XYZ in meters
    gof = head_pos[:, 7]  # Goodness of fit

    # Compute displacement from starting position
    start_pos = positions[0]
    displacement = (positions - start_pos) * 1000  # Convert to mm

    # Compute displacement magnitude (Euclidean distance)
    displacement_magnitude = np.sqrt(np.sum(displacement**2, axis=1))

    # Convert quaternions to Euler angles (pitch, roll, yaw)
    rotations_euler = quaternions_to_euler(rotations)

    metrics = {
        'times': times,
        'positions': positions,
        'rotations': rotations,
        'rotations_euler': rotations_euler,
        'displacement': displacement,
        'displacement_magnitude': displacement_magnitude,
        'goodness_of_fit': gof,
    }

    return metrics


def quaternions_to_euler(quaternions: np.ndarray) -> np.ndarray:
    """
    Convert quaternions to Euler angles (pitch, roll, yaw) in degrees.

    Parameters
    ----------
    quaternions : np.ndarray
        Array of quaternions with shape (N, 3) or (N, 4)
        If shape is (N, 3), assumes quaternions are [q1, q2, q3] with q0 implicit

    Returns
    -------
    euler : np.ndarray
        Euler angles in degrees with shape (N, 3)
        Columns: [pitch, roll, yaw]
    """
    n_samples = quaternions.shape[0]
    euler = np.zeros((n_samples, 3))

    for i, q in enumerate(quaternions):
        # If only 3 components, compute q0
        if len(q) == 3:
            q1, q2, q3 = q
            q0_squared = 1 - (q1**2 + q2**2 + q3**2)
            q0 = np.sqrt(max(0, q0_squared))  # Ensure non-negative
        else:
            q0, q1, q2, q3 = q

        # Convert to Euler angles
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (q0 * q1 + q2 * q3)
        cosr_cosp = 1 - 2 * (q1**2 + q2**2)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (q0 * q2 - q3 * q1)
        if abs(sinp) >= 1:
            pitch = np.sign(sinp) * np.pi / 2
        else:
            pitch = np.arcsin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (q0 * q3 + q1 * q2)
        cosy_cosp = 1 - 2 * (q2**2 + q3**2)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        euler[i] = [np.degrees(pitch), np.degrees(roll), np.degrees(yaw)]

    return euler


def extract_head_positions_from_run(subject_id: int, session_num: int, run_idx: int, data_path: str) -> tuple:
    """
    Extract head positions from a single run.

    This function loads the Raw data internally to avoid pickling issues
    when used with joblib.Parallel.

    Parameters
    ----------
    subject_id : int
        Subject ID
    session_num : int
        Session number
    run_idx : int
        Run index
    data_path : str
        Base data directory path

    Returns
    -------
    run_idx : int
        Run index (for tracking)
    head_pos : np.ndarray or None
        Head position array or None if extraction failed
    """
    #from pyavs.dataloader.meg import load_meg_run

    try:
        # Load the raw data for this specific run
        raw = load_meg_session(subject_id=subject_id, session=session_num, runs=[run_idx], data_path=data_path, preprocessed=False)
        raw = raw[run_idx]
        if raw is None:
            logger.warning(f"Run {run_idx}: Could not load raw data")
            return run_idx, None

        logger.info(f"Run {run_idx}: {raw.n_times} samples, {raw.info['nchan']} channels")

        # Extract head positions
        head_pos = extract_head_positions(raw)

        if head_pos is None:
            logger.warning(f"Run {run_idx}: No head position data extracted")

        return run_idx, head_pos

    except Exception as e:
        logger.error(f"Run {run_idx}: Failed to process - {e}")
        return run_idx, None


def process_subject_session(subject_id: int,
                            session_num: int,
                            data_path: Path,
                            output_dir: Path,
                            n_jobs: int = 1) -> Optional[Path]:
    """
    Process a single subject-session: extract head positions and save metrics.

    Parameters
    ----------
    subject_id : int
        Subject ID
    session_num : int
        Session number
    data_path : Path
        Base data directory
    output_dir : Path
        Output directory for results
    n_jobs : int
        Number of parallel jobs for processing runs (default: 1)

    Returns
    -------
    output_file : Path or None
        Path to saved NPZ file, or None if processing failed
    """
    logger.info(f"Processing subject {subject_id}, session {session_num}")

    # Determine which runs to process for this session
    # Session 1 has 10 runs (1-10), sessions 2-10 have 14 runs (1-14)
    if session_num == 1:
        run_indices = list(range(1, 11))  # 1-10
    else:
        run_indices = list(range(1, 15))  # 1-14

    logger.info(f"Processing {len(run_indices)} runs for session {session_num}")

    # Extract head positions from all runs in parallel
    # Each worker will load its own Raw data to avoid pickling issues
    logger.info(f"Extracting head positions from {len(run_indices)} runs using {n_jobs} parallel jobs")

    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(extract_head_positions_from_run)(subject_id, session_num, run_idx, str(data_path))
        for run_idx in run_indices
    )

    # Collect successful extractions and build run_ids array
    head_pos_list = []
    run_ids_list = []

    for run_idx, head_pos in results:
        if head_pos is not None:
            head_pos_list.append(head_pos)
            # Create run_id array for this run (same run_idx for all samples)
            run_ids_list.append(np.full(len(head_pos), run_idx, dtype=int))

    # Concatenate all head positions and run IDs
    head_pos = np.vstack(head_pos_list) if head_pos_list else None
    run_ids = np.concatenate(run_ids_list) if run_ids_list else None

    if head_pos is None:
        logger.error("Head position extraction failed")
        return None

    # Compute movement metrics
    metrics = compute_movement_metrics(head_pos)

    # Log summary statistics
    max_displacement = np.max(metrics['displacement_magnitude'])
    mean_displacement = np.mean(metrics['displacement_magnitude'])
    median_gof = np.median(metrics['goodness_of_fit'])
    logger.info(f"Mean displacement: {mean_displacement:.2f} mm, Max: {max_displacement:.2f} mm")
    logger.info(f"Median goodness of fit: {median_gof:.4f}")

    # Save results
    output_file = output_dir / f"sub-{subject_id:02d}" / f"sub-{subject_id:02d}_ses-{session_num:02d}_headpos.npz"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_file,
        subject_id=subject_id,
        session_num=session_num,
        times=metrics['times'],
        positions=metrics['positions'],
        rotations=metrics['rotations'],
        rotations_euler=metrics['rotations_euler'],
        displacement=metrics['displacement'],
        displacement_magnitude=metrics['displacement_magnitude'],
        goodness_of_fit=metrics['goodness_of_fit'],
        run_ids=run_ids,  # NEW: Track which run each sample belongs to
    )

    logger.info(f"Saved head position data to: {output_file}")
    return output_file


def main():
    """Main function for computing head position trajectories."""
    parser = argparse.ArgumentParser(
        description="Compute head movement trajectories from MEG data using HPI coils",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Data path configuration
    parser.add_argument('--data-path', type=str,
                       default="/share/klab/datasets/avs/",
                       help='Base data directory')

    # Subject and session selection
    parser.add_argument('--subjects', type=int, nargs='+',
                       default=[1,],
                       help='Subject IDs to process')
    parser.add_argument('--sessions', type=int, nargs='+',
                       default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                       help='Session numbers to process')

    # Output configuration
    parser.add_argument('--output-dir', type=str,
                       default="/share/klab/psulewski/psulewski/pyavs/stabilizer",
                       help='Output directory for results')

    # Processing options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Increase verbosity')
    parser.add_argument('--n-jobs', type=int,
                       default=1,
                       help='Number of parallel jobs for processing runs within each session (default: 1)')

    args = parser.parse_args()

    # Set up logging
    if args.verbose:
        logging.getLogger('pyavs').setLevel(logging.DEBUG)
        mne.set_log_level('INFO')
    else:
        mne.set_log_level('WARNING')

    # Validate data path
    data_path = Path(args.data_path)
    if not data_path.exists():
        parser.error(f"Data path does not exist: {data_path}")

    # Set up output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Process each subject-session combination
    n_processed = 0
    n_failed = 0

    for subject_id in args.subjects:
        for session_num in args.sessions:
            try:
                result = process_subject_session(
                    subject_id=subject_id,
                    session_num=session_num,
                    data_path=data_path,
                    output_dir=output_dir,
                    n_jobs=args.n_jobs
                )
                if result:
                    n_processed += 1
                else:
                    n_failed += 1

            except Exception as e:
                logger.error(f"Failed to process subject {subject_id}, session {session_num}: {e}")
                n_failed += 1

    # Summary
    logger.info(f"Processing complete: {n_processed} successful, {n_failed} failed")

    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
