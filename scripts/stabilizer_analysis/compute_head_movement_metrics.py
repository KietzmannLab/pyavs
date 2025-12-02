#!/usr/bin/env python3
"""
Compute head movement repositioning metrics from HPI coil data.

This script computes:
1. Within-session repositioning error (between runs)
2. Between-session repositioning error (between sessions)
3. Within-run stability (SD per run)

Usage:
    python compute_head_movement_metrics.py --data-dir /path/to/stabilizer --subjects 1 2 3 4 5

Author: pyAVS development team
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_head_position_data(data_dir: Path, subject_id: int, session_num: int) -> dict:
    """
    Load head position data for a subject-session.

    Parameters
    ----------
    data_dir : Path
        Base directory containing head position NPZ files
    subject_id : int
        Subject ID
    session_num : int
        Session number

    Returns
    -------
    data : dict or None
        Dictionary with head position data, or None if file not found
    """
    npz_file = data_dir / f"sub-{subject_id:02d}" / f"sub-{subject_id:02d}_ses-{session_num:02d}_headpos.npz"

    if not npz_file.exists():
        print(f"Warning: File not found: {npz_file}")
        return None

    try:
        data = np.load(npz_file)
        return {
            'subject_id': int(data['subject_id']),
            'session_num': int(data['session_num']),
            'times': data['times'],
            'positions': data['positions'],  # [N_times, 3] in meters
            'goodness_of_fit': data['goodness_of_fit'],  # [N_times]
            'run_ids': data.get('run_ids', None),  # [N_times] run numbers
        }
    except Exception as e:
        print(f"Error loading {npz_file}: {e}")
        return None


def compute_within_session_repositioning(data: dict, n_samples: int = 100) -> dict:
    """
    Compute repositioning error between runs within a session.

    For each run, compute median XYZ of first n_samples, then compute SD across runs.

    Parameters
    ----------
    data : dict
        Session data with positions and run_ids
    n_samples : int
        Number of samples to use from start of each run (default: 100)

    Returns
    -------
    metrics : dict
        Repositioning SD per axis
    """
    if data['run_ids'] is None:
        print("Warning: run_ids not found in data, cannot compute within-session repositioning")
        return None

    positions = data['positions'] * 1000  # Convert to mm
    run_ids = data['run_ids']
    unique_runs = np.unique(run_ids)

    # Extract median starting position for each run
    run_medians = []
    for run_id in unique_runs:
        run_mask = run_ids == run_id
        run_positions = positions[run_mask]

        if len(run_positions) < n_samples:
            print(f"Warning: Run {run_id} has only {len(run_positions)} samples, using all")
            run_medians.append(np.median(run_positions, axis=0))
        else:
            run_medians.append(np.median(run_positions[:n_samples], axis=0))

    run_medians = np.array(run_medians)  # [n_runs, 3]

    # Compute SD across runs for each axis
    sd_x, sd_y, sd_z = np.std(run_medians, axis=0)

    return {
        'repositioning_sd_x': sd_x,
        'repositioning_sd_y': sd_y,
        'repositioning_sd_z': sd_z,
        'n_runs': len(unique_runs),
    }


def compute_between_session_repositioning(session_data_list: list, n_samples: int = 100) -> dict:
    """
    Compute repositioning error between sessions.

    For each session, get median XYZ of first n_samples of first run.

    Parameters
    ----------
    session_data_list : list of dict
        List of session data dictionaries
    n_samples : int
        Number of samples to use from start (default: 100)

    Returns
    -------
    metrics : dict
        Repositioning SD per axis across sessions
    """
    session_medians = []

    for sess_data in session_data_list:
        if sess_data is None or sess_data['run_ids'] is None:
            continue

        positions = sess_data['positions'] * 1000  # Convert to mm
        run_ids = sess_data['run_ids']

        # Get first run
        first_run_id = np.min(run_ids)
        first_run_mask = run_ids == first_run_id
        first_run_positions = positions[first_run_mask]

        if len(first_run_positions) < n_samples:
            session_medians.append(np.median(first_run_positions, axis=0))
        else:
            session_medians.append(np.median(first_run_positions[:n_samples], axis=0))

    if len(session_medians) < 2:
        return {
            'repositioning_sd_x': np.nan,
            'repositioning_sd_y': np.nan,
            'repositioning_sd_z': np.nan,
            'n_sessions': len(session_medians),
        }

    session_medians = np.array(session_medians)  # [n_sessions, 3]
    sd_x, sd_y, sd_z = np.std(session_medians, axis=0)

    return {
        'repositioning_sd_x': sd_x,
        'repositioning_sd_y': sd_y,
        'repositioning_sd_z': sd_z,
        'n_sessions': len(session_medians),
    }


def compute_within_run_stability(data: dict, gof_threshold: float = 0.95) -> list:
    """
    Compute SD within each run.

    Parameters
    ----------
    data : dict
        Session data
    gof_threshold : float
        Minimum goodness-of-fit threshold

    Returns
    -------
    run_metrics : list of dict
        Per-run stability metrics
    """
    if data['run_ids'] is None:
        return []

    positions = data['positions'] * 1000  # Convert to mm
    run_ids = data['run_ids']
    gof = data['goodness_of_fit']
    unique_runs = np.unique(run_ids)

    run_metrics = []

    for run_id in unique_runs:
        run_mask = run_ids == run_id
        run_positions = positions[run_mask]
        run_gof = gof[run_mask]

        # Filter by GOF
        valid_mask = run_gof >= gof_threshold
        if np.sum(valid_mask) < 10:
            print(f"Warning: Run {run_id} has only {np.sum(valid_mask)} valid samples after GOF filtering")
            continue

        valid_positions = run_positions[valid_mask]

        # Mean-correct and compute SD (Meyer method)
        mean_pos = np.mean(valid_positions, axis=0)
        positions_centered = valid_positions - mean_pos
        sd_x, sd_y, sd_z = np.std(positions_centered, axis=0)

        run_metrics.append({
            'subject_id': data['subject_id'],
            'session_num': data['session_num'],
            'run_num': int(run_id),
            'sd_x': sd_x,
            'sd_y': sd_y,
            'sd_z': sd_z,
            'n_samples': np.sum(valid_mask),
        })

    return run_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Compute head movement repositioning metrics"
    )

    # Input/output paths
    parser.add_argument('--data-dir', type=str,
                       default="/share/klab/psulewski/psulewski/pyavs/stabilizer",
                       help='Directory containing head position NPZ files')
    parser.add_argument('--output-dir', type=str,
                       default="/share/klab/psulewski/psulewski/pyavs/stabilizer/analysis",
                       help='Output directory for metrics')

    # Subject/session selection
    parser.add_argument('--subjects', type=int, nargs='+',
                       default=[1, 2, 3, 4, 5],
                       help='Subject IDs to process')
    parser.add_argument('--sessions', type=int, nargs='+',
                       default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                       help='Session numbers to process')

    # Parameters
    parser.add_argument('--n-samples-start', type=int, default=100,
                       help='Number of samples from start to use for repositioning (default: 100)')
    parser.add_argument('--gof-threshold', type=float, default=0.95,
                       help='Goodness-of-fit threshold (default: 0.95)')

    args = parser.parse_args()

    # Setup paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("Head Movement Repositioning Metrics")
    print("="*70)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Subjects: {args.subjects}")
    print(f"Sessions: {args.sessions}")
    print()

    # Storage for metrics
    repositioning_metrics = []
    within_run_metrics = []

    # Process each subject
    for subject_id in args.subjects:
        print(f"\n{'='*70}")
        print(f"Processing Subject {subject_id}")
        print(f"{'='*70}")

        # Load all sessions for this subject
        session_data_list = []
        for session_num in args.sessions:
            data = load_head_position_data(data_dir, subject_id, session_num)
            session_data_list.append(data)

            if data is not None:
                # Within-session repositioning (between runs)
                within_metrics = compute_within_session_repositioning(data, args.n_samples_start)
                if within_metrics:
                    repositioning_metrics.append({
                        'subject_id': subject_id,
                        'session_num': session_num,
                        'axis': 'X',
                        'repositioning_error_mm': within_metrics['repositioning_sd_x'],
                        'metric_type': 'within_session',
                        'n_units': within_metrics['n_runs'],
                    })
                    repositioning_metrics.append({
                        'subject_id': subject_id,
                        'session_num': session_num,
                        'axis': 'Y',
                        'repositioning_error_mm': within_metrics['repositioning_sd_y'],
                        'metric_type': 'within_session',
                        'n_units': within_metrics['n_runs'],
                    })
                    repositioning_metrics.append({
                        'subject_id': subject_id,
                        'session_num': session_num,
                        'axis': 'Z',
                        'repositioning_error_mm': within_metrics['repositioning_sd_z'],
                        'metric_type': 'within_session',
                        'n_units': within_metrics['n_runs'],
                    })

                    print(f"  Session {session_num} within-session repositioning: "
                          f"X={within_metrics['repositioning_sd_x']:.3f}, "
                          f"Y={within_metrics['repositioning_sd_y']:.3f}, "
                          f"Z={within_metrics['repositioning_sd_z']:.3f} mm")

                # Within-run stability
                run_stability = compute_within_run_stability(data, args.gof_threshold)
                within_run_metrics.extend(run_stability)

        # Between-session repositioning
        between_metrics = compute_between_session_repositioning(session_data_list, args.n_samples_start)
        repositioning_metrics.append({
            'subject_id': subject_id,
            'session_num': np.nan,  # Across all sessions
            'axis': 'X',
            'repositioning_error_mm': between_metrics['repositioning_sd_x'],
            'metric_type': 'between_session',
            'n_units': between_metrics['n_sessions'],
        })
        repositioning_metrics.append({
            'subject_id': subject_id,
            'session_num': np.nan,
            'axis': 'Y',
            'repositioning_error_mm': between_metrics['repositioning_sd_y'],
            'metric_type': 'between_session',
            'n_units': between_metrics['n_sessions'],
        })
        repositioning_metrics.append({
            'subject_id': subject_id,
            'session_num': np.nan,
            'axis': 'Z',
            'repositioning_error_mm': between_metrics['repositioning_sd_z'],
            'metric_type': 'between_session',
            'n_units': between_metrics['n_sessions'],
        })

        print(f"\n  Between-session repositioning: "
              f"X={between_metrics['repositioning_sd_x']:.3f}, "
              f"Y={between_metrics['repositioning_sd_y']:.3f}, "
              f"Z={between_metrics['repositioning_sd_z']:.3f} mm")

    # Create DataFrames
    df_repositioning = pd.DataFrame(repositioning_metrics)
    df_within_run = pd.DataFrame(within_run_metrics)

    # Save CSVs
    print(f"\n{'='*70}")
    print("Saving Results")
    print(f"{'='*70}")

    csv_repositioning = output_dir / 'repositioning_metrics.csv'
    df_repositioning.to_csv(csv_repositioning, index=False)
    print(f"Saved repositioning metrics: {csv_repositioning}")

    csv_within_run = output_dir / 'within_run_stability.csv'
    df_within_run.to_csv(csv_within_run, index=False)
    print(f"Saved within-run stability: {csv_within_run}")

    # Print summary
    print(f"\n{'='*70}")
    print("Summary Statistics")
    print(f"{'='*70}")
    print("\nRepositioning Error (mean ± sem):")
    summary = df_repositioning.groupby(['metric_type', 'axis'])['repositioning_error_mm'].agg(['mean', 'sem'])
    print(summary.to_string())

    print("\nWithin-Run Stability (mean ± sem):")
    run_summary = df_within_run[['sd_x', 'sd_y', 'sd_z']].agg(['mean', 'sem'])
    print(run_summary.to_string())

    print(f"\n{'='*70}")
    print("Metrics computation complete!")
    print(f"{'='*70}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
