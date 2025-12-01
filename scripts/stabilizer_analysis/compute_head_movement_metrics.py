#!/usr/bin/env python3
"""
Compute Meyer-compatible head movement metrics from HPI coil data.

This script loads head position data and computes metrics following
Meyer et al. (2017) methodology for assessing head stabilizer efficacy.

Reference:
Meyer, S. S., et al. (2017). Flexible head-casts for high spatial precision MEG.
Journal of Neuroscience Methods, 276, 38-45.

Usage:
    python compute_head_movement_metrics.py --data-dir /path/to/stabilizer --subjects 1 2 3 4 5

Author: pyAVS development team
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Meyer et al. (2017) benchmarks
MEYER_BENCHMARKS = {
    'within_session_sd_threshold': 0.22,  # mm per axis
    'max_deviation_threshold': 0.75,  # mm
    'between_session_repositioning': 1.0,  # mm typical value
}


def load_head_position_data(data_dir: Path, subject_id: int, session_num: int) -> Dict:
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
            'displacement': data['displacement'],  # [N_times, 3] in mm
            'displacement_magnitude': data['displacement_magnitude'],  # [N_times] in mm
            'goodness_of_fit': data['goodness_of_fit'],  # [N_times]
        }
    except Exception as e:
        print(f"Error loading {npz_file}: {e}")
        return None


def compute_within_session_metrics(positions: np.ndarray, gof: np.ndarray = None, gof_threshold: float = 0.95) -> Dict:
    """
    Compute within-session stability metrics following Meyer et al. (2017).

    Meyer method: Mean-correct positions first, then compute SD.
    This measures variability around the mean position, not drift from start.

    Parameters
    ----------
    positions : np.ndarray
        Position array [N_times, 3] in meters
    gof : np.ndarray, optional
        Goodness of fit array [N_times]
    gof_threshold : float
        Minimum GOF to include samples (default: 0.95)

    Returns
    -------
    metrics : dict
        Dictionary with:
        - sd_x, sd_y, sd_z: Standard deviations in mm
        - sd_total: Total SD (Euclidean)
        - max_deviation: Maximum distance from mean position in mm
        - mean_position: Mean position [3] in meters
        - n_samples: Number of samples used
        - n_samples_excluded: Number of samples excluded by GOF
    """
    # Filter by goodness of fit if provided
    if gof is not None:
        valid_mask = gof >= gof_threshold
        positions_filtered = positions[valid_mask]
        n_excluded = np.sum(~valid_mask)
    else:
        positions_filtered = positions
        n_excluded = 0

    # Convert to mm
    positions_mm = positions_filtered * 1000

    # Mean-correct (Meyer method)
    mean_pos = np.mean(positions_mm, axis=0)
    positions_centered = positions_mm - mean_pos

    # Compute SD per axis
    sd_x, sd_y, sd_z = np.std(positions_centered, axis=0)

    # Compute total SD (Euclidean)
    sd_total = np.sqrt(sd_x**2 + sd_y**2 + sd_z**2)

    # Compute max deviation from mean
    deviations = np.sqrt(np.sum(positions_centered**2, axis=1))
    max_deviation = np.max(deviations)

    return {
        'sd_x': sd_x,
        'sd_y': sd_y,
        'sd_z': sd_z,
        'sd_total': sd_total,
        'max_deviation': max_deviation,
        'mean_position': mean_pos,
        'n_samples': len(positions_filtered),
        'n_samples_excluded': n_excluded,
    }


def compute_between_session_metrics(session_data_list: List[Dict]) -> Dict:
    """
    Compute between-session repositioning metrics.

    This measures how consistently the head is repositioned across sessions.

    Parameters
    ----------
    session_data_list : list of dict
        List of session data dictionaries

    Returns
    -------
    metrics : dict
        Dictionary with:
        - repositioning_sd_x, _y, _z: SD of starting positions in mm
        - repositioning_sd_total: Total repositioning SD in mm
        - n_sessions: Number of sessions
    """
    # Extract starting positions from each session
    start_positions = []
    for sess_data in session_data_list:
        if sess_data is not None and 'positions' in sess_data:
            start_pos = sess_data['positions'][0]  # First timepoint
            start_positions.append(start_pos)

    if len(start_positions) < 2:
        return {
            'repositioning_sd_x': np.nan,
            'repositioning_sd_y': np.nan,
            'repositioning_sd_z': np.nan,
            'repositioning_sd_total': np.nan,
            'n_sessions': len(start_positions),
        }

    # Convert to array [n_sessions, 3]
    start_positions = np.array(start_positions) * 1000  # Convert to mm

    # Compute SD across sessions per axis
    sd_x, sd_y, sd_z = np.std(start_positions, axis=0)

    # Compute total SD
    sd_total = np.sqrt(sd_x**2 + sd_y**2 + sd_z**2)

    return {
        'repositioning_sd_x': sd_x,
        'repositioning_sd_y': sd_y,
        'repositioning_sd_z': sd_z,
        'repositioning_sd_total': sd_total,
        'n_sessions': len(start_positions),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute Meyer-compatible head movement metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Input/output paths
    parser.add_argument('--data-dir', type=str,
                       default="/share/klab/psulewski/psulewski/pyavs/stabilizer",
                       help='Directory containing head position NPZ files')
    parser.add_argument('--output-dir', type=str,
                       default="/share/klab/psulewski/psulewski/pyavs/stabilizer/analysis",
                       help='Output directory for metrics and tables')

    # Subject/session selection
    parser.add_argument('--subjects', type=int, nargs='+',
                       default=[1, 2, 3, 4, 5],
                       help='Subject IDs to process')
    parser.add_argument('--sessions', type=int, nargs='+',
                       default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                       help='Session numbers to process')

    # Quality control
    parser.add_argument('--gof-threshold', type=float, default=0.95,
                       help='Minimum goodness-of-fit to include samples (default: 0.95)')

    args = parser.parse_args()

    # Setup paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("Head Movement Metrics Computation (Meyer et al. 2017 method)")
    print("="*70)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Subjects: {args.subjects}")
    print(f"Sessions: {args.sessions}")
    print(f"GOF threshold: {args.gof_threshold}")
    print()

    # Store all metrics
    within_session_metrics = []
    between_subject_metrics = []

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
                # Compute within-session metrics
                metrics = compute_within_session_metrics(
                    data['positions'],
                    data['goodness_of_fit'],
                    args.gof_threshold
                )

                # Store with metadata
                metrics_record = {
                    'subject_id': subject_id,
                    'session_num': session_num,
                    **metrics
                }
                within_session_metrics.append(metrics_record)

                # Print summary
                print(f"  Session {session_num}: "
                      f"SD = ({metrics['sd_x']:.3f}, {metrics['sd_y']:.3f}, {metrics['sd_z']:.3f}) mm, "
                      f"Max dev = {metrics['max_deviation']:.3f} mm "
                      f"({metrics['n_samples']} samples, {metrics['n_samples_excluded']} excluded)")

        # Compute between-session metrics for this subject
        between_metrics = compute_between_session_metrics(session_data_list)
        between_metrics['subject_id'] = subject_id
        between_subject_metrics.append(between_metrics)

        print(f"\n  Between-session repositioning: "
              f"SD = ({between_metrics['repositioning_sd_x']:.3f}, "
              f"{between_metrics['repositioning_sd_y']:.3f}, "
              f"{between_metrics['repositioning_sd_z']:.3f}) mm, "
              f"Total = {between_metrics['repositioning_sd_total']:.3f} mm")

    # Convert to DataFrames
    df_within = pd.DataFrame(within_session_metrics)
    df_between = pd.DataFrame(between_subject_metrics)

    # Compute summary statistics
    print(f"\n{'='*70}")
    print("Summary Statistics (AVS Dataset)")
    print(f"{'='*70}")

    summary_stats = {
        'within_session_sd_x': (df_within['sd_x'].mean(), df_within['sd_x'].std()),
        'within_session_sd_y': (df_within['sd_y'].mean(), df_within['sd_y'].std()),
        'within_session_sd_z': (df_within['sd_z'].mean(), df_within['sd_z'].std()),
        'max_deviation': (df_within['max_deviation'].mean(), df_within['max_deviation'].std()),
        'between_session_repositioning': (df_between['repositioning_sd_total'].mean(),
                                          df_between['repositioning_sd_total'].std()),
    }

    # Create comparison table
    comparison_data = []
    for metric, (mean_val, std_val) in summary_stats.items():
        meyer_value = MEYER_BENCHMARKS.get(metric, MEYER_BENCHMARKS.get(metric.replace('_', ' '), np.nan))

        comparison_data.append({
            'Metric': metric.replace('_', ' ').title(),
            'AVS Dataset': f"{mean_val:.3f} ± {std_val:.3f} mm",
            'Meyer et al. (2017)': f"< {meyer_value:.2f} mm" if not np.isnan(meyer_value) else "N/A"
        })

    df_comparison = pd.DataFrame(comparison_data)

    print("\n" + df_comparison.to_string(index=False))

    # Save results
    print(f"\n{'='*70}")
    print("Saving Results")
    print(f"{'='*70}")

    # Save metrics as NPZ
    metrics_file = output_dir / 'metrics_summary.npz'
    np.savez_compressed(
        metrics_file,
        within_session_metrics=df_within.to_dict('records'),
        between_subject_metrics=df_between.to_dict('records'),
        summary_stats=summary_stats,
        meyer_benchmarks=MEYER_BENCHMARKS,
    )
    print(f"Saved metrics: {metrics_file}")

    # Save within-session metrics CSV
    csv_within = output_dir / 'within_session_metrics.csv'
    df_within.to_csv(csv_within, index=False)
    print(f"Saved within-session CSV: {csv_within}")

    # Save between-session metrics CSV
    csv_between = output_dir / 'between_session_metrics.csv'
    df_between.to_csv(csv_between, index=False)
    print(f"Saved between-session CSV: {csv_between}")

    # Save comparison table
    csv_comparison = output_dir / 'meyer_comparison.csv'
    df_comparison.to_csv(csv_comparison, index=False)
    print(f"Saved comparison table: {csv_comparison}")

    print(f"\n{'='*70}")
    print("Metrics computation complete!")
    print(f"{'='*70}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
