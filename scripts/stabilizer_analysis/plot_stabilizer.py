#!/usr/bin/env python3
"""
Visualize head movement trajectories and displacement statistics.

This script creates publication-quality plots of head movement data computed
by compute_stabilizer.py, including displacement histograms and summary statistics.

Usage:
    python plot_stabilizer.py --subjects 1 2 3 --sessions 1 2 3

Author: pyAVS development team
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import logging

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add pyavs to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from pyavs.utils.logging import get_logger

# Initialize logger
logger = get_logger('scripts.stabilizer_analysis.plot_stabilizer')

# Set matplotlib style
sns.set_context("poster")
sns.set_style("white")

# Configuration
PLOT_CONFIG = {
    'figure_dpi': 300,
    'hist_bins': 50,
    'save_format': 'pdf',
}


def load_movement_data(subjects: List[int],
                      sessions: List[int],
                      stabilizer_dir: Path) -> Dict[Tuple[int, int], Dict]:
    """
    Load computed movement data for multiple subjects and sessions.

    Parameters
    ----------
    subjects : list of int
        Subject IDs
    sessions : list of int
        Session numbers
    stabilizer_dir : Path
        Directory containing computed results

    Returns
    -------
    data : dict
        Dictionary mapping (subject_id, session_num) -> metrics dict
    """
    data = {}

    for subject_id in subjects:
        for session_num in sessions:
            npz_file = stabilizer_dir / f"sub-{subject_id:02d}" / f"sub-{subject_id:02d}_ses-{session_num:02d}_headpos.npz"

            if not npz_file.exists():
                logger.warning(f"Missing data for subject {subject_id}, session {session_num}")
                continue

            try:
                npz_data = np.load(npz_file)
                data[(subject_id, session_num)] = {
                    'times': npz_data['times'],
                    'positions': npz_data['positions'],
                    'rotations': npz_data['rotations'],
                    'rotations_euler': npz_data['rotations_euler'],
                    'displacement': npz_data['displacement'],
                    'displacement_magnitude': npz_data['displacement_magnitude'],
                    'goodness_of_fit': npz_data['goodness_of_fit'] if 'goodness_of_fit' in npz_data else None,
                }
                logger.debug(f"Loaded: subject {subject_id}, session {session_num}")
            except Exception as e:
                logger.error(f"Failed to load {npz_file}: {e}")

    logger.info(f"Loaded data for {len(data)} subject-session pairs")
    return data


def plot_displacement_histogram(data: Dict[Tuple[int, int], Dict],
                                output_dir: Path,
                                save_fig: bool = True) -> plt.Figure:
    """
    Plot histogram of displacement magnitudes across all sessions.

    Parameters
    ----------
    data : dict
        Movement data from load_movement_data()
    output_dir : Path
        Output directory for plots
    save_fig : bool
        Whether to save the figure

    Returns
    -------
    fig : plt.Figure
        Created figure
    """
    # Collect all displacement magnitudes by session
    displacements_by_session = {}

    for (subject_id, session_num), metrics in data.items():
        if session_num not in displacements_by_session:
            displacements_by_session[session_num] = []
        displacements_by_session[session_num].extend(
            metrics['displacement_magnitude'].tolist()
        )

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Get unique sessions and sort
    sessions = sorted(displacements_by_session.keys())
    colors = sns.color_palette("husl", n_colors=len(sessions))

    # Plot histograms for each session
    for i, session_num in enumerate(sessions):
        if displacements_by_session[session_num]:
            ax.hist(
                displacements_by_session[session_num],
                bins=PLOT_CONFIG['hist_bins'],
                alpha=0.6,
                label=f'Session {session_num}',
                color=colors[i],
                density=True,
            )

    # Formatting
    ax.set_xlabel('Head displacement (mm)')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of Head Displacement Across Sessions')
    ax.legend(frameon=False)
    sns.despine(ax=ax)

    fig.tight_layout()

    if save_fig:
        filename = f"displacement_histogram.{PLOT_CONFIG['save_format']}"
        fig.savefig(output_dir / filename, dpi=PLOT_CONFIG['figure_dpi'])
        logger.info(f"Saved histogram: {output_dir / filename}")

    return fig


def plot_displacement_boxplot(data: Dict[Tuple[int, int], Dict],
                              output_dir: Path,
                              save_fig: bool = True) -> plt.Figure:
    """
    Plot boxplot of mean displacement across subjects and sessions.

    Parameters
    ----------
    data : dict
        Movement data from load_movement_data()
    output_dir : Path
        Output directory for plots
    save_fig : bool
        Whether to save the figure

    Returns
    -------
    fig : plt.Figure
        Created figure
    """
    # Compute summary statistics per subject-session
    summary = []

    for (subject_id, session_num), metrics in data.items():
        displacement = metrics['displacement_magnitude']
        summary.append({
            'subject_id': subject_id,
            'session_num': session_num,
            'mean_displacement': np.mean(displacement),
            'max_displacement': np.max(displacement),
            'median_displacement': np.median(displacement),
        })

    # Organize by session
    sessions = sorted(set(s['session_num'] for s in summary))
    mean_by_session = {session: [] for session in sessions}

    for s in summary:
        mean_by_session[s['session_num']].append(s['mean_displacement'])

    # Create box plot
    fig, ax = plt.subplots(figsize=(10, 6))

    positions = list(range(len(sessions)))
    data_to_plot = [mean_by_session[session] for session in sessions]

    bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6,
                    patch_artist=True, showmeans=True)

    # Color boxes
    colors = sns.color_palette("husl", n_colors=len(sessions))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Formatting
    ax.set_xticks(positions)
    ax.set_xticklabels([f'Session {s}' for s in sessions])
    ax.set_ylabel('Mean head displacement (mm)')
    ax.set_title('Head Movement Stability Across Sessions')
    sns.despine(ax=ax)

    fig.tight_layout()

    if save_fig:
        filename = f"displacement_boxplot.{PLOT_CONFIG['save_format']}"
        fig.savefig(output_dir / filename, dpi=PLOT_CONFIG['figure_dpi'])
        logger.info(f"Saved boxplot: {output_dir / filename}")

    return fig


def plot_xyz_displacement_histograms(data: Dict[Tuple[int, int], Dict],
                                    output_dir: Path,
                                    save_fig: bool = True) -> plt.Figure:
    """
    Plot separate histograms for X, Y, Z displacement components.

    Parameters
    ----------
    data : dict
        Movement data from load_movement_data()
    output_dir : Path
        Output directory for plots
    save_fig : bool
        Whether to save the figure

    Returns
    -------
    fig : plt.Figure
        Created figure
    """
    # Collect displacements for each axis
    displacements_x = []
    displacements_y = []
    displacements_z = []

    for (subject_id, session_num), metrics in data.items():
        displacement = metrics['displacement']  # (N_times, 3)
        displacements_x.extend(displacement[:, 0].tolist())
        displacements_y.extend(displacement[:, 1].tolist())
        displacements_z.extend(displacement[:, 2].tolist())

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Plot X displacement
    axes[0].hist(displacements_x, bins=PLOT_CONFIG['hist_bins'],
                alpha=0.7, color='#e74c3c', density=True)
    axes[0].set_xlabel('X displacement (mm)')
    axes[0].set_ylabel('Density')
    axes[0].set_title('X-axis (Left-Right)')
    axes[0].axvline(0, color='black', linestyle='--', alpha=0.3)

    # Plot Y displacement
    axes[1].hist(displacements_y, bins=PLOT_CONFIG['hist_bins'],
                alpha=0.7, color='#3498db', density=True)
    axes[1].set_xlabel('Y displacement (mm)')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Y-axis (Anterior-Posterior)')
    axes[1].axvline(0, color='black', linestyle='--', alpha=0.3)

    # Plot Z displacement
    axes[2].hist(displacements_z, bins=PLOT_CONFIG['hist_bins'],
                alpha=0.7, color='#2ecc71', density=True)
    axes[2].set_xlabel('Z displacement (mm)')
    axes[2].set_ylabel('Density')
    axes[2].set_title('Z-axis (Superior-Inferior)')
    axes[2].axvline(0, color='black', linestyle='--', alpha=0.3)

    # Despine all axes
    for ax in axes:
        sns.despine(ax=ax)

    fig.suptitle('Head Displacement by Spatial Dimension', y=1.02, fontsize=14)
    fig.tight_layout()

    if save_fig:
        filename = f"displacement_xyz_histograms.{PLOT_CONFIG['save_format']}"
        fig.savefig(output_dir / filename, dpi=PLOT_CONFIG['figure_dpi'], bbox_inches='tight')
        logger.info(f"Saved XYZ histograms: {output_dir / filename}")

    return fig


def plot_rotation_summary(data: Dict[Tuple[int, int], Dict],
                          output_dir: Path,
                          save_fig: bool = True) -> plt.Figure:
    """
    Plot summary of rotation angles (pitch, roll, yaw).

    Parameters
    ----------
    data : dict
        Movement data from load_movement_data()
    output_dir : Path
        Output directory for plots
    save_fig : bool
        Whether to save the figure

    Returns
    -------
    fig : plt.Figure
        Created figure
    """
    # Collect rotation angles
    pitch_list = []
    roll_list = []
    yaw_list = []

    for (subject_id, session_num), metrics in data.items():
        rotations_euler = metrics['rotations_euler']  # (N_times, 3)
        pitch_list.extend(rotations_euler[:, 0].tolist())
        roll_list.extend(rotations_euler[:, 1].tolist())
        yaw_list.extend(rotations_euler[:, 2].tolist())

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Plot Pitch
    axes[0].hist(pitch_list, bins=PLOT_CONFIG['hist_bins'],
                alpha=0.7, color='#9b59b6', density=True)
    axes[0].set_xlabel('Pitch (degrees)')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Pitch (Nodding)')
    axes[0].axvline(0, color='black', linestyle='--', alpha=0.3)

    # Plot Roll
    axes[1].hist(roll_list, bins=PLOT_CONFIG['hist_bins'],
                alpha=0.7, color='#f39c12', density=True)
    axes[1].set_xlabel('Roll (degrees)')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Roll (Tilting)')
    axes[1].axvline(0, color='black', linestyle='--', alpha=0.3)

    # Plot Yaw
    axes[2].hist(yaw_list, bins=PLOT_CONFIG['hist_bins'],
                alpha=0.7, color='#1abc9c', density=True)
    axes[2].set_xlabel('Yaw (degrees)')
    axes[2].set_ylabel('Density')
    axes[2].set_title('Yaw (Shaking)')
    axes[2].axvline(0, color='black', linestyle='--', alpha=0.3)

    # Despine all axes
    for ax in axes:
        sns.despine(ax=ax)

    fig.suptitle('Head Rotation Angles', y=1.02, fontsize=14)
    fig.tight_layout()

    if save_fig:
        filename = f"rotation_summary.{PLOT_CONFIG['save_format']}"
        fig.savefig(output_dir / filename, dpi=PLOT_CONFIG['figure_dpi'], bbox_inches='tight')
        logger.info(f"Saved rotation summary: {output_dir / filename}")

    return fig


def print_summary_statistics(data: Dict[Tuple[int, int], Dict]):
    """
    Print summary statistics to console.

    Parameters
    ----------
    data : dict
        Movement data from load_movement_data()
    """
    logger.info("\n" + "="*60)
    logger.info("HEAD MOVEMENT SUMMARY STATISTICS")
    logger.info("="*60)

    # Overall statistics
    all_displacements = []
    all_max_displacements = []

    for (subject_id, session_num), metrics in data.items():
        displacement = metrics['displacement_magnitude']
        all_displacements.extend(displacement.tolist())
        all_max_displacements.append(np.max(displacement))

    logger.info(f"\nOverall displacement statistics:")
    logger.info(f"  Mean: {np.mean(all_displacements):.2f} mm")
    logger.info(f"  Median: {np.median(all_displacements):.2f} mm")
    logger.info(f"  Std: {np.std(all_displacements):.2f} mm")
    logger.info(f"  Max: {np.max(all_displacements):.2f} mm")
    logger.info(f"  95th percentile: {np.percentile(all_displacements, 95):.2f} mm")

    logger.info(f"\nMaximum displacement per session:")
    logger.info(f"  Mean: {np.mean(all_max_displacements):.2f} mm")
    logger.info(f"  Median: {np.median(all_max_displacements):.2f} mm")
    logger.info(f"  Range: {np.min(all_max_displacements):.2f} - {np.max(all_max_displacements):.2f} mm")

    # Per-session statistics
    sessions = sorted(set(s[1] for s in data.keys()))
    logger.info(f"\nPer-session mean displacement:")

    for session_num in sessions:
        session_displacements = []
        for (subj, sess), metrics in data.items():
            if sess == session_num:
                session_displacements.extend(metrics['displacement_magnitude'].tolist())

        if session_displacements:
            logger.info(f"  Session {session_num}: {np.mean(session_displacements):.2f} mm "
                       f"(median: {np.median(session_displacements):.2f} mm)")

    logger.info("="*60 + "\n")


def main():
    """Main function for plotting stabilizer results."""
    parser = argparse.ArgumentParser(
        description="Visualize head movement trajectories and displacement statistics",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Input configuration
    parser.add_argument('--stabilizer-dir', type=str,
                       default="/share/klab/psulewski/psulewski/pyavs/stabilizer",
                       help='Directory containing computed stabilizer results')

    # Subject and session selection
    parser.add_argument('--subjects', type=int, nargs='+',
                       default=[1, 2, 3, 4, 5],
                       help='Subject IDs to include')
    parser.add_argument('--sessions', type=int, nargs='+',
                       default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                       help='Session numbers to include')

    # Output configuration
    parser.add_argument('--output-dir', type=str,
                       default="/share/klab/psulewski/psulewski/pyavs/stabilizer",
                       help='Output directory for plots')

    # Plot options
    parser.add_argument('--plot-types', type=str, nargs='+',
                       default=['histogram', 'boxplot', 'xyz', 'rotation'],
                       choices=['histogram', 'boxplot', 'xyz', 'rotation', 'all'],
                       help='Types of plots to generate')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Increase verbosity')

    args = parser.parse_args()

    # Set up logging
    if args.verbose:
        logging.getLogger('pyavs').setLevel(logging.DEBUG)

    # Validate stabilizer directory
    stabilizer_dir = Path(args.stabilizer_dir)
    if not stabilizer_dir.exists():
        parser.error(f"Stabilizer directory does not exist: {stabilizer_dir}")

    # Set up output directory for plots
    output_dir = Path(args.output_dir) / 'plots'
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving plots to: {output_dir}")

    # Load computed data
    logger.info("Loading movement data...")
    data = load_movement_data(
        subjects=args.subjects,
        sessions=args.sessions,
        stabilizer_dir=stabilizer_dir
    )

    if not data:
        logger.error("No data available for plotting")
        return 1

    # Print summary statistics
    print_summary_statistics(data)

    # Determine which plots to generate
    plot_types = args.plot_types
    if 'all' in plot_types:
        plot_types = ['histogram', 'boxplot', 'xyz', 'rotation']

    # Generate plots
    logger.info("Creating visualizations...")

    if 'histogram' in plot_types:
        plot_displacement_histogram(data, output_dir)

    if 'boxplot' in plot_types:
        plot_displacement_boxplot(data, output_dir)

    if 'xyz' in plot_types:
        plot_xyz_displacement_histograms(data, output_dir)

    if 'rotation' in plot_types:
        plot_rotation_summary(data, output_dir)

    logger.info("Plotting completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
