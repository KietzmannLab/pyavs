#!/usr/bin/env python3
"""
Visualize saccade main sequence temporal dynamics during scene exploration.

This script analyzes how the relationship between saccade amplitude and peak
velocity changes over time during the first 4 seconds of scene viewing. Uses
overlaid 2D KDE contours to show temporal evolution of the main sequence.

GENERATED FIGURES:
- saccade_main_sequence_temporal.png/pdf - Overlaid KDE contours for 4 temporal bins

Usage:
    python plot_saccade_main_sequence.py --verbose
    python plot_saccade_main_sequence.py --subjects 1 2 3

Author: P. Sulewski (psulewski@uos.de)
"""

import argparse
import os
import glob
import re
from typing import List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from pyavs.dataloader.eye import load_and_enrich_eye_events
from pyavs.utils.logging import get_logger

logger = get_logger('scripts.et_quality.saccade_main_sequence')


def discover_subjects(data_path: str) -> List[int]:
    """
    Scan data directory to find all available subjects.

    Parameters
    ----------
    data_path : str
        Base data directory path

    Returns
    -------
    List[int]
        Sorted list of subject IDs found in data directory
    """
    pattern = os.path.join(data_path, 'results', 'as[0-9][0-9]_*')
    dirs = glob.glob(pattern)

    subjects = set()
    for d in dirs:
        match = re.search(r'as(\d{2})_', os.path.basename(d))
        if match:
            subjects.add(int(match.group(1)))

    return sorted(list(subjects))


def load_saccade_data(subjects: List[int], sessions: List[int],
                     data_path: str, verbose: bool = False) -> pd.DataFrame:
    """
    Load and filter saccade events from scene viewing phase.

    Parameters
    ----------
    subjects : List[int]
        Subject IDs to load
    sessions : List[int]
        Session numbers to load
    data_path : str
        Base data directory path
    verbose : bool, default=False
        Enable verbose output

    Returns
    -------
    pd.DataFrame
        Saccade events with columns: amplitude, peak_velocity, time_in_trial,
        subject, session, trial, sceneID, recording
    """
    logger.info(f"Loading saccade data for {len(subjects)} subjects, {len(sessions)} sessions")

    try:
        explog, events = load_and_enrich_eye_events(
            subjects=subjects,
            sessions=sessions,
            data_path=data_path,
            preprocessed=True,
            verbose=verbose
        )

        # Filter to scene saccades only
        saccades = events[
            (events['type'] == 'saccade') &
            (events['recording'] == 'scene')
        ].copy()

        # Verify required columns exist
        required_cols = ['amplitude', 'peak_velocity', 'time_in_trial']
        missing_cols = [col for col in required_cols if col not in saccades.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # recompute amplitude in pixels from start_gx and start_gy and end_gx and end_gy
        saccades['amplitude'] = np.sqrt((saccades['end_gx'] - saccades['start_gx'])**2 + (saccades['end_gy'] - saccades['start_gy'])**2)

        logger.info(f"Total scene saccades loaded: {len(saccades)}")
        logger.info(f"  Per-subject counts: {saccades.groupby('subject').size().to_dict()}")

        return saccades

    except Exception as e:
        logger.error(f"Error loading saccade data: {e}")
        raise


def bin_saccades_by_time(saccades_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign temporal bins based on time_in_trial.

    Parameters
    ----------
    saccades_df : pd.DataFrame
        Saccade events dataframe

    Returns
    -------
    pd.DataFrame
        Saccades with added 'time_bin' column
    """
    logger.info("Binning saccades by temporal window:")

    bins = [0, 1, 2, 3, 4]
    labels = ["<1s", "1-2s", "2-3s", "3-4s"]

    saccades_df['time_bin'] = pd.cut(
        saccades_df['time_in_trial'],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=False  # [0, 1), [1, 2), etc.
    )

    # Exclude saccades >= 4.0s
    excluded = saccades_df['time_in_trial'] >= 4.0
    saccades_df.loc[excluded, 'time_bin'] = np.nan

    # Log distribution
    total = len(saccades_df)
    for label in labels:
        count = (saccades_df['time_bin'] == label).sum()
        pct = 100 * count / total
        bin_idx = labels.index(label)
        logger.info(f"  {label:10s} ({bins[bin_idx]}-{bins[bin_idx+1]}s): {count:6d} ({pct:.1f}%)")

    excluded_count = excluded.sum()
    if excluded_count > 0:
        logger.info(f"  Excluded (>= 4.0s): {excluded_count} ({100 * excluded_count / total:.1f}%)")

    return saccades_df


def preprocess_saccades(saccades_df: pd.DataFrame,
                       outlier_percentile: float = 100) -> pd.DataFrame:
    """
    Remove NaN values and clip outliers.

    Parameters
    ----------
    saccades_df : pd.DataFrame
        Saccade events dataframe
    outlier_percentile : float, default=98
        Percentile threshold for clipping outliers

    Returns
    -------
    pd.DataFrame
        Preprocessed saccades with clipped amplitude and velocity columns
    """
    logger.info("Preprocessing saccade data:")

    # Validate percentile
    if not 0 <= outlier_percentile <= 100:
        raise ValueError(f"outlier_percentile must be in [0, 100], got {outlier_percentile}")

    # Remove NaN values
    df = saccades_df.dropna(subset=['amplitude', 'peak_velocity', 'time_bin']).copy()
    nan_removed = len(saccades_df) - len(df)
    if nan_removed > 0:
        logger.info(f"  Removed {nan_removed} rows with NaN values")

    # Calculate percentile thresholds
    amp_cutoff = np.percentile(df['amplitude'], outlier_percentile)
    vel_cutoff = np.percentile(df['peak_velocity'], outlier_percentile)


    # Clip to percentile
    df['amplitude_clipped'] = df['amplitude'].clip(upper=amp_cutoff)
    df['peak_velocity_clipped'] = df['peak_velocity'].clip(upper=vel_cutoff)

    # Count clipped values
    amp_clipped = (df['amplitude'] > amp_cutoff).sum()
    vel_clipped = (df['peak_velocity'] > vel_cutoff).sum()
    logger.info(f"  Clipped values: {amp_clipped} amplitudes, {vel_clipped} velocities")

    logger.info(f"  Final dataset: {len(df)} saccades")

    return df


def plot_main_sequence_temporal(saccades_df: pd.DataFrame, output_dir: str,
                                dpi: int = 300, fmt: str = 'both'):
    """
    Create overlaid KDE contours for temporal bins.

    Parameters
    ----------
    saccades_df : pd.DataFrame
        Preprocessed saccades with time_bin and clipped metrics
    output_dir : str
        Output directory for saving figure
    dpi : int, default=300
        Resolution for raster output
    fmt : str, default='both'
        Output format ('png', 'pdf', or 'both')
    """
    logger.info("Creating figure: Saccade main sequence temporal analysis")

    # Setup styling
    sns.set_context("poster")

    
    # Color palette
    #bin_labels = ['early', 'mid-early', 'mid-late', 'late']
    colors = sns.color_palette("magma", n_colors=4)
    # convert pixels to degrees (assuming 33 px/deg as in AVS)
    pix_deg = 33
    #saccades_df['amplitude_clipped'] = saccades_df['amplitude_clipped'] / pix_deg
    saccades_df['peak_velocity_clipped'] = saccades_df['peak_velocity_clipped'] / pix_deg
    saccades_df['amplitude_clipped'] = saccades_df['amplitude_clipped'] / pix_deg
    # Plot KDE for each temporal bin
    g = sns.JointGrid(data=saccades_df, x='amplitude_clipped', y='peak_velocity_clipped')

    # Plot your binned scatter with hue on the joint axes
    for time_bin, color in zip(saccades_df['time_bin'].unique(), colors):
        subset = saccades_df[saccades_df['time_bin'] == time_bin]
        sns.regplot(
            data=subset,
            x='amplitude_clipped',
            y='peak_velocity_clipped',
            x_bins=20,
            fit_reg=True,
            lowess=True,
            scatter=False,
            #scatter_kws={'alpha': 0.3},
            ax=g.ax_joint,
            color=color,
            label=time_bin
        )
        # add legend
    g.ax_joint.legend(title='viewing time', frameon=False)

    # Marginal KDEs
    for time_bin, color in zip(saccades_df['time_bin'].unique(), colors):
        subset = saccades_df[saccades_df['time_bin'] == time_bin]
        sns.kdeplot(data=subset, x='amplitude_clipped', ax=g.ax_marg_x, color=color, fill=True, alpha=0.3)
        sns.kdeplot(data=subset, y='peak_velocity_clipped', ax=g.ax_marg_y, color=color, fill=True, alpha=0.3)
        # set figure size
    plt.gcf().set_size_inches(10, 8)
    # get the current axis
    #ax = plt.gca()
    # Labels (lowercase)
    g.ax_joint.set_xlabel('saccade amplitude [°]')
    g.ax_joint.set_ylabel('peak velocity [°/s]')

    # Legend and styling
    
    sns.despine()
    plt.tight_layout()

    # Save figure
    if fmt in ['png', 'both']:
        png_file = os.path.join(output_dir, 'saccade_main_sequence_temporal.png')
        plt.savefig(png_file, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved: {png_file}")

    if fmt in ['pdf', 'both']:
        pdf_file = os.path.join(output_dir, 'saccade_main_sequence_temporal.pdf')
        plt.savefig(pdf_file, format='pdf', bbox_inches='tight')
        logger.info(f"Saved: {pdf_file}")

    plt.close()


def main():
    """Command-line interface for saccade main sequence temporal analysis."""
    parser = argparse.ArgumentParser(
        description="Visualize saccade main sequence temporal dynamics during scene exploration"
    )

    parser.add_argument(
        '--data-path', '-d',
        type=str,
        default='/share/klab/datasets/avs/',
        help='Path to AVS data directory (default: /share/klab/datasets/avs/)'
    )

    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='/share/klab/psulewski/psulewski/pyavs/et_quality/',
        help='Output directory for figures (default: /share/klab/psulewski/psulewski/pyavs/et_quality/)'
    )

    parser.add_argument(
        '--subjects', '-s',
        nargs='+',
        type=int,
        default=[1,2,3,4,5],
        help='Subject IDs to process (default: all available subjects)'
    )

    parser.add_argument(
        '--sessions', '-sess',
        nargs='+',
        type=int,
        default=np.arange(1,11).tolist(),
        help='Sessions to include (default: 1 2 3 4)'
    )

    parser.add_argument(
        '--format', '-f',
        choices=['png', 'pdf', 'both'],
        default='both',
        help='Output format (default: both)'
    )

    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='Resolution for raster output (default: 300)'
    )

    parser.add_argument(
        '--outlier-percentile',
        type=float,
        default=99,
        help='Percentile threshold for outlier clipping (default: 99)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(levelname)s: %(message)s'
    )

    logger.info("=" * 70)
    logger.info("Saccade Main Sequence Temporal Analysis")
    logger.info("=" * 70)

    # Auto-discover subjects if not specified
    if args.subjects is None:
        logger.info("Auto-discovering subjects...")
        subjects = discover_subjects(args.data_path)
        if len(subjects) == 0:
            logger.error(f"No subjects found in {args.data_path}/results/")
            logger.error("Please check data path or specify subjects manually with --subjects")
            return
        logger.info(f"Found {len(subjects)} subjects: {subjects}")
    else:
        subjects = args.subjects
        logger.info(f"Processing specified subjects: {subjects}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")

    # Load saccade data
    saccades = load_saccade_data(
        subjects=subjects,
        sessions=args.sessions,
        data_path=args.data_path,
        verbose=args.verbose
    )

    if len(saccades) == 0:
        logger.error("No saccade data loaded. Cannot generate plot.")
        return

    # Bin saccades by time
    saccades = bin_saccades_by_time(saccades)

    # Preprocess (outlier removal, NaN handling)
    saccades = preprocess_saccades(saccades, outlier_percentile=args.outlier_percentile)

    if len(saccades) == 0:
        logger.error("No saccades remaining after preprocessing. Cannot generate plot.")
        return

    # Create plot
    plot_main_sequence_temporal(
        saccades_df=saccades,
        output_dir=args.output_dir,
        dpi=args.dpi,
        fmt=args.format
    )

    logger.info("=" * 70)
    logger.info("Saccade main sequence temporal analysis complete!")
    logger.info(f"Outputs saved to: {args.output_dir}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
