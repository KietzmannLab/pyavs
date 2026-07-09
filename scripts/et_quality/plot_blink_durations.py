#!/usr/bin/env python3
"""
Visualize blink duration distributions per subject.

This script analyzes blink duration distributions from eye tracking data,
generating per-subject violin plots and overall histograms.

GENERATED FIGURES:
- blink_durations_per_subject.png/pdf - Violin plot of blink durations per subject
- blink_durations_histogram.png/pdf - Overall histogram of blink durations

Usage:
    python plot_blink_durations.py --data-path /path/to/avs/
    python plot_blink_durations.py --subjects 1 2 3 --by-recording

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

logger = get_logger('scripts.et_quality.plot_blink_durations')


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


def load_blink_data(subjects: List[int], sessions: List[int],
                    data_path: str, verbose: bool = False) -> pd.DataFrame:
    """
    Load and filter blink events from eye tracking data.

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
        Blink events with columns: duration, subject, session, recording, etc.
    """
    logger.info(f"Loading blink data for {len(subjects)} subjects, {len(sessions)} sessions")

    try:
        explog, events = load_and_enrich_eye_events(
            subjects=subjects,
            sessions=sessions,
            data_path=data_path,
            preprocessed=True,
            verbose=verbose
        )

        # Filter to blinks only
        blinks = events[events['type'] == 'blink'].copy()

        # Filter to scene and caption recordings
        blinks = blinks[blinks['recording'].isin(['scene'])].copy()
        
        # exlude the most extreme 2 percent of blink durations
       
        upper_bound = blinks['end_time'].sub(blinks['start_time']).quantile(0.98)
        blinks = blinks[blinks['end_time'].sub(blinks['start_time']) <= upper_bound]

        # Compute duration in milliseconds if not already present
        if 'duration' not in blinks.columns:
            blinks['duration'] = (blinks['end_time'] - blinks['start_time']) * 1000
        else:
            # Convert to milliseconds if in seconds
            if blinks['duration'].median() < 1:
                blinks['duration'] = blinks['duration'] * 1000

        logger.info(f"Total blinks loaded: {len(blinks)}")
        logger.info(f"  Scene: {len(blinks[blinks['recording'] == 'scene'])}")
       

        return blinks

    except Exception as e:
        logger.error(f"Error loading blink data: {e}")
        raise


def compute_blink_statistics(blinks_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute summary statistics of blink durations per subject.

    Parameters
    ----------
    blinks_df : pd.DataFrame
        Blink events dataframe

    Returns
    -------
    pd.DataFrame
        Summary statistics with columns: subject, mean, median, std, count
    """
    stats = blinks_df.groupby('subject')['duration'].agg(
        ['mean', 'median', 'std', 'count']
    ).reset_index()

    logger.info("Per-subject blink duration statistics (ms):")
    for _, row in stats.iterrows():
        logger.info(f"  Subject {int(row['subject']):02d}: "
                   f"mean={row['mean']:.1f}, median={row['median']:.1f}, "
                   f"std={row['std']:.1f}, n={int(row['count'])}")

    return stats


def plot_blink_durations_per_subject(blinks_df: pd.DataFrame, output_dir: str,
                                     by_recording: bool = False,
                                     dpi: int = 300, fmt: str = 'both'):
    """
    Create violin plot of blink durations per subject.

    Parameters
    ----------
    blinks_df : pd.DataFrame
        Blink events dataframe
    output_dir : str
        Output directory for saving figure
    by_recording : bool, default=False
        Split violins by recording type (scene vs caption)
    dpi : int, default=300
        Resolution for raster output
    fmt : str, default='both'
        Output format ('png', 'pdf', or 'both')
    """
    logger.info("Creating figure: Blink durations per subject")

    sns.set_context("poster")

    plt.figure(figsize=(10, 6))

    if by_recording:
        sns.violinplot(
            data=blinks_df,
            x='subject',
            y='duration',
            hue='recording',
            split=True,
            palette={'scene': 'cornflowerblue', 'caption': 'salmon'},
            inner='quartile'
        )
        plt.legend(frameon=False)
    else:
        sns.violinplot(
            data=blinks_df,
            x='subject',
            y='duration',
            palette='colorblind',
            inner='quartile'
        )

    plt.xlabel('subject')
    plt.ylabel('blink duration [ms]')
    sns.despine()
    plt.tight_layout()

    # Save figure
    suffix = '_by_recording' if by_recording else ''
    if fmt in ['png', 'both']:
        png_file = os.path.join(output_dir, f'blink_durations_per_subject{suffix}.png')
        plt.savefig(png_file, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
        logger.info(f"Saved: {png_file}")

    if fmt in ['pdf', 'both']:
        pdf_file = os.path.join(output_dir, f'blink_durations_per_subject{suffix}.pdf')
        plt.savefig(pdf_file, format='pdf', bbox_inches='tight', facecolor='white', edgecolor='none')
        logger.info(f"Saved: {pdf_file}")

    plt.close()


def plot_blink_durations_histogram(blinks_df: pd.DataFrame, output_dir: str,
                                   by_recording: bool = False,
                                   dpi: int = 300, fmt: str = 'both'):
    """
    Create histogram of pooled blink durations.

    Parameters
    ----------
    blinks_df : pd.DataFrame
        Blink events dataframe
    output_dir : str
        Output directory for saving figure
    by_recording : bool, default=False
        Separate histograms by recording type
    dpi : int, default=300
        Resolution for raster output
    fmt : str, default='both'
        Output format ('png', 'pdf', or 'both')
    """
    logger.info("Creating figure: Blink durations histogram")

    sns.set_context("poster")

    plt.figure(figsize=(8, 6))

    if by_recording:
        sns.histplot(
            data=blinks_df,
            x='duration',
            hue='recording',
            palette={'scene': 'cornflowerblue', 'caption': 'salmon'},
            bins=50,
            alpha=0.7,
            edgecolor='none'
        )
        plt.legend(frameon=False)
    else:
        sns.histplot(
            data=blinks_df,
            x='duration',
            color='cornflowerblue',
            bins=50,
            edgecolor='none'
        )

    plt.xlabel('blink duration [ms]')
    plt.ylabel('count')
    sns.despine()
    plt.tight_layout()

    # Save figure
    suffix = '_by_recording' if by_recording else ''
    if fmt in ['png', 'both']:
        png_file = os.path.join(output_dir, f'blink_durations_histogram{suffix}.png')
        plt.savefig(png_file, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
        logger.info(f"Saved: {png_file}")

    if fmt in ['pdf', 'both']:
        pdf_file = os.path.join(output_dir, f'blink_durations_histogram{suffix}.pdf')
        plt.savefig(pdf_file, format='pdf', bbox_inches='tight', facecolor='white', edgecolor='none')
        logger.info(f"Saved: {pdf_file}")

    plt.close()


def generate_all_figures(subjects: List[int], sessions: List[int],
                         data_path: str, output_dir: str,
                         by_recording: bool = False,
                         dpi: int = 300, fmt: str = 'both',
                         verbose: bool = False):
    """
    Main orchestration function to generate all blink duration figures.

    Parameters
    ----------
    subjects : List[int]
        Subject IDs to process
    sessions : List[int]
        Session numbers to process
    data_path : str
        Base data directory path
    output_dir : str
        Output directory for figures
    by_recording : bool, default=False
        Split analysis by recording type
    dpi : int, default=300
        Resolution for raster output
    fmt : str, default='both'
        Output format ('png', 'pdf', or 'both')
    verbose : bool, default=False
        Enable verbose output
    """
    logger.info("=" * 70)
    logger.info("Generating Blink Duration Figures")
    logger.info("=" * 70)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Load blink data
    blinks = load_blink_data(subjects, sessions, data_path, verbose=verbose)

    if len(blinks) == 0:
        logger.error("No blink data loaded. Cannot generate figures.")
        return

    # Compute and log statistics
    compute_blink_statistics(blinks)

    # Generate plots
    plot_blink_durations_per_subject(blinks, output_dir, by_recording=by_recording,
                                     dpi=dpi, fmt=fmt)
    plot_blink_durations_histogram(blinks, output_dir, by_recording=by_recording,
                                   dpi=dpi, fmt=fmt)

    logger.info("=" * 70)
    logger.info("Blink duration figure generation complete!")
    logger.info(f"Figures saved to: {output_dir}")
    logger.info("=" * 70)


def main():
    """Command-line interface for blink duration visualization."""
    parser = argparse.ArgumentParser(
        description="Visualize blink duration distributions per subject"
    )

    parser.add_argument(
        '--data-path', '-d',
        type=str,
        default=None,
        help='Path to AVS data directory (default: /share/klab/datasets/avs/)'
    )

    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=None,
        help='Output directory for figures (default: /share/klab/psulewski/psulewski/pyavs/et_quality/figures)'
    )

    parser.add_argument(
        '--subjects', '-s',
        nargs='+',
        type=int,
        default=[1, 2, 3, 4, 5],
        help='Subject IDs to process (default: 1 2 3 4 5)'
    )

    parser.add_argument(
        '--sessions', '-sess',
        nargs='+',
        type=int,
        default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        help='Sessions to include (default: 1-10)'
    )

    parser.add_argument(
        '--by-recording',
        action='store_true',
        help='Split analysis by recording type (scene vs caption)'
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
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    if args.data_path is None:
        from pyavs import get_data_path as _get_dp
        args.data_path = _get_dp()
    if args.data_path is None:
        parser.error(
            "No data path configured. Run: pyavs configure --data-path /path/to/data"
        )
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(levelname)s: %(message)s'
    )

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

    # Generate figures
    generate_all_figures(
        subjects=subjects,
        sessions=args.sessions,
        data_path=args.data_path,
        output_dir=args.output_dir,
        by_recording=args.by_recording,
        dpi=args.dpi,
        fmt=args.format,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
