#!/usr/bin/env python3
"""
Visualize eye tracking event counts by type and recording phase.

This script counts and plots fixations, saccades, and blinks separately
for scene viewing and caption recording tasks across the AVS dataset.

GENERATED FIGURES:
- event_counts_dataset.png/pdf - Stacked bar plot of total counts across dataset
- event_counts_per_subject.png/pdf - Stacked bar plot of per-subject averages with error bars

Usage:
    python plot_event_counts.py --data-path /path/to/avs/

Author: P. Sulewski (psulewski@uos.de)
"""

import argparse
import os
import glob
import re
from typing import List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from pyavs.dataloader.eye import load_and_enrich_eye_events
from pyavs.utils.logging import get_logger

logger = get_logger('scripts.et_quality.event_counts')


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
    pattern = os.path.join(data_path, 'results', 'as[0-5][0-10]_*')
    dirs = glob.glob(pattern)

    subjects = set()
    for d in dirs:
        match = re.search(r'as(\d{2})_', os.path.basename(d))
        if match:
            subjects.add(int(match.group(1)))

    return sorted(list(subjects))


def load_all_events(subjects: List[int], sessions: List[int],
                   data_path: str, verbose: bool = False) -> pd.DataFrame:
    """
    Load and enrich eye tracking events for all subjects/sessions.

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
        Events dataframe with columns: type, recording, subject, session, etc.
        Filtered to only scene and caption recordings.
    """
    logger.info(f"Loading events for {len(subjects)} subjects, {len(sessions)} sessions")

    try:
        explog, events = load_and_enrich_eye_events(
            subjects=subjects,
            sessions=sessions,
            data_path=data_path,
            preprocessed=True,
            verbose=verbose
        )

        # Filter to only scene and caption recordings
        events = events[events['recording'].isin(['scene', 'caption'])].copy()

        logger.info(f"Total events loaded: {len(events)}")
        logger.info(f"  Scene: {len(events[events['recording'] == 'scene'])}")
        logger.info(f"  Caption: {len(events[events['recording'] == 'caption'])}")

        return events

    except Exception as e:
        logger.error(f"Error loading events: {e}")
        raise


def compute_event_counts_dataset(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute dataset-wide event counts.

    Parameters
    ----------
    events_df : pd.DataFrame
        Events dataframe

    Returns
    -------
    pd.DataFrame
        Counts dataframe with columns: event_type, recording, count
    """
    event_types = ['fixation', 'saccade', 'blink']
    recordings = ['scene', 'caption']

    counts = []
    for evt in event_types:
        for rec in recordings:
            mask = (events_df['type'] == evt) & (events_df['recording'] == rec)
            count = len(events_df[mask])
            counts.append({'event_type': evt, 'recording': rec, 'count': count})

    counts_df = pd.DataFrame(counts)

    logger.info("Dataset-wide counts:")
    for _, row in counts_df.iterrows():
        logger.info(f"  {row['event_type']:10s} {row['recording']:8s}: {row['count']:8d}")

    return counts_df


def compute_event_counts_per_subject(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-subject event counts with statistics.

    Parameters
    ----------
    events_df : pd.DataFrame
        Events dataframe

    Returns
    -------
    pd.DataFrame
        Summary dataframe with columns: event_type, recording, mean, std, sem, min, max
    """
    # Group by subject, event type, and recording
    grouped = events_df.groupby(['subject', 'type', 'recording']).size().reset_index(name='count')

    # Compute statistics across subjects
    summary = grouped.groupby(['type', 'recording'])['count'].agg(['mean', 'std', 'sem', 'min', 'max'])
    summary = summary.reset_index()
    summary.columns = ['event_type', 'recording', 'mean', 'std', 'sem', 'min', 'max']

    logger.info("Per-subject statistics:")
    for _, row in summary.iterrows():
        logger.info(f"  {row['event_type']:10s} {row['recording']:8s}: "
                   f"{row['mean']:.1f} ± {row['sem']:.1f} (range: {row['min']:.0f}-{row['max']:.0f})")

    return summary


def report_statistics(events_df: pd.DataFrame) -> None:
    """
    Print total counts and bootstrapped 95% CI per subject, by event type and task.

    Parameters
    ----------
    events_df : pd.DataFrame
        Events dataframe with columns: type, recording, subject
    """
    event_types = ['fixation', 'saccade', 'blink']
    recordings = ['scene', 'caption']

    rng = np.random.default_rng(0)
    n_boot = 1000

    print("\n" + "=" * 70)
    print("EVENT COUNT STATISTICS")
    print("=" * 70)

    for rec in recordings:
        rec_df = events_df[events_df['recording'] == rec]
        print(f"\nTask: {rec}")
        print(f"  {'event type':<12} {'total':>10}  {'mean [events/subject]':>22}  {'95% CI':>20}")
        print(f"  {'-'*12}  {'-'*10}  {'-'*22}  {'-'*20}")

        for evt in event_types:
            mask = rec_df['type'] == evt
            total = mask.sum()

            # Per-subject counts
            per_subj = rec_df[mask].groupby('subject').size().values.astype(float)

            if len(per_subj) == 0:
                print(f"  {evt:<12} {total:>10}  {'N/A':>22}  {'N/A':>20}")
                continue

            mean_val = per_subj.mean()

            # Bootstrapped 95% CI
            boot_means = np.array([
                rng.choice(per_subj, size=len(per_subj), replace=True).mean()
                for _ in range(n_boot)
            ])
            ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])

            print(f"  {evt:<12} {total:>10}  {mean_val:>22.1f}  [{ci_lo:>8.1f}, {ci_hi:>8.1f}]")

    print("\n" + "=" * 70 + "\n")


def plot_dataset_counts(counts_df: pd.DataFrame, output_dir: str,
                       dpi: int = 300, fmt: str = 'both'):
    """
    Create stacked bar plot of dataset-wide event counts.

    Parameters
    ----------
    counts_df : pd.DataFrame
        Counts dataframe from compute_event_counts_dataset()
    output_dir : str
        Output directory for saving figures
    dpi : int, default=300
        Resolution for raster output
    fmt : str, default='both'
        Output format ('png', 'pdf', or 'both')
    """
    logger.info("Creating figure: Dataset-wide event counts")

    # Setup styling
    sns.set_context("poster")
    # 

    # Prepare data for stacking
    event_types = ['fixation', 'saccade', 'blink']
    scene_counts = counts_df[counts_df['recording'] == 'scene'].set_index('event_type')['count'].reindex(event_types, fill_value=0)
    caption_counts = counts_df[counts_df['recording'] == 'caption'].set_index('event_type')['count'].reindex(event_types, fill_value=0)

    # Create plot
    plt.figure(figsize=(7, 7))
    x = np.arange(len(event_types))
    width = 0.6

    plt.bar(x, scene_counts, width, label='scene viewing', color='cornflowerblue')
    plt.bar(x, caption_counts, width, bottom=scene_counts, label='caption task', color='salmon')

    plt.xlabel('eye tracking event type')
    plt.ylabel('total count [events]')
    plt.xticks(x, event_types)
    plt.legend(frameon=False)
    #plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    # despine
    sns.despine()
    # Save figure
    if fmt in ['png', 'both']:
        png_file = os.path.join(output_dir, 'event_counts_dataset.png')
        plt.savefig(png_file, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
        logger.info(f"Saved: {png_file}")

    if fmt in ['pdf', 'both']:
        pdf_file = os.path.join(output_dir, 'event_counts_dataset.pdf')
        plt.savefig(pdf_file, format='pdf', bbox_inches='tight', facecolor='white', edgecolor='none')
        logger.info(f"Saved: {pdf_file}")

    plt.close()


def plot_per_subject_counts(summary_df: pd.DataFrame, output_dir: str,
                           dpi: int = 300, fmt: str = 'both'):
    """
    Create stacked bar plot of per-subject average counts with error bars.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Summary dataframe from compute_event_counts_per_subject()
    output_dir : str
        Output directory for saving figures
    dpi : int, default=300
        Resolution for raster output
    fmt : str, default='both'
        Output format ('png', 'pdf', or 'both')
    """
    logger.info("Creating figure: Per-subject average counts")

    # Setup styling
    sns.set_context("poster")
     

    # Prepare data for stacking
    event_types = ['fixation', 'saccade', 'blink']
    scene_means = summary_df[summary_df['recording'] == 'scene'].set_index('event_type')['mean'].reindex(event_types, fill_value=0)
    caption_means = summary_df[summary_df['recording'] == 'caption'].set_index('event_type')['mean'].reindex(event_types, fill_value=0)
    caption_sems = summary_df[summary_df['recording'] == 'caption'].set_index('event_type')['sem'].reindex(event_types, fill_value=0)

    # Create plot
    plt.figure(figsize=(8, 6))
    x = np.arange(len(event_types))
    width = 0.6

    plt.bar(x, scene_means, width, label='Scene', color='steelblue')
    plt.bar(x, caption_means, width, bottom=scene_means, label='Caption',
           color='coral', yerr=caption_sems, capsize=5)

    plt.xlabel('Event Type')
    plt.ylabel('Mean Count per Subject [events]')
    plt.xticks(x, event_types)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    # Save figure
    if fmt in ['png', 'both']:
        png_file = os.path.join(output_dir, 'event_counts_per_subject.png')
        plt.savefig(png_file, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
        logger.info(f"Saved: {png_file}")

    if fmt in ['pdf', 'both']:
        pdf_file = os.path.join(output_dir, 'event_counts_per_subject.pdf')
        plt.savefig(pdf_file, format='pdf', bbox_inches='tight', facecolor='white', edgecolor='none')
        logger.info(f"Saved: {pdf_file}")

    plt.close()


def generate_all_figures(subjects: List[int], sessions: List[int],
                        data_path: str, output_dir: str,
                        dpi: int = 300, fmt: str = 'both',
                        verbose: bool = False):
    """
    Main orchestration function to generate all event count figures.

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
    dpi : int, default=300
        Resolution for raster output
    fmt : str, default='both'
        Output format ('png', 'pdf', or 'both')
    verbose : bool, default=False
        Enable verbose output
    """
    logger.info("=" * 70)
    logger.info("Generating Event Count Figures")
    logger.info("=" * 70)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Load all events
    events = load_all_events(subjects, sessions, data_path, verbose=verbose)

    if len(events) == 0:
        logger.error("No events loaded. Cannot generate figures.")
        return

    # Compute dataset-wide counts
    counts_df = compute_event_counts_dataset(events)

    # Compute per-subject statistics
    summary_df = compute_event_counts_per_subject(events)

    # Generate plots
    plot_dataset_counts(counts_df, output_dir, dpi=dpi, fmt=fmt)
    plot_per_subject_counts(summary_df, output_dir, dpi=dpi, fmt=fmt)

    logger.info("=" * 70)
    logger.info("Figure generation complete!")
    logger.info(f"Figures saved to: {output_dir}")
    logger.info("=" * 70)


def main():
    """Command-line interface for event counts visualization."""
    parser = argparse.ArgumentParser(
        description="Visualize eye tracking event counts by type and recording phase"
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
        default='/share/klab/psulewski/psulewski/pyavs/et_quality/figures',
        help='Output directory for figures (default: /share/klab/psulewski/psulewski/pyavs/et_quality/figures)'
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
        default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
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
        dpi=args.dpi,
        fmt=args.format,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
