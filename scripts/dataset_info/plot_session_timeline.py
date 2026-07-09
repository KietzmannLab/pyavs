#!/usr/bin/env python3
"""
Extract MEG session timestamps and visualize temporal spacing between sessions.

This script reads MEG raw data files to extract recording timestamps and creates
a visualization showing the temporal spacing of sessions across participants.

GENERATED FIGURES:
- session_timeline.png/pdf - Scatter plot showing days between sessions
- meg_session_timestamps.csv - Cached timestamp data

Usage:
    python plot_session_timeline.py --verbose
    python plot_session_timeline.py --recompute

Author: P. Sulewski (psulewski@uos.de)
"""

import argparse
import os
from pathlib import Path
from typing import List, Optional
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mne
import logging

# Set plotting style
sns.set_context("poster")

# Import pyavs utilities
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from pyavs.utils.logging import get_logger

logger = get_logger('scripts.dataset_info.session_timeline')


def convert_session_to_letter(session: int) -> str:
    """
    Convert session number to letter (1→a, 2→b, etc.).

    Parameters
    ----------
    session : int
        Session number (1-10)

    Returns
    -------
    str
        Session letter (a-j)
    """
    return chr(ord('a') + session - 1)


def get_subject_session_id(subject: int, session: int) -> str:
    """
    Create subject-session ID string.

    Parameters
    ----------
    subject : int
        Subject ID (1-5)
    session : int
        Session number (1-10)

    Returns
    -------
    str
        Subject-session ID (e.g., 'as01a')
    """
    session_letter = convert_session_to_letter(session)
    return f"as{subject:02d}{session_letter}"



def build_meg_file_path(data_path: str, subject: int, session: int, block: int) -> str:
    """
    Construct path to MEG raw file.

    Parameters
    ----------
    data_path : str
        Base data directory path
    subject : int
        Subject ID (1-5)
    session : int
        Session number (1-10)
    block : int
        Block number

    Returns
    -------
    str
        Full path to MEG FIF file

    Examples
    --------
    >>> build_meg_file_path("/share/klab/datasets/avs/", 1, 1, 1)
    '/share/klab/datasets/avs/rawdir/as01a/as01a01.fif'
    """
    sub_sess_id = get_subject_session_id(subject, session)
    session_dir = os.path.join(data_path, "rawdir", sub_sess_id)

    meg_file = os.path.join(session_dir, f"{sub_sess_id}{block:02d}.fif")
    print(meg_file)
    return meg_file


def extract_session_timestamps(data_path: str, subjects: List[int],
                               sessions: List[int], verbose: bool = False) -> pd.DataFrame:
    """
    Extract recording timestamps from MEG raw files.

    Reads the first block of each session and extracts the measurement date
    from the MEG file header.

    Parameters
    ----------
    data_path : str
        Base data directory path
    subjects : List[int]
        Subject IDs to process (1-5)
    sessions : List[int]
        Session numbers to process (1-10)
    verbose : bool, default=False
        Enable verbose logging

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: subject, session, meas_date
    """
    logger.info(f"Extracting timestamps for {len(subjects)} subjects, {len(sessions)} sessions")

    timestamps = []
    success_count = 0
    total_count = len(subjects) * len(sessions)

    for subject in subjects:
        for session in sessions:
            # Get first block for this session
            block = 1
            meg_file = build_meg_file_path(data_path, subject, session, block)

            if verbose:
                logger.debug(f"Reading: {meg_file}")

            try:
                # Read MEG file (verbose=False to suppress MNE output)
                raw = mne.io.read_raw_fif(meg_file, preload=False, verbose=False)

                # Extract measurement date
                meas_date = raw.info['meas_date']

                # Convert to datetime if needed (MNE may return different types)
                if hasattr(meas_date, 'timestamp'):
                    # datetime object
                    timestamp = meas_date
                elif isinstance(meas_date, (int, float)):
                    # Unix timestamp
                    timestamp = datetime.fromtimestamp(meas_date)
                else:
                    timestamp = meas_date

                timestamps.append({
                    'subject': subject,
                    'session': session,
                    'meas_date': timestamp
                })

                success_count += 1

            except FileNotFoundError:
                logger.warning(f"Missing MEG file: {meg_file}")
            except Exception as e:
                logger.error(f"Failed to read {meg_file}: {e}")

    logger.info(f"Successfully extracted {success_count}/{total_count} timestamps")

    return pd.DataFrame(timestamps)


def calculate_days_from_first(timestamps_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate days elapsed from first session for each subject.

    Parameters
    ----------
    timestamps_df : pd.DataFrame
        DataFrame with columns: subject, session, meas_date

    Returns
    -------
    pd.DataFrame
        DataFrame with added 'days_from_first' column
    """
    logger.info("Calculating days from first session per subject")

    # Sort by subject and session
    df = timestamps_df.sort_values(['subject', 'session']).copy()

    # Calculate days from first session per subject
    days_from_first = []
    for subject in df['subject'].unique():
        subject_data = df[df['subject'] == subject]
        first_date = subject_data['meas_date'].min()

        for idx, row in subject_data.iterrows():
            delta = row['meas_date'] - first_date
            days_from_first.append(delta.days)

    df['days_from_first'] = days_from_first

    return df


def plot_session_timeline(data_df: pd.DataFrame, output_dir: str,
                          dpi: int = 300, fmt: str = 'pdf'):
    """
    Create scatter plot showing session temporal spacing.

    Parameters
    ----------
    data_df : pd.DataFrame
        DataFrame with columns: subject, session, days_from_first
    output_dir : str
        Output directory for saving figure
    dpi : int, default=300
        Resolution for raster output
    fmt : str, default='png'
        Output format ('png', 'pdf', or 'both')
    """
    logger.info("Creating figure: Session temporal spacing")

    # Create figure
    plt.figure(figsize=(8, 5))

    # Scatter plot with session coloring
    # add a session time rank variable for coloring
    data_df['session_rank'] = data_df.groupby('subject')['days_from_first'].rank(method='dense').astype(int)

    scatter = plt.scatter(
        data_df['days_from_first'],
        data_df['subject'],
        c=data_df['session_rank'],
        cmap='magma',
        s=300,
        alpha=0.9,
        vmin=1,
        vmax=10,
        edgecolors='black',
        #linewidth=0
    
    )

    # Labels (lowercase)
    plt.xlabel('time from first session [days]')
    plt.ylabel('participant')

    # Y-axis formatting
    plt.ylim(0.5, 5.5)
    plt.yticks([1, 2, 3, 4, 5])

    # Add colorbar for session numbers
    cbar = plt.colorbar(scatter)
    cbar.set_label('session')

    # Apply despine (remove top and right spines)
    sns.despine()

    plt.tight_layout()

    # Save figure
    if fmt in ['png', 'both']:
        png_file = os.path.join(output_dir, 'session_timeline.png')
        plt.savefig(png_file, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved: {png_file}")

    if fmt in ['pdf', 'both']:
        pdf_file = os.path.join(output_dir, 'session_timeline.pdf')
        plt.savefig(pdf_file, format='pdf', bbox_inches='tight')
        logger.info(f"Saved: {pdf_file}")

    plt.close()


def report_statistics(data_df: pd.DataFrame, output_dir: str) -> None:
    """
    Compute and save session timing statistics.

    Reports per-subject total recording span and inter-session intervals,
    providing the M ± SD values cited in the manuscript.

    Saves
    -----
    source_data/session_timeline_source_data.csv  — per-session rows
    source_data/session_timeline_stats.txt        — stats report

    Parameters
    ----------
    data_df : pd.DataFrame
        DataFrame with columns: subject, session, meas_date, days_from_first
    output_dir : str
        Output directory (source_data/ subdirectory is created here)
    """
    df = data_df.sort_values(['subject', 'session']).copy()

    # Per-subject total span: days from session 1 to session 10
    subject_spans = df.groupby('subject')['days_from_first'].max()

    span_mean = subject_spans.mean()
    span_sd = subject_spans.std(ddof=1)
    span_min = subject_spans.min()
    span_max = subject_spans.max()

    # Inter-session intervals per subject (sorted by actual recording date)
    intervals = []
    for subject, grp in df.groupby('subject'):
        grp = grp.sort_values('days_from_first')
        diffs = grp['days_from_first'].diff().dropna()
        for d in diffs:
            intervals.append({'subject': subject, 'interval_days': d})
    intervals_df = pd.DataFrame(intervals)

    interval_median = intervals_df['interval_days'].median()
    interval_q1 = intervals_df['interval_days'].quantile(0.25)
    interval_q3 = intervals_df['interval_days'].quantile(0.75)
    interval_min = intervals_df['interval_days'].min()
    interval_max = intervals_df['interval_days'].max()

    lines = [
        'Session Timeline Statistics',
        '=' * 65,
        f"  subjects:      {sorted(df['subject'].unique().tolist())}",
        f"  n_subjects:    {df['subject'].nunique()}",
        f"  sessions:      {sorted(df['session'].unique().tolist())}",
        f"  date range:    {df['meas_date'].min()} — {df['meas_date'].max()}",
        '',
        'Total recording span (days from session 1 to session 10):',
        f"  {'subject':<10} {'span_days':>12}",
        '  ' + '-' * 24,
    ]
    for subj, span in subject_spans.items():
        lines.append(f"  {subj:<10} {span:>12.0f}")
    lines += [
        '',
        f"  mean ± SD:  {span_mean:.1f} ± {span_sd:.1f} days",
        f"  range:      {span_min:.0f} – {span_max:.0f} days",
        '',
        'Inter-session intervals (days between consecutive sessions):',
        f"  median [IQR]:  {interval_median:.1f} [{interval_q1:.1f}, {interval_q3:.1f}] days",
        f"  range:         {interval_min:.0f} – {interval_max:.0f} days",
        '',
        'Manuscript placeholder:',
        f"  [PLACEHOLDER: M ± SD] = {span_mean:.1f} ± {span_sd:.1f}",
    ]

    print('\n' + '\n'.join(lines) + '\n')

    # Save files
    source_data_dir = os.path.join(output_dir, 'source_data')
    os.makedirs(source_data_dir, exist_ok=True)

    source_cols = [c for c in ['subject', 'session', 'meas_date', 'days_from_first']
                   if c in df.columns]
    df[source_cols].to_csv(
        os.path.join(source_data_dir, 'session_timeline_source_data.csv'),
        index=False
    )

    txt_path = os.path.join(source_data_dir, 'session_timeline_stats.txt')
    with open(txt_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    logger.info(f"Source data saved to: {source_data_dir}")


def load_or_extract_timestamps(cache_file: str, data_path: str,
                               subjects: List[int], sessions: List[int],
                               recompute: bool = False, verbose: bool = False) -> pd.DataFrame:
    """
    Load timestamps from cache or extract from MEG files.

    Parameters
    ----------
    cache_file : str
        Path to cache CSV file
    data_path : str
        Base data directory path
    subjects : List[int]
        Subject IDs to process
    sessions : List[int]
        Session numbers to process
    recompute : bool, default=False
        Force re-extraction even if cache exists
    verbose : bool, default=False
        Enable verbose logging

    Returns
    -------
    pd.DataFrame
        DataFrame with timestamps and days_from_first
    """
    # Check if cache exists and we're not forcing recomputation
    if os.path.exists(cache_file) and not recompute:
        logger.info(f"Loading cached timestamps from: {cache_file}")
        try:
            df = pd.read_csv(cache_file, parse_dates=['meas_date'])
            logger.info(f"Loaded {len(df)} cached timestamps")
            return df
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}. Falling back to extraction.")

    # Extract timestamps from MEG files
    logger.info("Extracting timestamps from MEG raw files")
    timestamps_df = extract_session_timestamps(data_path, subjects, sessions, verbose)

    # Calculate days from first session
    data_df = calculate_days_from_first(timestamps_df)

    # Save to cache
    logger.info(f"Saving timestamps to cache: {cache_file}")
    data_df.to_csv(cache_file, index=False)

    return data_df


def main():
    """Command-line interface for session timeline visualization."""
    parser = argparse.ArgumentParser(
        description="Extract MEG session timestamps and visualize temporal spacing"
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
        help='Output directory for figures and cache (default: /share/klab/psulewski/psulewski/pyavs/dataset_info/)'
    )

    parser.add_argument(
        '--recompute',
        action='store_true',
        
        help='Force re-extraction of timestamps (ignore cache)'
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
        default=list(range(1, 11)),
        help='Sessions to include (default: 1 2 3 4 5 6 7 8 9 10)'
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

    logger.info("=" * 70)
    logger.info("MEG Session Timeline Visualization")
    logger.info("=" * 70)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")

    # Define cache file path
    cache_file = os.path.join(args.output_dir, 'meg_session_timestamps.csv')

    # Load or extract timestamps
    data_df = load_or_extract_timestamps(
        cache_file=cache_file,
        data_path=args.data_path,
        subjects=args.subjects,
        sessions=args.sessions,
        recompute=args.recompute,
        verbose=args.verbose
    )

    if len(data_df) == 0:
        logger.error("No timestamps extracted. Cannot generate plot.")
        return

    # Log summary statistics
    logger.info(f"Data summary:")
    logger.info(f"  Total sessions: {len(data_df)}")
    logger.info(f"  Subjects: {sorted(data_df['subject'].unique())}")
    logger.info(f"  Sessions per subject: {data_df.groupby('subject').size().to_dict()}")
    logger.info(f"  Date range: {data_df['meas_date'].min()} to {data_df['meas_date'].max()}")
    logger.info(f"  Max days from first: {data_df['days_from_first'].max()}")

    # Report statistics and save source data
    report_statistics(
        data_df=data_df,
        output_dir=args.output_dir,
    )

    # Create plot
    plot_session_timeline(
        data_df=data_df,
        output_dir=args.output_dir,
    )

    logger.info("=" * 70)
    logger.info("Session timeline visualization complete!")
    logger.info(f"Outputs saved to: {args.output_dir}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
