#!/usr/bin/env python3
"""
Create publication-quality visualizations of eye tracking calibration and drift correction quality.

This script generates minimal, concise figures suitable for a dataset paper,
showing calibration quality distributions and drift correction statistics
across all subjects and sessions.

STYLE PREFERENCES:
- Single plot figures only (no subplots)
- No plt.text() annotations
- Units always in square brackets []
- Never define fontsizes - use sns.poster defaults
- Minimal, clean design

Usage:
    python plot_et_quality_summary.py --derivatives-dir /path/to/derivatives

Author: P. Sulewski (psulewski@uos.de)
"""

import argparse
import os
import glob
from typing import Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger(__name__)


def load_et_quality_data(derivatives_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all ET quality CSVs and summary file.

    Parameters
    ----------
    derivatives_dir : str
        Path to ET quality derivatives directory containing per-session CSVs

    Returns
    -------
    calibration_df : pd.DataFrame
        All calibration events across all sessions
    drift_df : pd.DataFrame
        All drift correction events across all sessions
    summary_df : pd.DataFrame
        Per-session summary statistics

    Raises
    ------
    FileNotFoundError
        If no CSV files are found in derivatives directory
    """
    logger.info(f"Loading ET quality data from: {derivatives_dir}")

    # Find all per-session CSV files
    csv_pattern = os.path.join(derivatives_dir, 'sub-*_ses-*_et_quality.csv')
    csv_files = glob.glob(csv_pattern)

    if len(csv_files) == 0:
        raise FileNotFoundError(
            f"No ET quality CSV files found in {derivatives_dir}. "
            "Run extract_calibration_quality.py first."
        )

    logger.info(f"Found {len(csv_files)} per-session CSV files")

    # Load and concatenate all session files
    all_data = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

    # Split by event type
    cal_df = all_data[all_data['event_type'] == 'calibration'].copy()
    drift_df = all_data[all_data['event_type'] == 'drift_correction'].copy()

    logger.info(f"Loaded {len(cal_df)} calibration events")
    logger.info(f"Loaded {len(drift_df)} drift correction events")

    # Load summary file
    summary_file = os.path.join(derivatives_dir, 'all_subjects_et_quality_summary.csv')
    if os.path.exists(summary_file):
        summary_df = pd.read_csv(summary_file)
        logger.info(f"Loaded summary for {len(summary_df)} sessions")
    else:
        logger.warning(f"Summary file not found: {summary_file}")
        summary_df = pd.DataFrame()

    return cal_df, drift_df, summary_df


def plot_calibration_quality(cal_df: pd.DataFrame, output_dir: str,
                             dpi: int = 300, fmt: str = 'both'):
    """
    Create calibration quality overview violin plot.

    Parameters
    ----------
    cal_df : pd.DataFrame
        Calibration events dataframe
    output_dir : str
        Output directory for saving figures
    dpi : int, default=300
        Resolution for raster output
    fmt : str, default='both'
        Output format ('png', 'pdf', or 'both')
    """
    logger.info("Creating figure: Calibration quality overview")

    # Setup styling
    sns.set_context("poster")
    sns.set_style("white")

    # Create figure
    plt.figure(figsize=(8, 6))

    # Filter out unknown quality
    cal_plot = cal_df[cal_df['quality'].isin(['GOOD', 'FAIR', 'POOR'])].copy()

    # Violin plot
    sns.violinplot(
        data=cal_plot,
        x='quality',
        y='avg_error_deg',
        order=['GOOD', 'FAIR', 'POOR'],
        color='lightblue'
    )

    # Add threshold line
    plt.axhline(0.5, ls='--', color='red', linewidth=2, alpha=0.7)

    # Labels
    plt.ylabel('Average Calibration Error [°]')
    plt.xlabel('Calibration Quality')
    plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    # Save figure
    if fmt in ['png', 'both']:
        png_file = os.path.join(output_dir, 'calibration_quality.png')
        plt.savefig(png_file, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
        logger.info(f"Saved: {png_file}")

    if fmt in ['pdf', 'both']:
        pdf_file = os.path.join(output_dir, 'calibration_quality.pdf')
        plt.savefig(pdf_file, format='pdf', bbox_inches='tight', facecolor='white', edgecolor='none')
        logger.info(f"Saved: {pdf_file}")

    plt.close()


def plot_drift_histogram(drift_df: pd.DataFrame, output_dir: str,
                        dpi: int = 300, fmt: str = 'both'):
    """
    Create drift correction magnitude histogram.

    Parameters
    ----------
    drift_df : pd.DataFrame
        Drift correction events dataframe
    output_dir : str
        Output directory for saving figures
    dpi : int, default=300
        Resolution for raster output
    fmt : str, default='both'
        Output format ('png', 'pdf', or 'both')
    """
    logger.info("Creating figure: Drift correction histogram")

    # Setup styling
    sns.set_context("poster")
    sns.set_style("white")

    # Create figure
    plt.figure(figsize=(8, 6))

    # Get drift magnitudes
    drift_magnitudes = drift_df['offset_total_deg'].dropna()

    # Histogram
    plt.hist(drift_magnitudes, bins=20, range=(0, 2),
            color='steelblue', edgecolor='white', linewidth=0.5)

    # Add exclusion threshold line
    plt.axvline(1.0, ls='--', color='red', linewidth=2)

    # Labels
    plt.xlabel('Drift Correction Magnitude [°]')
    plt.ylabel('Frequency [count]')
    plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    # Save figure
    if fmt in ['png', 'both']:
        png_file = os.path.join(output_dir, 'drift_histogram.png')
        plt.savefig(png_file, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
        logger.info(f"Saved: {png_file}")

    if fmt in ['pdf', 'both']:
        pdf_file = os.path.join(output_dir, 'drift_histogram.pdf')
        plt.savefig(pdf_file, format='pdf', bbox_inches='tight', facecolor='white', edgecolor='none')
        logger.info(f"Saved: {pdf_file}")

    plt.close()


def plot_drift_cdf(drift_df: pd.DataFrame, output_dir: str,
                  dpi: int = 300, fmt: str = 'both'):
    """
    Create drift correction cumulative distribution.

    Parameters
    ----------
    drift_df : pd.DataFrame
        Drift correction events dataframe
    output_dir : str
        Output directory for saving figures
    dpi : int, default=300
        Resolution for raster output
    fmt : str, default='both'
        Output format ('png', 'pdf', or 'both')
    """
    logger.info("Creating figure: Drift correction CDF")

    # Setup styling
    sns.set_context("poster")
    sns.set_style("white")

    # Create figure
    plt.figure(figsize=(8, 6))

    # Get drift magnitudes
    drift_magnitudes = drift_df['offset_total_deg'].dropna()

    # Compute CDF
    sorted_drifts = np.sort(drift_magnitudes)
    cdf = np.arange(1, len(sorted_drifts) + 1) / len(sorted_drifts)

    # Plot CDF
    plt.plot(sorted_drifts, cdf, color='steelblue', linewidth=2.5)

    # Shade good range (< 0.5°)
    plt.axvspan(0, 0.5, alpha=0.2, color='green')

    # Add threshold line
    plt.axvline(1.0, ls='--', color='red', linewidth=2, alpha=0.7)

    # Labels
    plt.xlabel('Drift Correction Magnitude [°]')
    plt.ylabel('Cumulative Probability')
    plt.xlim(0, min(2.0, sorted_drifts.max()))
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)

    plt.tight_layout()

    # Save figure
    if fmt in ['png', 'both']:
        png_file = os.path.join(output_dir, 'drift_cdf.png')
        plt.savefig(png_file, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
        logger.info(f"Saved: {png_file}")

    if fmt in ['pdf', 'both']:
        pdf_file = os.path.join(output_dir, 'drift_cdf.pdf')
        plt.savefig(pdf_file, format='pdf', bbox_inches='tight', facecolor='white', edgecolor='none')
        logger.info(f"Saved: {pdf_file}")

    plt.close()


def plot_session_heatmap(summary_df: pd.DataFrame, output_dir: str,
                        dpi: int = 300, fmt: str = 'both'):
    """
    Create per-session quality heatmap.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Per-session summary statistics dataframe
    output_dir : str
        Output directory for saving figures
    dpi : int, default=300
        Resolution for raster output
    fmt : str, default='both'
        Output format ('png', 'pdf', or 'both')
    """
    logger.info("Creating figure: Per-session quality heatmap")

    # Setup styling
    sns.set_context("poster")
    sns.set_style("white")

    # Create pivot table
    pivot_table = summary_df.pivot(index='subject', columns='session',
                                   values='avg_cal_error_deg')

    # Create figure
    plt.figure(figsize=(10, max(6, len(pivot_table) * 0.4)))

    # Create heatmap
    sns.heatmap(pivot_table, cmap='RdYlGn_r', vmin=0, vmax=1.0,
               cbar_kws={'label': 'Avg. Calibration Error [°]'},
               linewidths=0.5, linecolor='white',
               annot=True, fmt='.2f')

    plt.xlabel('Session')
    plt.ylabel('Subject')

    plt.tight_layout()

    # Save figure
    if fmt in ['png', 'both']:
        png_file = os.path.join(output_dir, 'session_heatmap.png')
        plt.savefig(png_file, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
        logger.info(f"Saved: {png_file}")

    if fmt in ['pdf', 'both']:
        pdf_file = os.path.join(output_dir, 'session_heatmap.pdf')
        plt.savefig(pdf_file, format='pdf', bbox_inches='tight', facecolor='white', edgecolor='none')
        logger.info(f"Saved: {pdf_file}")

    plt.close()


def generate_all_figures(derivatives_dir: str, output_dir: str,
                        include_heatmap: bool = False,
                        dpi: int = 300, fmt: str = 'both'):
    """
    Main function to generate all ET quality figures.

    Parameters
    ----------
    derivatives_dir : str
        Path to ET quality derivatives directory
    output_dir : str
        Output directory for saving figures
    include_heatmap : bool, default=False
        Whether to include optional per-session heatmap
    dpi : int, default=300
        Resolution for raster output
    fmt : str, default='both'
        Output format ('png', 'pdf', or 'both')
    """
    logger.info("=" * 70)
    logger.info("Generating ET Quality Summary Figures")
    logger.info("=" * 70)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Load data
    try:
        cal_df, drift_df, summary_df = load_et_quality_data(derivatives_dir)
    except FileNotFoundError as e:
        logger.error(str(e))
        return

    # Generate calibration quality plot
    if len(cal_df) > 0:
        plot_calibration_quality(cal_df, output_dir, dpi=dpi, fmt=fmt)
    else:
        logger.warning("No calibration data found. Skipping calibration plot.")

    # Generate drift histogram
    if len(drift_df) > 0:
        plot_drift_histogram(drift_df, output_dir, dpi=dpi, fmt=fmt)
    else:
        logger.warning("No drift correction data found. Skipping drift histogram.")

    # Generate drift CDF
    if len(drift_df) > 0:
        plot_drift_cdf(drift_df, output_dir, dpi=dpi, fmt=fmt)
    else:
        logger.warning("No drift correction data found. Skipping drift CDF.")

    # Generate session heatmap (optional)
    if include_heatmap and len(summary_df) > 0:
        plot_session_heatmap(summary_df, output_dir, dpi=dpi, fmt=fmt)
    elif include_heatmap:
        logger.warning("No summary data found. Skipping session heatmap.")

    logger.info("=" * 70)
    logger.info("Figure generation complete!")
    logger.info(f"Figures saved to: {output_dir}")
    logger.info("=" * 70)


def main():
    """Command-line interface for ET quality visualization."""
    parser = argparse.ArgumentParser(
        description="Generate publication-quality ET quality figures for dataset paper"
    )

    parser.add_argument(
        '--derivatives-dir', '-d',
        type=str,
        default='/share/klab/datasets/avs/derivatives/pyavs/et_quality',
        help='Path to ET quality derivatives directory (default: /share/klab/datasets/avs/derivatives/pyavs/et_quality)'
    )

    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='./figures',
        help='Output directory for figures (default: ./figures)'
    )

    parser.add_argument(
        '--include-heatmap',
        action='store_true',
        help='Include optional per-session quality heatmap'
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

    # Generate figures
    generate_all_figures(
        derivatives_dir=args.derivatives_dir,
        output_dir=args.output_dir,
        include_heatmap=args.include_heatmap,
        dpi=args.dpi,
        fmt=args.format
    )


if __name__ == "__main__":
    main()
