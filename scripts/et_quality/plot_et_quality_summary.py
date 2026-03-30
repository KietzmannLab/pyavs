#!/usr/bin/env python3
"""
Create publication-quality visualizations of eye tracking calibration and drift correction quality.

This script generates minimal, concise figures suitable for a dataset paper,
showing calibration quality distributions and drift correction statistics
across all subjects and sessions.

GENERATED FIGURES:
- calibration_quality.png/pdf - Violin plot of calibration errors by quality category
- calibration_avg_error_hist.png/pdf - Histogram + KDE of average calibration errors
- calibration_max_error_hist.png/pdf - Histogram + KDE of maximum calibration errors
- drift_histogram.png/pdf - Histogram of drift correction magnitudes
- drift_cdf.png/pdf - Cumulative distribution of drift corrections
- session_heatmap.png/pdf - Per-session quality heatmap (optional, with --include-heatmap)

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
from scipy.stats import bootstrap as scipy_bootstrap

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
    #sns.set_style("white")

    # Create figure
    plt.figure(figsize=(7,7))

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
    #plt.axhline(0.5, ls='--', color='red', linewidth=2, alpha=0.7)

    # Labels
    plt.ylabel('average 9-point calibration\nerror [°]')
    plt.xlabel('calibration quality')
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
    #sns.set_style("white")

    # Create figure
    plt.figure(figsize=(6,7))

    # Get drift magnitudes
    drift_magnitudes = drift_df['offset_total_deg'].dropna()

    # Histogram
    sns.histplot(drift_magnitudes, color='cornflowerblue',
                edgecolor='white', bins=20)

    # Add exclusion threshold line
    #plt.axvline(1.0, ls='--', color='red', linewidth=2)

    # Labels
    plt.xlabel('pre-scene drift correction\nmagnitude [°]')
    plt.ylabel('frequency [count]')
    #plt.grid(axis='y', alpha=0.3)
    # despine
    sns.despine()
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
    #sns.set_style("white")

    # Create figure
    plt.figure(figsize=(6,7))

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
    plt.xlabel('drift correction magnitude [°]')
    plt.ylabel('cumulative Probability')
    plt.xlim(0, min(2.0, sorted_drifts.max()))
    plt.ylim(0, 1)
    #plt.grid(alpha=0.3)
    sns.despine()
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


def plot_calibration_avg_error_hist(cal_df: pd.DataFrame, output_dir: str,
                                    dpi: int = 300, fmt: str = 'both'):
    """
    Create calibration average error histogram with KDE.

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
    logger.info("Creating figure: Calibration average error histogram with KDE")

    # Setup styling
    sns.set_context("poster")
  
    # Create figure
    plt.figure(figsize=(6,7))

    # Get average errors
    avg_errors = cal_df['avg_error_deg'].dropna()

    # Histogram with KDE
    sns.histplot(avg_errors, color='steelblue',
                edgecolor='white', bins=20)

    # Add threshold line
    #plt.axvline(0.5, ls='--', color='red', linewidth=2)

    # Labels
    plt.xlabel('average 9-point calibration\nerror [°]')
    plt.ylabel('frequency [count]')
    #plt.grid(axis='y', alpha=0.3)
    sns.despine()
    plt.tight_layout()

    # Save figure
    if fmt in ['png', 'both']:
        png_file = os.path.join(output_dir, 'calibration_avg_error_hist.png')
        plt.savefig(png_file, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
        logger.info(f"Saved: {png_file}")

    if fmt in ['pdf', 'both']:
        pdf_file = os.path.join(output_dir, 'calibration_avg_error_hist.pdf')
        plt.savefig(pdf_file, format='pdf', bbox_inches='tight', facecolor='white', edgecolor='none')
        logger.info(f"Saved: {pdf_file}")

    plt.close()


def plot_calibration_max_error_hist(cal_df: pd.DataFrame, output_dir: str,
                                    dpi: int = 300, fmt: str = 'both'):
    """
    Create calibration maximum error histogram with KDE.

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
    logger.info("Creating figure: Calibration maximum error histogram with KDE")

    # Setup styling
    sns.set_context("poster")
    #sns.set_style("white")

    # Create figure
    plt.figure(figsize=(6,7))

    # Get maximum errors
    max_errors = cal_df['max_error_deg'].dropna()

    # Histogram with KDE
    sns.histplot(max_errors, color='steelblue',
                edgecolor='white', bins=20)

    # Add threshold line
    #plt.axvline(1.0, ls='--', color='red', linewidth=2)

    # Labels
    plt.xlabel('maximum 9-point calibration\nerror [°]')
    plt.ylabel('frequency [count]')
    #plt.grid(axis='y', alpha=0.3)
    sns.despine()
    plt.tight_layout()

    # Save figure
    if fmt in ['png', 'both']:
        png_file = os.path.join(output_dir, 'calibration_max_error_hist.png')
        plt.savefig(png_file, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
        logger.info(f"Saved: {png_file}")

    if fmt in ['pdf', 'both']:
        pdf_file = os.path.join(output_dir, 'calibration_max_error_hist.pdf')
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
    #sns.set_style("white")

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



def report_et_quality_stats(
    cal_df: pd.DataFrame,
    drift_df: pd.DataFrame,
    output_dir: str,
) -> None:
    """
    Compute and save calibration accuracy and drift correction stats.

    CIs are bootstrapped BCa across subjects (biological replicates).
    Per-subject means are computed first; CI is then over those N=5 values.

    Saves
    -----
    source_data/et_quality_calibration_source_data.csv  — per-calibration-event rows
    source_data/et_quality_drift_source_data.csv        — per-drift-event rows
    source_data/et_quality_stats.txt                    — full stats report
    """
    N_BOOTSTRAP = 10_000

    def _bca(vals):
        res = scipy_bootstrap(
            (np.asarray(vals, dtype=float),), np.mean,
            n_resamples=N_BOOTSTRAP, confidence_level=0.95, method='BCa',
        )
        return res.confidence_interval.low, res.confidence_interval.high

    # Identify subject column (handles both 'subject' and 'subject_id')
    sub_col = 'subject' if 'subject' in cal_df.columns else 'subject_id'
    subjects = sorted(cal_df[sub_col].dropna().unique())
    n_subjects = len(subjects)

    # ------------------------------------------------------------------
    # Calibration: per-subject means → BCa CI across subjects
    # ------------------------------------------------------------------
    subj_cal = (
        cal_df.groupby(sub_col)[['avg_error_deg', 'max_error_deg']]
        .mean()
        .reindex(subjects)
    )

    ci_avg = _bca(subj_cal['avg_error_deg'].values)
    ci_max = _bca(subj_cal['max_error_deg'].values)

    # Pooled descriptives
    avg_vals = cal_df['avg_error_deg'].dropna()
    max_vals = cal_df['max_error_deg'].dropna()

    # Quality category counts per subject
    quality_rows = []
    for subj in subjects:
        subj_data = cal_df[cal_df[sub_col] == subj]
        total = len(subj_data)
        for cat in ['GOOD', 'FAIR', 'POOR']:
            n = (subj_data['quality'] == cat).sum() if 'quality' in subj_data.columns else 0
            quality_rows.append({
                'subject': subj,
                'quality': cat,
                'count': int(n),
                'percent': 100.0 * n / total if total > 0 else 0.0,
            })
    quality_df = pd.DataFrame(quality_rows)

    quality_summary = []
    for cat in ['GOOD', 'FAIR', 'POOR']:
        subj_pcts = quality_df[quality_df['quality'] == cat]['percent'].values
        if len(subj_pcts) >= 2:
            ci = _bca(subj_pcts)
        else:
            ci = (np.nan, np.nan)
        quality_summary.append({
            'quality': cat,
            'mean_pct': float(np.mean(subj_pcts)),
            'sd_pct': float(np.std(subj_pcts)),
            'ci_low': float(ci[0]),
            'ci_high': float(ci[1]),
            'total_count': int(quality_df[quality_df['quality'] == cat]['count'].sum()),
        })

    # ------------------------------------------------------------------
    # Drift: per-subject means → BCa CI across subjects
    # ------------------------------------------------------------------
    drift_sub_col = 'subject' if 'subject' in drift_df.columns else 'subject_id'
    subj_drift = (
        drift_df.groupby(drift_sub_col)['offset_total_deg']
        .mean()
        .reindex(subjects)
    )
    ci_drift_mean = _bca(subj_drift.values)

    drift_vals = drift_df['offset_total_deg'].dropna()
    pct_lt_05 = 100.0 * (drift_vals < 0.5).mean()
    pct_lt_10 = 100.0 * (drift_vals < 1.0).mean()

    # Per-subject % < 0.5° and < 1.0°
    subj_drift_lt05 = drift_df.groupby(drift_sub_col)['offset_total_deg'].apply(
        lambda x: 100.0 * (x.dropna() < 0.5).mean()
    ).reindex(subjects)
    subj_drift_lt10 = drift_df.groupby(drift_sub_col)['offset_total_deg'].apply(
        lambda x: 100.0 * (x.dropna() < 1.0).mean()
    ).reindex(subjects)
    ci_lt05 = _bca(subj_drift_lt05.values)
    ci_lt10 = _bca(subj_drift_lt10.values)

    # ------------------------------------------------------------------
    # Save source data CSVs
    # ------------------------------------------------------------------
    source_data_dir = os.path.join(output_dir, 'source_data')
    os.makedirs(source_data_dir, exist_ok=True)

    cal_cols = [c for c in [sub_col, 'session', 'quality', 'avg_error_deg', 'max_error_deg']
                if c in cal_df.columns]
    cal_df[cal_cols].to_csv(
        os.path.join(source_data_dir, 'et_quality_calibration_source_data.csv'), index=False
    )

    drift_cols = [c for c in [drift_sub_col, 'session', 'offset_total_deg']
                  if c in drift_df.columns]
    drift_df[drift_cols].to_csv(
        os.path.join(source_data_dir, 'et_quality_drift_source_data.csv'), index=False
    )

    # ------------------------------------------------------------------
    # Stats txt
    # ------------------------------------------------------------------
    lines = [
        'Eye Tracking Quality Stats',
        '=' * 65,
        'Configuration:',
        f'  subjects:       {list(subjects)}',
        f'  n_subjects:     {n_subjects}',
        f'  n_calibrations: {len(cal_df)}',
        f'  n_drift_events: {len(drift_df)}',
        f'  ci_method:      bootstrap BCa (n={N_BOOTSTRAP}) across subjects',
        f'  ci_unit:        per-subject mean → BCa CI over N={n_subjects} subjects',
        '',
        'Calibration accuracy:',
        '-' * 65,
        'Pooled distribution:',
        f'  avg error — median: {np.median(avg_vals):.3f}°  '
        f'IQR: [{np.percentile(avg_vals,25):.3f}, {np.percentile(avg_vals,75):.3f}]°  '
        f'mean: {avg_vals.mean():.3f}°  SD: {avg_vals.std():.3f}°',
        f'  max error — median: {np.median(max_vals):.3f}°  '
        f'IQR: [{np.percentile(max_vals,25):.3f}, {np.percentile(max_vals,75):.3f}]°  '
        f'mean: {max_vals.mean():.3f}°  SD: {max_vals.std():.3f}°',
        '',
        'Mean across subjects (BCa 95% CI over subjects):',
        f'  avg error: {subj_cal["avg_error_deg"].mean():.3f}°  '
        f'[{ci_avg[0]:.3f}, {ci_avg[1]:.3f}]°',
        f'  max error: {subj_cal["max_error_deg"].mean():.3f}°  '
        f'[{ci_max[0]:.3f}, {ci_max[1]:.3f}]°',
        '',
        'Quality category breakdown (mean % across subjects ± BCa 95% CI):',
        f"  {'quality':<8} {'total_n':>8} {'mean_%':>8} {'SD_%':>6} {'CI_low':>8} {'CI_high':>8}",
        '  ' + '-' * 48,
    ]
    for row in quality_summary:
        lines.append(
            f"  {row['quality']:<8} {row['total_count']:>8} "
            f"{row['mean_pct']:>8.1f} {row['sd_pct']:>6.1f} "
            f"{row['ci_low']:>8.1f} {row['ci_high']:>8.1f}"
        )

    lines += [
        '',
        'Drift correction magnitude:',
        '-' * 65,
        'Pooled distribution:',
        f'  median: {np.median(drift_vals):.3f}°  '
        f'IQR: [{np.percentile(drift_vals,25):.3f}, {np.percentile(drift_vals,75):.3f}]°  '
        f'mean: {drift_vals.mean():.3f}°  SD: {drift_vals.std():.3f}°',
        '',
        'Mean across subjects (BCa 95% CI over subjects):',
        f'  mean drift: {subj_drift.mean():.3f}°  '
        f'[{ci_drift_mean[0]:.3f}, {ci_drift_mean[1]:.3f}]°',
        '',
        'Coverage thresholds (mean % across subjects ± BCa 95% CI):',
        f'  < 0.5°: {subj_drift_lt05.mean():.1f}%  [{ci_lt05[0]:.1f}, {ci_lt05[1]:.1f}]%',
        f'  < 1.0°: {subj_drift_lt10.mean():.1f}%  [{ci_lt10[0]:.1f}, {ci_lt10[1]:.1f}]%',
        '',
        'Per-subject summary:',
        '-' * 65,
        f"  {'subject':<10} {'n_cal':>6} {'avg_cal_err':>12} {'max_cal_err':>12} "
        f"{'n_drift':>8} {'mean_drift':>12}",
        '  ' + '-' * 56,
    ]
    for subj in subjects:
        n_cal   = len(cal_df[cal_df[sub_col] == subj])
        n_drift = len(drift_df[drift_df[drift_sub_col] == subj])
        avg_e   = subj_cal.loc[subj, 'avg_error_deg'] if subj in subj_cal.index else np.nan
        max_e   = subj_cal.loc[subj, 'max_error_deg'] if subj in subj_cal.index else np.nan
        dr_m    = subj_drift.loc[subj] if subj in subj_drift.index else np.nan
        lines.append(
            f'  {subj:<10} {n_cal:>6} {avg_e:>12.3f} {max_e:>12.3f} '
            f'{n_drift:>8} {dr_m:>12.3f}'
        )

    txt_path = os.path.join(source_data_dir, 'et_quality_stats.txt')
    with open(txt_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    logger.info(f"ET quality stats saved: {txt_path}")


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

    # Export stats and source data
    if len(cal_df) > 0 and len(drift_df) > 0:
        report_et_quality_stats(cal_df, drift_df, output_dir)

    # Generate calibration quality plot
    if len(cal_df) > 0:
        plot_calibration_quality(cal_df, output_dir, dpi=dpi, fmt=fmt)
    else:
        logger.warning("No calibration data found. Skipping calibration plot.")

    # Generate calibration average error histogram
    if len(cal_df) > 0:
        plot_calibration_avg_error_hist(cal_df, output_dir, dpi=dpi, fmt=fmt)
    else:
        logger.warning("No calibration data found. Skipping average error histogram.")

    # Generate calibration maximum error histogram
    if len(cal_df) > 0:
        plot_calibration_max_error_hist(cal_df, output_dir, dpi=dpi, fmt=fmt)
    else:
        logger.warning("No calibration data found. Skipping maximum error histogram.")

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
